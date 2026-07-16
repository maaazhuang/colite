#include "colite/core/parallel.h"
#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <exception>
#include <memory>
#include <thread>
#include <mutex>
#include <vector>

#ifndef MIN
#define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

//#define USE_OPENMP
#define USE_THREAD

#ifdef USE_OPENMP
#undef USE_THREAD
#elif defined(USE_THREAD)
#undef USE_OPENMP
#endif

#if defined USE_OPENMP
#  define MV_PARALLEL_FRAMEWORK "openmp"
#elif defined USE_THREAD
#  define MV_PARALLEL_FRAMEWORK "thread"
#endif


#if defined USE_OPENMP
#include <omp.h>
static int numThreadsMax = omp_get_max_threads();
#endif


namespace parallel {

thread_local int currentThreadIndex = 0;

/* ================================   Range  ================================ */

Range::Range()
    : start(0), end(0) {}

Range::Range(int _start, int _end)
    : start(_start), end(_end) {}

int Range::size() const
{
    return end - start;
}

bool Range::empty() const
{
    return start == end;
}


/* ================================   ParallelLoopBody  ================================ */


static inline int mvRound(double value)
{
    return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

unsigned defaultNumberOfThreads()
{
    unsigned int detected = std::thread::hardware_concurrency();

    const unsigned int result = (unsigned int)std::max(1u, detected);

    return result;
}

ParallelLoopBody::~ParallelLoopBody() {}

class ParallelLoopBodyWrapperContext
{
public:
	ParallelLoopBodyWrapperContext(const ParallelLoopBody& _body, const Range& _r, double _nstripes)
	{
		body = &_body;
		wholeRange = _r;
		double len = wholeRange.end - wholeRange.start;
		nstripes = mvRound(_nstripes <= 0 ? len : MIN(MAX(_nstripes, 1.), len));
	}

	~ParallelLoopBodyWrapperContext() {}

	const ParallelLoopBody* body;
	Range wholeRange;
	int nstripes;

private:
	ParallelLoopBodyWrapperContext(const ParallelLoopBodyWrapperContext&); // disabled
	ParallelLoopBodyWrapperContext& operator=(const ParallelLoopBodyWrapperContext&); // disabled
};

class ParallelLoopBodyWrapper : public ParallelLoopBody
{
public:
    ParallelLoopBodyWrapper(ParallelLoopBodyWrapperContext& ctx_) :
        ctx(ctx_)
    {
    }
    ~ParallelLoopBodyWrapper()
    {
    }
    void operator()(const Range& sr) const override
    {
        Range r;
        Range wholeRange = ctx.wholeRange;
        int nstripes = ctx.nstripes;
        r.start = (int)(wholeRange.start +
            ((uint64_t)sr.start * (wholeRange.end - wholeRange.start) + nstripes / 2) / nstripes);
        r.end = sr.end >= nstripes ? wholeRange.end : (int)(wholeRange.start +
            ((uint64_t)sr.end * (wholeRange.end - wholeRange.start) + nstripes / 2) / nstripes);

        //{
        //    std::lock_guard<std::mutex> gd(global_mtx);
        //    std::cout << " r.start： " << r.start << std::endl;
        //    std::cout << " r.end： " << r.end << std::endl;
        //}

        (*ctx.body)(r);

        //try
        //{
        //    (*ctx.body)(r);
        //}

        //catch (const std::exception& e)
        //{
        //    ctx.recordException(e.what());
        //}

    }
    Range stripeRange() const { return Range(0, ctx.nstripes); }

protected:
    ParallelLoopBodyWrapperContext& ctx;
};

class ParallelLoopBodyLambdaWrapper : public ParallelLoopBody
{
private:
    std::function<void(const Range&)> m_functor;
public:
    inline
        ParallelLoopBodyLambdaWrapper(std::function<void(const Range&)> functor)
        : m_functor(functor)
    {
        // nothing
    }

    virtual void operator() (const Range& range) const override
    {
        m_functor(range);
    }
};


/* ================================   parallel_for_  ================================ */

int numThreads = -1;

static void parallel_for_impl(const Range& range, const ParallelLoopBody& body, double nstripes);

void parallel_for_(const Range& range, const ParallelLoopBody& body, double nstripes)
{
    if (range.empty())
        return;

    static std::atomic<bool> flagNestedParallelFor(false);
    bool isNotNestedRegion = !flagNestedParallelFor.load();
    if (isNotNestedRegion)
        isNotNestedRegion = !flagNestedParallelFor.exchange(true);
    if (isNotNestedRegion)
    {
        try
        {
            parallel_for_impl(range, body, nstripes);
            flagNestedParallelFor = false;
        }
        catch (...)
        {
            flagNestedParallelFor = false;
            throw;
        }
    }
    else // nested parallel_for_() calls are not parallelized
    {
        body(range);
    }
}

void parallel_for_(const Range& range, std::function<void(const Range&)> functor, double nstripes)
{
    parallel_for_(range, ParallelLoopBodyLambdaWrapper(functor), nstripes);
}

void parallel_for_thread(const Range& range, const ParallelLoopBody& body, double nstripes);

static void parallel_for_impl(const Range& range, const ParallelLoopBody& body, double nstripes)
{
    if (range.end - range.start > 1)
    {
        ParallelLoopBodyWrapperContext ctx(body, range, nstripes);
        ParallelLoopBodyWrapper pbody(ctx);
        Range stripeRange = pbody.stripeRange();
        if (stripeRange.end - stripeRange.start == 1)
        {
            body(range);
            return;
        }

#ifdef MV_PARALLEL_FRAMEWORK
#if defined USE_OPENMP

#pragma omp parallel for schedule(dynamic) num_threads(numThreads > 0 ? numThreads : numThreadsMax)
        //for (int i = stripeRange.start; i < stripeRange.end; ++i)
        //    pbody(Range(i, i + 1));

        for (int i = range.start; i < range.end; ++i)
            pbody(Range(i, i + 1));


#elif defined USE_THREAD

        // 适用于range与nstripe一致，或nstripe大于threadnum
        parallel_for_thread(pbody.stripeRange(), pbody, pbody.stripeRange().size());
        // 适用于nstripe设置为threadnum
        //parallel_for_thread(range, body, nstripes);

#else

#error You have hacked and compiling with unsupported parallel framework

#endif

        return;
#endif // MV_PARALLEL_FRAMEWORK
    }

    body(range);
}

/* ================================   ThreadPool  ================================ */

#ifdef USE_THREAD

class WorkerThread;
class ParallelJob;

class ThreadPool {
public:
    static ThreadPool& instance()
    {
        static ThreadPool* pool = new ThreadPool();
        return *pool;
    }

    void run(const Range& range, const ParallelLoopBody& body, double nstripes);
    std::size_t get_num_threads() const;
    void set_num_threads(unsigned num_threads);

private:
    friend class WorkerThread;
    friend class ParallelJob;

    ThreadPool()
        : num_threads_(defaultNumberOfThreads()) {
    }

    ~ThreadPool()
    {
        reconfigure(0);
    }

    ThreadPool(const ThreadPool&);
    ThreadPool& operator=(const ThreadPool&);

    void reconfigure(unsigned new_threads_count) {
        std::lock_guard<std::mutex> lock(mutex_);
        reconfigure_(new_threads_count);
    }
    bool reconfigure_(unsigned new_threads_count);

    unsigned configured_threads() const
    {
        return std::max(1u, num_threads_.load(std::memory_order_acquire));
    }

    static unsigned normalize_nstripes(const Range& range, double nstripes)
    {
        if (range.size() <= 0) {
            return 1;
        }
        if (nstripes <= 0.0) {
            return static_cast<unsigned>(range.size());
        }
        return std::max(1u, static_cast<unsigned>(nstripes));
    }

    void notify_job_complete()
    {
        std::lock_guard<std::mutex> lock(mutex_notify_);
        cond_thread_task_complete_.notify_all();
    }

    std::atomic<unsigned> num_threads_;
    mutable std::mutex mutex_;
    std::mutex mutex_notify_;
    std::condition_variable cond_thread_task_complete_;
    std::vector<std::shared_ptr<WorkerThread> > threads_;
    std::shared_ptr<ParallelJob> job_;
};

class ParallelJob {
public:
    ParallelJob(const ThreadPool& thread_pool,
        const Range& range,
        const ParallelLoopBody& body,
        unsigned nstripes,
        unsigned expected_worker_count)
        : thread_pool_(thread_pool),
        body_(body),
        range_(range),
        nstripes_(std::max(1u, nstripes)),
        expected_worker_count_(expected_worker_count),
        current_task_(0),
        active_thread_count_(0),
        completed_thread_count_(0),
        is_completed_(false),
        is_canceled_(false) {
    }

    unsigned execute(bool is_worker_thread) {
        unsigned executed_tasks = 0;
        const int task_count = range_.size();
        const unsigned configured = std::max(1u, thread_pool_.configured_threads());
        const unsigned remaining_multiplier = std::min(
            nstripes_, std::max(std::min(100u, configured * 4u), configured * 2u));

        for (;;) {
            if (is_canceled_.load(std::memory_order_acquire)) {
                break;
            }
            const int next = current_task_.load(std::memory_order_relaxed);
            if (next >= task_count) {
                break;
            }
            const int chunk_size = std::max(1, (task_count - next) / static_cast<int>(remaining_multiplier));

            const int begin = current_task_.fetch_add(chunk_size, std::memory_order_relaxed);
            if (begin >= task_count) {
                break;
            }
            const int end = std::min(task_count, begin + chunk_size);
            executed_tasks += static_cast<unsigned>(end - begin);

            try {
                body_(Range(range_.start + begin, range_.start + end));
            }
            catch (...) {
                set_exception(std::current_exception());
                is_canceled_.store(true, std::memory_order_release);
                break;
            }
            if (is_worker_thread && is_completed_.load(std::memory_order_acquire)) {
                break;
            }
        }
        return executed_tasks;
    }

    bool has_remaining_tasks() const
    {
        return current_task_.load(std::memory_order_acquire) < range_.size();
    }

    void on_worker_start()
    {
        active_thread_count_.fetch_add(1, std::memory_order_relaxed);
    }

    bool on_worker_finish()
    {
        const unsigned completed =
            static_cast<unsigned>(completed_thread_count_.fetch_add(1, std::memory_order_acq_rel) + 1);
        if (completed >= expected_worker_count_) {
            bool expected = false;
            return is_completed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
        }
        return false;
    }

    bool try_complete_from_main()
    {
        if (expected_worker_count_ != 0) {
            return false;
        }
        bool expected = false;
        return is_completed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
    }

    bool completed() const
    {
        return is_completed_.load(std::memory_order_acquire);
    }

    void rethrow_if_needed() const
    {
        std::lock_guard<std::mutex> lock(exception_mutex_);
        if (first_exception_) {
            std::rethrow_exception(first_exception_);
        }
    }

private:

    void set_exception(std::exception_ptr e)
    {
        std::lock_guard<std::mutex> lock(exception_mutex_);
        if (!first_exception_) {
            first_exception_ = e;
        }
    }

    const ThreadPool& thread_pool_;
    const ParallelLoopBody& body_;
    const Range range_;
    const unsigned nstripes_;
    const unsigned expected_worker_count_;

    std::atomic<int> current_task_;
    std::atomic<int> active_thread_count_;
    std::atomic<int> completed_thread_count_;
    std::atomic<bool> is_completed_;
    std::atomic<bool> is_canceled_;
    mutable std::mutex exception_mutex_;
    std::exception_ptr first_exception_;
};

class WorkerThread {
public:
    WorkerThread(ThreadPool& thread_pool, unsigned id)
        : thread_pool_(thread_pool),
        id_(id),
        stop_thread_(false),
        has_wake_signal_(false)
    {
        worker_ = std::thread(&WorkerThread::thread_body, this);
    }

    ~WorkerThread()
    {
        request_stop();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void assign_job(const std::shared_ptr<ParallelJob>& job)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = job;
            has_wake_signal_.store(true, std::memory_order_release);
        }
        cond_thread_wake_.notify_one();
    }

    void request_stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_thread_.store(true, std::memory_order_release);
            has_wake_signal_.store(true, std::memory_order_release);
        }
        cond_thread_wake_.notify_all();
    }

    void thread_body() {
        currentThreadIndex = static_cast<int>(id_) + 1;
        while (!stop_thread_.load(std::memory_order_acquire)) {

            std::shared_ptr<ParallelJob> local_job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                while (!has_wake_signal_.load(std::memory_order_acquire) &&
                    !stop_thread_.load(std::memory_order_acquire)) {
                    cond_thread_wake_.wait(lock);
                }
                if (stop_thread_.load(std::memory_order_acquire)) {
                    break;
                }
                local_job.swap(job_);
                has_wake_signal_.store(false, std::memory_order_release);
            }

            if (!local_job) {
                continue;
            }

            local_job->on_worker_start();
            local_job->execute(true);
            if (local_job->on_worker_finish()) {
                thread_pool_.notify_job_complete();
            }
        }
    }

private:
    ThreadPool& thread_pool_;
    unsigned id_;
    std::thread worker_;
    std::mutex mutex_;
    std::atomic<bool> stop_thread_;
    std::atomic<bool> has_wake_signal_;
    std::condition_variable cond_thread_wake_;
    std::shared_ptr<ParallelJob> job_;
};

bool ThreadPool::reconfigure_(unsigned new_threads_count)
{
    if (new_threads_count == threads_.size()) {
        return false;
    }

    if (new_threads_count < threads_.size()) {
        std::vector<std::shared_ptr<WorkerThread>> release_threads;
        release_threads.reserve(threads_.size() - new_threads_count);
        for (std::size_t i = new_threads_count; i < threads_.size(); ++i) {
            threads_[i]->request_stop();
            release_threads.push_back(threads_[i]);
        }
        threads_.resize(new_threads_count);
        release_threads.clear();
        return false;
    }

    for (std::size_t i = threads_.size(); i < new_threads_count; ++i) {
        threads_.push_back(std::shared_ptr<WorkerThread>(new WorkerThread(*this, static_cast<unsigned>(i))));
    }
    return false;
}

void ThreadPool::run(const Range& range, const ParallelLoopBody& body, double nstripes) {
    if (range.size() <= 0) {
        body(range);
        return;
    }
    const std::size_t requested_threads = get_num_threads();
    const bool enough_work = (range.size() * nstripes >= 2.0) || (range.size() > 1 && nstripes <= 0.0);

    if (requested_threads <= 1 || !enough_work) {
        body(range);
        return;
    }

    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        body(range);
        return;
    }
    if (job_) {
        lock.unlock();
        body(range);
        return;
    }

    const unsigned worker_count = requested_threads > 0 ? static_cast<unsigned>(requested_threads - 1) : 0u;
    reconfigure_(worker_count);
    const std::size_t workers_to_wake = std::min(static_cast<std::size_t>(range.size()), threads_.size());
    std::shared_ptr<ParallelJob> local_job(
        new ParallelJob(*this, range, body, normalize_nstripes(range, nstripes),
                        static_cast<unsigned>(workers_to_wake)));
    job_ = local_job;
    lock.unlock();

    for (std::size_t i = 0; i < workers_to_wake; ++i) {
        threads_[i]->assign_job(local_job);
    }

    local_job->execute(false);

    if (!local_job->try_complete_from_main()) {

        std::unique_lock<std::mutex> notify_lock(mutex_notify_);
        cond_thread_task_complete_.wait(notify_lock, [&local_job]() {
            return local_job->completed();
            });
    }

    {
        std::lock_guard<std::mutex> clear_lock(mutex_);
        if (job_ == local_job) {
            job_.reset();
        }
    }
    local_job->rethrow_if_needed();
}

std::size_t ThreadPool::get_num_threads() const
{
    return static_cast<std::size_t>(configured_threads());
}

void ThreadPool::set_num_threads(unsigned num_threads)
{
    const unsigned normalized = std::max(1u, num_threads);
    num_threads_.store(normalized, std::memory_order_release);
    if (normalized == 1u) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!job_) {
            reconfigure_(0);
        }
    }
}

void parallel_for_thread(const Range& range, const ParallelLoopBody& body, double nstripes)
{
    ThreadPool::instance().run(range, body, nstripes);
}

#endif


int getNumThreads(void)
{
    if (numThreads == 0)
        return 1;

#if defined USE_OPENMP

    return numThreads > 0 ? numThreads : numThreadsMax;

#elif defined USE_THREAD

    return static_cast<int>(ThreadPool::instance().get_num_threads());

#else

    return 1;

#endif
}

int getThreadIndex(void)
{
    return currentThreadIndex;
}

void setNumThreads(int threads_)
{
    int threads = (threads_ < 0)
        ? static_cast<int>(defaultNumberOfThreads())
        : std::max(threads_, 1);
    numThreads = threads;

#if defined USE_OPENMP
    return;

#elif defined USE_THREAD
    ThreadPool::instance().set_num_threads(static_cast<unsigned>(threads));
#endif
}

int getThreadNum()
{
#if defined USE_OPENMP
    return omp_get_thread_num();
#elif defined USE_THREAD
    return 0;
#else
    return 0;
#endif
}

}
