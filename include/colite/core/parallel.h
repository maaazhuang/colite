#ifndef __PARALLEL_H__
#define __PARALLEL_H__

#include <functional>
#include <type_traits>

namespace parallel {

class Range
{
public:
    Range();
    Range(int _start, int _end);
    int size() const;
    bool empty() const;

    int start, end;
};

class ParallelLoopBody
{
public:
    virtual ~ParallelLoopBody();
    virtual void operator() (const Range& range) const = 0;
};

void parallel_for_(const Range& range, const ParallelLoopBody& body, double nstripes = -1.);

void parallel_for_(const Range& range, std::function<void(const Range&)> functor, double nstripes = -1.);

template <typename Functor>
typename std::enable_if<
    !std::is_base_of<ParallelLoopBody, typename std::decay<Functor>::type>::value,
    void>::type
parallel_for_(const Range& range, const Functor& functor, double nstripes = -1.)
{
    class LambdaBody final : public ParallelLoopBody {
    public:
        explicit LambdaBody(const Functor& f) : functor_(f) {}
        void operator()(const Range& r) const override { functor_(r); }

    private:
        const Functor& functor_;
    };

    LambdaBody body(functor);
    parallel_for_(range, body, nstripes);
}

void setNumThreads(int n);

int getNumThreads();

int getThreadIndex();

} // namespace parallel

#endif
