#include "sift_opencv.hpp"
#include "colite/core/parallel.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <algorithm>
#include <chrono>
#include <climits>
#include <memory>
#include <vector>

#if defined(COLITE_ENABLE_AVX2)
#include <immintrin.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace colite {
namespace features {
namespace backends {
namespace {

/* ===== Constants matching OpenCV sift.simd.hpp ===== */
constexpr int   SIFT_DESCR_WIDTH      = 4;
constexpr int   SIFT_DESCR_HIST_BINS  = 8;
constexpr float SIFT_INIT_SIGMA       = 0.5f;
constexpr int   SIFT_IMG_BORDER       = 5;
constexpr int   SIFT_MAX_INTERP_STEPS = 5;
constexpr int   SIFT_ORI_HIST_BINS    = 36;
constexpr float SIFT_ORI_SIG_FCTR     = 1.5f;
constexpr float SIFT_ORI_RADIUS       = 4.5f;
constexpr float SIFT_ORI_PEAK_RATIO   = 0.8f;
constexpr float SIFT_DESCR_SCL_FCTR   = 3.f;
constexpr float SIFT_DESCR_MAG_THR    = 0.2f;
constexpr float SIFT_INT_DESCR_FCTR   = 512.f;
constexpr int   SIFT_FIXPT_SCALE      = 1;
constexpr float FAST_ATAN2_P1 = 0.9997878412794807f * 57.29577951308232f;
constexpr float FAST_ATAN2_P3 = -0.3258083974640975f * 57.29577951308232f;
constexpr float FAST_ATAN2_P5 = 0.1555786518463281f * 57.29577951308232f;
constexpr float FAST_ATAN2_P7 = -0.04432655554792128f * 57.29577951308232f;
using sift_wt = float;

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

inline float fastAtan2Degrees(float y, float x)
{
    // 与 OpenCV core::fastAtan2 相同的 7 次多项式近似。
    // SIFT 原版 OpenCV 路径在 orientation/descriptor 中也使用 fastAtan2，
    // 这里避免 CRT atan2f 热点，同时让数值行为更贴近 OpenCV。
    const float ax = std::fabs(x);
    const float ay = std::fabs(y);
    float c;
    float a;
    if (ax >= ay) {
        c = ay / (ax + FLT_EPSILON);
        const float c2 = c * c;
        a = (((FAST_ATAN2_P7 * c2 + FAST_ATAN2_P5) * c2 + FAST_ATAN2_P3) * c2 + FAST_ATAN2_P1) * c;
    } else {
        c = ax / (ay + FLT_EPSILON);
        const float c2 = c * c;
        a = 90.f - (((FAST_ATAN2_P7 * c2 + FAST_ATAN2_P5) * c2 + FAST_ATAN2_P3) * c2 + FAST_ATAN2_P1) * c;
    }
    if (x < 0.f)
        a = 180.f - a;
    if (y < 0.f)
        a = 360.f - a;
    if (a >= 360.f)
        a -= 360.f;
    return a;
}

inline int fastFloorFloat(float value)
{
    const int ivalue = (int)value;
    return ivalue - (ivalue > value);
}

/* ===== Improved Gaussian blur matching OpenCV precisely ===== */

/*
 * BORDER_REFLECT_101: reflects at boundary without repeating edge pixel.
 * Matches OpenCV's default border mode for GaussianBlur.
 */
inline int reflect101(int x, int max_val)
{
    if (max_val <= 1) return 0;
    // 周期折叠可一次处理任意大的卷积半径；最小金字塔层宽度小于核半径时，
    // 单次反射仍可能落在区间外，曾导致真实图像特征提取越界。
    const int period = 2 * max_val - 2;
    x %= period;
    if (x < 0) x += period;
    return x < max_val ? x : period - x;
}

/*
 * Build 1D Gaussian kernel matching OpenCV's createGaussianKernels.
 * Uses double precision (like OpenCV's softdouble) for accuracy.
 * Kernel size: ksize = round(sigma * 8 + 1) | 1 for float images.
 */
void buildGaussianKernel(std::vector<float> &kernel, float sigma)
{
    int ksize = (int)std::lround(sigma * 8.0 + 1.0) | 1;
    if (ksize < 1) ksize = 1;

    kernel.resize(ksize);
    int half = ksize / 2;
    double sum = 0.0;

    for (int i = 0; i < ksize; i++) {
        double x = (double)(i - half);
        double v = std::exp(-0.5 * (x * x) / (sigma * sigma));
        kernel[i] = (float)v;
        sum += v;
    }
    for (int i = 0; i < ksize; i++)
        kernel[i] = (float)(kernel[i] / sum);
}

struct GaussianWorkspace {
    // 高斯卷积的行缓冲提升到整次 SIFT 调用级别复用，避免每一层卷积都重新分配。
    // 这里保留 std::vector 是为了和现有 Image 生命周期一致；它不在并行任务内部分配。
    std::vector<float> temp;
    std::vector<std::vector<float>> bandTemps;

    float *ensureTemp(int count)
    {
        if ((int)temp.size() < count)
            temp.resize(static_cast<size_t>(count));
        return temp.data();
    }

    void prepareBandTemps(int threadCount)
    {
        if ((int)bandTemps.size() < threadCount)
            bandTemps.resize(static_cast<size_t>(threadCount));
    }

    float *ensureBandTemp(int slot, int count)
    {
        if (slot < 0)
            slot = 0;
        if (slot >= (int)bandTemps.size())
            slot = 0;
        std::vector<float> &buffer = bandTemps[static_cast<size_t>(slot)];
        if ((int)buffer.size() < count)
            buffer.resize(static_cast<size_t>(count));
        return buffer.data();
    }
};

/*
 * Ground Truth: scalar separable GaussianBlur. This path defines the expected
 * numeric behavior for the AVX2 implementation.
 */
void gaussianBlurScalar(float *__restrict dst, const float *__restrict src,
                        int w, int h, const std::vector<float> &kernel,
                        GaussianWorkspace &workspace)
{
    int ksize = (int)kernel.size();
    int half = ksize / 2;
    float *temp = workspace.ensureTemp(w * h);

    /* Row convolution (horizontal) */
    parallel::parallel_for_(parallel::Range(0, h), [&](const parallel::Range &rows) {
        for (int y = rows.start; y < rows.end; y++) {
            const float *row = src + y * w;
            for (int x = 0; x < w; x++) {
                double sum = 0.0;
                for (int k = -half; k <= half; k++) {
                    int xx = reflect101(x + k, w);
                    sum += (double)row[xx] * kernel[k + half];
                }
                temp[y * w + x] = (float)sum;
            }
        }
    });

    /* Column convolution (vertical) */
    parallel::parallel_for_(parallel::Range(0, h), [&](const parallel::Range &rows) {
        for (int y = rows.start; y < rows.end; y++) {
            float *dstRow = dst + y * w;
            for (int x = 0; x < w; x++) {
                double sum = 0.0;
                for (int k = -half; k <= half; k++) {
                    int yy = reflect101(y + k, h);
                    sum += (double)temp[yy * w + x] * kernel[k + half];
                }
                dstRow[x] = (float)sum;
            }
        }
    });
}

#if defined(COLITE_ENABLE_AVX2)
void convolveRowAvx2(float *__restrict dstRow, const float *__restrict srcRow,
                     int w, int half, const std::vector<float> &kernel,
                     const __m256 *__restrict coeffs)
{
    int x = 0;
    const int leftBorder = std::min(half, w);
    for (; x < leftBorder; x++) {
        double sum = 0.0;
        for (int k = -half; k <= half; k++) {
            int xx = reflect101(x + k, w);
            sum += (double)srcRow[xx] * kernel[k + half];
        }
        dstRow[x] = (float)sum;
    }

    const int vecEnd = std::max(leftBorder, w - half);
    for (; x + 16 <= vecEnd; x += 16) {
        // 同一行连续 16 个像素拆成两个 AVX2 累加器，提升 ILP 并降低循环控制开销。
        __m256 vsum0 = _mm256_setzero_ps();
        __m256 vsum1 = _mm256_setzero_ps();
        for (int k = -half; k <= half; k++) {
            const float *base = srcRow + x + k;
            __m256 v0 = _mm256_loadu_ps(base);
            __m256 v1 = _mm256_loadu_ps(base + 8);
            const __m256 c = coeffs[k + half];
            vsum0 = _mm256_add_ps(vsum0, _mm256_mul_ps(v0, c));
            vsum1 = _mm256_add_ps(vsum1, _mm256_mul_ps(v1, c));
        }
        _mm256_storeu_ps(dstRow + x, vsum0);
        _mm256_storeu_ps(dstRow + x + 8, vsum1);
    }
    for (; x + 8 <= vecEnd; x += 8) {
        __m256 vsum = _mm256_setzero_ps();
        for (int k = -half; k <= half; k++) {
            // 水平方向是连续内存访问，预广播 kernel 系数，避免热循环内重复 set1。
            __m256 v = _mm256_loadu_ps(srcRow + x + k);
            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(v, coeffs[k + half]));
        }
        _mm256_storeu_ps(dstRow + x, vsum);
    }

    for (; x < w; x++) {
        double sum = 0.0;
        for (int k = -half; k <= half; k++) {
            int xx = reflect101(x + k, w);
            sum += (double)srcRow[xx] * kernel[k + half];
        }
        dstRow[x] = (float)sum;
    }
}

void gaussianBlurAvx2Banded(float *__restrict dst, const float *__restrict src,
                            int w, int h, const std::vector<float> &kernel,
                            GaussianWorkspace &workspace,
                            const __m256 *__restrict coeffs)
{
    const int ksize = (int)kernel.size();
    const int half = ksize / 2;
    const int threadCount = std::max(1, parallel::getNumThreads());
    const int bandRows = (w >= 4096) ? 48 : 96;
    const int numBands = (h + bandRows - 1) / bandRows;
    const int maxTempRows = std::min(h + 2 * half, bandRows + 2 * half);

    workspace.prepareBandTemps(threadCount);

    parallel::parallel_for_(parallel::Range(0, numBands), [&](const parallel::Range &bands) {
        const int slot = parallel::getThreadIndex();
        float *bandTemp = workspace.ensureBandTemp(slot, maxTempRows * w);

        for (int band = bands.start; band < bands.end; band++) {
            const int y0 = band * bandRows;
            const int y1 = std::min(h, y0 + bandRows);
            const int firstY = y0 - half;
            const int tempRows = (y1 - y0) + 2 * half;

            for (int ty = 0; ty < tempRows; ty++) {
                const int srcY = reflect101(firstY + ty, h);
                convolveRowAvx2(bandTemp + ty * w, src + srcY * w,
                                w, half, kernel, coeffs);
            }

            for (int y = y0; y < y1; y++) {
                float *dstRow = dst + y * w;
                const int center = y - firstY;
                int x = 0;
                for (; x + 8 <= w; x += 8) {
                    __m256 vsum = _mm256_setzero_ps();
                    for (int k = -half; k <= half; k++) {
                        // bandTemp 只保留当前输出 band 需要的水平卷积行，
                        // 垂直聚合时这些行在短时间内复用，减少整图 temp 的缓存压力。
                        const float *row = bandTemp + (center + k) * w;
                        __m256 v = _mm256_loadu_ps(row + x);
                        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(v, coeffs[k + half]));
                    }
                    _mm256_storeu_ps(dstRow + x, vsum);
                }
                for (; x < w; x++) {
                    double sum = 0.0;
                    for (int k = -half; k <= half; k++)
                        sum += (double)bandTemp[(center + k) * w + x] * kernel[k + half];
                    dstRow[x] = (float)sum;
                }
            }
        }
    }, std::min(numBands, threadCount));
}

void gaussianBlurAvx2(float *__restrict dst, const float *__restrict src,
                      int w, int h, const std::vector<float> &kernel,
                      GaussianWorkspace &workspace)
{
    int ksize = (int)kernel.size();
    int half = ksize / 2;
    alignas(32) __m256 coeffsStack[64];
    std::vector<__m256> coeffsHeap;
    __m256 *coeffs = coeffsStack;
    if (ksize > 64) {
        coeffsHeap.resize(static_cast<size_t>(ksize));
        coeffs = coeffsHeap.data();
    }
    for (int i = 0; i < ksize; i++)
        coeffs[i] = _mm256_set1_ps(kernel[i]);

    if (h >= 64 && w >= 64) {
        gaussianBlurAvx2Banded(dst, src, w, h, kernel, workspace, coeffs);
        _mm256_zeroupper();
        return;
    }

    float *temp = workspace.ensureTemp(w * h);

    parallel::parallel_for_(parallel::Range(0, h), [&](const parallel::Range &rows) {
        for (int y = rows.start; y < rows.end; y++) {
            const float *row = src + y * w;
            float *tmpRow = temp + y * w;
            convolveRowAvx2(tmpRow, row, w, half, kernel, coeffs);
        }
    });

    parallel::parallel_for_(parallel::Range(0, h), [&](const parallel::Range &rows) {
        for (int y = rows.start; y < rows.end; y++) {
            float *dstRow = dst + y * w;
            int x = 0;
            for (; x + 8 <= w; x += 8) {
                __m256 vsum = _mm256_setzero_ps();
                for (int k = -half; k <= half; k++) {
                    int yy = reflect101(y + k, h);
                    // 垂直卷积按 x 连续 8 点并行，读取 row buffer，避免 gather。
                    __m256 v = _mm256_loadu_ps(temp + yy * w + x);
                    vsum = _mm256_add_ps(vsum, _mm256_mul_ps(v, coeffs[k + half]));
                }
                _mm256_storeu_ps(dstRow + x, vsum);
            }
            for (; x < w; x++) {
                double sum = 0.0;
                for (int k = -half; k <= half; k++) {
                    int yy = reflect101(y + k, h);
                    sum += (double)temp[yy * w + x] * kernel[k + half];
                }
                dstRow[x] = (float)sum;
            }
        }
    });

    _mm256_zeroupper();
}
#endif

/*
 * Separable GaussianBlur: rows first, then columns (matching OpenCV's FilterEngine order).
 * Uses BORDER_REFLECT_101.
 */
void gaussianBlur(float *__restrict dst, const float *__restrict src,
                  int w, int h, const std::vector<float> &kernel,
                  bool useAvx2, GaussianWorkspace &workspace)
{
#if defined(COLITE_ENABLE_AVX2)
    if (useAvx2 && w >= 8 && h >= 2) {
        gaussianBlurAvx2(dst, src, w, h, kernel, workspace);
        return;
    }
#else
    (void)useAvx2;
#endif
    gaussianBlurScalar(dst, src, w, h, kernel, workspace);
}

/* BORDER_REFLECT: reflect including the edge pixel. Used by OpenCV's warpAffine. */
inline int reflectBorder(int x, int max_val)
{
    if (max_val <= 1) return 0;
    // BORDER_REFLECT 的周期包含两个端点，使用取模避免极小图像上的多次反射。
    const int period = 2 * max_val;
    x %= period;
    if (x < 0) x += period;
    return x < max_val ? x : period - x - 1;
}

void doubleImagePrecise(const float *src, int sw, int sh, float *dst, int dw, int dh)
{
    parallel::parallel_for_(parallel::Range(0, dh), [&](const parallel::Range &rows) {
    for (int y = rows.start; y < rows.end; y++) {
        float sy = y * 0.5f;
        int sy0 = (int)std::floor(sy);
        int sy1 = sy0 + 1;
        float fy = sy - (float)sy0;

        for (int x = 0; x < dw; x++) {
            float sx = x * 0.5f;
            int sx0 = (int)std::floor(sx);
            int sx1 = sx0 + 1;
            float fx = sx - (float)sx0;

            float v00 = src[reflectBorder(sy0, sh) * sw + reflectBorder(sx0, sw)];
            float v01 = src[reflectBorder(sy0, sh) * sw + reflectBorder(sx1, sw)];
            float v10 = src[reflectBorder(sy1, sh) * sw + reflectBorder(sx0, sw)];
            float v11 = src[reflectBorder(sy1, sh) * sw + reflectBorder(sx1, sw)];

            dst[y * dw + x] = v00 * (1-fx)*(1-fy) + v01*fx*(1-fy) +
                              v10 * (1-fx)*fy    + v11*fx*fy;
        }
    }
    });
}

/* OpenCV INTER_LINEAR uses half-pixel coordinates for the default SIFT upscale. */
void doubleImageLinear(const float *src, int sw, int sh, float *dst, int dw, int dh)
{
    const size_t tempCount = static_cast<size_t>(dw) * static_cast<size_t>(sh);
    std::unique_ptr<float[]> temp(new float[tempCount]);

    parallel::parallel_for_(parallel::Range(0, sh), [&](const parallel::Range &rows) {
    for (int y = rows.start; y < rows.end; y++) {
        const float *srcRow = src + y * sw;
        float *dstRow = temp.get() + y * dw;
        for (int x = 0; x < dw; x++) {
            float fx = (x + 0.5f) * 0.5f - 0.5f;
            int sx = (int)std::floor(fx);
            fx -= sx;
            if (sx < 0) {
                sx = 0;
                fx = 0;
            } else if (sx >= sw - 1) {
                sx = sw - 1;
                fx = 0;
            }
            dstRow[x] = srcRow[sx] * (1.0f - fx) +
                        srcRow[std::min(sx + 1, sw - 1)] * fx;
        }
    }
    });

    parallel::parallel_for_(parallel::Range(0, dh), [&](const parallel::Range &rows) {
    for (int y = rows.start; y < rows.end; y++) {
        float fy = (y + 0.5f) * 0.5f - 0.5f;
        int sy = (int)std::floor(fy);
        fy -= sy;
        if (sy < 0) {
            sy = 0;
            fy = 0;
        } else if (sy >= sh - 1) {
            sy = sh - 1;
            fy = 0;
        }
        const float *row0 = temp.get() + sy * dw;
        const float *row1 = temp.get() + std::min(sy + 1, sh - 1) * dw;
        float *dstRow = dst + y * dw;
        for (int x = 0; x < dw; x++)
            dstRow[x] = row0[x] * (1.0f - fy) + row1[x] * fy;
    }
    });
}

/* Downsample by 2 (nearest neighbor, matching OpenCV's INTER_NEAREST). */
void downsample2(const float *src, int sw, int /*sh*/, float *dst, int dw, int dh)
{
    parallel::parallel_for_(parallel::Range(0, dh), [&](const parallel::Range &rows) {
    for (int y = rows.start; y < rows.end; y++) {
        int sy = y * 2;
        for (int x = 0; x < dw; x++) {
            dst[y * dw + x] = src[sy * sw + x * 2];
        }
    }
    });
}

/* Ground Truth: scalar DoG layer subtraction. */
void subtractImagesScalar(const float *__restrict a, const float *__restrict b,
                          float *__restrict dst, int count)
{
    for (int i = 0; i < count; i++)
        dst[i] = a[i] - b[i];
}

#if defined(COLITE_ENABLE_AVX2)
void subtractImagesAvx2(const float *__restrict a, const float *__restrict b,
                        float *__restrict dst, int count)
{
    int i = 0;
    for (; i + 32 <= count; i += 32) {
        // DoG 是三条连续流：两读一写。展开到 32 个 float 可减少分支/计数开销，
        // 轻量预取提前拉入后续 cache line，缓解 VTune 中的内存带宽压力。
        _mm_prefetch(reinterpret_cast<const char*>(a + i + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b + i + 128), _MM_HINT_T0);
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        __m256 a2 = _mm256_loadu_ps(a + i + 16);
        __m256 b2 = _mm256_loadu_ps(b + i + 16);
        __m256 a3 = _mm256_loadu_ps(a + i + 24);
        __m256 b3 = _mm256_loadu_ps(b + i + 24);
        _mm256_storeu_ps(dst + i, _mm256_sub_ps(a0, b0));
        _mm256_storeu_ps(dst + i + 8, _mm256_sub_ps(a1, b1));
        _mm256_storeu_ps(dst + i + 16, _mm256_sub_ps(a2, b2));
        _mm256_storeu_ps(dst + i + 24, _mm256_sub_ps(a3, b3));
    }
    for (; i + 8 <= count; i += 8) {
        // DoG 是逐元素连续内存差分，AVX2 一次处理 8 个 float，无跨 lane 依赖。
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(dst + i, _mm256_sub_ps(va, vb));
    }
    for (; i < count; i++)
        dst[i] = a[i] - b[i];
    _mm256_zeroupper();
}
#endif

void subtractImages(const float *__restrict a, const float *__restrict b,
                    float *__restrict dst, int count, bool useAvx2)
{
#if defined(COLITE_ENABLE_AVX2)
    if (useAvx2) {
        subtractImagesAvx2(a, b, dst, count);
        return;
    }
#else
    (void)useAvx2;
#endif
    subtractImagesScalar(a, b, dst, count);
}

/* ===== Image buffer helper ===== */
struct Image {
    std::unique_ptr<float[]> data;
    int width = 0, height = 0;

    Image() = default;
    Image(int w, int h)
        : data((w > 0 && h > 0)
                   ? new float[static_cast<size_t>(w) * static_cast<size_t>(h)]
                   : nullptr),
          width(w),
          height(h)
    {}

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&&) noexcept = default;
    Image& operator=(Image&&) noexcept = default;

    float *ptr() { return data.get(); }
    const float *ptr() const { return data.get(); }
    float &at(int x, int y) { return data[static_cast<size_t>(y) * width + x]; }
    float  at(int x, int y) const { return data[static_cast<size_t>(y) * width + x]; }
};

#if defined(COLITE_ENABLE_AVX2)
inline __m256 fastAtan2DegreesAvx2(__m256 y, __m256 x)
{
    const __m256 signMask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 ax = _mm256_and_ps(x, signMask);
    const __m256 ay = _mm256_and_ps(y, signMask);
    const __m256 eps = _mm256_set1_ps(FLT_EPSILON);
    const __m256 p1 = _mm256_set1_ps(FAST_ATAN2_P1);
    const __m256 p3 = _mm256_set1_ps(FAST_ATAN2_P3);
    const __m256 p5 = _mm256_set1_ps(FAST_ATAN2_P5);
    const __m256 p7 = _mm256_set1_ps(FAST_ATAN2_P7);
    const __m256 ninety = _mm256_set1_ps(90.f);
    const __m256 oneEighty = _mm256_set1_ps(180.f);
    const __m256 full = _mm256_set1_ps(360.f);

    const __m256 axGeAy = _mm256_cmp_ps(ax, ay, _CMP_GE_OQ);
    __m256 c0 = _mm256_div_ps(ay, _mm256_add_ps(ax, eps));
    __m256 c1 = _mm256_div_ps(ax, _mm256_add_ps(ay, eps));
    __m256 c02 = _mm256_mul_ps(c0, c0);
    __m256 c12 = _mm256_mul_ps(c1, c1);

    // 与标量 fastAtan2Degrees 使用同一多项式；这里只把 8 个采样点批量计算。
    __m256 a0 = _mm256_mul_ps(
        _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(p7, c02), p5), c02), p3), c02), p1),
        c0);
    __m256 a1 = _mm256_sub_ps(ninety, _mm256_mul_ps(
        _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(p7, c12), p5), c12), p3), c12), p1),
        c1));
    __m256 a = _mm256_blendv_ps(a1, a0, axGeAy);

    const __m256 zero = _mm256_setzero_ps();
    const __m256 xNeg = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    const __m256 yNeg = _mm256_cmp_ps(y, zero, _CMP_LT_OQ);
    a = _mm256_blendv_ps(a, _mm256_sub_ps(oneEighty, a), xNeg);
    a = _mm256_blendv_ps(a, _mm256_sub_ps(full, a), yNeg);
    const __m256 geFull = _mm256_cmp_ps(a, full, _CMP_GE_OQ);
    a = _mm256_sub_ps(a, _mm256_and_ps(geFull, full));
    return a;
}

inline void computeDescriptorGradients8(const Image &img, int c, int r,
                                        float *__restrict magOut,
                                        float *__restrict angleOut)
{
    const int w = img.width;
    const float *__restrict prev = img.ptr() + static_cast<size_t>(r - 1) * w + c;
    const float *__restrict curr = img.ptr() + static_cast<size_t>(r) * w + c;
    const float *__restrict next = img.ptr() + static_cast<size_t>(r + 1) * w + c;

    // descriptor 采样点在行内连续，dx/dy/magnitude/angle 可用 AVX2 批量计算；
    // 后续三线性投票仍保持标量 scatter，避免改变直方图累加顺序。
    const __m256 dx = _mm256_sub_ps(_mm256_loadu_ps(curr + 1),
                                    _mm256_loadu_ps(curr - 1));
    const __m256 dy = _mm256_sub_ps(_mm256_loadu_ps(prev),
                                    _mm256_loadu_ps(next));
    const __m256 mag = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx),
                                                    _mm256_mul_ps(dy, dy)));
    const __m256 angle = fastAtan2DegreesAvx2(dy, dx);
    _mm256_storeu_ps(magOut, mag);
    _mm256_storeu_ps(angleOut, angle);
}
#endif

/* ===== 3x3 linear system solver ===== */
bool solve3x3(const float A[9], const float b[3], float x[3])
{
    float a[9], bb[3];
    std::memcpy(a, A, 9 * sizeof(float));
    std::memcpy(bb, b, 3 * sizeof(float));

    /* LU decomposition with partial pivoting */
    for (int j = 0; j < 3; j++) {
        float maxabs = 0;
        int maxi = -1;
        for (int i = j; i < 3; i++) {
            float absa = std::fabs(a[i * 3 + j]);
            if (absa > maxabs) { maxabs = absa; maxi = i; }
        }
        if (maxabs < 1e-10f) return false;

        if (maxi != j) {
            for (int k = 0; k < 3; k++)
                std::swap(a[j * 3 + k], a[maxi * 3 + k]);
            std::swap(bb[j], bb[maxi]);
        }

        float piv = a[j * 3 + j];
        for (int i = j + 1; i < 3; i++) {
            float factor = a[i * 3 + j] / piv;
            for (int k = j; k < 3; k++)
                a[i * 3 + k] -= factor * a[j * 3 + k];
            bb[i] -= factor * bb[j];
        }
    }

    for (int i = 2; i >= 0; i--) {
        float sum = bb[i];
        for (int j = i + 1; j < 3; j++)
            sum -= a[i * 3 + j] * x[j];
        x[i] = sum / a[i * 3 + i];
    }
    return true;
}

/* ===== Compute orientation histogram (matching OpenCV's calcOrientationHist) ===== */
float calcOrientationHist(const Image &img, int cx, int cy, int radius,
                          float sigma, float *hist, int n)
{
    float expf_scale = -1.f / (2.f * sigma * sigma);
    float temphist[SIFT_ORI_HIST_BINS + 4] = {};
    float *th = temphist + 2;
    float nd360 = n / 360.f;

    for (int i = -radius; i <= radius; i++) {
        int y = cy + i;
        if (y <= 0 || y >= img.height - 1) continue;
        for (int j = -radius; j <= radius; j++) {
            int x = cx + j;
            if (x <= 0 || x >= img.width - 1) continue;

            float dx = img.at(x+1, y) - img.at(x-1, y);
            float dy = img.at(x, y-1) - img.at(x, y+1);
            float mag = std::sqrt(dx*dx + dy*dy);
            float angle = fastAtan2Degrees(dy, dx);
            float w = std::exp((float)(i*i + j*j) * expf_scale);
            int bin = (int)std::lround(nd360 * angle);
            if (bin >= n) bin -= n;
            if (bin < 0) bin += n;
            th[bin] += w * mag;
        }
    }

    /* Smooth with [1,4,6,4,1]/16 */
    th[-1] = th[n-1]; th[-2] = th[n-2];
    th[n] = th[0]; th[n+1] = th[1];

    for (int i = 0; i < n; i++) {
        hist[i] = (th[i-2] + th[i+2]) * (1.f/16.f) +
                  (th[i-1] + th[i+1]) * (4.f/16.f) +
                   th[i] * (6.f/16.f);
    }

    float maxval = hist[0];
    for (int i = 1; i < n; i++)
        maxval = std::max(maxval, hist[i]);

    return maxval;
}

/* ===== Subpixel refinement (matching OpenCV's adjustLocalExtrema) ===== */
bool adjustLocalExtrema(const std::vector<Image> &dog_pyr,
                        int nOctaveLayers,
                        KeyPoint &kpt, int octv,
                        int &layer, int &r, int &c,
                        float contrastThreshold,
                        float edgeThreshold, float sigma)
{
    const float img_scale = 1.f / (255 * SIFT_FIXPT_SCALE);
    const float deriv_scale = img_scale * 0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale * 0.25f;

    float xi = 0, xr = 0, xc = 0, contr = 0;
    int nDogPerOct = nOctaveLayers + 2;
    int ci = c, ri = r, li = layer;
    int iter;

    for (iter = 0; iter < SIFT_MAX_INTERP_STEPS; iter++) {
        int idx0 = octv * nDogPerOct + li;
        const Image &img  = dog_pyr[idx0];
        const Image &prev = dog_pyr[idx0 - 1];
        const Image &next = dog_pyr[idx0 + 1];
        int w = img.width;

        float dD[3];
        dD[0] = (img.at(ci+1, ri) - img.at(ci-1, ri)) * deriv_scale;
        dD[1] = (img.at(ci, ri+1) - img.at(ci, ri-1)) * deriv_scale;
        dD[2] = (next.at(ci, ri) - prev.at(ci, ri)) * deriv_scale;

        float center_twice = img.at(ci, ri) * 2;
        float dxx = (img.at(ci+1, ri) + img.at(ci-1, ri) - center_twice) * second_deriv_scale;
        float dyy = (img.at(ci, ri+1) + img.at(ci, ri-1) - center_twice) * second_deriv_scale;
        float dss = (next.at(ci, ri) + prev.at(ci, ri) - center_twice) * second_deriv_scale;
        float dxy = (img.at(ci+1, ri+1) - img.at(ci+1, ri-1) -
                     img.at(ci-1, ri+1) + img.at(ci-1, ri-1)) * cross_deriv_scale;
        float dxs = (next.at(ci+1, ri) - next.at(ci-1, ri) -
                     prev.at(ci+1, ri) + prev.at(ci-1, ri)) * cross_deriv_scale;
        float dys = (next.at(ci, ri+1) - next.at(ci, ri-1) -
                     prev.at(ci, ri+1) + prev.at(ci, ri-1)) * cross_deriv_scale;

        float H[9] = {dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss};
        float X[3];
        if (!solve3x3(H, dD, X)) return false;

        xi = -X[2]; xr = -X[1]; xc = -X[0];

        if (std::fabs(xi) < 0.5f && std::fabs(xr) < 0.5f && std::fabs(xc) < 0.5f)
            break;

        if (std::fabs(xi) > (float)(INT_MAX/3) ||
            std::fabs(xr) > (float)(INT_MAX/3) ||
            std::fabs(xc) > (float)(INT_MAX/3))
            return false;

        ci += (int)std::lround(xc);
        ri += (int)std::lround(xr);
        li += (int)std::lround(xi);

        if (li < 1 || li > nOctaveLayers ||
            ci < SIFT_IMG_BORDER || ci >= w - SIFT_IMG_BORDER ||
            ri < SIFT_IMG_BORDER || ri >= (int)img.height - SIFT_IMG_BORDER)
            return false;
    }

    if (iter >= SIFT_MAX_INTERP_STEPS) return false;

    /* Check contrast and edge */
    {
        int idx0 = octv * nDogPerOct + li;
        const Image &img  = dog_pyr[idx0];
        const Image &prev = dog_pyr[idx0 - 1];
        const Image &next = dog_pyr[idx0 + 1];

        float dD[3];
        dD[0] = (img.at(ci+1, ri) - img.at(ci-1, ri)) * deriv_scale;
        dD[1] = (img.at(ci, ri+1) - img.at(ci, ri-1)) * deriv_scale;
        dD[2] = (next.at(ci, ri) - prev.at(ci, ri)) * deriv_scale;

        float t = dD[0]*xc + dD[1]*xr + dD[2]*xi;
        contr = img.at(ci, ri) * img_scale + t * 0.5f;

        if (std::fabs(contr) * nOctaveLayers < contrastThreshold)
            return false;

        float center_twice = img.at(ci, ri) * 2.f;
        float dxx = (img.at(ci+1, ri) + img.at(ci-1, ri) - center_twice) * second_deriv_scale;
        float dyy = (img.at(ci, ri+1) + img.at(ci, ri-1) - center_twice) * second_deriv_scale;
        float dxy = (img.at(ci+1, ri+1) - img.at(ci+1, ri-1) -
                     img.at(ci-1, ri+1) + img.at(ci-1, ri-1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if (det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det)
            return false;
    }

    kpt.x = (ci + xc) * (float)(1 << octv);
    kpt.y = (ri + xr) * (float)(1 << octv);
    kpt.octave = octv + (li << 8) + ((int)std::lround((xi + 0.5f)*255) << 16);
    kpt.size = sigma * std::pow(2.f, (li + xi) / nOctaveLayers) * (float)(1 << octv) * 2;
    kpt.response = std::fabs(contr);

    c = ci; r = ri; layer = li;
    return true;
}

/* ===== Detect keypoints in scale space ===== */
void findScaleSpaceExtrema(const std::vector<Image> &gauss_pyr,
                            int nOctaves, int nOctaveLayers,
                            const std::vector<Image> &dog_pyr,
                            float contrastThreshold,
                            float edgeThreshold, float sigma,
                            std::vector<KeyPoint> &kpts)
{
    int nLayersPerOct = nOctaveLayers + 3;
    int nDogPerOct = nOctaveLayers + 2;

    float threshold = std::floor(0.5f * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);

    for (int o = 0; o < nOctaves; o++) {
        for (int i = 1; i <= nOctaveLayers; i++) {
            int idx = o * nDogPerOct + i;
            const Image &img  = dog_pyr[idx];
            const Image &prev = dog_pyr[idx - 1];
            const Image &next = dog_pyr[idx + 1];
            int w = img.width, h = img.height, step = w;
            std::vector<std::vector<KeyPoint>> rowKeypoints(h);

            parallel::parallel_for_(parallel::Range(SIFT_IMG_BORDER, h - SIFT_IMG_BORDER),
                [&](const parallel::Range &rows) {
            for (int r = rows.start; r < rows.end; r++) {
                std::vector<KeyPoint> &rowOut = rowKeypoints[static_cast<size_t>(r)];
                const float *currptr = img.ptr() + r * w;
                const float *prevptr = prev.ptr() + r * w;
                const float *nextptr = next.ptr() + r * w;

                for (int c = SIFT_IMG_BORDER; c < w - SIFT_IMG_BORDER; c++) {
                    float val = currptr[c];
                    if (std::fabs(val) <= threshold) continue;

                    bool calculate = false;

                    float _00=currptr[c-step-1], _01=currptr[c-step], _02=currptr[c-step+1];
                    float _10=currptr[c-1],                        _12=currptr[c+1];
                    float _20=currptr[c+step-1], _21=currptr[c+step], _22=currptr[c+step+1];

                    if (val > 0) {
                        float vmax = std::max({_00,_01,_02,_10,_12,_20,_21,_22});
                        if (val >= vmax) {
                            _00=prevptr[c-step-1]; _01=prevptr[c-step]; _02=prevptr[c-step+1];
                            _10=prevptr[c-1];                       _12=prevptr[c+1];
                            _20=prevptr[c+step-1]; _21=prevptr[c+step]; _22=prevptr[c+step+1];
                            vmax = std::max({_00,_01,_02,_10,_12,_20,_21,_22});
                            if (val >= vmax) {
                                _00=nextptr[c-step-1]; _01=nextptr[c-step]; _02=nextptr[c-step+1];
                                _10=nextptr[c-1];                       _12=nextptr[c+1];
                                _20=nextptr[c+step-1]; _21=nextptr[c+step]; _22=nextptr[c+step+1];
                                vmax = std::max({_00,_01,_02,_10,_12,_20,_21,_22});
                                if (val >= vmax) {
                                    calculate = (val >= std::max(prevptr[c], nextptr[c]));
                                }
                            }
                        }
                    } else {
                        float vmin = std::min({_00,_01,_02,_10,_12,_20,_21,_22});
                        if (val <= vmin) {
                            _00=prevptr[c-step-1]; _01=prevptr[c-step]; _02=prevptr[c-step+1];
                            _10=prevptr[c-1];                       _12=prevptr[c+1];
                            _20=prevptr[c+step-1]; _21=prevptr[c+step]; _22=prevptr[c+step+1];
                            vmin = std::min({_00,_01,_02,_10,_12,_20,_21,_22});
                            if (val <= vmin) {
                                _00=nextptr[c-step-1]; _01=nextptr[c-step]; _02=nextptr[c-step+1];
                                _10=nextptr[c-1];                       _12=nextptr[c+1];
                                _20=nextptr[c+step-1]; _21=nextptr[c+step]; _22=nextptr[c+step+1];
                                vmin = std::min({_00,_01,_02,_10,_12,_20,_21,_22});
                                if (val <= vmin) {
                                    calculate = (val <= std::min(prevptr[c], nextptr[c]));
                                }
                            }
                        }
                    }

                    if (calculate) {
                        KeyPoint kpt;
                        int r1 = r, c1 = c, layer = i;
                        if (!adjustLocalExtrema(dog_pyr, nOctaveLayers, kpt, o,
                                                layer, r1, c1,
                                                contrastThreshold, edgeThreshold, sigma))
                            continue;

                        float scl_octv = kpt.size * 0.5f / (float)(1 << o);
                        float hist[SIFT_ORI_HIST_BINS];
                        float omax = calcOrientationHist(
                            gauss_pyr[o * nLayersPerOct + layer],
                            c1, r1,
                            (int)std::lround(SIFT_ORI_RADIUS * scl_octv),
                            SIFT_ORI_SIG_FCTR * scl_octv,
                            hist, SIFT_ORI_HIST_BINS);

                        float mag_thr = omax * SIFT_ORI_PEAK_RATIO;
                        for (int j = 0; j < SIFT_ORI_HIST_BINS; j++) {
                            int l = j > 0 ? j - 1 : SIFT_ORI_HIST_BINS - 1;
                            int r2 = j < SIFT_ORI_HIST_BINS - 1 ? j + 1 : 0;

                            if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr) {
                                float bin = j + 0.5f * (hist[l] - hist[r2]) /
                                            (hist[l] - 2*hist[j] + hist[r2]);
                                if (bin < 0) bin += SIFT_ORI_HIST_BINS;
                                else if (bin >= SIFT_ORI_HIST_BINS) bin -= SIFT_ORI_HIST_BINS;
                                kpt.angle = 360.f - (360.f / SIFT_ORI_HIST_BINS) * bin;
                                if (std::fabs(kpt.angle - 360.f) < FLT_EPSILON)
                                    kpt.angle = 0.f;
                                rowOut.push_back(kpt);
                            }
                        }
                    }
                }
            }
            });
            for (int r = SIFT_IMG_BORDER; r < h - SIFT_IMG_BORDER; r++) {
                kpts.insert(kpts.end(), rowKeypoints[static_cast<size_t>(r)].begin(),
                            rowKeypoints[static_cast<size_t>(r)].end());
            }
        }
    }
}

/* ===== Compute SIFT descriptor (matching OpenCV's calcSIFTDescriptor BASELINE) ===== */
void calcSIFTDescriptor(const Image &img, float x, float y,
                         float ori, float scl,
                         float *dst,
                         bool vectorizeGradients = false)
{
    int pt_x = (int)std::lround(x);
    int pt_y = (int)std::lround(y);
    int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

    float cos_t = std::cos(ori * (float)(M_PI / 180.0));
    float sin_t = std::sin(ori * (float)(M_PI / 180.0));
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f / (d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    const float halfDescr = d * 0.5f - 0.5f;
    int radius = (int)std::lround(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    int w = img.width, h_img = img.height;
    radius = std::min(radius, (int)std::sqrt((float)(w*w + h_img*h_img)));

    cos_t /= hist_width;
    sin_t /= hist_width;

    int len_hist = (d+2)*(d+2)*(n+2);
    int len_ddn = d * d * n;
    float hist[(SIFT_DESCR_WIDTH + 2) * (SIFT_DESCR_WIDTH + 2) *
               (SIFT_DESCR_HIST_BINS + 2)] = {};

    for (int ri = -radius; ri <= radius; ri++) {
        const int r = pt_y + ri;
        float c_rot = (-radius) * cos_t - ri * sin_t;
        float r_rot = (-radius) * sin_t + ri * cos_t;
        int c = pt_x - radius;
        int ci = -radius;
        for (; ci <= radius;) {
#if defined(COLITE_ENABLE_AVX2)
            if (vectorizeGradients &&
                ci + 7 <= radius &&
                r > 0 && r < h_img - 1 &&
                c > 0 && c + 7 < w - 1) {
                alignas(32) float mags[8];
                alignas(32) float angles[8];
                computeDescriptorGradients8(img, c, r, mags, angles);

                float laneCRot = c_rot;
                float laneRRot = r_rot;
                for (int lane = 0; lane < 8; lane++) {
                    const float rbin = laneRRot + halfDescr;
                    const float cbin = laneCRot + halfDescr;
                    if (rbin > -1 && rbin < d && cbin > -1 && cbin < d) {
                        float mag = mags[lane];
                        float angle = angles[lane];
                        float obin = (angle - ori) * bins_per_rad;
                        float wgt = std::exp((laneCRot*laneCRot + laneRRot*laneRRot) * exp_scale);
                        mag *= wgt;

                        int r0 = fastFloorFloat(rbin);
                        int c0 = fastFloorFloat(cbin);
                        int o0 = fastFloorFloat(obin);
                        float dr = rbin - r0;
                        float dc = cbin - c0;
                        float dob = obin - o0;

                        if (o0 < 0) o0 += n;
                        if (o0 >= n) o0 -= n;

                        float v_r1 = mag * dr, v_r0 = mag - v_r1;
                        float v_rc11 = v_r1 * dc, v_rc10 = v_r1 - v_rc11;
                        float v_rc01 = v_r0 * dc, v_rc00 = v_r0 - v_rc01;
                        float v_rco111 = v_rc11 * dob, v_rco110 = v_rc11 - v_rco111;
                        float v_rco101 = v_rc10 * dob, v_rco100 = v_rc10 - v_rco101;
                        float v_rco011 = v_rc01 * dob, v_rco010 = v_rc01 - v_rco011;
                        float v_rco001 = v_rc00 * dob, v_rco000 = v_rc00 - v_rco001;

                        int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
                        hist[idx] += v_rco000;
                        hist[idx+1] += v_rco001;
                        hist[idx+(n+2)] += v_rco010;
                        hist[idx+(n+3)] += v_rco011;
                        hist[idx+(d+2)*(n+2)] += v_rco100;
                        hist[idx+(d+2)*(n+2)+1] += v_rco101;
                        hist[idx+(d+3)*(n+2)] += v_rco110;
                        hist[idx+(d+3)*(n+2)+1] += v_rco111;
                    }
                    laneCRot += cos_t;
                    laneRRot += sin_t;
                }

                c_rot += cos_t * 8.f;
                r_rot += sin_t * 8.f;
                c += 8;
                ci += 8;
                continue;
            }
#else
            (void)vectorizeGradients;
#endif
            const float rbin = r_rot + halfDescr;
            const float cbin = c_rot + halfDescr;

            if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < h_img - 1 && c > 0 && c < w - 1)
            {
                float dx = img.at(c+1, r) - img.at(c-1, r);
                float dy = img.at(c, r-1) - img.at(c, r+1);
                float mag = std::sqrt(dx*dx + dy*dy);
                float angle = fastAtan2Degrees(dy, dx);
                float obin = (angle - ori) * bins_per_rad;
                float wgt = std::exp((c_rot*c_rot + r_rot*r_rot) * exp_scale);
                mag *= wgt;

                int r0 = fastFloorFloat(rbin);
                int c0 = fastFloorFloat(cbin);
                int o0 = fastFloorFloat(obin);
                float dr = rbin - r0;
                float dc = cbin - c0;
                float dob = obin - o0;

                if (o0 < 0) o0 += n;
                if (o0 >= n) o0 -= n;

                float v_r1 = mag * dr, v_r0 = mag - v_r1;
                float v_rc11 = v_r1 * dc, v_rc10 = v_r1 - v_rc11;
                float v_rc01 = v_r0 * dc, v_rc00 = v_r0 - v_rc01;
                float v_rco111 = v_rc11 * dob, v_rco110 = v_rc11 - v_rco111;
                float v_rco101 = v_rc10 * dob, v_rco100 = v_rc10 - v_rco101;
                float v_rco011 = v_rc01 * dob, v_rco010 = v_rc01 - v_rco011;
                float v_rco001 = v_rc00 * dob, v_rco000 = v_rc00 - v_rco001;

                int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
                hist[idx] += v_rco000;
                hist[idx+1] += v_rco001;
                hist[idx+(n+2)] += v_rco010;
                hist[idx+(n+3)] += v_rco011;
                hist[idx+(d+2)*(n+2)] += v_rco100;
                hist[idx+(d+2)*(n+2)+1] += v_rco101;
                hist[idx+(d+3)*(n+2)] += v_rco110;
                hist[idx+(d+3)*(n+2)+1] += v_rco111;
            }
            c_rot += cos_t;
            r_rot += sin_t;
            c++;
            ci++;
        }
    }

    /* Finalize histogram, wrap orientation bins */
    float rawDst[SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS];
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for (int k = 0; k < n; k++)
                rawDst[(i*d + j)*n + k] = hist[idx+k];
        }
    }

    /* Normalize, threshold, normalize */
    float nrm2 = 0;
    for (int k = 0; k < len_ddn; k++)
        nrm2 += rawDst[k] * rawDst[k];

    float thr = std::sqrt(nrm2) * SIFT_DESCR_MAG_THR;

    nrm2 = 0;
    for (int k = 0; k < len_ddn; k++) {
        float val = std::min(rawDst[k], thr);
        rawDst[k] = val;
        nrm2 += val * val;
    }

    float nrm2_final = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

    for (int k = 0; k < len_ddn; k++) {
        dst[k] = std::min(std::max((float)std::nearbyint(rawDst[k] * nrm2_final), 0.0f), 255.0f);
    }
}

/* ===== Unpack octave ===== */
void unpackOctave(const KeyPoint &kpt, int &octave, int &layer, float &scale)
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

} // anonymous namespace

/* ===== Public API ===== */

Sift Sift::create(int nfeatures, int nOctaveLayers,
                       double contrastThreshold, double edgeThreshold,
                       double sigma, bool enablePreciseUpscale, bool useAvx2)
{
    Sift sift;
    sift.nfeatures = nfeatures;
    sift.nOctaveLayers = nOctaveLayers;
    sift.contrastThreshold = contrastThreshold;
    sift.edgeThreshold = edgeThreshold;
    sift.sigma = sigma;
    sift.enablePreciseUpscale = enablePreciseUpscale;
    sift.useAvx2 = useAvx2;
    return sift;
}

void Sift::detectAndCompute(const float *image, int width, int height,
                               std::vector<KeyPoint> &keypoints,
                               std::vector<float> &descriptors)
{
    const auto totalStart = Clock::now();
    lastProfile = SiftProfile{};

    keypoints.clear();
    descriptors.clear();

    int firstOctave = -1;

    /* Create initial image */
    float sig_diff = std::sqrt(std::max(sigma*sigma - SIFT_INIT_SIGMA*SIFT_INIT_SIGMA*4, 0.01));
    GaussianWorkspace gaussianWorkspace;
    std::vector<float> initialKernel;
    buildGaussianKernel(initialKernel, sig_diff);

    Image base;
    if (firstOctave < 0) {
        int dw = width * 2, dh = height * 2;
        std::unique_ptr<float[]> dbl(new float[static_cast<size_t>(dw) * static_cast<size_t>(dh)]);
        if (enablePreciseUpscale)
            doubleImagePrecise(image, width, height, dbl.get(), dw, dh);
        else
            doubleImageLinear(image, width, height, dbl.get(), dw, dh);

        base = Image(dw, dh);
        gaussianBlur(base.ptr(), dbl.get(), dw, dh, initialKernel, useAvx2, gaussianWorkspace);
    } else {
        base = Image(width, height);
        gaussianBlur(base.ptr(), image, width, height, initialKernel, useAvx2, gaussianWorkspace);
    }
    const auto initialEnd = Clock::now();

    /* Number of octaves */
    int nOctaves = std::max((int)(std::log(std::min(base.width, base.height)) / std::log(2.0) - 2), 1);
    nOctaves -= firstOctave; /* subtract -1 = add 1 */

    int nLayers = nOctaveLayers + 3;

    /* Precompute sigmas */
    std::vector<double> sig(nLayers);
    sig[0] = sigma;
    double k = std::pow(2.0, 1.0 / nOctaveLayers);
    for (int i = 1; i < nLayers; i++) {
        double sig_prev = std::pow(k, (double)(i-1)) * sigma;
        double sig_total = sig_prev * k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }
    std::vector<std::vector<float>> gaussianKernels(nLayers);
    for (int i = 1; i < nLayers; i++)
        buildGaussianKernel(gaussianKernels[i], (float)sig[i]);

    /* Build Gaussian pyramid */
    std::vector<Image> gpyr(nOctaves * nLayers);
    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < nLayers; i++) {
            if (o == 0 && i == 0) {
                gpyr[o*nLayers + i] = Image(base.width, base.height);
                std::memcpy(gpyr[o*nLayers + i].ptr(), base.ptr(),
                           base.width * base.height * sizeof(float));
            } else if (i == 0) {
                const Image &src = gpyr[(o-1)*nLayers + nOctaveLayers];
                int nw = src.width / 2, nh = src.height / 2;
                gpyr[o*nLayers + i] = Image(nw, nh);
                downsample2(src.ptr(), src.width, src.height,
                           gpyr[o*nLayers + i].ptr(), nw, nh);
            } else {
                const Image &src = gpyr[o*nLayers + i - 1];
                gpyr[o*nLayers + i] = Image(src.width, src.height);
                gaussianBlur(gpyr[o*nLayers + i].ptr(), src.ptr(),
                            src.width, src.height, gaussianKernels[i], useAvx2, gaussianWorkspace);
            }
        }
    }
    const auto gaussianEnd = Clock::now();

    /* Build DoG pyramid */
    int nDog = nOctaveLayers + 2;
    std::vector<Image> dogpyr(nOctaves * nDog);
    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < nDog; i++) {
            const Image &src1 = gpyr[o*nLayers + i];
            dogpyr[o*nDog + i] = Image(src1.width, src1.height);
        }
    }
    parallel::parallel_for_(parallel::Range(0, nOctaves * nDog),
        [&](const parallel::Range &range) {
        for (int idx = range.start; idx < range.end; idx++) {
            const int o = idx / nDog;
            const int i = idx - o * nDog;
            const Image &src1 = gpyr[o*nLayers + i];
            const Image &src2 = gpyr[o*nLayers + i + 1];
            subtractImages(src2.ptr(), src1.ptr(), dogpyr[idx].ptr(),
                           src1.width * src1.height, useAvx2);
        }
    });
    const auto dogEnd = Clock::now();

    /* Detect keypoints */
    findScaleSpaceExtrema(gpyr, nOctaves, nOctaveLayers,
                           dogpyr, (float)contrastThreshold,
                           (float)edgeThreshold, (float)sigma,
                           keypoints);
    const auto extremaEnd = Clock::now();

    std::sort(keypoints.begin(), keypoints.end(),
              [](const KeyPoint &a, const KeyPoint &b) {
                  if (a.x != b.x) return a.x < b.x;
                  if (a.y != b.y) return a.y < b.y;
                  if (a.size != b.size) return a.size > b.size;
                  if (a.angle != b.angle) return a.angle < b.angle;
                  if (a.response != b.response) return a.response > b.response;
                  return a.octave > b.octave;
              });
    keypoints.erase(
        std::unique(keypoints.begin(), keypoints.end(),
                    [](const KeyPoint &a, const KeyPoint &b) {
                        return a.x == b.x && a.y == b.y &&
                               a.size == b.size && a.angle == b.angle;
                    }),
        keypoints.end());

    /* Adjust for firstOctave < 0 */
    if (firstOctave < 0) {
        float scale = 1.f / (float)(1 << (-firstOctave));
        for (auto &kpt : keypoints) {
            kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
            kpt.x *= scale;
            kpt.y *= scale;
            kpt.size *= scale;
        }
    }

    /* Keep only best nfeatures */
    if (nfeatures > 0 && (int)keypoints.size() > nfeatures) {
        std::partial_sort(keypoints.begin(), keypoints.begin() + nfeatures,
                         keypoints.end(),
                         [](const KeyPoint &a, const KeyPoint &b) {
                             return a.response > b.response;
                         });
        keypoints.resize(nfeatures);
    }
    const auto postprocessEnd = Clock::now();

    /* Compute descriptors */
    int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    int descSize = d * d * n;
    descriptors.resize(keypoints.size() * descSize);

    parallel::parallel_for_(parallel::Range(0, static_cast<int>(keypoints.size())),
        [&](const parallel::Range &range) {
        for (int i = range.start; i < range.end; i++) {
            int octave, layer;
            float scale;
            unpackOctave(keypoints[static_cast<size_t>(i)], octave, layer, scale);

            layer = std::min(layer, nOctaveLayers + 2);
            octave = std::max(std::min(octave - firstOctave, nOctaves - 1), 0);

            float size = keypoints[static_cast<size_t>(i)].size * scale;
            float pt_x = keypoints[static_cast<size_t>(i)].x * scale;
            float pt_y = keypoints[static_cast<size_t>(i)].y * scale;
            float angle = 360.f - keypoints[static_cast<size_t>(i)].angle;
            if (std::fabs(angle - 360.f) < FLT_EPSILON) angle = 0.f;

            const Image &img = gpyr[octave * nLayers + layer];
            calcSIFTDescriptor(img, pt_x, pt_y, angle, size * 0.5f,
                              descriptors.data() + static_cast<size_t>(i) * descSize,
                              useAvx2);
        }
    });
    const auto descriptorEnd = Clock::now();

    lastProfile.valid = true;
    lastProfile.initialImageMs = elapsedMs(totalStart, initialEnd);
    lastProfile.gaussianPyramidMs = elapsedMs(initialEnd, gaussianEnd);
    lastProfile.dogPyramidMs = elapsedMs(gaussianEnd, dogEnd);
    lastProfile.extremaMs = elapsedMs(dogEnd, extremaEnd);
    lastProfile.postprocessMs = elapsedMs(extremaEnd, postprocessEnd);
    lastProfile.descriptorMs = elapsedMs(postprocessEnd, descriptorEnd);
    lastProfile.totalMs = elapsedMs(totalStart, descriptorEnd);
}

}  // namespace backends
}  // namespace features
}  // namespace colite
