#define COLITE_VLFEAT_IMPLEMENTATION 1
#include "sift_vlfeat.h"
#if defined(COLITE_VLFEAT_ENABLE_PARALLEL) && defined(__cplusplus)
#include "colite/core/parallel.h"
#include <vector>
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#if defined(COLITE_VLFEAT_ENABLE_AVX2)
#include <immintrin.h>
#endif

/* ===== Minimal macros replacing VLFeat ===== */
#define VL_MAX(a,b) (((a)>(b))?(a):(b))
#define VL_MIN(a,b) (((a)<(b))?(a):(b))
#define VL_SHIFT_LEFT(x,n) (((n)>=0)?((x)<<(n)):((x)>>(-(n))))
#define COLITE_VLFEAT_PAD_BY_ZERO       0x0
#define COLITE_VLFEAT_PAD_BY_CONTINUITY 0x1
#define COLITE_VLFEAT_PAD_MASK          0x3
#define COLITE_VLFEAT_TRANSPOSE         0x4

#if defined(_MSC_VER)
#define COLITE_RESTRICT __restrict
#else
#define COLITE_RESTRICT __restrict__
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_LN2
#define M_LN2 0.69314718055994530942
#endif

#define VL_EPSILON_F 1.19209290E-07F
#define VL_EPSILON_D 2.220446049250313e-16
#define VL_PI 3.141592653589793

#define log2(x) (log(x)/M_LN2)

static inline int vl_floor_d(double x);

static void colite_vlfeat_imconvcol_f_range(float *COLITE_RESTRICT dst,
                                            int dst_stride,
                                            const float *COLITE_RESTRICT src,
                                            int src_width, int src_height,
                                            int src_stride,
                                            const float *COLITE_RESTRICT filt,
                                            int filt_begin, int filt_end,
                                            int step, unsigned int flags,
                                            int x_begin, int x_end)
{
    int x;
    int transp = (flags & COLITE_VLFEAT_TRANSPOSE) != 0;
    int zeropad =
        (flags & COLITE_VLFEAT_PAD_MASK) == COLITE_VLFEAT_PAD_BY_ZERO;
    const float *filt_base = filt + filt_end - filt_begin;

    for (x = x_begin; x < x_end; ++x) {
        float *dstp = transp ? dst + x * dst_stride : dst + x;
        int y;

        for (y = 0; y < src_height; y += step) {
            float acc = 0;
            float v = 0, c;
            const float *srci;
            const float *filti;
            int stop;

            filti = filt_base;
            stop = filt_end - y;
            srci = src + x - stop * src_stride;

            if (stop > 0) {
                if (zeropad) {
                    v = 0;
                } else {
                    v = *(src + x);
                }
                while (filti > filt_base - stop) {
                    c = *filti--;
                    acc += v * c;
                    srci += src_stride;
                }
            }

            stop = filt_end -
                   (filt_begin > (y - src_height + 1)
                        ? filt_begin
                        : (y - src_height + 1)) +
                   1;
            while (filti > filt_base - stop) {
                v = *srci;
                c = *filti--;
                acc += v * c;
                srci += src_stride;
            }

            if (zeropad) v = 0;

            stop = filt_end - filt_begin + 1;
            while (filti > filt_base - stop) {
                c = *filti--;
                acc += v * c;
            }

            if (transp) {
                *dstp = acc; dstp += 1;
            } else {
                *dstp = acc; dstp += dst_stride;
            }
        }
    }
}

static void colite_vlfeat_imconvcol_f(float *COLITE_RESTRICT dst, int dst_stride,
                                      const float *COLITE_RESTRICT src,
                                      int src_width, int src_height,
                                      int src_stride,
                                      const float *COLITE_RESTRICT filt,
                                      int filt_begin, int filt_end, int step,
                                      unsigned int flags)
{
#if defined(COLITE_VLFEAT_ENABLE_PARALLEL) && defined(__cplusplus)
    if (src_width >= 64 && src_height >= 64) {
        /*
         * VLFeat 的列卷积在外层 x 维没有写共享；每个任务只写自己的输出列
         * 或转置后的输出行。并行化这一层可保持逐点累加顺序完全不变。
         */
        parallel::parallel_for_(parallel::Range(0, src_width),
            [&](const parallel::Range &range) {
                colite_vlfeat_imconvcol_f_range(dst, dst_stride, src,
                                                src_width, src_height, src_stride,
                                                filt, filt_begin, filt_end, step,
                                                flags, range.start, range.end);
            });
        return;
    }
#endif
    colite_vlfeat_imconvcol_f_range(dst, dst_stride, src,
                                    src_width, src_height, src_stride,
                                    filt, filt_begin, filt_end, step,
                                    flags, 0, src_width);
}

#if defined(COLITE_VLFEAT_ENABLE_AVX2)
static inline int clamp_int(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static void smooth_vertical_rowmajor_avx2(float *COLITE_RESTRICT tmp,
                                          const float *COLITE_RESTRICT src,
                                          int width, int height,
                                          const float *COLITE_RESTRICT kernel,
                                          int half_width,
                                          int row_begin, int row_end,
                                          int temp_row_origin)
{
    int y;
    for (y = row_begin; y < row_end; ++y) {
        float *COLITE_RESTRICT dst_row = tmp + (size_t)(y - temp_row_origin) * width;
        int j, x = 0;

        if (y >= half_width && y < height - half_width) {
            for (; x + 8 <= width; x += 8) {
                /* 内部行的每个 tap 都在合法图像范围内；直接计算地址可移除热循环中的边界 clamp，
                 * 仍按 VLFeat 的 tap 顺序累积；FMA 把乘加合并为一条指令减少依赖链，
                 * 允许最后一位 float 舍入差异，由特征匹配阈值回归验证。 */
                __m256 acc = _mm256_setzero_ps();
                for (j = -half_width; j <= half_width; ++j) {
                    const __m256 pix = _mm256_loadu_ps(src + (size_t)(y + j) * width + x);
                    const __m256 coeff = _mm256_set1_ps(kernel[half_width - j]);
                    acc = _mm256_fmadd_ps(pix, coeff, acc);
                }
                _mm256_storeu_ps(dst_row + x, acc);
            }

            for (; x < width; ++x) {
                float acc = 0.0f;
                for (j = -half_width; j <= half_width; ++j) {
                    acc += src[(size_t)(y + j) * width + x] * kernel[half_width - j];
                }
                dst_row[x] = acc;
            }
        } else {
            for (; x + 8 <= width; x += 8) {
                /* 边界行继续使用连续填充，保留原有 VLFeat 边界语义。 */
                __m256 acc = _mm256_setzero_ps();
                for (j = -half_width; j <= half_width; ++j) {
                    const int yy = clamp_int(y + j, 0, height - 1);
                    const __m256 pix = _mm256_loadu_ps(src + (size_t)yy * width + x);
                    const __m256 coeff = _mm256_set1_ps(kernel[half_width - j]);
                    acc = _mm256_fmadd_ps(pix, coeff, acc);
                }
                _mm256_storeu_ps(dst_row + x, acc);
            }

            for (; x < width; ++x) {
                float acc = 0.0f;
                for (j = -half_width; j <= half_width; ++j) {
                    const int yy = clamp_int(y + j, 0, height - 1);
                    acc += src[(size_t)yy * width + x] * kernel[half_width - j];
                }
                dst_row[x] = acc;
            }
        }
    }
}

static void smooth_horizontal_rowmajor_avx2(float *COLITE_RESTRICT dst,
                                            const float *COLITE_RESTRICT tmp,
                                            float *COLITE_RESTRICT dog_output,
                                            const float *COLITE_RESTRICT dog_base,
                                            int width, int height,
                                            const float *COLITE_RESTRICT kernel,
                                            int half_width,
                                            int row_begin, int row_end,
                                            int temp_row_origin)
{
    int y;
    const int interior_begin = half_width;
    const int interior_end = width - half_width;

    for (y = row_begin; y < row_end; ++y) {
        const float *COLITE_RESTRICT src_row = tmp + (size_t)(y - temp_row_origin) * width;
        float *COLITE_RESTRICT dst_row = dst + (size_t)y * width;
        int x, j;

        for (x = 0; x < width && x < interior_begin; ++x) {
            float acc = 0.0f;
            for (j = -half_width; j <= half_width; ++j) {
                const int xx = clamp_int(x + j, 0, width - 1);
                acc += src_row[xx] * kernel[half_width - j];
            }
            dst_row[x] = acc;
            if (dog_output)
                dog_output[(size_t)y * width + x] =
                    acc - dog_base[(size_t)y * width + x];
        }

        x = interior_begin;
        for (; x + 8 <= interior_end; x += 8) {
            /*
             * AVX2: 水平方向内部区域无需边界判断，每个 tap 是连续 load。
             * 按 VLFeat 的 tap 顺序进行 FMA 累积，以一条融合乘加指令缩短计算依赖链。
             */
            __m256 acc = _mm256_setzero_ps();
            for (j = -half_width; j <= half_width; ++j) {
                const __m256 pix = _mm256_loadu_ps(src_row + x + j);
                const __m256 coeff = _mm256_set1_ps(kernel[half_width - j]);
                acc = _mm256_fmadd_ps(pix, coeff, acc);
            }
            _mm256_storeu_ps(dst_row + x, acc);
            if (dog_output) {
                const __m256 base = _mm256_loadu_ps(dog_base + (size_t)y * width + x);
                _mm256_storeu_ps(dog_output + (size_t)y * width + x,
                                 _mm256_sub_ps(acc, base));
            }
        }

        for (; x < width; ++x) {
            float acc = 0.0f;
            for (j = -half_width; j <= half_width; ++j) {
                const int xx = clamp_int(x + j, 0, width - 1);
                acc += src_row[xx] * kernel[half_width - j];
            }
            dst_row[x] = acc;
            if (dog_output)
                dog_output[(size_t)y * width + x] =
                    acc - dog_base[(size_t)y * width + x];
        }
    }
}

static int ensure_band_temp_capacity(VlSiftFilt *f, size_t required)
{
    if (required <= (size_t)f->band_temp_capacity) return 1;
    if (required > (size_t)0x7fffffff) return 0;
    {
        float *expanded = (float *)realloc(f->band_temp, required * sizeof(float));
        if (!expanded) return 0;
        f->band_temp = expanded;
        f->band_temp_capacity = (int)required;
    }
    return 1;
}

static int vl_sift_smooth_rowmajor_avx2(VlSiftFilt *f,
                                        float *COLITE_RESTRICT output,
                                        float *COLITE_RESTRICT temp,
                                        const float *COLITE_RESTRICT input,
                                        float *COLITE_RESTRICT dog_output,
                                        int width, int height,
                                        const float *COLITE_RESTRICT kernel,
                                        int half_width)
{
    if (width <= 0 || height <= 0 || half_width <= 0) return 0;
    if (width < 16 || height < 16) return 0;

#if defined(COLITE_VLFEAT_ENABLE_PARALLEL) && defined(__cplusplus)
    if (output != input && height >= 64) {
        const int thread_count = VL_MAX(1, parallel::getNumThreads());
        const int band_rows = width >= 4096 ? 48 : 96;
        const int num_bands = (height + band_rows - 1) / band_rows;
        const size_t slot_size = (size_t)width;
        const size_t required = slot_size * (size_t)thread_count;

        if (ensure_band_temp_capacity(f, required)) {
            /*
             * band 只负责调度粒度；每个线程实际只保留一行垂直卷积结果并立即完成水平卷积。
             * 首层每线程工作集由约 1 MB 降到约 24 KB，使中间行停留在 L1/L2，减少回写和重读流量。
             * continuity 边界和 tap 累加顺序不变；仅在输入输出不同时启用，避免原地平滑覆盖相邻 band 的输入。
             */
            parallel::parallel_for_(parallel::Range(0, num_bands),
                [&](const parallel::Range &range) {
                    int band;
                    int slot = parallel::getThreadIndex();
                    if (slot < 0 || slot >= thread_count) slot = 0;
                    float *local_temp = f->band_temp + (size_t)slot * slot_size;

                    for (band = range.start; band < range.end; ++band) {
                        const int y0 = band * band_rows;
                        const int y1 = VL_MIN(height, y0 + band_rows);
                        int y;
                        for (y = y0; y < y1; ++y) {
                            smooth_vertical_rowmajor_avx2(local_temp, input, width, height,
                                                          kernel, half_width, y, y + 1, y);
                            smooth_horizontal_rowmajor_avx2(output, local_temp,
                                                            dog_output, input, width, height,
                                                            kernel, half_width, y, y + 1, y);
                        }
                    }
                }, VL_MIN(num_bands, thread_count));
            _mm256_zeroupper();
            return 1;
        }
    }

    parallel::parallel_for_(parallel::Range(0, height),
        [&](const parallel::Range &range) {
            smooth_vertical_rowmajor_avx2(temp, input, width, height,
                                          kernel, half_width,
                                          range.start, range.end, 0);
        });
    parallel::parallel_for_(parallel::Range(0, height),
        [&](const parallel::Range &range) {
            smooth_horizontal_rowmajor_avx2(output, temp, dog_output, input,
                                            width, height,
                                            kernel, half_width,
                                            range.start, range.end, 0);
        });
#else
    smooth_vertical_rowmajor_avx2(temp, input, width, height,
                                  kernel, half_width, 0, height, 0);
    smooth_horizontal_rowmajor_avx2(output, temp, dog_output, input,
                                    width, height,
                                    kernel, half_width, 0, height, 0);
#endif
    _mm256_zeroupper();
    return 1;
}
#endif

/* ===== Fast exp(-x) table ===== */
#define EXPN_SZ  256
#define EXPN_MAX 25.0
static double expn_tab[EXPN_SZ+1];

static void fast_expn_init(void)
{
    int k;
    for (k = 0; k < EXPN_SZ + 1; ++k) {
        expn_tab[k] = exp(-(double)k * (EXPN_MAX / EXPN_SZ));
    }
}

static inline double fast_expn(double x)
{
    double a, b, r;
    int i;
    if (x > EXPN_MAX) return 0.0;
    x *= EXPN_SZ / EXPN_MAX;
    i = vl_floor_d(x);
    r = x - i;
    a = expn_tab[i];
    b = expn_tab[i + 1];
    return a + r * (b - a);
}

/* ===== Inline math helpers ===== */
static inline float vl_mod_2pi_f(float x)
{
    while (x > (float)(2 * VL_PI)) x -= (float)(2 * VL_PI);
    while (x < 0.0F) x += (float)(2 * VL_PI);
    return x;
}

static inline float vl_fast_resqrt_f(float x)
{
    union {
        float x;
        int i;
    } u;

    float xhalf = (float)0.5 * x;
    u.x = x;
    u.i = 0x5f3759df - (u.i >> 1);
    u.x = u.x * ((float)1.5 - xhalf*u.x*u.x);
    u.x = u.x * ((float)1.5 - xhalf*u.x*u.x);
    return u.x;
}

static inline float vl_fast_sqrt_f(float x)
{
    return (x < 1e-8f) ? 0.0f : x * vl_fast_resqrt_f(x);
}

static inline float vl_fast_atan2_f(float y, float x)
{
    float angle, r;
    float const c3 = 0.1821F;
    float const c1 = 0.9675F;
    float abs_y = fabsf(y) + VL_EPSILON_F;

    if (x >= 0) {
        r = (x - abs_y) / (x + abs_y);
        angle = (float)(VL_PI / 4);
    } else {
        r = (x + abs_y) / (abs_y - x);
        angle = (float)(3 * VL_PI / 4);
    }
    angle += (c3*r*r - c1) * r;
    return (y < 0) ? -angle : angle;
}

static inline float vl_abs_f(float x)
{
    return fabsf(x);
}

static inline double vl_abs_d(double x)
{
    return fabs(x);
}

static inline int vl_floor_f(float x)
{
    int xi = (int)x;
    if (x >= 0 || (float)xi == x) return xi;
    else return xi - 1;
}

static inline int vl_floor_d(double x)
{
    int xi = (int)x;
    if (x >= 0 || (double)xi == x) return xi;
    else return xi - 1;
}

/* ===== Original VLFeat separable Gaussian convolution ===== */

/*
 * Exact original VLFeat _vl_sift_smooth using the embedded column
 * convolution helper.
 * The two-pass approach: columns+transpose, then columns+transpose.
 * VL_TRANSPOSE makes the column convolution effectively operate on rows.
 */
static void vl_sift_smooth(VlSiftFilt *f,
                           float *output, float *temp,
                           const float *input,
                           float *dog_output,
                           int width, int height,
                           const float *kernel, int half_width)
{
#if defined(COLITE_VLFEAT_ENABLE_AVX2)
    if (vl_sift_smooth_rowmajor_avx2(f, output, temp, input, dog_output,
                                     width, height, kernel, half_width)) {
        return;
    }
#endif

    /* Pass 1: convolve columns of input, store transposed in temp */
    colite_vlfeat_imconvcol_f(temp, height,
                              input, width, height, width,
                              kernel, -half_width, half_width,
                              1, COLITE_VLFEAT_PAD_BY_CONTINUITY |
                                     COLITE_VLFEAT_TRANSPOSE);

    /* Pass 2: convolve columns of transposed temp, store transposed back */
    colite_vlfeat_imconvcol_f(output, width,
                              temp, height, width, height,
                              kernel, -half_width, half_width,
                              1, COLITE_VLFEAT_PAD_BY_CONTINUITY |
                                     COLITE_VLFEAT_TRANSPOSE);

    if (dog_output) {
        size_t i;
        const size_t count = (size_t)width * height;
        for (i = 0; i < count; ++i)
            dog_output[i] = output[i] - input[i];
    }
}

// dst and src intentionally alias after the first upsampled octave.  Marking
// either one restrict makes that in-place path undefined under GCC/Clang.
static void copy_and_upsample_2x(float *dst,
                                 float *COLITE_RESTRICT temp,
                                 const float *src,
                                 int width, int height)
{
    const int doubled_width = 2 * width;

    /*
     * Ground Truth 等价变换：先完成原实现第一次“行插值+转置”中的行插值，
     * 但保持为连续行布局；第二阶段再按同样的 0.5*(a+b) 顺序做垂直插值。
     */
#if defined(COLITE_VLFEAT_ENABLE_PARALLEL) && defined(__cplusplus)
    parallel::parallel_for_(parallel::Range(0, height),
        [&](const parallel::Range &range) {
            int y;
            for (y = range.start; y < range.end; ++y) {
                const float *src_row = src + (size_t)y * width;
                float *temp_row = temp + (size_t)y * doubled_width;
                int x;
                for (x = 0; x < width - 1; ++x) {
                    const float a = src_row[x];
                    const float b = src_row[x + 1];
                    temp_row[2 * x] = a;
                    temp_row[2 * x + 1] = 0.5f * (a + b);
                }
                temp_row[2 * width - 2] = src_row[width - 1];
                temp_row[2 * width - 1] = src_row[width - 1];
            }
        });

    parallel::parallel_for_(parallel::Range(0, height),
        [&](const parallel::Range &range) {
            int y;
            for (y = range.start; y < range.end; ++y) {
                const float *row0 = temp + (size_t)y * doubled_width;
                const float *row1 = temp + (size_t)VL_MIN(y + 1, height - 1) * doubled_width;
                float *dst0 = dst + (size_t)(2 * y) * doubled_width;
                float *dst1 = dst0 + doubled_width;
                int x = 0;
#if defined(COLITE_VLFEAT_ENABLE_AVX2)
                for (; x + 8 <= doubled_width; x += 8) {
                    const __m256 a = _mm256_loadu_ps(row0 + x);
                    const __m256 b = _mm256_loadu_ps(row1 + x);
                    _mm256_storeu_ps(dst0 + x, a);
                    _mm256_storeu_ps(dst1 + x,
                        _mm256_mul_ps(_mm256_add_ps(a, b), _mm256_set1_ps(0.5f)));
                }
#endif
                for (; x < doubled_width; ++x) {
                    dst0[x] = row0[x];
                    dst1[x] = 0.5f * (row0[x] + row1[x]);
                }
            }
        });
#else
    {
        int y, x;
        for (y = 0; y < height; ++y) {
            const float *src_row = src + (size_t)y * width;
            float *temp_row = temp + (size_t)y * doubled_width;
            for (x = 0; x < width - 1; ++x) {
                temp_row[2 * x] = src_row[x];
                temp_row[2 * x + 1] = 0.5f * (src_row[x] + src_row[x + 1]);
            }
            temp_row[2 * width - 2] = src_row[width - 1];
            temp_row[2 * width - 1] = src_row[width - 1];
        }
        for (y = 0; y < height; ++y) {
            const float *row0 = temp + (size_t)y * doubled_width;
            const float *row1 = temp + (size_t)VL_MIN(y + 1, height - 1) * doubled_width;
            float *dst0 = dst + (size_t)(2 * y) * doubled_width;
            float *dst1 = dst0 + doubled_width;
            for (x = 0; x < doubled_width; ++x) {
                dst0[x] = row0[x];
                dst1[x] = 0.5f * (row0[x] + row1[x]);
            }
        }
    }
#endif
#if defined(COLITE_VLFEAT_ENABLE_AVX2)
    _mm256_zeroupper();
#endif
}

/* ===== Internal: copy and downsample ===== */
static void copy_and_downsample(float *dst, const float *src,
                                int width, int height, int d)
{
    int x, y;
    d = 1 << d;
    for (y = 0; y < height; y += d) {
        const float *srcrowp = src + y * width;
        for (x = 0; x < width - (d - 1); x += d) {
            *dst++ = *srcrowp;
            srcrowp += d;
        }
    }
}

static VlSiftKeypoint *append_keypoint(VlSiftFilt *f)
{
    if (f->nkeys >= f->keys_res) {
        f->keys_res += 500;
        if (f->keys) {
            f->keys = (VlSiftKeypoint *)realloc(f->keys,
                f->keys_res * sizeof(VlSiftKeypoint));
        } else {
            f->keys = (VlSiftKeypoint *)malloc(
                f->keys_res * sizeof(VlSiftKeypoint));
        }
    }
    return f->keys + (f->nkeys++);
}

static int ensure_key_capacity(VlSiftFilt *f, int capacity)
{
    if (capacity <= f->keys_res) return 1;
    VlSiftKeypoint *keys = (VlSiftKeypoint *)realloc(
        f->keys, (size_t)capacity * sizeof(VlSiftKeypoint));
    if (!keys) return 0;
    f->keys = keys;
    f->keys_res = capacity;
    return 1;
}

#if defined(COLITE_VLFEAT_ENABLE_AVX2)
static inline int extrema_mask8_avx2(const float *pt, int yo, int so,
                                     float pos_thresh, float neg_thresh)
{
    const __m256 v = _mm256_loadu_ps(pt);
    __m256 gt = _mm256_cmp_ps(v, _mm256_set1_ps(pos_thresh), _CMP_GE_OQ);
    __m256 lt = _mm256_cmp_ps(v, _mm256_set1_ps(neg_thresh), _CMP_LE_OQ);

    /* 8 个相邻 x 位置共享同一组邻域偏移；逐个邻域载入后更新极大/极小 mask。 */
#define UPDATE_EXTREMA_MASK(OFF)                                      \
    do {                                                              \
        const __m256 n = _mm256_loadu_ps(pt + (OFF));                 \
        gt = _mm256_and_ps(gt, _mm256_cmp_ps(v, n, _CMP_GT_OQ));      \
        lt = _mm256_and_ps(lt, _mm256_cmp_ps(v, n, _CMP_LT_OQ));      \
    } while (0)

    UPDATE_EXTREMA_MASK(+1);
    UPDATE_EXTREMA_MASK(-1);
    UPDATE_EXTREMA_MASK(+so);
    UPDATE_EXTREMA_MASK(-so);
    UPDATE_EXTREMA_MASK(+yo);
    UPDATE_EXTREMA_MASK(-yo);

    UPDATE_EXTREMA_MASK(+yo + 1);
    UPDATE_EXTREMA_MASK(+yo - 1);
    UPDATE_EXTREMA_MASK(-yo + 1);
    UPDATE_EXTREMA_MASK(-yo - 1);

    UPDATE_EXTREMA_MASK(+1      + so);
    UPDATE_EXTREMA_MASK(-1      + so);
    UPDATE_EXTREMA_MASK(+yo     + so);
    UPDATE_EXTREMA_MASK(-yo     + so);
    UPDATE_EXTREMA_MASK(+yo + 1 + so);
    UPDATE_EXTREMA_MASK(+yo - 1 + so);
    UPDATE_EXTREMA_MASK(-yo + 1 + so);
    UPDATE_EXTREMA_MASK(-yo - 1 + so);

    UPDATE_EXTREMA_MASK(+1      - so);
    UPDATE_EXTREMA_MASK(-1      - so);
    UPDATE_EXTREMA_MASK(+yo     - so);
    UPDATE_EXTREMA_MASK(-yo     - so);
    UPDATE_EXTREMA_MASK(+yo + 1 - so);
    UPDATE_EXTREMA_MASK(+yo - 1 - so);
    UPDATE_EXTREMA_MASK(-yo + 1 - so);
    UPDATE_EXTREMA_MASK(-yo - 1 - so);

#undef UPDATE_EXTREMA_MASK

    return _mm256_movemask_ps(_mm256_or_ps(gt, lt));
}
#endif

#if defined(COLITE_VLFEAT_ENABLE_PARALLEL) && defined(__cplusplus)
static void scan_extrema_row(const float *dog,
                             int w, int h,
                             int s, int y, int s_min,
                             double peak_thresh,
                             std::vector<VlSiftKeypoint> &row_out)
{
    const int yo = w;
    const int so = w * h;
    int x = 1;

#if defined(COLITE_VLFEAT_ENABLE_AVX2)
    for (; x + 8 <= w - 1; x += 8) {
        const float *pt = dog + x + yo * y + so * (s - s_min);
        const int mask = extrema_mask8_avx2(
            pt, yo, so, (float)(+0.8 * peak_thresh), (float)(-0.8 * peak_thresh));
        int bit;
        for (bit = 0; bit < 8; ++bit) {
            if (mask & (1 << bit)) {
                VlSiftKeypoint candidate = {};
                candidate.ix = x + bit;
                candidate.iy = y;
                candidate.is = s;
                row_out.push_back(candidate);
            }
        }
    }
    _mm256_zeroupper();
#endif

    for (; x < w - 1; ++x) {
        const float *pt = dog + x + yo * y + so * (s - s_min);
        const float v = *pt;

#define CHECK_ROW_NEIGHBORS(CMP,SGN)                                  \
        ( v CMP ## = SGN 0.8 * peak_thresh &&                         \
          v CMP *(pt + 1) && v CMP *(pt - 1) &&                       \
          v CMP *(pt + so) && v CMP *(pt - so) &&                     \
          v CMP *(pt + yo) && v CMP *(pt - yo) &&                     \
          v CMP *(pt + yo + 1) && v CMP *(pt + yo - 1) &&             \
          v CMP *(pt - yo + 1) && v CMP *(pt - yo - 1) &&             \
          v CMP *(pt + 1 + so) && v CMP *(pt - 1 + so) &&             \
          v CMP *(pt + yo + so) && v CMP *(pt - yo + so) &&           \
          v CMP *(pt + yo + 1 + so) && v CMP *(pt + yo - 1 + so) &&   \
          v CMP *(pt - yo + 1 + so) && v CMP *(pt - yo - 1 + so) &&   \
          v CMP *(pt + 1 - so) && v CMP *(pt - 1 - so) &&             \
          v CMP *(pt + yo - so) && v CMP *(pt - yo - so) &&           \
          v CMP *(pt + yo + 1 - so) && v CMP *(pt + yo - 1 - so) &&   \
          v CMP *(pt - yo + 1 - so) && v CMP *(pt - yo - 1 - so) )

        if (CHECK_ROW_NEIGHBORS(>,+) || CHECK_ROW_NEIGHBORS(<,-)) {
            VlSiftKeypoint candidate = {};
            candidate.ix = x;
            candidate.iy = y;
            candidate.is = s;
            row_out.push_back(candidate);
        }
#undef CHECK_ROW_NEIGHBORS
    }
}
#endif

static int refine_candidate_keypoint(const VlSiftFilt *f,
                                     const VlSiftKeypoint *candidate,
                                     VlSiftKeypoint *out,
                                     double xper,
                                     int w, int h,
                                     int s_min, int s_max,
                                     int xo, int yo, int so,
                                     double te, double tp)
{
    float *dog = f->dog;
    int xk = candidate->ix;
    int yk = candidate->iy;
    int sk = candidate->is;

    double Dx=0,Dy=0,Ds=0,Dxx=0,Dyy=0,Dss=0,Dxy=0,Dxs=0,Dys=0;
    double A[3*3], b[3];

    int dx = 0, dy = 0;
    int iter, j, ii, jj;
    float *pt = NULL;

    for (iter = 0; iter < 5; ++iter) {
        xk += dx;
        yk += dy;

        pt = dog + xo * xk + yo * yk + so * (sk - s_min);

#define at(dx_,dy_,ds_) (*( pt + (dx_)*xo + (dy_)*yo + (ds_)*so))
#define Aat(i_,j_)     (A[(i_)+(j_)*3])

        Dx = 0.5 * (at(+1,0,0) - at(-1,0,0));
        Dy = 0.5 * (at(0,+1,0) - at(0,-1,0));
        Ds = 0.5 * (at(0,0,+1) - at(0,0,-1));

        Dxx = (at(+1,0,0) + at(-1,0,0) - 2.0 * at(0,0,0));
        Dyy = (at(0,+1,0) + at(0,-1,0) - 2.0 * at(0,0,0));
        Dss = (at(0,0,+1) + at(0,0,-1) - 2.0 * at(0,0,0));

        Dxy = 0.25 * (at(+1,+1,0) + at(-1,-1,0) - at(-1,+1,0) - at(+1,-1,0));
        Dxs = 0.25 * (at(+1,0,+1) + at(-1,0,-1) - at(-1,0,+1) - at(+1,0,-1));
        Dys = 0.25 * (at(0,+1,+1) + at(0,-1,-1) - at(0,-1,+1) - at(0,+1,-1));

        Aat(0,0) = Dxx;
        Aat(1,1) = Dyy;
        Aat(2,2) = Dss;
        Aat(0,1) = Aat(1,0) = Dxy;
        Aat(0,2) = Aat(2,0) = Dxs;
        Aat(1,2) = Aat(2,1) = Dys;

        b[0] = -Dx;
        b[1] = -Dy;
        b[2] = -Ds;

        /* 3x3 高斯消元保持原 VLFeat 的标量运算顺序；候选点之间彼此独立，可安全并行。 */
        for (j = 0; j < 3; ++j) {
            double maxa    = 0;
            double maxabsa = 0;
            int    maxi    = -1;
            double tmp;

            for (ii = j; ii < 3; ++ii) {
                double a = Aat(ii, j);
                double absa = vl_abs_d(a);
                if (absa > maxabsa) {
                    maxa    = a;
                    maxabsa = absa;
                    maxi    = ii;
                }
            }

            if (maxabsa < 1e-10) {
                b[0] = 0; b[1] = 0; b[2] = 0;
                break;
            }

            ii = maxi;

            for (jj = j; jj < 3; ++jj) {
                tmp = Aat(ii, jj); Aat(ii, jj) = Aat(j, jj); Aat(j, jj) = tmp;
                Aat(j, jj) /= maxa;
            }
            tmp = b[j]; b[j] = b[ii]; b[ii] = tmp;
            b[j] /= maxa;

            for (ii = j + 1; ii < 3; ++ii) {
                double xv = Aat(ii, j);
                for (jj = j; jj < 3; ++jj) {
                    Aat(ii, jj) -= xv * Aat(j, jj);
                }
                b[ii] -= xv * b[j];
            }
        }

        for (ii = 2; ii > 0; --ii) {
            double xv = b[ii];
            for (jj = ii - 1; jj >= 0; --jj) {
                b[jj] -= xv * Aat(jj, ii);
            }
        }

        dx = ((b[0] >  0.6 && xk < w - 2) ?  1 : 0)
           + ((b[0] < -0.6 && xk > 1    ) ? -1 : 0);

        dy = ((b[1] >  0.6 && yk < h - 2) ?  1 : 0)
           + ((b[1] < -0.6 && yk > 1    ) ? -1 : 0);

        if (dx == 0 && dy == 0) break;
    }

    {
        double val   = at(0,0,0) + 0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]);
        double score = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy);
        double xn = xk + b[0];
        double yn = yk + b[1];
        double sn = sk + b[2];

        int good =
            vl_abs_d(val)  > tp                  &&
            score           < (te+1)*(te+1)/te    &&
            score           >= 0                  &&
            vl_abs_d(b[0]) <  1.5                &&
            vl_abs_d(b[1]) <  1.5                &&
            vl_abs_d(b[2]) <  1.5                &&
            xn              >= 0                  &&
            xn              <= w - 1              &&
            yn              >= 0                  &&
            yn              <= h - 1              &&
            sn              >= s_min              &&
            sn              <= s_max;

        if (good) {
            out->o     = f->o_cur;
            out->ix    = xk;
            out->iy    = yk;
            out->is    = sk;
            out->s     = (float)sn;
            out->x     = (float)(xn * xper);
            out->y     = (float)(yn * xper);
            out->sigma = (float)(f->sigma0 * pow(2.0, sn/f->S) * xper);
            return 1;
        }
    }

#undef at
#undef Aat
    return 0;
}

/* ===== VlSiftFilt implementation ===== */

VlSiftFilt *vl_sift_new(int width, int height, int noctaves, int nlevels, int o_min)
{
    VlSiftFilt *f = (VlSiftFilt *)malloc(sizeof(VlSiftFilt));
    int w = VL_SHIFT_LEFT(width,  -o_min);
    int h = VL_SHIFT_LEFT(height, -o_min);
    int nel = w * h;
    int grad_layers = VL_MAX(nlevels, 1);

    if (noctaves < 0) {
        noctaves = VL_MAX((int)floor(log2(VL_MIN(width, height))) - o_min - 3, 1);
    }

    f->width   = width;
    f->height  = height;
    f->O       = noctaves;
    f->S       = nlevels;
    f->o_min   = o_min;
    f->s_min   = -1;
    f->s_max   = nlevels + 1;
    f->o_cur   = o_min;
    f->xper    = ldexp(1.0, o_min);

    f->temp    = (float *)malloc(sizeof(float) * nel);
    f->band_temp = NULL;
    f->band_temp_capacity = 0;
    f->octave  = (float *)malloc(sizeof(float) * nel * (f->s_max - f->s_min + 1));
    f->dog     = (float *)malloc(sizeof(float) * nel * (f->s_max - f->s_min));
    f->grad    = (float *)malloc(sizeof(float) * nel * 2 * grad_layers);

    f->sigman  = 0.5;
    f->sigmak  = pow(2.0, 1.0 / nlevels);
    f->sigma0  = 1.6 * f->sigmak;
    f->dsigma0 = f->sigma0 * sqrt(1.0 - 1.0 / (f->sigmak * f->sigmak));

    f->gaussFilter = NULL;
    f->gaussFilterSigma = 0;
    f->gaussFilterWidth = 0;

    f->octave_width  = 0;
    f->octave_height = 0;

    f->keys     = NULL;
    f->nkeys    = 0;
    f->keys_res = 0;

    f->peak_thresh = 0.0;
    f->edge_thresh = 10.0;
    f->norm_thresh = 0.0;
    f->magnif      = 3.0;
    f->windowSize  = VL_SIFT_NBP / 2;

    f->grad_o  = o_min - 1;
    f->grad_valid_mask = 0;

    fast_expn_init();

    return f;
}

void vl_sift_delete(VlSiftFilt *f)
{
    if (f) {
        free(f->keys);
        free(f->grad);
        free(f->dog);
        free(f->octave);
        free(f->temp);
        free(f->band_temp);
        free(f->gaussFilter);
        free(f);
    }
}

int colite_vl_sift_get_octave_index(const VlSiftFilt *f) { return f->o_cur; }
int colite_vl_sift_get_noctaves(const VlSiftFilt *f) { return f->O; }
int colite_vl_sift_get_octave_first(const VlSiftFilt *f) { return f->o_min; }
int colite_vl_sift_get_octave_width(const VlSiftFilt *f) { return f->octave_width; }
int colite_vl_sift_get_octave_height(const VlSiftFilt *f) { return f->octave_height; }
int colite_vl_sift_get_nlevels(const VlSiftFilt *f) { return f->S; }
int colite_vl_sift_get_nkeypoints(const VlSiftFilt *f) { return f->nkeys; }
double colite_vl_sift_get_peak_thresh(const VlSiftFilt *f) { return f->peak_thresh; }
double colite_vl_sift_get_edge_thresh(const VlSiftFilt *f) { return f->edge_thresh; }
double colite_vl_sift_get_norm_thresh(const VlSiftFilt *f) { return f->norm_thresh; }
double colite_vl_sift_get_magnif(const VlSiftFilt *f) { return f->magnif; }
double colite_vl_sift_get_window_size(const VlSiftFilt *f) { return f->windowSize; }
const VlSiftKeypoint *colite_vl_sift_get_keypoints(const VlSiftFilt *f) {
    return f->keys;
}
void colite_vl_sift_set_peak_thresh(VlSiftFilt *f, double value) {
    f->peak_thresh = value;
}
void colite_vl_sift_set_edge_thresh(VlSiftFilt *f, double value) {
    f->edge_thresh = value;
}
void colite_vl_sift_set_norm_thresh(VlSiftFilt *f, double value) {
    f->norm_thresh = value;
}
void colite_vl_sift_set_magnif(VlSiftFilt *f, double value) {
    f->magnif = value;
}
void colite_vl_sift_set_window_size(VlSiftFilt *f, double value) {
    f->windowSize = value;
}

/* Get pointer to octave data at level s */
static inline float *get_octave(const VlSiftFilt *f, int s)
{
    int w = f->octave_width;
    int h = f->octave_height;
    return f->octave + w * h * (s - f->s_min);
}

int vl_sift_process_first_octave(VlSiftFilt *f, const float *im)
{
    int o, s, h, w;
    double sa, sb;
    float *octave;

    float *temp   = f->temp;
    int width     = f->width;
    int height    = f->height;
    int o_min     = f->o_min;
    int s_min     = f->s_min;
    int s_max     = f->s_max;
    double sigma0 = f->sigma0;
    double sigmak = f->sigmak;
    double sigman = f->sigman;
    double dsigma0 = f->dsigma0;

    f->o_cur = o_min;
    f->xper = ldexp(1.0, o_min);
    f->nkeys = 0;
    w = f->octave_width  = VL_SHIFT_LEFT(f->width,  -f->o_cur);
    h = f->octave_height = VL_SHIFT_LEFT(f->height, -f->o_cur);

    if (f->O == 0)
        return VL_ERR_EOF;

    octave = get_octave(f, s_min);

    if (o_min < 0) {
        copy_and_upsample_2x(octave, temp, im, width, height);

        for (o = -1; o > o_min; --o) {
            copy_and_upsample_2x(octave, temp, octave,
                                 width << -o, height << -o);
        }
    }
    else if (o_min > 0) {
        copy_and_downsample(octave, im, width, height, o_min);
    }
    else {
        memcpy(octave, im, sizeof(float) * width * height);
    }

    /* Adjust initial smoothing */
    sa = sigma0 * pow(sigmak, s_min);
    sb = sigman * pow(2.0, -o_min);

    if (sa > sb) {
        double sd = sqrt(sa*sa - sb*sb);
        int ksize;
        float *kernel;
        int j;

        ksize = VL_MAX((int)ceil(4.0 * sd), 1);
        kernel = (float *)malloc(sizeof(float) * (2 * ksize + 1));
        {
            float acc = 0;
            for (j = 0; j < 2 * ksize + 1; ++j) {
                float d = (float)(j - ksize) / (float)sd;
                kernel[j] = (float)exp(-0.5 * (d*d));
                acc += kernel[j];
            }
            for (j = 0; j < 2 * ksize + 1; ++j) {
                kernel[j] /= acc;
            }
        }

        if (ksize > 0) {
            vl_sift_smooth(f, octave, temp, octave, NULL, w, h, kernel, ksize);
        }
        free(kernel);
    }

    /* Compute subsequent levels of the first octave */
    for (s = s_min + 1; s <= s_max; ++s) {
        double sd = dsigma0 * pow(sigmak, s);
        int ksize = VL_MAX((int)ceil(4.0 * sd), 1);
        float *kernel;
        int j;

        kernel = (float *)malloc(sizeof(float) * (2 * ksize + 1));
        {
            float acc = 0;
            for (j = 0; j < 2 * ksize + 1; ++j) {
                float d = (float)(j - ksize) / (float)sd;
                kernel[j] = (float)exp(-0.5 * (d*d));
                acc += kernel[j];
            }
            for (j = 0; j < 2 * ksize + 1; ++j) {
                kernel[j] /= acc;
            }
        }

        if (ksize > 0) {
            float *previous = get_octave(f, s - 1);
            float *current = get_octave(f, s);
            float *dog_level = f->dog + (size_t)(s - 1 - s_min) * w * h;
            vl_sift_smooth(f, current, temp, previous, dog_level,
                           w, h, kernel, ksize);
        } else {
            memcpy(get_octave(f, s), get_octave(f, s - 1), sizeof(float) * w * h);
        }
        free(kernel);
    }

    return VL_ERR_OK;
}

int vl_sift_process_next_octave(VlSiftFilt *f)
{
    int s, h, w, s_best;
    double sa, sb;
    float *octave, *pt;

    float *temp   = f->temp;
    int O         = f->O;
    int S         = f->S;
    int o_min     = f->o_min;
    int s_min     = f->s_min;
    int s_max     = f->s_max;
    double sigma0 = f->sigma0;
    double sigmak = f->sigmak;
    double dsigma0 = f->dsigma0;

    if (f->o_cur == o_min + O - 1)
        return VL_ERR_EOF;

    s_best = VL_MIN(s_min + S, s_max);
    w      = f->octave_width;
    h      = f->octave_height;
    pt     = get_octave(f, s_best);
    octave = get_octave(f, s_min);

    copy_and_downsample(octave, pt, w, h, 1);

    f->o_cur            += 1;
    f->xper              *= 2.0;
    f->nkeys             = 0;
    w = f->octave_width  = VL_SHIFT_LEFT(f->width,  -f->o_cur);
    h = f->octave_height = VL_SHIFT_LEFT(f->height, -f->o_cur);

    sa = sigma0 * pow(sigmak, s_min);
    sb = sigma0 * pow(sigmak, s_best - S);

    if (sa > sb) {
        double sd = sqrt(sa*sa - sb*sb);
        int ksize = VL_MAX((int)ceil(4.0 * sd), 1);
        float *kernel;
        int j;

        kernel = (float *)malloc(sizeof(float) * (2 * ksize + 1));
        {
            float acc = 0;
            for (j = 0; j < 2 * ksize + 1; ++j) {
                float d = (float)(j - ksize) / (float)sd;
                kernel[j] = (float)exp(-0.5 * (d*d));
                acc += kernel[j];
            }
            for (j = 0; j < 2 * ksize + 1; ++j) {
                kernel[j] /= acc;
            }
        }

        if (ksize > 0) {
            vl_sift_smooth(f, octave, temp, octave, NULL, w, h, kernel, ksize);
        }
        free(kernel);
    }

    for (s = s_min + 1; s <= s_max; ++s) {
        double sd = dsigma0 * pow(sigmak, s);
        int ksize = VL_MAX((int)ceil(4.0 * sd), 1);
        float *kernel;
        int j;

        kernel = (float *)malloc(sizeof(float) * (2 * ksize + 1));
        {
            float acc = 0;
            for (j = 0; j < 2 * ksize + 1; ++j) {
                float d = (float)(j - ksize) / (float)sd;
                kernel[j] = (float)exp(-0.5 * (d*d));
                acc += kernel[j];
            }
            for (j = 0; j < 2 * ksize + 1; ++j) {
                kernel[j] /= acc;
            }
        }

        if (ksize > 0) {
            float *previous = get_octave(f, s - 1);
            float *current = get_octave(f, s);
            float *dog_level = f->dog + (size_t)(s - 1 - s_min) * w * h;
            vl_sift_smooth(f, current, temp, previous, dog_level,
                           w, h, kernel, ksize);
        } else {
            memcpy(get_octave(f, s), get_octave(f, s - 1), sizeof(float) * w * h);
        }
        free(kernel);
    }

    return VL_ERR_OK;
}

void vl_sift_detect(VlSiftFilt *f)
{
    float *dog   = f->dog;
    int    s_min = f->s_min;
    int    s_max = f->s_max;
    int    w     = f->octave_width;
    int    h     = f->octave_height;
    double te    = f->edge_thresh;
    double tp    = f->peak_thresh;

    int const xo = 1;
    int const yo = w;
    int const so = w * h;

    double xper = f->xper;

    int x, y, s, i, ii, jj;
    float *pt, v;
    VlSiftKeypoint *k;

    f->nkeys = 0;
    if (f->keys_res == 0) {
        int expected = VL_MAX(4096, VL_MIN(w * h / 64, 262144));
        ensure_key_capacity(f, expected);
    }

    /* Find local maxima of DoG */
#if defined(COLITE_VLFEAT_ENABLE_PARALLEL) && defined(__cplusplus)
    {
        const int scale_count = s_max - s_min - 2;
        const int rows_per_scale = h - 2;
        const int task_count = scale_count * rows_per_scale;

        if (task_count >= 256) {
            std::vector<std::vector<VlSiftKeypoint>> row_candidates((size_t)task_count);

            /*
             * 每个任务只扫描一个 (scale,row)，写入独立 vector，因此无锁且无伪共享写入。
             * 最后按 task 索引合并，恢复 VLFeat 原始的 scale -> row -> x 候选顺序。
             */
            parallel::parallel_for_(parallel::Range(0, task_count),
                [&](const parallel::Range &range) {
                    int task;
                    for (task = range.start; task < range.end; ++task) {
                        const int scale_index = task / rows_per_scale;
                        const int row_index = task - scale_index * rows_per_scale;
                        const int ss = s_min + 1 + scale_index;
                        const int yy = 1 + row_index;
                        scan_extrema_row(dog, w, h, ss, yy, s_min, tp,
                                         row_candidates[(size_t)task]);
                    }
                });

            {
                size_t total = 0;
                int task;
                for (task = 0; task < task_count; ++task) {
                    total += row_candidates[(size_t)task].size();
                }
                if (total > (size_t)0x7fffffff ||
                    !ensure_key_capacity(f, (int)total)) {
                    f->nkeys = 0;
                    return;
                }
                for (task = 0; task < task_count; ++task) {
                    const std::vector<VlSiftKeypoint> &row = row_candidates[(size_t)task];
                    if (!row.empty()) {
                        memcpy(f->keys + f->nkeys, row.data(),
                               row.size() * sizeof(VlSiftKeypoint));
                        f->nkeys += (int)row.size();
                    }
                }
            }
        } else
#endif
        {
    for (s = s_min + 1; s <= s_max - 2; ++s) {
        for (y = 1; y < h - 1; ++y) {
            x = 1;
#if defined(COLITE_VLFEAT_ENABLE_AVX2)
            for (; x + 8 <= w - 1; x += 8) {
                int mask, bit;
                pt = dog + xo * x + yo * y + so * (s - s_min);
                mask = extrema_mask8_avx2(pt, yo, so,
                    (float)(+0.8 * tp), (float)(-0.8 * tp));
                for (bit = 0; bit < 8; ++bit) {
                    if (mask & (1 << bit)) {
                        k = append_keypoint(f);
                        k->ix = x + bit;
                        k->iy = y;
                        k->is = s;
                    }
                }
            }
            _mm256_zeroupper();
#endif
            for (; x < w - 1; ++x) {
                pt = dog + xo * x + yo * y + so * (s - s_min);
                v = *pt;

#define CHECK_NEIGHBORS(CMP,SGN)                                      \
                ( v CMP ## = SGN 0.8 * tp &&                          \
                  v CMP *(pt + xo) &&                                  \
                  v CMP *(pt - xo) &&                                  \
                  v CMP *(pt + so) &&                                  \
                  v CMP *(pt - so) &&                                  \
                  v CMP *(pt + yo) &&                                  \
                  v CMP *(pt - yo) &&                                  \
                                                                       \
                  v CMP *(pt + yo + xo) &&                             \
                  v CMP *(pt + yo - xo) &&                             \
                  v CMP *(pt - yo + xo) &&                             \
                  v CMP *(pt - yo - xo) &&                             \
                                                                       \
                  v CMP *(pt + xo      + so) &&                        \
                  v CMP *(pt - xo      + so) &&                        \
                  v CMP *(pt + yo      + so) &&                        \
                  v CMP *(pt - yo      + so) &&                        \
                  v CMP *(pt + yo + xo + so) &&                        \
                  v CMP *(pt + yo - xo + so) &&                        \
                  v CMP *(pt - yo + xo + so) &&                        \
                  v CMP *(pt - yo - xo + so) &&                        \
                                                                       \
                  v CMP *(pt + xo      - so) &&                        \
                  v CMP *(pt - xo      - so) &&                        \
                  v CMP *(pt + yo      - so) &&                        \
                  v CMP *(pt - yo      - so) &&                        \
                  v CMP *(pt + yo + xo - so) &&                        \
                  v CMP *(pt + yo - xo - so) &&                        \
                  v CMP *(pt - yo + xo - so) &&                        \
                  v CMP *(pt - yo - xo - so) )

                if (CHECK_NEIGHBORS(>,+) ||
                    CHECK_NEIGHBORS(<,-)) {
                    k = append_keypoint(f);
                    k->ix = x;
                    k->iy = y;
                    k->is = s;
                }
#undef CHECK_NEIGHBORS
            }
        }
    }
        }
#if defined(COLITE_VLFEAT_ENABLE_PARALLEL) && defined(__cplusplus)
    }
#endif

    /* Refine local maxima */
    {
        const int candidate_count = f->nkeys;
        int refined_count = 0;

#if defined(COLITE_VLFEAT_ENABLE_PARALLEL) && defined(__cplusplus)
        if (candidate_count >= 2048) {
            VlSiftKeypoint *refined = (VlSiftKeypoint *)malloc(
                (size_t)candidate_count * sizeof(VlSiftKeypoint));
            unsigned char *accepted = (unsigned char *)calloc(
                (size_t)candidate_count, sizeof(unsigned char));
            if (refined && accepted) {
                /*
                 * 每个候选点 refinement 只读 DoG、只写自己的槽位；最后按原候选顺序压缩，
                 * 因而结果顺序与串行 VLFeat 保持一致，同时利用全部逻辑核心。
                 */
                parallel::parallel_for_(parallel::Range(0, candidate_count),
                    [&](const parallel::Range &range) {
                        int idx;
                        for (idx = range.start; idx < range.end; ++idx) {
                            accepted[idx] = (unsigned char)refine_candidate_keypoint(
                                f, f->keys + idx, refined + idx,
                                xper, w, h, s_min, s_max, xo, yo, so, te, tp);
                        }
                    });

                for (i = 0; i < candidate_count; ++i) {
                    if (accepted[i]) {
                        f->keys[refined_count++] = refined[i];
                    }
                }
                free(accepted);
                free(refined);
                f->nkeys = refined_count;
                return;
            }
            free(accepted);
            free(refined);
        }
#endif

        k = f->keys;
        for (i = 0; i < candidate_count; ++i) {
            VlSiftKeypoint refined;
            if (refine_candidate_keypoint(f, f->keys + i, &refined,
                                          xper, w, h, s_min, s_max,
                                          xo, yo, so, te, tp)) {
                *k++ = refined;
            }
        }
        f->nkeys = (int)(k - f->keys);
    }
}

/* ===== Gradient update ===== */
static inline void save_gradient_sample(float *grad, float gx, float gy)
{
    grad[0] = vl_fast_sqrt_f(gx*gx + gy*gy);
    grad[1] = vl_mod_2pi_f(vl_fast_atan2_f(gy, gx) + (float)(2*VL_PI));
}

#if defined(COLITE_VLFEAT_ENABLE_AVX2)
static inline __m256 vl_fast_sqrt_avx2(__m256 x)
{
    const __m256 tiny = _mm256_set1_ps(1e-8f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 three_halves = _mm256_set1_ps(1.5f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 active = _mm256_cmp_ps(x, tiny, _CMP_GE_OQ);

    __m256 xhalf = _mm256_mul_ps(half, x);
    __m256i bits = _mm256_castps_si256(x);
    bits = _mm256_sub_epi32(_mm256_set1_epi32(0x5f3759df),
                            _mm256_srli_epi32(bits, 1));
    __m256 y = _mm256_castsi256_ps(bits);
    y = _mm256_mul_ps(y, _mm256_sub_ps(three_halves,
        _mm256_mul_ps(xhalf, _mm256_mul_ps(y, y))));
    y = _mm256_mul_ps(y, _mm256_sub_ps(three_halves,
        _mm256_mul_ps(xhalf, _mm256_mul_ps(y, y))));
    return _mm256_blendv_ps(zero, _mm256_mul_ps(x, y), active);
}

static inline __m256 vl_fast_atan2_avx2(__m256 y, __m256 x)
{
    const __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 abs_y = _mm256_add_ps(_mm256_and_ps(y, signmask),
                                       _mm256_set1_ps(VL_EPSILON_F));
    const __m256 zero = _mm256_setzero_ps();
    const __m256 xpos = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);

    const __m256 r_pos = _mm256_div_ps(_mm256_sub_ps(x, abs_y),
                                       _mm256_add_ps(x, abs_y));
    const __m256 r_neg = _mm256_div_ps(_mm256_add_ps(x, abs_y),
                                       _mm256_sub_ps(abs_y, x));
    const __m256 r = _mm256_blendv_ps(r_neg, r_pos, xpos);
    __m256 angle = _mm256_blendv_ps(_mm256_set1_ps((float)(3 * VL_PI / 4)),
                                    _mm256_set1_ps((float)(VL_PI / 4)),
                                    xpos);
    const __m256 r2 = _mm256_mul_ps(r, r);
    angle = _mm256_add_ps(angle,
        _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(0.1821f), r2),
                                    _mm256_set1_ps(0.9675f)),
                      r));

    const __m256 yneg = _mm256_cmp_ps(y, zero, _CMP_LT_OQ);
    return _mm256_blendv_ps(angle, _mm256_sub_ps(zero, angle), yneg);
}

static inline void store_gradient8(float *grad, __m256 mod, __m256 angle)
{
    const __m256 lo = _mm256_unpacklo_ps(mod, angle);
    const __m256 hi = _mm256_unpackhi_ps(mod, angle);
    const __m256 out0 = _mm256_permute2f128_ps(lo, hi, 0x20);
    const __m256 out1 = _mm256_permute2f128_ps(lo, hi, 0x31);
    _mm256_storeu_ps(grad, out0);
    _mm256_storeu_ps(grad + 8, out1);
}
#endif

static void update_gradient_row(const float *COLITE_RESTRICT src,
                                float *COLITE_RESTRICT grad,
                                int w, int h, int y)
{
    int x = 0;
    const int row = y * w;
    if (w <= 0 || h <= 0) return;
    if (w <= 1 || h <= 1) {
        memset(grad + 2*row, 0, sizeof(float) * 2 * (size_t)w);
        return;
    }

    if (y == 0) {
        save_gradient_sample(grad, src[1] - src[0], src[w] - src[0]);
        for (x = 1; x < w - 1; ++x)
            save_gradient_sample(grad + 2*x,
                                 0.5f * (src[x + 1] - src[x - 1]),
                                 src[w + x] - src[x]);
        save_gradient_sample(grad + 2*(w - 1),
                             src[w - 1] - src[w - 2],
                             src[w + w - 1] - src[w - 1]);
        return;
    }

    if (y == h - 1) {
        const float *cur = src + row;
        const float *up = src + row - w;
        save_gradient_sample(grad + 2*row, cur[1] - cur[0], cur[0] - up[0]);
        for (x = 1; x < w - 1; ++x)
            save_gradient_sample(grad + 2*(row + x),
                                 0.5f * (cur[x + 1] - cur[x - 1]),
                                 cur[x] - up[x]);
        save_gradient_sample(grad + 2*(row + w - 1),
                             cur[w - 1] - cur[w - 2],
                             cur[w - 1] - up[w - 1]);
        return;
    }

    {
        const float *up = src + row - w;
        const float *cur = src + row;
        const float *down = src + row + w;
        float *g = grad + 2*row;

        save_gradient_sample(g, cur[1] - cur[0], 0.5f * (down[0] - up[0]));
        x = 1;

#if defined(COLITE_VLFEAT_ENABLE_AVX2)
        for (; x + 8 <= w - 1; x += 8) {
            // 中间行的中间像素全部是连续访问，8 个像素批量计算梯度幅值和方向。
            const __m256 gx = _mm256_mul_ps(_mm256_set1_ps(0.5f),
                _mm256_sub_ps(_mm256_loadu_ps(cur + x + 1),
                              _mm256_loadu_ps(cur + x - 1)));
            const __m256 gy = _mm256_mul_ps(_mm256_set1_ps(0.5f),
                _mm256_sub_ps(_mm256_loadu_ps(down + x),
                              _mm256_loadu_ps(up + x)));
            float gx_values[8];
            float gy_values[8];
            int lane;
            _mm256_storeu_ps(gx_values, gx);
            _mm256_storeu_ps(gy_values, gy);
            /*
             * VLFeat's scalar fast sqrt/atan2 are part of its numerical
             * contract. Reusing them per lane keeps orientations and
             * descriptors reference-compatible; AVX2 still computes the
             * central differences and the heavier pyramid kernels.
             */
            for (lane = 0; lane < 8; ++lane)
                save_gradient_sample(g + 2*(x + lane),
                                     gx_values[lane], gy_values[lane]);
        }
        _mm256_zeroupper();
#endif

        for (; x < w - 1; ++x)
            save_gradient_sample(g + 2*x,
                                 0.5f * (cur[x + 1] - cur[x - 1]),
                                 0.5f * (down[x] - up[x]));
        save_gradient_sample(g + 2*(w - 1),
                             cur[w - 1] - cur[w - 2],
                             0.5f * (down[w - 1] - up[w - 1]));
    }
}

static void update_gradient_mask(VlSiftFilt *f, unsigned int required_mask)
{
    int s_min = f->s_min;
    int s_max = f->s_max;
    int w     = f->octave_width;
    int h     = f->octave_height;
    int so    = h * w;
    int grad_layers = s_max - s_min - 2;
    unsigned int all_mask;
    unsigned int missing_mask;
    int active_layers[32];
    int active_count = 0;
    int layer;

    if (grad_layers <= 0) {
        f->grad_o = f->o_cur;
        f->grad_valid_mask = 0;
        return;
    }
    all_mask = grad_layers >= 32 ? 0xffffffffu : ((1u << grad_layers) - 1u);
    required_mask &= all_mask;
    if (f->grad_o != f->o_cur) {
        f->grad_o = f->o_cur;
        f->grad_valid_mask = 0;
    }
    missing_mask = required_mask & ~f->grad_valid_mask;
    if (!missing_mask) return;
    for (layer = 0; layer < grad_layers; ++layer) {
        if (missing_mask & (1u << layer))
            active_layers[active_count++] = layer;
    }

#if defined(COLITE_VLFEAT_ENABLE_PARALLEL) && defined(__cplusplus)
    parallel::parallel_for_(parallel::Range(0, active_count * h),
        [&](const parallel::Range &range) {
            int job;
            for (job = range.start; job < range.end; ++job) {
                int active_index = job / h;
                int gradient_layer = active_layers[active_index];
                int y = job - active_index * h;
                int s = s_min + 1 + gradient_layer;
                const float *src = get_octave(f, s);
                float *grad = f->grad + 2 * so * gradient_layer;
                update_gradient_row(src, grad, w, h, y);
            }
        });
#else
    {
        int active_index, y;
        for (active_index = 0; active_index < active_count; ++active_index) {
            int gradient_layer = active_layers[active_index];
            int s = s_min + 1 + gradient_layer;
            const float *src = get_octave(f, s);
            float *grad = f->grad + 2 * so * gradient_layer;
            for (y = 0; y < h; ++y)
                update_gradient_row(src, grad, w, h, y);
        }
    }
#endif
    f->grad_valid_mask |= missing_mask;
}

void vl_sift_update_gradient(VlSiftFilt *f)
{
    int grad_layers = f->s_max - f->s_min - 2;
    unsigned int all_mask = grad_layers >= 32 ? 0xffffffffu : ((1u << grad_layers) - 1u);
    update_gradient_mask(f, all_mask);
}

void vl_sift_update_gradient_for_detected_keypoints(VlSiftFilt *f)
{
    unsigned int required_mask = 0;
    int i;
    for (i = 0; i < f->nkeys; ++i) {
        int layer = f->keys[i].is - f->s_min - 1;
        if (layer >= 0 && layer < 32)
            required_mask |= 1u << layer;
    }
    update_gradient_mask(f, required_mask);
}

/* ===== Normalize histogram ===== */
static float normalize_histogram(float *begin, float *end)
{
#if defined(COLITE_VLFEAT_ENABLE_AVX2)
    if (end - begin == VL_SIFT_DESCR_LEN) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        int i;
        for (i = 0; i < VL_SIFT_DESCR_LEN; i += 16) {
            const __m256 v0 = _mm256_loadu_ps(begin + i);
            const __m256 v1 = _mm256_loadu_ps(begin + i + 8);
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(v0, v0));
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(v1, v1));
        }
        {
            const __m256 sum = _mm256_add_ps(sum0, sum1);
            __m128 total = _mm_add_ps(_mm256_castps256_ps128(sum),
                                     _mm256_extractf128_ps(sum, 1));
            total = _mm_hadd_ps(total, total);
            total = _mm_hadd_ps(total, total);
            {
                const float norm = sqrtf(_mm_cvtss_f32(total)) + VL_EPSILON_F;
                const __m256 vnorm = _mm256_set1_ps(norm);
                for (i = 0; i < VL_SIFT_DESCR_LEN; i += 8) {
                    const __m256 v = _mm256_loadu_ps(begin + i);
                    _mm256_storeu_ps(begin + i, _mm256_div_ps(v, vnorm));
                }
                return norm;
            }
        }
    }
#endif
    float *iter;
    float norm = 0.0f;

    for (iter = begin; iter != end; ++iter)
        norm += (*iter) * (*iter);

    norm = sqrtf(norm) + VL_EPSILON_F;

    for (iter = begin; iter != end; ++iter)
        *iter /= norm;

    return norm;
}

static void truncate_histogram_128(float *descr, float limit)
{
#if defined(COLITE_VLFEAT_ENABLE_AVX2)
    int i;
    const __m256 vmax = _mm256_set1_ps(limit);
    /* descriptor 是连续 128 个 float，AVX2 一次截断 8 个 bin，避免标量分支。 */
    for (i = 0; i < VL_SIFT_DESCR_LEN; i += 8) {
        __m256 v = _mm256_loadu_ps(descr + i);
        _mm256_storeu_ps(descr + i, _mm256_min_ps(v, vmax));
    }
#else
    int i;
    for (i = 0; i < VL_SIFT_DESCR_LEN; ++i) {
        if (descr[i] > limit) descr[i] = limit;
    }
#endif
}

/* ===== Orientations ===== */
int vl_sift_calc_keypoint_orientations(VlSiftFilt *f,
                                        double angles[4],
                                        const VlSiftKeypoint *k)
{
    double const winf = 1.5;
    double xper = f->xper;

    int w      = f->octave_width;
    int h      = f->octave_height;
    int const xo = 2;
    int const yo = 2 * w;
    int const so = 2 * w * h;
    double x    = k->x / xper;
    double y    = k->y / xper;
    double sigma = k->sigma / xper;

    int xi = (int)(x + 0.5);
    int yi = (int)(y + 0.5);
    int si = k->is;

    double const sigmaw = winf * sigma;
    int    W = VL_MAX((int)floor(3.0 * sigmaw), 1);
    int    nangles = 0;

    enum { nbins = 36 };
    double hist[nbins], maxh;
    const float *pt;
    int xs, ys, iter, i;

    if (k->o != f->o_cur)  return 0;
    if (xi < 0 || xi > w - 1 || yi < 0 || yi > h - 1 ||
        si < f->s_min + 1 || si > f->s_max - 2) return 0;

    update_gradient_mask(f, 1u << (si - f->s_min - 1));

    memset(hist, 0, sizeof(double) * nbins);

    pt = f->grad + xo*xi + yo*yi + so*(si - f->s_min - 1);

#undef at
#define at(dx,dy) (*(pt + xo * (dx) + yo * (dy)))

    for (ys  = VL_MAX(-W,       -yi);
         ys <= VL_MIN(+W, h - 1 - yi); ++ys) {
        for (xs  = VL_MAX(-W,       -xi);
             xs <= VL_MIN(+W, w - 1 - xi); ++xs) {

            double dx = (double)(xi + xs) - x;
            double dy = (double)(yi + ys) - y;
            double r2 = dx*dx + dy*dy;
            double wgt, mod, ang, fbin;

            if (r2 >= W*W + 0.6) continue;

            wgt  = fast_expn(r2 / (2*sigmaw*sigmaw));
            mod  = *(pt + xs*xo + ys*yo);
            ang  = *(pt + xs*xo + ys*yo + 1);
            fbin = nbins * ang / (2 * VL_PI);

            /* Bilinear orientation interpolation */
            {
                int bin = (int)vl_floor_d(fbin - 0.5);
                double rbin = fbin - bin - 0.5;
                hist[(bin + nbins) % nbins] += (1 - rbin) * mod * wgt;
                hist[(bin + 1    ) % nbins] += (    rbin) * mod * wgt;
            }
        }
    }

    /* Smooth histogram (6 iterations) */
    for (iter = 0; iter < 6; iter++) {
        double prev  = hist[nbins - 1];
        double first = hist[0];
        for (i = 0; i < nbins - 1; i++) {
            double newh = (prev + hist[i] + hist[(i+1) % nbins]) / 3.0;
            prev = hist[i];
            hist[i] = newh;
        }
        hist[i] = (prev + hist[i] + first) / 3.0;
    }

    /* Find histogram maximum */
    maxh = 0;
    for (i = 0; i < nbins; ++i)
        maxh = VL_MAX(maxh, hist[i]);

    /* Find peaks within 80% from max */
    nangles = 0;
    for (i = 0; i < nbins; ++i) {
        double h0 = hist[i];
        double hm = hist[(i - 1 + nbins) % nbins];
        double hp = hist[(i + 1 + nbins) % nbins];

        if (h0 > 0.8*maxh && h0 > hm && h0 > hp) {
            double di = -0.5 * (hp - hm) / (hp + hm - 2 * h0);
            double th = 2 * VL_PI * (i + di + 0.5) / nbins;
            angles[nangles++] = th;
            if (nangles == 4)
                goto enough_angles;
        }
    }
enough_angles:
    return nangles;
}

/* ===== Keypoint descriptor ===== */
void vl_sift_calc_keypoint_descriptor(VlSiftFilt *f,
                                       float *descr,
                                       const VlSiftKeypoint *k,
                                       double angle0)
{
    double const magnif = f->magnif;
    double xper = f->xper;

    int w        = f->octave_width;
    int h        = f->octave_height;
    int const xo = 2;
    int const yo = 2 * w;
    int const so = 2 * w * h;
    double x     = k->x / xper;
    double y     = k->y / xper;
    double sigma = k->sigma / xper;

    int xi = (int)(x + 0.5);
    int yi = (int)(y + 0.5);
    int si = k->is;

    double const st0 = sin(angle0);
    double const ct0 = cos(angle0);
    double const SBP = magnif * sigma + VL_EPSILON_D;
    double const invSBP = 1.0 / SBP;
    int    const W   = (int)floor(sqrt(2.0) * SBP * (VL_SIFT_NBP + 1) / 2.0 + 0.5);
    float  const wsigma = (float)f->windowSize;
    float  const inv_2wsigma2 = 1.0f / (2.0f * wsigma * wsigma);
    float  const two_pi = (float)(2 * VL_PI);
    float  const nbo_over_2pi = (float)(VL_SIFT_NBO / (2 * VL_PI));

    int const binto = 1;
    int const binyo = VL_SIFT_NBO * VL_SIFT_NBP;
    int const binxo = VL_SIFT_NBO;

    int bin, dxi, dyi;
    const float *pt;
    float *dpt;

    if (k->o  != f->o_cur        ||
        xi    <  0               ||
        xi    >= w               ||
        yi    <  0               ||
        yi    >= h -    1        ||
        si    <  f->s_min + 1    ||
        si    >  f->s_max - 2)
        return;

    update_gradient_mask(f, 1u << (si - f->s_min - 1));

    memset(descr, 0, sizeof(float) * VL_SIFT_NBO * VL_SIFT_NBP * VL_SIFT_NBP);

    pt  = f->grad + xi*xo + yi*yo + (si - f->s_min - 1)*so;
    dpt = descr + (VL_SIFT_NBP/2) * binyo + (VL_SIFT_NBP/2) * binxo;

#undef atd
#define atd(dbinx,dbiny,dbint) (*(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo))

    for (dyi = VL_MAX(- W, 1 - yi);
         dyi <= VL_MIN(+ W, h - yi - 2); ++dyi) {
        for (dxi = VL_MAX(- W, 1 - xi);
             dxi <= VL_MIN(+ W, w - xi - 2); ++dxi) {

            float mod   = *(pt + dxi*xo + dyi*yo + 0);
            float angle = *(pt + dxi*xo + dyi*yo + 1);
            float theta = angle - (float)angle0;
            if (theta < 0.0f) theta += two_pi;
            else if (theta > two_pi) theta -= two_pi;

            float dx = (float)(xi + dxi - x);
            float dy = (float)(yi + dyi - y);

            float nx = (float)(( ct0 * dx + st0 * dy) * invSBP);
            float ny = (float)((-st0 * dx + ct0 * dy) * invSBP);
            float nt = nbo_over_2pi * theta;

            float win = (float)fast_expn((nx*nx + ny*ny) * inv_2wsigma2);

            int binx = vl_floor_f(nx - 0.5f);
            int biny = vl_floor_f(ny - 0.5f);
            int bint = vl_floor_f(nt);
            float rbinx = nx - (binx + 0.5f);
            float rbiny = ny - (biny + 0.5f);
            float rbint = nt - bint;
            const float wx0 = vl_abs_f(1.0f - rbinx);
            const float wx1 = vl_abs_f(-rbinx);
            const float wy0 = vl_abs_f(1.0f - rbiny);
            const float wy1 = vl_abs_f(-rbiny);
            const float wt0 = vl_abs_f(1.0f - rbint);
            const float wt1 = vl_abs_f(-rbint);
            const float weighted_mod = win * mod;
            int tb0 = bint;
            int tb1 = bint + 1;
            if (tb0 >= VL_SIFT_NBO) tb0 -= VL_SIFT_NBO;
            if (tb1 >= VL_SIFT_NBO) tb1 -= VL_SIFT_NBO;

            if (binx >= -(VL_SIFT_NBP/2) && binx + 1 < (VL_SIFT_NBP/2) &&
                biny >= -(VL_SIFT_NBP/2) && biny + 1 < (VL_SIFT_NBP/2)) {
                /*
                 * 内部样本的八个目标 bin 全部有效，展开后消除三层小循环和八次边界判断。
                 * 每项仍按 VLFeat 原顺序执行 weighted_mod*wx*wy*wt，保持浮点结果一致。
                 */
                atd(binx,     biny,     tb0) += weighted_mod * wx0 * wy0 * wt0;
                atd(binx,     biny,     tb1) += weighted_mod * wx0 * wy0 * wt1;
                atd(binx,     biny + 1, tb0) += weighted_mod * wx0 * wy1 * wt0;
                atd(binx,     biny + 1, tb1) += weighted_mod * wx0 * wy1 * wt1;
                atd(binx + 1, biny,     tb0) += weighted_mod * wx1 * wy0 * wt0;
                atd(binx + 1, biny,     tb1) += weighted_mod * wx1 * wy0 * wt1;
                atd(binx + 1, biny + 1, tb0) += weighted_mod * wx1 * wy1 * wt0;
                atd(binx + 1, biny + 1, tb1) += weighted_mod * wx1 * wy1 * wt1;
            } else {
                const float wx[2] = {wx0, wx1};
                const float wy[2] = {wy0, wy1};
                const float wt[2] = {wt0, wt1};
                const int tb[2] = {tb0, tb1};
                int dbinx, dbiny, dbint;
                for (dbinx = 0; dbinx < 2; ++dbinx) {
                    for (dbiny = 0; dbiny < 2; ++dbiny) {
                        if (binx + dbinx >= -(VL_SIFT_NBP/2) &&
                            binx + dbinx <   (VL_SIFT_NBP/2) &&
                            biny + dbiny >= -(VL_SIFT_NBP/2) &&
                            biny + dbiny <   (VL_SIFT_NBP/2)) {
                            for (dbint = 0; dbint < 2; ++dbint) {
                                atd(binx + dbinx, biny + dbiny, tb[dbint]) +=
                                    weighted_mod * wx[dbinx] * wy[dbiny] * wt[dbint];
                            }
                        }
                    }
                }
            }
        }
    }

    /* Normalize, truncate, normalize again */
    {
        float norm = normalize_histogram(descr, descr + VL_SIFT_NBO*VL_SIFT_NBP*VL_SIFT_NBP);

        if (f->norm_thresh && norm < f->norm_thresh) {
            for (bin = 0; bin < VL_SIFT_NBO*VL_SIFT_NBP*VL_SIFT_NBP; ++bin)
                descr[bin] = 0;
        } else {
            truncate_histogram_128(descr, 0.2f);
            normalize_histogram(descr, descr + VL_SIFT_NBO*VL_SIFT_NBP*VL_SIFT_NBP);
        }
    }
#if defined(COLITE_VLFEAT_ENABLE_AVX2)
    _mm256_zeroupper();
#endif
}

/* ===== Raw descriptor ===== */
void vl_sift_calc_raw_descriptor(const VlSiftFilt *f,
                                  const float *grad,
                                  float *descr,
                                  int width, int height,
                                  double x, double y,
                                  double sigma, double angle0)
{
    double const magnif = f->magnif;

    int w        = width;
    int h        = height;
    int const xo = 2;
    int const yo = 2 * w;

    int xi = (int)(x + 0.5);
    int yi = (int)(y + 0.5);

    double const st0 = sin(angle0);
    double const ct0 = cos(angle0);
    double const SBP = magnif * sigma + VL_EPSILON_D;
    int    const W   = (int)floor(sqrt(2.0) * SBP * (VL_SIFT_NBP + 1) / 2.0 + 0.5);

    int const binto = 1;
    int const binyo = VL_SIFT_NBO * VL_SIFT_NBP;
    int const binxo = VL_SIFT_NBO;

    int bin, dxi, dyi;
    const float *pt;
    float *dpt;

    if (xi < 0 || xi >= w || yi < 0 || yi >= h - 1)
        return;

    memset(descr, 0, sizeof(float) * VL_SIFT_NBO * VL_SIFT_NBP * VL_SIFT_NBP);

    pt  = grad + xi*xo + yi*yo;
    dpt = descr + (VL_SIFT_NBP/2) * binyo + (VL_SIFT_NBP/2) * binxo;

#undef atd
#define atd(dbinx,dbiny,dbint) (*(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo))

    for (dyi = VL_MAX(- W,   - yi);
         dyi <= VL_MIN(+ W, h - yi - 1); ++dyi) {
        for (dxi = VL_MAX(- W,   - xi);
             dxi <= VL_MIN(+ W, w - xi - 1); ++dxi) {

            float mod   = *(pt + dxi*xo + dyi*yo + 0);
            float angle = *(pt + dxi*xo + dyi*yo + 1);
            float theta = vl_mod_2pi_f(angle - (float)angle0);

            float dx = (float)(xi + dxi - x);
            float dy = (float)(yi + dyi - y);

            float nx = (float)(( ct0 * dx + st0 * dy) / SBP);
            float ny = (float)((-st0 * dx + ct0 * dy) / SBP);
            float nt = VL_SIFT_NBO * theta / (float)(2 * VL_PI);

            float wsigma = (float)f->windowSize;
            float win = (float)fast_expn((nx*nx + ny*ny) / (2.0 * wsigma * wsigma));

            int binx = vl_floor_f(nx - 0.5f);
            int biny = vl_floor_f(ny - 0.5f);
            int bint = vl_floor_f(nt);
            float rbinx = nx - (binx + 0.5f);
            float rbiny = ny - (biny + 0.5f);
            float rbint = nt - bint;
            int dbinx, dbiny, dbint;

            for (dbinx = 0; dbinx < 2; ++dbinx) {
                for (dbiny = 0; dbiny < 2; ++dbiny) {
                    for (dbint = 0; dbint < 2; ++dbint) {
                        if (binx + dbinx >= -(VL_SIFT_NBP/2) &&
                            binx + dbinx <   (VL_SIFT_NBP/2) &&
                            biny + dbiny >= -(VL_SIFT_NBP/2) &&
                            biny + dbiny <   (VL_SIFT_NBP/2)) {
                            float weight = win * mod
                                * vl_abs_f(1 - dbinx - rbinx)
                                * vl_abs_f(1 - dbiny - rbiny)
                                * vl_abs_f(1 - dbint - rbint);

                            atd(binx+dbinx, biny+dbiny, (bint+dbint) % VL_SIFT_NBO) += weight;
                        }
                    }
                }
            }
        }
    }

    /* Normalize, truncate, normalize */
    {
        int numSamples =
            (VL_MIN(W, w - xi - 1) - VL_MAX(-W, -xi) + 1) *
            (VL_MIN(W, h - yi - 1) - VL_MAX(-W, -yi) + 1);

        float norm = normalize_histogram(descr, descr + VL_SIFT_NBO*VL_SIFT_NBP*VL_SIFT_NBP);

        if (f->norm_thresh && norm < f->norm_thresh * numSamples) {
            for (bin = 0; bin < VL_SIFT_NBO*VL_SIFT_NBP*VL_SIFT_NBP; ++bin)
                descr[bin] = 0;
        } else {
            for (bin = 0; bin < VL_SIFT_NBO*VL_SIFT_NBP*VL_SIFT_NBP; ++bin) {
                if (descr[bin] > 0.2f) descr[bin] = 0.2f;
            }
            normalize_histogram(descr, descr + VL_SIFT_NBO*VL_SIFT_NBP*VL_SIFT_NBP);
        }
    }
#if defined(COLITE_VLFEAT_ENABLE_AVX2)
    _mm256_zeroupper();
#endif
}

/* ===== Keypoint init from position and scale ===== */
void vl_sift_keypoint_init(const VlSiftFilt *f,
                            VlSiftKeypoint *k,
                            double x, double y, double sigma)
{
    int    o, ix, iy, is;
    double s, phi, xper;

    phi = log2((sigma + VL_EPSILON_D) / f->sigma0);
    o   = (int)vl_floor_d(phi - ((double)f->s_min + 0.5) / f->S);
    o   = VL_MIN(o, f->o_min + f->O - 1);
    o   = VL_MAX(o, f->o_min);
    s   = f->S * (phi - o);

    is  = (int)(s + 0.5);
    is  = VL_MIN(is, f->s_max - 2);
    is  = VL_MAX(is, f->s_min + 1);

    xper = pow(2.0, o);
    ix   = (int)(x / xper + 0.5);
    iy   = (int)(y / xper + 0.5);

    k->o  = o;
    k->ix = ix;
    k->iy = iy;
    k->is = is;
    k->x = x;
    k->y = y;
    k->s = s;
    k->sigma = sigma;
}
