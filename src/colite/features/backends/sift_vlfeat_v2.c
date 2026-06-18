#include "sift_vlfeat_v2.h"
#include "vl_imconv.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ===== Minimal macros replacing VLFeat ===== */
#define VL_MAX(a,b) (((a)>(b))?(a):(b))
#define VL_MIN(a,b) (((a)<(b))?(a):(b))
#define VL_SHIFT_LEFT(x,n) (((n)>=0)?((x)<<(n)):((x)>>(-(n))))

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
 * Exact original VLFeat _vl_sift_smooth using vl_imconvcol_vf.
 * The two-pass approach: columns+transpose, then columns+transpose.
 * VL_TRANSPOSE makes the column convolution effectively operate on rows.
 */
static void vl_sift_smooth(float *output, float *temp,
                           const float *input,
                           int width, int height,
                           const float *kernel, int half_width)
{
    /* Pass 1: convolve columns of input, store transposed in temp */
    vl_imconvcol_vf(temp, height,
                    input, width, height, width,
                    kernel, -half_width, half_width,
                    1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE);

    /* Pass 2: convolve columns of transposed temp, store transposed back */
    vl_imconvcol_vf(output, width,
                    temp, height, width, height,
                    kernel, -half_width, half_width,
                    1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE);
}

/* ===== Internal: copy and upsample rows ===== */
static void copy_and_upsample_rows(float *dst, const float *src,
                                   int width, int height)
{
    int x, y;
    float a, b;

    for (y = 0; y < height; ++y) {
        b = a = *src++;
        for (x = 0; x < width - 1; ++x) {
            b = *src++;
            *dst = a;             dst += height;
            *dst = 0.5f * (a + b); dst += height;
            a = b;
        }
        *dst = b; dst += height;
        *dst = b; dst += height;
        dst += 1 - width * 2 * height;
    }
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

/* ===== VlSiftFilt implementation ===== */

VlSiftFilt *vl_sift_new(int width, int height, int noctaves, int nlevels, int o_min)
{
    VlSiftFilt *f = (VlSiftFilt *)malloc(sizeof(VlSiftFilt));
    int w = VL_SHIFT_LEFT(width,  -o_min);
    int h = VL_SHIFT_LEFT(height, -o_min);
    int nel = w * h;

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

    f->temp    = (float *)malloc(sizeof(float) * nel);
    f->octave  = (float *)malloc(sizeof(float) * nel * (f->s_max - f->s_min + 1));
    f->dog     = (float *)malloc(sizeof(float) * nel * (f->s_max - f->s_min));
    f->grad    = (float *)malloc(sizeof(float) * nel * 2 * (f->s_max - f->s_min));

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
        free(f->gaussFilter);
        free(f);
    }
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
    f->nkeys = 0;
    w = f->octave_width  = VL_SHIFT_LEFT(f->width,  -f->o_cur);
    h = f->octave_height = VL_SHIFT_LEFT(f->height, -f->o_cur);

    if (f->O == 0)
        return VL_ERR_EOF;

    octave = get_octave(f, s_min);

    if (o_min < 0) {
        copy_and_upsample_rows(temp,   im,   width,      height);
        copy_and_upsample_rows(octave, temp, height, 2 * width);

        for (o = -1; o > o_min; --o) {
            copy_and_upsample_rows(temp, octave,
                                   width << -o,      height << -o);
            copy_and_upsample_rows(octave, temp,
                                   width << -o, 2 * (height << -o));
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
            vl_sift_smooth(octave, temp, octave, w, h, kernel, ksize);
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
            vl_sift_smooth(get_octave(f, s), temp,
                           get_octave(f, s - 1), w, h, kernel, ksize);
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
            vl_sift_smooth(octave, temp, octave, w, h, kernel, ksize);
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
            vl_sift_smooth(get_octave(f, s), temp,
                           get_octave(f, s - 1), w, h, kernel, ksize);
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

    double xper = pow(2.0, f->o_cur);

    int x, y, s, i, ii, jj;
    float *pt, v;
    VlSiftKeypoint *k;

    f->nkeys = 0;

    /* Compute DoG */
    pt = f->dog;
    for (s = s_min; s <= s_max - 1; ++s) {
        float *src_a = get_octave(f, s);
        float *src_b = get_octave(f, s + 1);
        float *end_a = src_a + w * h;
        while (src_a != end_a) {
            *pt++ = *src_b++ - *src_a++;
        }
    }

    /* Find local maxima of DoG */
    pt = dog + xo + yo + so;

    for (s = s_min + 1; s <= s_max - 2; ++s) {
        for (y = 1; y < h - 1; ++y) {
            for (x = 1; x < w - 1; ++x) {
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

                    k = f->keys + (f->nkeys++);
                    k->ix = x;
                    k->iy = y;
                    k->is = s;
                }
                pt += 1;
            }
            pt += 2;
        }
        pt += 2 * yo;
    }

    /* Refine local maxima */
    k = f->keys;

    for (i = 0; i < f->nkeys; ++i) {
        int xk = f->keys[i].ix;
        int yk = f->keys[i].iy;
        int sk = f->keys[i].is;

        double Dx=0,Dy=0,Ds=0,Dxx=0,Dyy=0,Dss=0,Dxy=0,Dxs=0,Dys=0;
        double A[3*3], b[3];

        int dx = 0, dy = 0;
        int iter, j;

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

            /* Gauss elimination */
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

            /* Backward substitution */
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

        /* Check threshold and edge */
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
                k->o     = f->o_cur;
                k->ix    = xk;
                k->iy    = yk;
                k->is    = sk;
                k->s     = sn;
                k->x     = (float)(xn * xper);
                k->y     = (float)(yn * xper);
                k->sigma = (float)(f->sigma0 * pow(2.0, sn/f->S) * xper);
                ++k;
            }
        }
    }

    f->nkeys = (int)(k - f->keys);
}

/* ===== Gradient update ===== */
static void update_gradient(VlSiftFilt *f)
{
    int s_min = f->s_min;
    int s_max = f->s_max;
    int w     = f->octave_width;
    int h     = f->octave_height;
    int const xo = 1;
    int const yo = w;
    int const so = h * w;
    int y, s;

    if (f->grad_o == f->o_cur) return;

    for (s = s_min + 1; s <= s_max - 2; ++s) {
        float *src, *end, *grad, gx, gy;

#define SAVE_BACK                                         \
        *grad++ = vl_fast_sqrt_f(gx*gx + gy*gy);          \
        *grad++ = vl_mod_2pi_f(vl_fast_atan2_f(gy, gx) + (float)(2*VL_PI)); \
        ++src;

        src  = get_octave(f, s);
        grad = f->grad + 2 * so * (s - s_min - 1);

        /* First pixel of first row */
        gx = src[+xo] - src[0];
        gy = src[+yo] - src[0];
        SAVE_BACK;

        /* Middle pixels of first row */
        end = (src - 1) + w - 1;
        while (src < end) {
            gx = 0.5f * (src[+xo] - src[-xo]);
            gy =        src[+yo] - src[0];
            SAVE_BACK;
        }

        /* Last pixel of first row */
        gx = src[0]   - src[-xo];
        gy = src[+yo] - src[0];
        SAVE_BACK;

        for (y = 1; y < h - 1; ++y) {
            /* First pixel of middle rows */
            gx =        src[+xo] - src[0];
            gy = 0.5f * (src[+yo] - src[-yo]);
            SAVE_BACK;

            /* Middle pixels of middle rows */
            end = (src - 1) + w - 1;
            while (src < end) {
                gx = 0.5f * (src[+xo] - src[-xo]);
                gy = 0.5f * (src[+yo] - src[-yo]);
                SAVE_BACK;
            }

            /* Last pixel of middle row */
            gx =        src[0]   - src[-xo];
            gy = 0.5f * (src[+yo] - src[-yo]);
            SAVE_BACK;
        }

        /* First pixel of last row */
        gx = src[+xo] - src[0];
        gy = src[  0] - src[-yo];
        SAVE_BACK;

        /* Middle pixels of last row */
        end = (src - 1) + w - 1;
        while (src < end) {
            gx = 0.5f * (src[+xo] - src[-xo]);
            gy =        src[0]   - src[-yo];
            SAVE_BACK;
        }

        /* Last pixel of last row */
        gx = src[0]   - src[-xo];
        gy = src[0]   - src[-yo];
        SAVE_BACK;

#undef SAVE_BACK
    }
    f->grad_o = f->o_cur;
}

/* ===== Normalize histogram ===== */
static float normalize_histogram(float *begin, float *end)
{
    float *iter;
    float norm = 0.0f;

    for (iter = begin; iter != end; ++iter)
        norm += (*iter) * (*iter);

    norm = sqrtf(norm) + VL_EPSILON_F;

    for (iter = begin; iter != end; ++iter)
        *iter /= norm;

    return norm;
}

/* ===== Orientations ===== */
int vl_sift_calc_keypoint_orientations(VlSiftFilt *f,
                                        double angles[4],
                                        const VlSiftKeypoint *k)
{
    double const winf = 1.5;
    double xper = pow(2.0, f->o_cur);

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

    update_gradient(f);

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
    double xper = pow(2.0, f->o_cur);

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
    int    const W   = (int)floor(sqrt(2.0) * SBP * (VL_SIFT_NBP + 1) / 2.0 + 0.5);

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

    update_gradient(f);

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

    /* Normalize, truncate, normalize again */
    {
        float norm = normalize_histogram(descr, descr + VL_SIFT_NBO*VL_SIFT_NBP*VL_SIFT_NBP);

        if (f->norm_thresh && norm < f->norm_thresh) {
            for (bin = 0; bin < VL_SIFT_NBO*VL_SIFT_NBP*VL_SIFT_NBP; ++bin)
                descr[bin] = 0;
        } else {
            for (bin = 0; bin < VL_SIFT_NBO*VL_SIFT_NBP*VL_SIFT_NBP; ++bin) {
                if (descr[bin] > 0.2f) descr[bin] = 0.2f;
            }
            normalize_histogram(descr, descr + VL_SIFT_NBO*VL_SIFT_NBP*VL_SIFT_NBP);
        }
    }
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
