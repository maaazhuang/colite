#include "sift_opencv.hpp"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <algorithm>
#include <climits>
#include <vector>

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
using sift_wt = float;

/* ===== Improved Gaussian blur matching OpenCV precisely ===== */

/*
 * BORDER_REFLECT_101: reflects at boundary without repeating edge pixel.
 * Matches OpenCV's default border mode for GaussianBlur.
 */
inline int reflect101(int x, int max_val)
{
    if (x < 0) return -x;
    if (x >= max_val) return 2 * max_val - x - 2;
    return x;
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

/*
 * Separable GaussianBlur: rows first, then columns (matching OpenCV's FilterEngine order).
 * Uses BORDER_REFLECT_101.
 */
void gaussianBlur(float *dst, const float *src, int w, int h, float sigma)
{
    std::vector<float> kernel;
    buildGaussianKernel(kernel, sigma);

    int ksize = (int)kernel.size();
    int half = ksize / 2;
    std::vector<float> temp(w * h);

    /* Row convolution (horizontal) */
    for (int y = 0; y < h; y++) {
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

    /* Column convolution (vertical) */
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            double sum = 0.0;
            for (int k = -half; k <= half; k++) {
                int yy = reflect101(y + k, h);
                sum += (double)temp[yy * w + x] * kernel[k + half];
            }
            dst[y * w + x] = (float)sum;
        }
    }
}

/* BORDER_REFLECT: reflect including the edge pixel. Used by OpenCV's warpAffine. */
inline int reflectBorder(int x, int max_val)
{
    if (x < 0) return -x - 1;
    if (x >= max_val) return 2 * max_val - x - 1;
    return x;
}

/*
 * Image doubling matching OpenCV's precise upscale (warpAffine with WARP_INVERSE_MAP).
 * Matrix [[0.5, 0, 0], [0, 0.5, 0]] maps dst鈫抯rc: src_x = 0.5 * dst_x.
 * Bilinear interpolation with BORDER_REFLECT.
 */
void doubleImage(const float *src, int sw, int sh, float *dst, int dw, int dh)
{
    for (int y = 0; y < dh; y++) {
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
}

/* Downsample by 2 (nearest neighbor, matching OpenCV's INTER_NEAREST). */
void downsample2(const float *src, int sw, int /*sh*/, float *dst, int dw, int dh)
{
    for (int y = 0; y < dh; y++) {
        int sy = y * 2;
        for (int x = 0; x < dw; x++) {
            dst[y * dw + x] = src[sy * sw + x * 2];
        }
    }
}

/* ===== Image buffer helper ===== */
struct Image {
    std::vector<float> data;
    int width = 0, height = 0;

    Image() = default;
    Image(int w, int h) : data(w * h), width(w), height(h) {}

    float *ptr() { return data.data(); }
    const float *ptr() const { return data.data(); }
    float &at(int x, int y) { return data[y * width + x]; }
    float  at(int x, int y) const { return data[y * width + x]; }
};

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
    std::vector<float> X, Y, Mag, W;
    int area = (2*radius+1)*(2*radius+1);
    X.resize(area); Y.resize(area); Mag.resize(area); W.resize(area);

    float expf_scale = -1.f / (2.f * sigma * sigma);
    int k = 0;

    for (int i = -radius; i <= radius; i++) {
        int y = cy + i;
        if (y <= 0 || y >= img.height - 1) continue;
        for (int j = -radius; j <= radius; j++) {
            int x = cx + j;
            if (x <= 0 || x >= img.width - 1) continue;

            float dx = img.at(x+1, y) - img.at(x-1, y);
            float dy = img.at(x, y-1) - img.at(x, y+1);
            X[k] = dx; Y[k] = dy;
            W[k] = (float)(i*i + j*j) * expf_scale;
            k++;
        }
    }
    int len = k;

    /* Compute magnitude, orientation, weight */
    for (k = 0; k < len; k++) {
        Mag[k] = std::sqrt(X[k]*X[k] + Y[k]*Y[k]);
        float angle = std::atan2(Y[k], X[k]) * 180.f / (float)M_PI;
        if (angle < 0) angle += 360.f;
        float &ori = X[k]; /* reuse X as Ori */
        ori = angle;
        W[k] = std::exp(W[k]);
    }

    /* Accumulate histogram */
    std::vector<float> temphist(n + 4, 0.0f);
    float *th = temphist.data() + 2;
    float nd360 = n / 360.f;

    for (k = 0; k < len; k++) {
        int bin = (int)std::lround(nd360 * X[k]);
        if (bin >= n) bin -= n;
        if (bin < 0) bin += n;
        th[bin] += W[k] * Mag[k];
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
    float hist[SIFT_ORI_HIST_BINS];

    float threshold = std::floor(0.5f * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);

    for (int o = 0; o < nOctaves; o++) {
        for (int i = 1; i <= nOctaveLayers; i++) {
            int idx = o * nDogPerOct + i;
            const Image &img  = dog_pyr[idx];
            const Image &prev = dog_pyr[idx - 1];
            const Image &next = dog_pyr[idx + 1];
            int w = img.width, h = img.height, step = w;

            for (int r = SIFT_IMG_BORDER; r < h - SIFT_IMG_BORDER; r++) {
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
                                kpts.push_back(kpt);
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ===== Compute SIFT descriptor (matching OpenCV's calcSIFTDescriptor BASELINE) ===== */
void calcSIFTDescriptor(const Image &img, float x, float y,
                         float ori, float scl,
                         float *dst)
{
    int pt_x = (int)std::lround(x);
    int pt_y = (int)std::lround(y);
    int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

    float cos_t = std::cos(ori * (float)(M_PI / 180.0));
    float sin_t = std::sin(ori * (float)(M_PI / 180.0));
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f / (d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = (int)std::lround(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    int w = img.width, h_img = img.height;
    radius = std::min(radius, (int)std::sqrt((float)(w*w + h_img*h_img)));

    cos_t /= hist_width;
    sin_t /= hist_width;

    int len_hist = (d+2)*(d+2)*(n+2);
    int len_ddn = d * d * n;
    std::vector<float> hist(len_hist, 0.0f);

    for (int ri = -radius; ri <= radius; ri++) {
        for (int ci = -radius; ci <= radius; ci++) {
            float c_rot = ci * cos_t - ri * sin_t;
            float r_rot = ci * sin_t + ri * cos_t;
            float rbin = r_rot + d/2.f - 0.5f;
            float cbin = c_rot + d/2.f - 0.5f;
            int r = pt_y + ri, c = pt_x + ci;

            if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < h_img - 1 && c > 0 && c < w - 1)
            {
                float dx = img.at(c+1, r) - img.at(c-1, r);
                float dy = img.at(c, r-1) - img.at(c, r+1);
                float mag = std::sqrt(dx*dx + dy*dy);
                float angle = std::atan2(dy, dx) * 180.f / (float)M_PI;
                if (angle < 0) angle += 360.f;
                float obin = (angle - ori) * bins_per_rad;
                float wgt = std::exp((c_rot*c_rot + r_rot*r_rot) * exp_scale);
                mag *= wgt;

                int r0 = (int)std::floor(rbin);
                int c0 = (int)std::floor(cbin);
                int o0 = (int)std::floor(obin);
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
        }
    }

    /* Finalize histogram, wrap orientation bins */
    std::vector<float> rawDst(len_ddn);
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
        dst[k] = std::min(rawDst[k] * nrm2_final, 255.0f);
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
                       double sigma)
{
    Sift sift;
    sift.nfeatures = nfeatures;
    sift.nOctaveLayers = nOctaveLayers;
    sift.contrastThreshold = contrastThreshold;
    sift.edgeThreshold = edgeThreshold;
    sift.sigma = sigma;
    return sift;
}

void Sift::detectAndCompute(const float *image, int width, int height,
                               std::vector<KeyPoint> &keypoints,
                               std::vector<float> &descriptors)
{
    keypoints.clear();
    descriptors.clear();

    int firstOctave = -1;

    /* Create initial image */
    float sig_diff = std::sqrt(std::max(sigma*sigma - SIFT_INIT_SIGMA*SIFT_INIT_SIGMA*4, 0.01));

    Image base;
    if (firstOctave < 0) {
        /* Double image size (matching OpenCV's precise upscale) */
        int dw = width * 2, dh = height * 2;
        std::vector<float> gray_fpt(image, image + width * height);
        std::vector<float> dbl(dw * dh);
        doubleImage(gray_fpt.data(), width, height, dbl.data(), dw, dh);

        base = Image(dw, dh);
        gaussianBlur(base.ptr(), dbl.data(), dw, dh, sig_diff);
    } else {
        base = Image(width, height);
        gaussianBlur(base.ptr(), image, width, height, sig_diff);
    }

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
                            src.width, src.height, (float)sig[i]);
            }
        }
    }

    /* Build DoG pyramid */
    int nDog = nOctaveLayers + 2;
    std::vector<Image> dogpyr(nOctaves * nDog);
    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < nDog; i++) {
            const Image &src1 = gpyr[o*nLayers + i];
            const Image &src2 = gpyr[o*nLayers + i + 1];
            dogpyr[o*nDog + i] = Image(src1.width, src1.height);
            for (int j = 0; j < src1.width * src1.height; j++)
                dogpyr[o*nDog + i].ptr()[j] = src2.ptr()[j] - src1.ptr()[j];
        }
    }

    /* Detect keypoints */
    findScaleSpaceExtrema(gpyr, nOctaves, nOctaveLayers,
                           dogpyr, (float)contrastThreshold,
                           (float)edgeThreshold, (float)sigma,
                           keypoints);

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

    /* Compute descriptors */
    int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    int descSize = d * d * n;
    descriptors.resize(keypoints.size() * descSize);

    for (size_t i = 0; i < keypoints.size(); i++) {
        int octave, layer;
        float scale;
        unpackOctave(keypoints[i], octave, layer, scale);

        layer = std::min(layer, nOctaveLayers + 2);
        octave = std::max(std::min(octave - firstOctave, nOctaves - 1), 0);

        float size = keypoints[i].size * scale;
        float pt_x = keypoints[i].x * scale;
        float pt_y = keypoints[i].y * scale;
        float angle = 360.f - keypoints[i].angle;
        if (std::fabs(angle - 360.f) < FLT_EPSILON) angle = 0.f;

        const Image &img = gpyr[octave * nLayers + layer];
        calcSIFTDescriptor(img, pt_x, pt_y, angle, size * 0.5f,
                          descriptors.data() + i * descSize);
    }
}

}  // namespace backends
}  // namespace features
}  // namespace colite
