#include "vl_imconv.h"
#include <stdlib.h>
#include <string.h>

/*
 * Extracted from vlfeat-0.9.21/vl/imopv.c — vl_imconvcol_vf
 *
 * Convolves image columns with a 1D filter. With VL_TRANSPOSE, the output
 * is stored transposed, enabling a two-pass separable 2D convolution:
 *   pass 1: vl_imconvcol_vf(temp, h, src, w, h, w, kernel, -half, half,
 *                           1, PAD_CONTINUITY | VL_TRANSPOSE);
 *   pass 2: vl_imconvcol_vf(dst, w, temp, h, w, h, kernel, -half, half,
 *                           1, PAD_CONTINUITY | VL_TRANSPOSE);
 *
 * This is the exact scalar implementation from the original VLFeat library.
 */

void vl_imconvcol_vf(float *dst, int dst_stride,
                     const float *src,
                     int src_width, int src_height, int src_stride,
                     const float *filt, int filt_begin, int filt_end,
                     int step, unsigned int flags)
{
    int x = 0;
    int y;
    int dheight = (src_height - 1) / step + 1;
    int transp = (flags & VL_TRANSPOSE) != 0;
    int zeropad = (flags & VL_PAD_MASK) == VL_PAD_BY_ZERO;

    /* let filt point to the last sample of the filter */
    filt += filt_end - filt_begin;

    while (x < src_width) {
        /* Calculate dest[x,y] = sum_p image[x,p] filt[y - p]
         * where supp(filt) = [filt_begin, filt_end] = [fb,fe].
         *
         * CHUNK_A: y - fe <= p < 0
         *          completes VL_MAX(fe - y, 0) samples
         * CHUNK_B: VL_MAX(y - fe, 0) <= p < VL_MIN(y - fb, height - 1)
         *          completes fe - VL_MAX(fb, height - y) + 1 samples
         * CHUNK_C: completes all samples
         */
        const float *filti;
        int stop;

        for (y = 0; y < src_height; y += step) {
            float acc = 0;
            float v = 0, c;
            const float *srci;

            filti = filt;
            stop = filt_end - y;
            srci = src + x - stop * src_stride;

            if (stop > 0) {
                if (zeropad) {
                    v = 0;
                } else {
                    v = *(src + x);
                }
                while (filti > filt - stop) {
                    c = *filti--;
                    acc += v * c;
                    srci += src_stride;
                }
            }

            stop = filt_end - (filt_begin > (y - src_height + 1) ? filt_begin : (y - src_height + 1)) + 1;
            while (filti > filt - stop) {
                v = *srci;
                c = *filti--;
                acc += v * c;
                srci += src_stride;
            }

            if (zeropad) v = 0;

            stop = filt_end - filt_begin + 1;
            while (filti > filt - stop) {
                c = *filti--;
                acc += v * c;
            }

            if (transp) {
                *dst = acc; dst += 1;
            } else {
                *dst = acc; dst += dst_stride;
            }
        } /* next y */
        if (transp) {
            dst += 1 * dst_stride - dheight * 1;
        } else {
            dst += 1 * 1 - dheight * dst_stride;
        }
        x += 1;
    } /* next x */
}
