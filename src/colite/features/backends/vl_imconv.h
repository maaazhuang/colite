#ifndef VL_IMCONV_H
#define VL_IMCONV_H

#ifdef __cplusplus
extern "C" {
#endif

/* Padding modes */
#define VL_PAD_BY_ZERO       0x0
#define VL_PAD_BY_CONTINUITY 0x1
#define VL_PAD_MASK          0x3
#define VL_TRANSPOSE         0x4

/*
 * Convolve image columns with a 1D filter.
 *
 * dst         - destination image
 * dst_stride  - width of destination image including padding
 * src         - source image
 * src_width   - width of source image
 * src_height  - height of source image
 * src_stride  - width of source image including padding
 * filt        - filter kernel
 * filt_begin  - coordinate of the first filter element
 * filt_end    - coordinate of the last filter element
 * step        - sub-sampling step
 * flags       - VL_PAD_BY_ZERO | VL_PAD_BY_CONTINUITY | VL_TRANSPOSE
 */
void vl_imconvcol_vf(float *dst, int dst_stride,
                     const float *src,
                     int src_width, int src_height, int src_stride,
                     const float *filt, int filt_begin, int filt_end,
                     int step, unsigned int flags);

#ifdef __cplusplus
}
#endif

#endif /* VL_IMCONV_H */
