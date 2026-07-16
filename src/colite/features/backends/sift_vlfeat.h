#ifndef COLITE_FEATURES_BACKENDS_SIFT_VLFEAT_H_
#define COLITE_FEATURES_BACKENDS_SIFT_VLFEAT_H_

#ifdef __cplusplus
extern "C" {
#endif

/* SIFT descriptor dimensions */
#define VL_SIFT_NBO 8
#define VL_SIFT_NBP 4
#define VL_SIFT_DESCR_LEN (VL_SIFT_NBO * VL_SIFT_NBP * VL_SIFT_NBP) /* 128 */

/* Keypoint structure */
typedef struct {
    int   o;         /* octave index */
    int   ix;        /* integer x coordinate in octave */
    int   iy;        /* integer y coordinate in octave */
    int   is;        /* integer scale level index */
    float x;         /* x coordinate in original image */
    float y;         /* y coordinate in original image */
    float s;         /* continuous scale level */
    float sigma;     /* keypoint scale */
} VlSiftKeypoint;

/* SIFT filter. Public callers only see an opaque type, so optimized workspace
   fields can evolve without changing the C ABI. */
#if defined(COLITE_VLFEAT_IMPLEMENTATION)
typedef struct VlSiftFilt {
    double sigman;
    double sigma0;
    double sigmak;
    double dsigma0;

    int   width;
    int   height;
    int   O;            /* number of octaves */
    int   S;            /* levels per octave */
    int   o_min;        /* minimum octave index */
    int   s_min;        /* minimum level index */
    int   s_max;        /* maximum level index */
    int   o_cur;        /* current octave */
    double xper;        /* cached pixel spacing: 2^o_cur */

    float *temp;        /* temporary buffer */
    float *band_temp;   /* reusable per-thread Gaussian band workspace */
    int    band_temp_capacity;
    float *octave;      /* current Gaussian scale space data */
    float *dog;         /* current DoG data */
    int    octave_width;
    int    octave_height;

    float *gaussFilter;       /* Gaussian filter kernel */
    double gaussFilterSigma;
    int    gaussFilterWidth;

    VlSiftKeypoint *keys;     /* detected keypoints */
    int   nkeys;
    int   keys_res;

    double peak_thresh;
    double edge_thresh;
    double norm_thresh;
    double magnif;
    double windowSize;

    float *grad;        /* gradient data (magnitude + angle interleaved) */
    int    grad_o;      /* octave for which gradient is computed */
    unsigned int grad_valid_mask; /* valid gradient scale layers in grad_o */
} VlSiftFilt;
#else
typedef struct VlSiftFilt VlSiftFilt;
#endif

/* Create a new SIFT filter */
VlSiftFilt *vl_sift_new(int width, int height, int noctaves, int nlevels, int o_min);

/* Delete a SIFT filter */
void vl_sift_delete(VlSiftFilt *f);

/* Process the first octave of a new image */
int vl_sift_process_first_octave(VlSiftFilt *f, const float *im);

/* Process the next octave */
int vl_sift_process_next_octave(VlSiftFilt *f);

/* Detect keypoints in the current octave */
void vl_sift_detect(VlSiftFilt *f);

/* Compute orientation(s) for a keypoint. Returns number of orientations (up to 4). */
int vl_sift_calc_keypoint_orientations(VlSiftFilt *f, double angles[4],
                                       const VlSiftKeypoint *k);

/* Compute the descriptor for a keypoint and orientation */
void vl_sift_calc_keypoint_descriptor(VlSiftFilt *f, float *descr,
                                      const VlSiftKeypoint *k, double angle);

/* Compute raw descriptor at arbitrary position and scale */
void vl_sift_calc_raw_descriptor(const VlSiftFilt *f, const float *grad,
                                 float *descr, int width, int height,
                                 double x, double y, double sigma, double angle0);

/* Compute gradient cache for the current octave. Useful before batch descriptor work. */
void vl_sift_update_gradient(VlSiftFilt *f);

/* Compute only gradient layers referenced by the current detected keypoints. */
void vl_sift_update_gradient_for_detected_keypoints(VlSiftFilt *f);

/* Initialize a keypoint from position and scale */
void vl_sift_keypoint_init(const VlSiftFilt *f, VlSiftKeypoint *k,
                           double x, double y, double sigma);

/* 带项目名前缀的访问函数保证过滤器布局不泄漏，也不与原始 VLFeat 符号冲突。 */
int colite_vl_sift_get_octave_index(const VlSiftFilt *f);
int colite_vl_sift_get_noctaves(const VlSiftFilt *f);
int colite_vl_sift_get_octave_first(const VlSiftFilt *f);
int colite_vl_sift_get_octave_width(const VlSiftFilt *f);
int colite_vl_sift_get_octave_height(const VlSiftFilt *f);
int colite_vl_sift_get_nlevels(const VlSiftFilt *f);
int colite_vl_sift_get_nkeypoints(const VlSiftFilt *f);
double colite_vl_sift_get_peak_thresh(const VlSiftFilt *f);
double colite_vl_sift_get_edge_thresh(const VlSiftFilt *f);
double colite_vl_sift_get_norm_thresh(const VlSiftFilt *f);
double colite_vl_sift_get_magnif(const VlSiftFilt *f);
double colite_vl_sift_get_window_size(const VlSiftFilt *f);
const VlSiftKeypoint *colite_vl_sift_get_keypoints(const VlSiftFilt *f);
void colite_vl_sift_set_peak_thresh(VlSiftFilt *f, double value);
void colite_vl_sift_set_edge_thresh(VlSiftFilt *f, double value);
void colite_vl_sift_set_norm_thresh(VlSiftFilt *f, double value);
void colite_vl_sift_set_magnif(VlSiftFilt *f, double value);
void colite_vl_sift_set_window_size(VlSiftFilt *f, double value);

#define vl_sift_get_octave_index colite_vl_sift_get_octave_index
#define vl_sift_get_noctaves colite_vl_sift_get_noctaves
#define vl_sift_get_octave_first colite_vl_sift_get_octave_first
#define vl_sift_get_octave_width colite_vl_sift_get_octave_width
#define vl_sift_get_octave_height colite_vl_sift_get_octave_height
#define vl_sift_get_nlevels colite_vl_sift_get_nlevels
#define vl_sift_get_nkeypoints colite_vl_sift_get_nkeypoints
#define vl_sift_get_peak_thresh colite_vl_sift_get_peak_thresh
#define vl_sift_get_edge_thresh colite_vl_sift_get_edge_thresh
#define vl_sift_get_norm_thresh colite_vl_sift_get_norm_thresh
#define vl_sift_get_magnif colite_vl_sift_get_magnif
#define vl_sift_get_window_size colite_vl_sift_get_window_size
#define vl_sift_get_keypoints colite_vl_sift_get_keypoints
#define vl_sift_set_peak_thresh colite_vl_sift_set_peak_thresh
#define vl_sift_set_edge_thresh colite_vl_sift_set_edge_thresh
#define vl_sift_set_norm_thresh colite_vl_sift_set_norm_thresh
#define vl_sift_set_magnif colite_vl_sift_set_magnif
#define vl_sift_set_window_size colite_vl_sift_set_window_size

/* Error codes */
#define VL_ERR_OK   0
#define VL_ERR_EOF  5

#ifdef __cplusplus
}
#endif

#endif /* COLITE_FEATURES_BACKENDS_SIFT_VLFEAT_H_ */
