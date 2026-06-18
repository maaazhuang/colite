#ifndef SIFT_VLFEAT_V2_H
#define SIFT_VLFEAT_V2_H

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

/* SIFT filter */
typedef struct {
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

    float *temp;        /* temporary buffer */
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
} VlSiftFilt;

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

/* Initialize a keypoint from position and scale */
void vl_sift_keypoint_init(const VlSiftFilt *f, VlSiftKeypoint *k,
                           double x, double y, double sigma);

/* Inline accessors */
static inline int    vl_sift_get_octave_index(const VlSiftFilt *f)   { return f->o_cur; }
static inline int    vl_sift_get_noctaves(const VlSiftFilt *f)       { return f->O; }
static inline int    vl_sift_get_octave_first(const VlSiftFilt *f)   { return f->o_min; }
static inline int    vl_sift_get_octave_width(const VlSiftFilt *f)   { return f->octave_width; }
static inline int    vl_sift_get_octave_height(const VlSiftFilt *f)  { return f->octave_height; }
static inline int    vl_sift_get_nlevels(const VlSiftFilt *f)        { return f->S; }
static inline int    vl_sift_get_nkeypoints(const VlSiftFilt *f)     { return f->nkeys; }
static inline double vl_sift_get_peak_thresh(const VlSiftFilt *f)    { return f->peak_thresh; }
static inline double vl_sift_get_edge_thresh(const VlSiftFilt *f)    { return f->edge_thresh; }
static inline double vl_sift_get_norm_thresh(const VlSiftFilt *f)    { return f->norm_thresh; }
static inline double vl_sift_get_magnif(const VlSiftFilt *f)         { return f->magnif; }
static inline double vl_sift_get_window_size(const VlSiftFilt *f)    { return f->windowSize; }

static inline const VlSiftKeypoint *vl_sift_get_keypoints(const VlSiftFilt *f) { return f->keys; }

static inline void vl_sift_set_peak_thresh(VlSiftFilt *f, double t) { f->peak_thresh = t; }
static inline void vl_sift_set_edge_thresh(VlSiftFilt *f, double t) { f->edge_thresh = t; }
static inline void vl_sift_set_norm_thresh(VlSiftFilt *f, double t) { f->norm_thresh = t; }
static inline void vl_sift_set_magnif(VlSiftFilt *f, double m)      { f->magnif = m; }
static inline void vl_sift_set_window_size(VlSiftFilt *f, double x) { f->windowSize = x; }

/* Error codes */
#define VL_ERR_OK   0
#define VL_ERR_EOF  5

#ifdef __cplusplus
}
#endif

#endif /* SIFT_VLFEAT_V2_H */
