#ifndef COLITE_FEATURES_BACKENDS_SIFT_OPENCV_HPP_
#define COLITE_FEATURES_BACKENDS_SIFT_OPENCV_HPP_

#include <vector>

namespace colite {
namespace features {
namespace backends {

/* KeyPoint structure used by the copied CVSIFT backend. */
struct KeyPoint {
    float x;         /* x coordinate */
    float y;         /* y coordinate */
    float size;      /* keypoint diameter */
    float angle;     /* orientation in degrees [0, 360) */
    float response;  /* strength / contrast */
    int   octave;    /* encoded octave, layer, scale */

    KeyPoint() : x(0), y(0), size(0), angle(-1), response(0), octave(0) {}
    KeyPoint(float _x, float _y, float _size, float _angle=0,
             float _resp=0, int _oct=0)
        : x(_x), y(_y), size(_size), angle(_angle), response(_resp), octave(_oct) {}
};

struct SiftProfile {
    bool valid = false;
    double initialImageMs = 0.0;
    double gaussianPyramidMs = 0.0;
    double dogPyramidMs = 0.0;
    double extremaMs = 0.0;
    double postprocessMs = 0.0;
    double descriptorMs = 0.0;
    double totalMs = 0.0;
};

/* SIFT implementation matching OpenCV's SIFT_Impl */
class Sift {
public:
    /* Factory method for the copied CVSIFT backend. */
    static Sift create(int nfeatures = 0, int nOctaveLayers = 3,
                       double contrastThreshold = 0.04,
                       double edgeThreshold = 10.0,
                       double sigma = 1.6,
                       bool enablePreciseUpscale = false,
                       bool useAvx2 = false);

    /* Detect keypoints and compute descriptors.
       image: grayscale float array [0, 255] of size width*height
       keypoints: output keypoints
       descriptors: output descriptors, row-major, nkeypoints * 128 floats
    */
    void detectAndCompute(const float *image, int width, int height,
                          std::vector<KeyPoint> &keypoints,
                          std::vector<float> &descriptors);

    int descriptorSize() const { return 128; }
    int descriptorType() const { return 5; } /* CV_32F */

    void setNFeatures(int n)        { nfeatures = n; }
    int  getNFeatures() const       { return nfeatures; }
    void setNOctaveLayers(int n)    { nOctaveLayers = n; }
    int  getNOctaveLayers() const   { return nOctaveLayers; }
    void setContrastThreshold(double t) { contrastThreshold = t; }
    double getContrastThreshold() const { return contrastThreshold; }
    void setEdgeThreshold(double t) { edgeThreshold = t; }
    double getEdgeThreshold() const { return edgeThreshold; }
    void setSigma(double s)         { sigma = s; }
    double getSigma() const         { return sigma; }
    void setEnablePreciseUpscale(bool enabled) { enablePreciseUpscale = enabled; }
    bool getEnablePreciseUpscale() const { return enablePreciseUpscale; }
    void setUseAvx2(bool enabled) { useAvx2 = enabled; }
    bool getUseAvx2() const { return useAvx2; }
    SiftProfile getLastProfile() const { return lastProfile; }

private:
    Sift() = default;

    int    nfeatures = 0;
    int    nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10.0;
    double sigma = 1.6;
    bool enablePreciseUpscale = false;
    bool useAvx2 = false;
    SiftProfile lastProfile;
};

}  // namespace backends
}  // namespace features
}  // namespace colite

#endif  // COLITE_FEATURES_BACKENDS_SIFT_OPENCV_HPP_
