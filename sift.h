#ifndef __SIFT_H__
#define __SIFT_H__

#include <opencv2/core.hpp>

/** default number of sampled intervals per octave */
#define SIFT_INTVLS 3

/** default sigma for initial gaussian smoothing */
#define SIFT_SIGMA 1.6

/** default threshold on keypoint contrast |D(x)| */
#define SIFT_CONTR_THR 0.04

/** default threshold on keypoint ratio of principle curvatures */
#define SIFT_CURV_THR 10

/** double image size before pyramid construction? */
#define SIFT_IMG_DBL 1

/** default width of descriptor histogram array */
#define SIFT_DESCR_WIDTH 4

/** default number of bins per histogram in descriptor array */
#define SIFT_DESCR_HIST_BINS 8

/* assumed gaussian blur for input image */
#define SIFT_INIT_SIGMA 0.5

/* width of border in which to ignore keypoints */
#define SIFT_IMG_BORDER 5

/* maximum steps of keypoint interpolation before failure */
#define SIFT_MAX_INTERP_STEPS 5

/* default number of bins in histogram for orientation assignment */
#define SIFT_ORI_HIST_BINS 36

/* determines gaussian sigma for orientation assignment */
#define SIFT_ORI_SIG_FCTR 1.5

/* determines the radius of the region used in orientation assignment */
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR

/* number of passes of orientation histogram smoothing */
#define SIFT_ORI_SMOOTH_PASSES 2

/* orientation magnitude relative to max that results in new feature */
#define SIFT_ORI_PEAK_RATIO 0.8

/* determines the size of a single descriptor orientation histogram */
#define SIFT_DESCR_SCL_FCTR 3.0

/* threshold on magnitude of elements of descriptor vector */
#define SIFT_DESCR_MAG_THR 0.2

/* factor used to convert floating-point descriptor to unsigned char */
#define SIFT_INT_DESCR_FCTR 512.0



#define FEATURE_MAX_D 128

using namespace cv;
using namespace std;

struct detection_data
{
  int r;
  int c;
  int octv;
  int intvl;
  double subintvl;
  double scl_octv;
};

struct feature
{
  double x;                      /**< x coord */
  double y;                      /**< y coord */
  double a;                      /**< Oxford-type affine region parameter */
  double b;                      /**< Oxford-type affine region parameter */
  double c;                      /**< Oxford-type affine region parameter */
  double scl;                    /**< scale of a Lowe-style feature */
  double ori;                    /**< orientation of a Lowe-style feature */
  int d;                         /**< descriptor length */
  double descr[FEATURE_MAX_D];   /**< descriptor */
  int type;                      /**< feature type, OXFD or LOWE */
  int category;                  /**< all-purpose feature category */
  struct feature* fwd_match;     /**< matching feature from forward image */
  struct feature* bck_match;     /**< matching feature from backmward image */
  struct feature* mdl_match;     /**< matching feature from model */
  Point2d img_pt;                /**< location in image */
  Point2d mdl_pt;           /**< location in model */
  detection_data feature_data;            /**< user-definable data */
};

class Sift
{
public:

	Sift();
	void compute(const cv::Mat& img);

private:
    void create_init_img(Mat& dbl, const Mat& img);
    void downsample(Mat& smaller,  Mat& img);
    void build_gauss_pyr(vector<vector<Mat> >& gauss_pyr, Mat& base, int octvs);
    void build_dog_pyr(vector<vector<Mat> >& dog_pyr, vector<vector<Mat> >& gauss_pyr, int octvs);

    void deriv_3D(Mat&dI, vector<vector<Mat> >& dog_pyr, int octv, int intvl, int r, int c );
    void hessian_3D(Mat& H, vector<vector<Mat> >& dog_pyr, int octv, int intvl, int r, int c);
    int is_extremum(vector<vector<Mat> >& dog_pyr, int octv, int intvl, int r, int c );
    double interp_contr(vector<vector<Mat> >&  dog_pyr, int octv, int intvl, int r,
                        int c, double xi, double xr, double xc);
    void interp_step(vector<vector<Mat> >& dog_pyr, int octv, int intvl, int r, int c,
                        double* xi, double* xr, double* xc);

    int interp_extremum(feature& feat, vector<vector<Mat> >& dog_pyr, int octv,
                        int intvl, int r, int c, int intvls, double contr_thr);
    void scale_space_extrema(vector<feature>& features, vector<vector<Mat> >& dog_pyr, int octvs, double contr_thr, int curv_thr);
    void calc_feature_scales(vector<feature>& features, double sigma, int intvls);
    void adjust_for_img_dbl(vector<feature>& features);
    void smooth_ori_hist(double* hist, int n);
    int calc_grad_mag_ori(Mat& img, int r, int c, double* mag, double* ori);
    void ori_hist(double* hist, Mat& img, int r, int c, int n, int rad, double sigma);
    void calc_feature_oris(vector<feature>& features, vector<vector<Mat> >& gauss_pyr);

    void add_good_ori_features(vector<feature>& features, double* hist, int n, 
                               double mag_thr, struct feature* feat);
    double dominant_ori(double* hist, int n);

    void interp_hist_entry(double*** hist, double rbin, double cbin,
                            double obin, double mag, int d, int n);

    double*** descr_hist(Mat& img, int r, int c, double ori,
                            double scl, int d, int n);

    int is_too_edge_like(Mat& dog_img, int r, int c, int curv_thr);
    void hist_to_descr(double*** hist, int d, int n, struct feature* feat);
    void release_descr_hist( double**** hist, int d);
    void normalize_descr( struct feature* feat);

    void compute_descriptors(vector<feature>& features, vector<vector<Mat> >& gauss_pyr, int d, int n);

    //parameters
    int mDbl;              //double image size before pyramid construction
    double mSigma;         //default sigma for initial gaussian smoothing   
    int mSampleIntervals;  //default number of sampled intervals per octave
    double mContr_thr;     //default threshold on keypoint contrast |D(x)|
    double mCurv_thr;      //default threshold on keypoint ratio of principle curvatures

    int mDescr_width;
    int mDescr_hist_bins;
};

#endif // !__SIFT_H__
