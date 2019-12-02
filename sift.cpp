#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/core.hpp>

#include "sift.h"

using namespace std;
using namespace cv;

Sift::Sift()
{
    mDbl = SIFT_IMG_DBL;       
    mSigma = SIFT_SIGMA;     
    mSampleIntervals = SIFT_INTVLS;
    mContr_thr = SIFT_CONTR_THR;
    mCurv_thr = SIFT_CURV_THR;
    mDescr_width = SIFT_DESCR_WIDTH;
    mDescr_hist_bins = SIFT_DESCR_HIST_BINS;
}

/*
   Computes the partial derivatives in x, y, and scale of a pixel in the DoG
   scale space pyramid.

   @param dog_pyr DoG scale space pyramid
   @param octv pixel's octave in dog_pyr
   @param intvl pixel's interval in octv
   @param r pixel's image row
   @param c pixel's image col

   @return Returns the vector of partial derivatives for pixel I
   { dI/dx, dI/dy, dI/ds }^T as a CvMat*
   */
void Sift::deriv_3D(Mat&dI, vector<vector<Mat> >& dog_pyr, int octv, int intvl, int r, int c )
{
    double dx, dy, ds;

    dx = (dog_pyr[octv][intvl].at<float>(r, c+1) -
            dog_pyr[octv][intvl].at<float>(r, c-1)) / 2.0;
    dy = (dog_pyr[octv][intvl].at<float>(r+1, c) -
            dog_pyr[octv][intvl].at<float>(r-1, c)) / 2.0;
    ds = (dog_pyr[octv][intvl+1].at<float>(r, c) -
            dog_pyr[octv][intvl-1].at<float>(r, c)) / 2.0;

    dI = Mat::zeros( 3, 1, CV_64FC1 );
    dI.at<double>(0, 0) = dx;
    dI.at<double>(1, 0) = dy;
    dI.at<double>(2, 0) = ds;
}

/*
   Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.

   @param dog_pyr DoG scale space pyramid
   @param octv pixel's octave in dog_pyr
   @param intvl pixel's interval in octv
   @param r pixel's image row
   @param c pixel's image col

   @return Returns the Hessian matrix (below) for pixel I as a CvMat*

   / Ixx  Ixy  Ixs \ <BR>
   | Ixy  Iyy  Iys | <BR>
   \ Ixs  Iys  Iss /
   */
void Sift::hessian_3D(Mat& H, vector<vector<Mat> >& dog_pyr, int octv, int intvl, int r,
        int c )
{
    double v, dxx, dyy, dss, dxy, dxs, dys;

    v = dog_pyr[octv][intvl].at<float>(r, c);

    dxx = (dog_pyr[octv][intvl].at<float>(r, c+1) + 
            dog_pyr[octv][intvl].at<float>(r, c-1) - 2 * v);

    dyy = (dog_pyr[octv][intvl].at<float>(r+1, c) +
            dog_pyr[octv][intvl].at<float>(r-1, c) - 2 * v);

    dss = (dog_pyr[octv][intvl+1].at<float>(r, c) +
            dog_pyr[octv][intvl-1].at<float>(r, c) - 2 * v);

    dxy = (dog_pyr[octv][intvl].at<float>(r+1, c+1) -
            dog_pyr[octv][intvl].at<float>(r+1, c-1) -
            dog_pyr[octv][intvl].at<float>(r-1, c+1) +
            dog_pyr[octv][intvl].at<float>(r-1, c-1)) / 4.0;

    dxs = (dog_pyr[octv][intvl+1].at<float>(r, c+1) -
            dog_pyr[octv][intvl+1].at<float>(r, c-1) -
            dog_pyr[octv][intvl-1].at<float>(r, c+1) +
            dog_pyr[octv][intvl-1].at<float>(r, c-1)) / 4.0;

    dys = (dog_pyr[octv][intvl+1].at<float>(r+1, c) -
            dog_pyr[octv][intvl+1].at<float>(r-1, c) -
            dog_pyr[octv][intvl-1].at<float>(r+1, c) +
            dog_pyr[octv][intvl-1].at<float>(r-1, c)) / 4.0;

    H = Mat::zeros(3, 3, CV_64FC1);

    H.at<double>(0, 0) = dxx;
    H.at<double>(0, 1) = dxy;
    H.at<double>(0, 2) = dxs;
    H.at<double>(1, 0) = dxy;
    H.at<double>(1, 1) = dyy;
    H.at<double>(1, 2) = dys;
    H.at<double>(2, 0) = dxs;
    H.at<double>(2, 1) = dys;
    H.at<double>(2, 2) = dss;
}


/**
  Downsamples an image to a quarter of its size (half in each dimension)
  using nearest-neighbor interpolation

  @param img an image

  @return Returns an image whose dimensions are half those of img
  */
void Sift::downsample(Mat& smaller,  Mat& img)
{
    resize(img, smaller, Size(), 0.5, 0.5, INTER_NEAREST);
}

/**
  Converts an image to 32-bit grayscale and Gaussian-smooths it.  The image is
  optionally doubled in size prior to smoothing.

  @param img input image
  @param img_dbl if true, image is doubled in size prior to smoothing
  @param sigma total std of Gaussian smoothing
  */
void Sift::create_init_img(Mat& dbl, const Mat& img)
{
    Mat gray;
    //convert to 32 bit float 0.0~1.0
    img.convertTo(gray, CV_32F, 1.0/255, 0);

    double sig_diff;

    if (mDbl)
    {
        sig_diff = sqrt(mSigma * mSigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4);

        resize(gray, dbl, Size(), 2, 2, INTER_CUBIC);

        GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
    }
    else
    {
        sig_diff = sqrt( mSigma * mSigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA );
        GaussianBlur(gray, dbl, Size(), sig_diff, sig_diff);
    }
}

/**
  Builds Gaussian scale space pyramid from an image

  @param base base image of the pyramid
  @param octvs number of octaves of scale space
  @param intvls number of intervals per octave
  @param sigma amount of Gaussian smoothing per octave

  @return Returns a Gaussian scale space pyramid as an octvs x (intvls + 3)
  array
  */
void Sift::build_gauss_pyr(vector<vector<Mat> >& gauss_pyr, Mat& base, int octvs)
{
    double sig[mSampleIntervals+3];
    double sig_total, sig_prev, k;

    gauss_pyr.resize(octvs);

    for(int i=0; i<octvs; i++)
    {
        gauss_pyr[i].resize(mSampleIntervals+3);
    }

    /**
      precompute Gaussian sigmas using the following formula:

      \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2

      sig[i] is the incremental sigma value needed to compute 
      the actual sigma of level i. Keeping track of incremental
      sigmas vs. total sigmas keeps the gaussian kernel small.
      */

    k = pow( 2.0, 1.0 / mSampleIntervals );

    sig[0] = mSigma;
    sig[1] = mSigma * sqrt( k*k- 1 );

    for (int i = 2; i < mSampleIntervals + 3; i++)
    {
        sig[i] = sig[i-1] * k;
    }

    for(int o = 0; o < octvs; o++ )
    {
        for( int i = 0; i < mSampleIntervals + 3; i++ )
        {
            if(o == 0  &&  i == 0)
            {
                gauss_pyr[o][i] = base.clone();
            }

            /* base of new octvave is halved image from end of previous octave */
            else if(i == 0)
            {
                downsample(gauss_pyr[o][i], gauss_pyr[o-1][mSampleIntervals] );
            }

            /* blur the current octave's last image to create the next one */
            else
            {
                GaussianBlur(gauss_pyr[o][i-1], gauss_pyr[o][i], Size(), sig[i], sig[i]);
            }
        }
    }
}

/**
  Builds a difference of Gaussians scale space pyramid by subtracting adjacent
  intervals of a Gaussian pyramid

  @param gauss_pyr Gaussian scale-space pyramid
  @param octvs number of octaves of scale space
  @param intvls number of intervals per octave

  @return Returns a difference of Gaussians scale space pyramid as an
  octvs x (intvls + 2) array
  */
void Sift::build_dog_pyr(vector<vector<Mat> >& dog_pyr, vector<vector<Mat> >& gauss_pyr, int octvs)
{
    dog_pyr.resize(octvs);

    for(int i = 0; i < octvs; i++ )
    {
        dog_pyr[i].resize(mSampleIntervals+2);
    }

    for(int o = 0; o < octvs; o++ )
    {
        for(int i = 0; i < mSampleIntervals + 2; i++ )
        {
            dog_pyr[o][i] = gauss_pyr[o][i+1] - gauss_pyr[o][i];
        }
    }
}

/*
   Determines whether a pixel is a scale-space extremum by comparing it to it's
   3x3x3 pixel neighborhood.

   @param dog_pyr DoG scale space pyramid
   @param octv pixel's scale space octave
   @param intvl pixel's within-octave interval
   @param r pixel's image row
   @param c pixel's image col

   @return Returns 1 if the specified pixel is an extremum (max or min) among
   it's 3x3x3 pixel neighborhood.
   */
int Sift::is_extremum(vector<vector<Mat> >& dog_pyr, int octv, int intvl, int r, int c )
{
    double val = dog_pyr[octv][intvl].at<float>(r, c);
    int i, j, k;

    /* check for maximum */
    if( val > 0 )
    {
        for( i = -1; i <= 1; i++ )
        {
            for( j = -1; j <= 1; j++ )
            {
                for( k = -1; k <= 1; k++ )
                {
                    double neighbor = dog_pyr[octv][intvl+i].at<float>(r+j, c+k);
                    if( val < neighbor )
                        return 0;
                }
            }
        }
    }

    /* check for minimum */
    else
    {
        for( i = -1; i <= 1; i++ )
        {
            for( j = -1; j <= 1; j++ )
            {
                for( k = -1; k <= 1; k++ )
                {
                    double neighbor = dog_pyr[octv][intvl+i].at<float>(r+j, c+k);
                    if( val > neighbor )
                        return 0;
                }
            }
        }
    }

    return 1;
}

/*
   Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's
   paper.

   @param dog_pyr difference of Gaussians scale space pyramid
   @param octv octave of scale space
   @param intvl within-octave interval
   @param r pixel row
   @param c pixel column
   @param xi interpolated subpixel increment to interval
   @param xr interpolated subpixel increment to row
   @param xc interpolated subpixel increment to col

   @param Returns interpolated contrast.
   */
double Sift::interp_contr(vector<vector<Mat> >&  dog_pyr, int octv, int intvl, int r,
        int c, double xi, double xr, double xc )
{
    Mat dD, X, T;

    X = Mat::zeros(3, 1, CV_64FC1);
    X.at<double>(0,0) =xc;
    X.at<double>(1,0) =xr;
    X.at<double>(2,0) =xi;

    deriv_3D(dD, dog_pyr, octv, intvl, r, c );
    T = dD.t()*X;

    return dog_pyr[octv][intvl].at<float>(r, c) + T.at<double>(0,0) * 0.5;
}

/*
   Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's
   paper.

   @param dog_pyr difference of Gaussians scale space pyramid
   @param octv octave of scale space
   @param intvl interval being interpolated
   @param r row being interpolated
   @param c column being interpolated
   @param xi output as interpolated subpixel increment to interval
   @param xr output as interpolated subpixel increment to row
   @param xc output as interpolated subpixel increment to col
   */

void Sift::interp_step(vector<vector<Mat> >& dog_pyr, int octv, int intvl, int r, int c,
        double* xi, double* xr, double* xc )
{
    Mat dD, H, H_inv, X;
    X = Mat::zeros(3, 1, CV_64FC1);
    double x[3] = { 0 };

    deriv_3D(dD, dog_pyr, octv, intvl, r, c);

    hessian_3D(H, dog_pyr, octv, intvl, r, c);

    invert(H, H_inv, DECOMP_SVD);

    X=H_inv*(-1)*dD;

    *xi = X.at<double>(2, 0);
    *xr = X.at<double>(1, 0);
    *xc = X.at<double>(0, 0);
}

/*
   Interpolates a scale-space extremum's location and scale to subpixel
   accuracy to form an image feature.  Rejects features with low contrast.
   Based on Section 4 of Lowe's paper.  

   @param dog_pyr DoG scale space pyramid
   @param octv feature's octave of scale space
   @param intvl feature's within-octave interval
   @param r feature's image row
   @param c feature's image column
   @param intvls total intervals per octave
   @param contr_thr threshold on feature contrast

   @return Returns the feature resulting from interpolation of the given
   parameters or NULL if the given location could not be interpolated or
   if contrast at the interpolated loation was too low.  If a feature is
   returned, its scale, orientation, and descriptor are yet to be determined.
   */
int Sift::interp_extremum(feature& feat, vector<vector<Mat> >& dog_pyr, int octv,
        int intvl, int r, int c, int intvls,
        double contr_thr )
{
    //struct feature* feat;
    //struct detection_data* ddata;
    double xi, xr, xc, contr;
    int i = 0;

    while( i < SIFT_MAX_INTERP_STEPS )
    {
        interp_step( dog_pyr, octv, intvl, r, c, &xi, &xr, &xc );
        if( fabs( xi ) < 0.5  &&  fabs( xr ) < 0.5  &&  fabs( xc ) < 0.5 )
            break;

        c += cvRound( xc );
        r += cvRound( xr );
        intvl += cvRound( xi );

        if( intvl < 1  ||
                intvl > intvls  ||
                c < SIFT_IMG_BORDER  ||
                r < SIFT_IMG_BORDER  ||
                c >= dog_pyr[octv][0].cols - SIFT_IMG_BORDER  ||
                r >= dog_pyr[octv][0].rows - SIFT_IMG_BORDER )
        {
            return -1;
        }

        i++;
    }

    /* ensure convergence of interpolation */
    if( i >= SIFT_MAX_INTERP_STEPS )
        return -1;

    contr = interp_contr( dog_pyr, octv, intvl, r, c, xi, xr, xc );
    if( fabs( contr ) < contr_thr / intvls )
        return -1;

    feat.img_pt.x = feat.x = ( c + xc ) * pow( 2.0, octv );
    feat.img_pt.y = feat.y = ( r + xr ) * pow( 2.0, octv );
    feat.feature_data.r = r;
    feat.feature_data.c = c;
    feat.feature_data.octv = octv;
    feat.feature_data.intvl = intvl;
    feat.feature_data.subintvl = xi;

    return 0;
}

/*
  Determines whether a feature is too edge like to be stable by computing the
  ratio of principal curvatures at that feature.  Based on Section 4.1 of
  Lowe's paper.

  @param dog_img image from the DoG pyramid in which feature was detected
  @param r feature row
  @param c feature col
  @param curv_thr high threshold on ratio of principal curvatures

  @return Returns 0 if the feature at (r,c) in dog_img is sufficiently
    corner-like or 1 otherwise.
*/
int Sift::is_too_edge_like(Mat& dog_img, int r, int c, int curv_thr)
{
  double d, dxx, dyy, dxy, tr, det;

  /* principal curvatures are computed using the trace and det of Hessian */
  d = dog_img.at<float>(r, c);

  dxx = dog_img.at<float>(r, c+1) + dog_img.at<float>(r, c-1) - 2 * d;
  dyy = dog_img.at<float>(r+1, c) + dog_img.at<float>(r-1, c ) - 2 * d;
  dxy = (dog_img.at<float>(r+1, c+1) - dog_img.at<float>(r+1, c-1) -
	  dog_img.at<float>(r-1, c+1) + dog_img.at<float>(r-1, c-1)) / 4.0;
  tr = dxx + dyy;
  det = dxx * dyy - dxy * dxy;

  /* negative determinant -> curvatures have different signs; reject feature */
  if( det <= 0 )
    return 1;

  if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
    return 0;
  return 1;
}


/*
   Detects features at extrema in DoG scale space.  Bad features are discarded
   based on contrast and ratio of principal curvatures.

   @param dog_pyr DoG scale space pyramid
   @param octvs octaves of scale space represented by dog_pyr
   @param intvls intervals per octave
   @param contr_thr low threshold on feature contrast
   @param curv_thr high threshold on feature ratio of principal curvatures
   @param storage memory storage in which to store detected features

   @return Returns an array of detected features whose scales, orientations,
   and descriptors are yet to be determined.
   */
void Sift::scale_space_extrema(vector<feature>& features, vector<vector<Mat> >& dog_pyr, int octvs,
        double contr_thr, int curv_thr)
{
    double prelim_contr_thr = 0.5 * mContr_thr / mSampleIntervals;
    struct feature feat;
    struct detection_data* ddata;
    Mat feature_mat;

    for( int o = 0; o < octvs; o++ )
    {
        //feature_mat = calloc( dog_pyr[o][0]->height * dog_pyr[o][0]->width, sizeof(unsigned long) );
        feature_mat = Mat::zeros(dog_pyr[o][0].rows, dog_pyr[o][0].cols, CV_32S);
        for(int i = 1; i <= mSampleIntervals; i++ )
        {
            for(int r = SIFT_IMG_BORDER; r < dog_pyr[o][0].rows-SIFT_IMG_BORDER; r++)
            {
                for(int c = SIFT_IMG_BORDER; c < dog_pyr[o][0].cols-SIFT_IMG_BORDER; c++)
                {
                    /* perform preliminary check on contrast */
                    float contr = fabs(dog_pyr[o][i].at<float>(r, c));
                    if( contr > prelim_contr_thr )
                    {
                        if( is_extremum( dog_pyr, o, i, r, c ) )
                        {
                            int ret = interp_extremum(feat, dog_pyr, o, i, r, c, mSampleIntervals, contr_thr);
                            if(ret == 0)
                            {
                                ddata = &(feat.feature_data);
                                if( ! is_too_edge_like( dog_pyr[ddata->octv][ddata->intvl],
                                            ddata->r, ddata->c, curv_thr ) )
                                {
                                    //FIXME: double check here
                                    if( ddata->intvl > sizeof(unsigned long) )
                                    {
                                        features.push_back(feat);
                                    }
                                    else if( (feature_mat.at<int>(ddata->r, ddata->c) & (1 << ddata->intvl-1)) == 0 )
                                    {
                                        features.push_back(feat);
                                        feature_mat.at<int>(ddata->r, ddata->c) += 1 << ddata->intvl-1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
   Calculates characteristic scale for each feature in an array.

   @param features array of features
   @param sigma amount of Gaussian smoothing per octave of scale space
   @param intvls intervals per octave of scale space
   */
void Sift::calc_feature_scales(vector<feature>& features, double sigma, int intvls)
{
    struct feature feat;
    struct detection_data* ddata;
    double intvl;
    int i, n;

    n = features.size();

    for( i = 0; i < n; i++ )
    {
        //feat = CV_GET_SEQ_ELEM( struct feature, features, i );
        //ddata = feat_detection_data( feat );

        intvl = features[i].feature_data.intvl + features[i].feature_data.subintvl;

        features[i].scl = sigma * pow( 2.0, features[i].feature_data.octv + intvl / intvls );

        features[i].feature_data.scl_octv = sigma * pow( 2.0, intvl / intvls );
    }
}

/*
   Halves feature coordinates and scale in case the input image was doubled
   prior to scale space construction.

   @param features array of features
   */
void Sift::adjust_for_img_dbl(vector<feature>& features)
{
    int i, n;

    n = features.size();

    for( i = 0; i < n; i++ )
    {
        features[i].x /= 2.0;
        features[i].y /= 2.0;
        features[i].scl /= 2.0;
        features[i].img_pt.x /= 2.0;
        features[i].img_pt.y /= 2.0;
    }
}

/*
   Gaussian smooths an orientation histogram.

   @param hist an orientation histogram
   @param n number of bins
   */
void Sift::smooth_ori_hist( double* hist, int n )
{
    double prev, tmp, h0 = hist[0];
    int i;

    prev = hist[n-1];
    for( i = 0; i < n; i++ )
    {
        tmp = hist[i];
        hist[i] = 0.25 * prev + 0.5 * hist[i] + 
            0.25 * ( ( i+1 == n )? h0 : hist[i+1] );
        prev = tmp;
    }
}

/*
   Calculates the gradient magnitude and orientation at a given pixel.

   @param img image
   @param r pixel row
   @param c pixel col
   @param mag output as gradient magnitude at pixel (r,c)
   @param ori output as gradient orientation at pixel (r,c)

   @return Returns 1 if the specified pixel is a valid one and sets mag and
   ori accordingly; otherwise returns 0
   */
int Sift::calc_grad_mag_ori(Mat& img, int r, int c, double* mag,
        double* ori)
{
    double dx, dy;

    if( r > 0  &&  r < img.rows - 1  &&  c > 0  &&  c < img.cols - 1 )
    {
        dx = img.at<float>(r, c+1) - img.at<float>(r, c-1);
        dy = img.at<float>(r-1, c) - img.at<float>(r+1, c);

        *mag = sqrt(dx*dx + dy*dy);
        *ori = atan2(dy, dx);
        return 1;
    }

    else
        return 0;
}


/*
   Computes a gradient orientation histogram at a specified pixel.

   @param img image
   @param r pixel row
   @param c pixel col
   @param n number of histogram bins
   @param rad radius of region over which histogram is computed
   @param sigma std for Gaussian weighting of histogram entries

   @return Returns an n-element array containing an orientation histogram
   representing orientations between 0 and 2 PI.
   */
void Sift::ori_hist(double* hist, Mat& img, int r, int c, int n, int rad,
        double sigma)
{
    double mag, ori, w, exp_denom, PI2 = CV_PI * 2.0;
    int bin, i, j;

    exp_denom = 2.0 * sigma * sigma;
    for( i = -rad; i <= rad; i++ )
    {
        for( j = -rad; j <= rad; j++ )
        {
            if( calc_grad_mag_ori( img, r + i, c + j, &mag, &ori ) )
            {
                w = exp( -( i*i + j*j ) / exp_denom );
                bin = cvRound( n * ( ori + CV_PI ) / PI2 );
                bin = ( bin < n )? bin : 0;
                hist[bin] += w * mag;
            }
        }
    }
}


/*
   Computes a canonical orientation for each image feature in an array.  Based
   on Section 5 of Lowe's paper.  This function adds features to the array when
   there is more than one dominant orientation at a given feature location.

   @param features an array of image features
   @param gauss_pyr Gaussian scale space pyramid
   */
void Sift::calc_feature_oris(vector<feature>& features, vector<vector<Mat> >& gauss_pyr )
{
    //  struct feature* feat;
    //  struct detection_data* ddata;
    struct feature* feat;
    double* hist;
    double omax;
    int i, j, n = features.size();

    for( i = 0; i < n; i++ )
    {
        //        feat = malloc( sizeof( struct feature ) );
        feat = &features[i];

        //cvSeqPopFront( features, feat );

        //ddata = feat_detection_data( feat );
        double hist[SIFT_ORI_HIST_BINS];

        ori_hist(hist, gauss_pyr[feat->feature_data.octv][feat->feature_data.intvl],
                feat->feature_data.r, feat->feature_data.c, SIFT_ORI_HIST_BINS,
                cvRound(SIFT_ORI_RADIUS * feat->feature_data.scl_octv),
                SIFT_ORI_SIG_FCTR * feat->feature_data.scl_octv );

        for( j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++ )
            smooth_ori_hist(hist, SIFT_ORI_HIST_BINS);

        omax = dominant_ori(hist, SIFT_ORI_HIST_BINS);

        add_good_ori_features( features, hist, SIFT_ORI_HIST_BINS,
                omax * SIFT_ORI_PEAK_RATIO, feat );
    }
}

/*
   Interpolates a histogram peak from left, center, and right values
   */
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )

/*
   Adds features to an array for every orientation in a histogram greater than
   a specified threshold.

   @param features new features are added to the end of this array
   @param hist orientation histogram
   @param n number of bins in hist
   @param mag_thr new features are added for entries in hist greater than this
   @param feat new features are clones of this with different orientations
   */
void Sift::add_good_ori_features(vector<feature>& features, double* hist, int n,
        double mag_thr, struct feature* feat )
{
    struct feature new_feat;
    double bin, PI2 = CV_PI * 2.0;
    int l, r, i;

    for( i = 0; i < n; i++ )
    {
        l = ( i == 0 )? n - 1 : i-1;
        r = ( i + 1 ) % n;

        if( hist[i] > hist[l]  &&  hist[i] > hist[r]  &&  hist[i] >= mag_thr )
        {
            bin = i + interp_hist_peak( hist[l], hist[i], hist[r] );
            bin = ( bin < 0 )? n + bin : ( bin >= n )? bin - n : bin;
            new_feat = *feat;
            new_feat.ori = ( ( PI2 * bin ) / n ) - CV_PI;
            features.push_back(new_feat);
        }
    }
}

/*
   Finds the magnitude of the dominant orientation in a histogram

   @param hist an orientation histogram
   @param n number of bins

   @return Returns the value of the largest bin in hist
   */
double Sift::dominant_ori(double* hist, int n)
{
    double omax;
    int maxbin, i;

    omax = hist[0];
    maxbin = 0;
    for( i = 1; i < n; i++ )
        if( hist[i] > omax )
        {
            omax = hist[i];
            maxbin = i;
        }
    return omax;
}

/*
   Interpolates an entry into the array of orientation histograms that form
   the feature descriptor.

   @param hist 2D array of orientation histograms
   @param rbin sub-bin row coordinate of entry
   @param cbin sub-bin column coordinate of entry
   @param obin sub-bin orientation coordinate of entry
   @param mag size of entry
   @param d width of 2D array of orientation histograms
   @param n number of bins per orientation histogram
   */
void Sift::interp_hist_entry( double*** hist, double rbin, double cbin,
        double obin, double mag, int d, int n )
{
    double d_r, d_c, d_o, v_r, v_c, v_o;
    double** row, * h;
    int r0, c0, o0, rb, cb, ob, r, c, o;

    r0 = cvFloor( rbin );
    c0 = cvFloor( cbin );
    o0 = cvFloor( obin );
    d_r = rbin - r0;
    d_c = cbin - c0;
    d_o = obin - o0;

    /*
       The entry is distributed into up to 8 bins.  Each entry into a bin
       is multiplied by a weight of 1 - d for each dimension, where d is the
       distance from the center value of the bin measured in bin units.
       */
    for( r = 0; r <= 1; r++ )
    {
        rb = r0 + r;
        if( rb >= 0  &&  rb < d )
        {
            v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );
            row = hist[rb];
            for( c = 0; c <= 1; c++ )
            {
                cb = c0 + c;
                if( cb >= 0  &&  cb < d )
                {
                    v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
                    h = row[cb];
                    for( o = 0; o <= 1; o++ )
                    {
                        ob = ( o0 + o ) % n;
                        v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );
                        h[ob] += v_o;
                    }
                }
            }
        }
    }
}

/*
   Computes the 2D array of orientation histograms that form the feature
   descriptor.  Based on Section 6.1 of Lowe's paper.

   @param img image used in descriptor computation
   @param r row coord of center of orientation histogram array
   @param c column coord of center of orientation histogram array
   @param ori canonical orientation of feature whose descr is being computed
   @param scl scale relative to img of feature whose descr is being computed
   @param d width of 2d array of orientation histograms
   @param n bins per orientation histogram

   @return Returns a d x d array of n-bin orientation histograms.
   */
double*** Sift::descr_hist(Mat& img, int r, int c, double ori,
        double scl, int d, int n)
{
    double*** hist;
    double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag,
           grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * CV_PI;
    int radius, i, j;

    hist = (double***)calloc( d, sizeof( double** ) );
    for( i = 0; i < d; i++ )
    {
        hist[i] = (double**)calloc( d, sizeof( double* ) );
        for( j = 0; j < d; j++ )
            hist[i][j] = (double*)calloc( n, sizeof( double ) );
    }

    cos_t = cos( ori );
    sin_t = sin( ori );
    bins_per_rad = n / PI2;
    exp_denom = d * d * 0.5;
    hist_width = SIFT_DESCR_SCL_FCTR * scl;
    radius = hist_width * sqrt(2) * ( d + 1.0 ) * 0.5 + 0.5;
    for( i = -radius; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            /*
               Calculate sample's histogram array coords rotated relative to ori.
               Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
               r_rot = 1.5) have full weight placed in row 1 after interpolation.
               */
            c_rot = ( j * cos_t - i * sin_t ) / hist_width;
            r_rot = ( j * sin_t + i * cos_t ) / hist_width;
            rbin = r_rot + d / 2 - 0.5;
            cbin = c_rot + d / 2 - 0.5;

            if( rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d )
                if( calc_grad_mag_ori( img, r + i, c + j, &grad_mag, &grad_ori ))
                {
                    grad_ori -= ori;
                    while( grad_ori < 0.0 )
                        grad_ori += PI2;
                    while( grad_ori >= PI2 )
                        grad_ori -= PI2;

                    obin = grad_ori * bins_per_rad;
                    w = exp( -(c_rot * c_rot + r_rot * r_rot) / exp_denom );
                    interp_hist_entry( hist, rbin, cbin, obin, grad_mag * w, d, n );
                }
        }

    return hist;
}

/*
  Normalizes a feature's descriptor vector to unitl length

  @param feat feature
*/
void Sift::normalize_descr( struct feature* feat )
{
  double cur, len_inv, len_sq = 0.0;
  int i, d = feat->d;

  for( i = 0; i < d; i++ )
    {
      cur = feat->descr[i];
      len_sq += cur*cur;
    }
  len_inv = 1.0 / sqrt( len_sq );
  for( i = 0; i < d; i++ )
    feat->descr[i] *= len_inv;
}


/*
  Converts the 2D array of orientation histograms into a feature's descriptor
  vector.
  
  @param hist 2D array of orientation histograms
  @param d width of hist
  @param n bins per histogram
  @param feat feature into which to store descriptor
*/
void Sift::hist_to_descr( double*** hist, int d, int n, struct feature* feat )
{
  int int_val, i, r, c, o, k = 0;

  for( r = 0; r < d; r++ )
    for( c = 0; c < d; c++ )
      for( o = 0; o < n; o++ )
	feat->descr[k++] = hist[r][c][o];

  feat->d = k;
  normalize_descr( feat );
  for( i = 0; i < k; i++ )
    if( feat->descr[i] > SIFT_DESCR_MAG_THR )
      feat->descr[i] = SIFT_DESCR_MAG_THR;
  normalize_descr( feat );

  /* convert floating-point descriptor to integer valued descriptor */
  for( i = 0; i < k; i++ )
    {
      int_val = SIFT_INT_DESCR_FCTR * feat->descr[i];
      feat->descr[i] = MIN( 255, int_val );
    }
}

/*
  De-allocates memory held by a descriptor histogram

  @param hist pointer to a 2D array of orientation histograms
  @param d width of hist
*/
void Sift::release_descr_hist( double**** hist, int d )
{
  int i, j;

  for( i = 0; i < d; i++)
    {
      for( j = 0; j < d; j++ )
	free( (*hist)[i][j] );
      free( (*hist)[i] );
    }
  free( *hist );
  *hist = NULL;
}


/*
   Computes feature descriptors for features in an array.  Based on Section 6
   of Lowe's paper.

   @param features array of features
   @param gauss_pyr Gaussian scale space pyramid
   @param d width of 2D array of orientation histograms
   @param n number of bins per orientation histogram
   */
void Sift::compute_descriptors(vector<feature>& features, vector<vector<Mat> >& gauss_pyr, int d,
        int n)
{
    //    struct feature* feat;
    //    struct detection_data* ddata;
    double*** hist;
    int i, k = features.size();

    for( i = 0; i < k; i++ )
    {
        // feat = CV_GET_SEQ_ELEM( struct feature, features, i );
        // ddata = feat_detection_data( feat );
        struct feature* feat = &features[i];

        hist = descr_hist( gauss_pyr[feat->feature_data.octv][feat->feature_data.intvl], 
                feat->feature_data.r, feat->feature_data.c, feat->ori, 
                feat->feature_data.scl_octv, d, n );
        hist_to_descr( hist, d, n, feat );
        release_descr_hist( &hist, d );
    }
}


void Sift::compute(const cv::Mat& img)
{
    Mat init_img;

    create_init_img(init_img, img);

    //build scale space pyramid; smallest dimension of top level is ~4 pixels
    int octvs = log(MIN(init_img.rows, init_img.cols)) / log(2) - 2;

    vector<vector<Mat> > gauss_pyr;
    build_gauss_pyr(gauss_pyr, init_img, octvs);

    //show gauss pyr
#if 0
    for(int i=0; i<octvs; i++)
    {
        int n = gauss_pyr[i].size();
        for(int j=0; j<n; j++)
        {
            Mat show;
            gauss_pyr[i][j].convertTo(show, CV_8U, 255, 0);

            std::stringstream ss;
            ss<<"gauss["<<i<<"]["<<j<<"]"<<line;

            imshow(ss.str(), show);

            waitKey(50);
        }
    }
#endif

    vector<vector<Mat> > dog_pyr;
    build_dog_pyr(dog_pyr, gauss_pyr, octvs);

#if 0
    //show dog pyr
    for(int i=0; i<octvs; i++)
    {
        int n = dog_pyr[i].size();
        for(int j=0; j<n; j++)
        {
            Mat show;

            double vmin, vmax;

            minMaxIdx(dog_pyr[i][j], &vmin, &vmax);

            double den = 1;
            if (vmin!=vmax)
            {
                den = vmax-vmin;
            }
            dog_pyr[i][j].convertTo(show, CV_8U, 255.0/den, -255.0*vmin/den);

            std::stringstream ss;
            ss<<"dog["<<i<<"]["<<j<<"]";

            imshow(ss.str(), show);

            waitKey();
        }
    }
#endif

    vector<feature> features;
    scale_space_extrema(features, dog_pyr, octvs, mContr_thr, mCurv_thr);

    calc_feature_scales(features, mSigma, mSampleIntervals);

    if(mDbl)
        adjust_for_img_dbl(features);

    calc_feature_oris(features, gauss_pyr);
    compute_descriptors(features, gauss_pyr, mDescr_width, mDescr_hist_bins);

    cout<<"octvs: "<<octvs<<endl;
}
