// #include <torch/torch.h>
// #include <torch/script.h>
// #include <iostream>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>

#include "Tracking.h"

#include <vector>

namespace ORB_SLAM3
{
    class Deblur
    {
    public:
        Deblur(float gammaFactor, float threshold);

        // Destructor
        // ~Deblur();

        void VarLaplacian(const cv::Mat &im);
        void ComputeKenrelsMasksMono(const cv::Mat &im);
        void ComputeKernelsMasksStereo(const cv::Mat &imLeft, const cv::Mat &imRight);
        // void RepostionPoints(cv::Mat &pixLoc, torch::Tensor &masks, torch::Tensor &kernels);
        // void RepositionPoints(const cv::Mat &points);
        void BlurKeyPointCulling(Tracking *pTracker);
        bool ImageBlurReport();

    protected:
        bool mbIsBlur;
        float mfGammaFactor;
        float mfVarLaplacianThreshold;
        int mnGpuID;
        int mnBasisKernel;
        std::string msModelFile;
        std::mutex mMutex;

        cv::Mat mImBlur, mImBlurRight, mKeyIm;
        cv::Mat mDeblurPoints;
        std::vector<cv::KeyPoint> mvCurrentKeys, mvCurrentKeysRight;

        torch::jit::script::Module twoHeadNet;
        torch::Tensor blurryTensor, blurryTensorNormalized, blurryTensorCorrected, mKernels, mMasks;
    };
} // namespace ORB_SLAM3