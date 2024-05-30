#include "Deblur.h"
#include "MapPoint.h"

#include <mutex>

namespace ORB_SLAM3
{
    Deblur::Deblur(float gammaFactor, float threshold) : mfGammaFactor(gammaFactor), mfVarLaplacianThreshold(threshold), mnBasisKernel(25), mnGpuID(0), mbIsBlur(false)
    {
        torch::Device device(torch::kCUDA, mnGpuID);
    }

    void Deblur::VarLaplacian(const cv::Mat &im)
    {
        cv::Mat laplacian;
        cv::Laplacian(im, laplacian, CV_64F);
        cv::Scalar mean, stddev;
        cv::meanStdDev(laplacian, mean, stddev);
        float var = stddev.val[0] * stddev.val[0];
        std::cout << "Var of Laplacian: " << var << std::endl;
        // std::cout << " Threshold: " << mfVarLaplacianThreshold << std::endl;
        if (var <= mfVarLaplacianThreshold)
        {
            mbIsBlur = true;
        }
        else
        {
            mbIsBlur = false;
        }
    }

    void Deblur::ComputeKenrelsMasksMono(const cv::Mat &im)
    {
        torch::Tensor blurryTensor, blurryTensorNormalized, blurryTensorCorrected, cumulativeKernel, masks;

        std::string model_file("/root/datasets/TwoHeads.pt");
        torch::jit::script::Module two_heads;
        two_heads = torch::jit::load(model_file);
        two_heads.to(at::kCUDA);

        two_heads.eval();
        torch::InferenceMode no_grad;
        std::cout << "Kernel Eval Completed" << std::endl;

        // Blurry image is transformed to PyTorch format
        torch::Tensor blurry_tensor = torch::from_blob(im.data, {im.rows, im.cols, im.channels()}, torch::kByte);
        // blurry_tensor = blurry_tensor.to(torch::kFloat); // Convert to float tensor

        // Kernels and masks are estimated

        blurryTensorNormalized = blurry_tensor.div(255.0);
        blurryTensorCorrected = torch::pow(blurryTensorNormalized, mfGammaFactor) - 0.5;

        blurryTensorCorrected = blurryTensorCorrected.unsqueeze(0);
        blurryTensorCorrected = blurryTensorCorrected.permute({0, 3, 1, 2});
        
        blurryTensorCorrected = blurryTensorCorrected.to(torch::kCUDA);
        torch::Tensor kernels_estimated, masks_estimated;
        // Forward pass through the network

        // std::cout << blurryTensorCorrected[0][0][0] << std::endl;

        auto outputs = two_heads.forward({blurryTensorCorrected}).toTuple();
        mKernels = outputs->elements()[0].toTensor();
        mMasks = outputs->elements()[1].toTensor();
    }

    
    void Deblur::BlurKeyPointCulling(Tracking *pTracker)
    {
        unique_lock<mutex> lock(mMutex);
        int N, K, blurRowMax, blurRowMin, blurColMax, blurColMin, kernelDim;
        int lowestNonZeroRow, highestNonZeroRow, lowestNonZeroCol, highestNonZeroCol;
        int rowSpread, colSpread;
        float x, y;

        pTracker->mImGray.copyTo(mKeyIm);
        mvCurrentKeys = pTracker->mCurrentFrame.mvKeys;
        N = mvCurrentKeys.size();

        ComputeKenrelsMasksMono(mKeyIm);
        K = mMasks.size(1);

        // Assumes square Kernel
        kernelDim = mKernels.size(2);
        // int kernelH = mKernels.size(2);
        // int kernelW = mKernels.size(3);

        int countFilter = 0;
        int cullCount = 0;

        std::cout << "Total Keypoints: " << N << std::endl;
        for (int i = 0; i < N; i++)
        {
            MapPoint *pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if (!pMP)
                continue;

            if (pTracker->mCurrentFrame.mvbOutlier[i])
                continue;

            countFilter++;

            blurRowMax = 0;
            blurRowMin = 0;
            blurColMax = 0;
            blurColMin = 0;

            y = mvCurrentKeys[i].pt.x;
            x = mvCurrentKeys[i].pt.y;

            torch::Tensor cumulativeKernel = torch::zeros({kernelDim, kernelDim}, mKernels.options());

            for (int k = 0; k < K; ++k)
            {
                // std::cout << mMasks[0][k][x][y] << std::endl;
                // Select the value at (x, y) for each k from masks tensor
                float maskValue = mMasks[0][k][x][y].item<float>();
                // std::cout << maskValue << std::endl;
                // Get the corresponding kernel
                torch::Tensor weightedKernel = mKernels[0][k] * maskValue;
                // Multiply the mask value with the kernel value
                cumulativeKernel = cumulativeKernel + weightedKernel;
            }

            auto cpu_tensor = cumulativeKernel.to(torch::kCPU);

            auto min_val = cpu_tensor.min().item<float>();
            auto max_val = cpu_tensor.max().item<float>();

            std::cout.precision(5);
            // std::cout << std::scientific;
            // std::cout << "Kernel max_val, min_val " << max_val << min_val << std::endl;

            auto indices = torch::argmax(cpu_tensor, 0);

            // Normalize the tensor values between 0 and 1
            auto normalized_tensor = (cpu_tensor - min_val) / (max_val - min_val);
            normalized_tensor = normalized_tensor * 255.0;

            // If needed, convert the normalized tensor to a different data type (e.g., float to uint8)
            auto uint8_kernel = normalized_tensor.to(torch::kU8);

            lowestNonZeroRow = -1;  // Initialize with an invalid value
            highestNonZeroRow = -1; // Initialize with an invalid value
            lowestNonZeroCol = -1;  // Initialize with an invalid value
            highestNonZeroCol = -1; // Initialize with an invalid value

            for (int k = 0; k < kernelDim; ++k)
            {
                // Extract the row as a sub-tensor
                torch::Tensor rowTensor = uint8_kernel[k];
                torch::Tensor colTensor = uint8_kernel.select(1, k);

                // Check if any non-zero elements are present in the row
                // std::cout << rowTensor << std::endl;

                if (rowTensor.max().item<uint8_t>() > 150)
                {
                    if (lowestNonZeroRow == -1)
                    {
                        lowestNonZeroRow = k;
                    }
                    highestNonZeroRow = k; // Update highestNonZeroRow with the current row
                }

                if (colTensor.max().item<uint8_t>() > 150)
                {
                    if (lowestNonZeroCol == -1)
                    {
                        lowestNonZeroCol = k;
                    }
                    highestNonZeroCol = k; // Update highestNonZeroCol with the current column
                }

                // if (rowTensor.nonzero().size(0) > 0)
                // {
                //     if (lowestNonZeroRow == -1)
                //     {
                //         lowestNonZeroRow = k;
                //     }
                //     highestNonZeroRow = k; // Update highestNonZeroRow with the current row
                // }

                // if (colTensor.nonzero().size(0) > 0)
                // {
                //     if (lowestNonZeroCol == -1)
                //     {
                //         lowestNonZeroCol = k;
                //     }
                //     highestNonZeroCol = k; // Update highestNonZeroCol with the current column
                // }
            }

            rowSpread = highestNonZeroRow - lowestNonZeroRow;
            colSpread = highestNonZeroCol - lowestNonZeroCol;

            if (rowSpread > 9 || colSpread > 9)
            {
                // std::cout << "RowColSpread: " << rowSpread << colSpread << std::endl;

                if (!pMP->isBlur())
                {
                    // std::cout << "Before Cull:  " << pMP->isBlur() << "id: " << pMP->mnId;
                    pMP->SetBlurFlag();
                    // std::cout << " After Cull: " << pMP->isBlur() << "id: " << pMP->mnId << std::endl;
                    // pMP->SetBadFlag();
                }

                // std::cout << "ToDo::::Delete this Keypoint" << std::endl;
                cullCount++;
            }
        }
        std::cout << "Filtered Keypoints: " << countFilter << std::endl;
        std::cout << "Culled Keypoints: " << cullCount << std::endl;
    }

    bool Deblur::ImageBlurReport()
    {
        return mbIsBlur;
    }
}