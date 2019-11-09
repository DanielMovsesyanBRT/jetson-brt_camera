#include <cuda_debayering.h>

#include <string>
#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>

#define IMG_WIDTH 1920
#define IMG_HEIGHT 1208
#define HEADER_BYTES 12

void test_bilinear_debayering()
{
    const bool outputBGR = true;
    const bool doDownsample = true;

    std::string pathToRawImage = "../sample_data/example.raw";
    FILE *file = std::fopen(pathToRawImage.c_str(), "rb");
    std::fseek(file, 0, SEEK_END);
    size_t fileSize = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);

    assert((int) fileSize / (int) IMG_WIDTH / (int) IMG_HEIGHT == sizeof(uint16_t));

    uint16_t *inputImgWithHeaderBuffer = (uint16_t*)malloc(fileSize);
    std::fread(inputImgWithHeaderBuffer, sizeof(uint16_t), fileSize / sizeof(uint16_t), file);

    /* Bilinear debayering */
    size_t outputImgWidth;
    size_t outputImgHeight;
    uint8_t* outputImgBuffer;

    if (doDownsample)
    {
        outputImgWidth = IMG_WIDTH / 2;
        outputImgHeight = IMG_HEIGHT / 2;
    } else
    {
        outputImgWidth = IMG_WIDTH;
        outputImgHeight = IMG_HEIGHT;

    }

    outputImgBuffer = (uint8_t*)malloc(outputImgWidth * outputImgHeight * 3 * sizeof(uint8_t));
    memset(outputImgBuffer, 128, outputImgWidth * outputImgHeight * 3 * sizeof(uint8_t));

    brt::cuda::debayerUsingBilinearInterpolation(inputImgWithHeaderBuffer + HEADER_BYTES, outputImgBuffer, outputBGR, doDownsample);

    cv::Mat outputImage = cv::Mat(outputImgHeight, outputImgWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    uint8_t *itr = outputImgBuffer;
    for (size_t i = 0; i < outputImgHeight; ++i)
    {
        for (size_t j = 0; j < outputImgWidth; ++j)
        {
            uint8_t b = *itr++;
            uint8_t g = *itr++;
            uint8_t r = *itr++;

            outputImage.at<cv::Vec3b>(cv::Point(j, i)) = {b, g, r};
        }
    }
    cv::imwrite("debayered_with_bilinear_interpolation.png", outputImage);
}

void test_downsample_debayering()
{
    const bool outputBGR = true;

    std::string pathToRawImage = "../sample_data/example.raw";
    FILE *file = std::fopen(pathToRawImage.c_str(), "rb");
    std::fseek(file, 0, SEEK_END);
    size_t fileSize = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);

    assert((int) fileSize / (int) IMG_WIDTH / (int) IMG_HEIGHT == sizeof(uint16_t));

    uint16_t *inputImgWithHeaderBuffer = (uint16_t*)malloc(fileSize);
    std::fread(inputImgWithHeaderBuffer, sizeof(uint16_t), fileSize / sizeof(uint16_t), file);

    /* Downsample debayering */
    const size_t outputImgWidth = IMG_WIDTH / 2;
    const size_t outputImgHeight = IMG_HEIGHT / 2;
    uint8_t *outputImgBuffer = (uint8_t*)malloc(outputImgWidth * outputImgHeight * 3 * sizeof(uint8_t));
    memset(outputImgBuffer, 128, outputImgWidth * outputImgHeight * 3 * sizeof(uint8_t));

    brt::cuda::debayerUsingDownsample(inputImgWithHeaderBuffer + HEADER_BYTES, outputImgBuffer, outputBGR);

    cv::Mat outputImage = cv::Mat(outputImgHeight, outputImgWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    uint8_t *itr = outputImgBuffer;
    for (size_t i = 0; i < outputImgHeight; ++i)
    {
        for (size_t j = 0; j < outputImgWidth; ++j)
        {
            uint8_t b = *itr++;
            uint8_t g = *itr++;
            uint8_t r = *itr++;

            outputImage.at<cv::Vec3b>(cv::Point(j, i)) = {b, g, r};
        }
    }
    cv::imwrite("debayered_with_downsample.png", outputImage);
}

int main(int argc, char** argv)
{
    test_bilinear_debayering();
    test_downsample_debayering();
    printf("Cuda debayering unit test finished. Output debayered images are saved to the build folder.\n");
}
