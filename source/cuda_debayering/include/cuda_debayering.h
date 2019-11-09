/**
 * Demosaicing using CUDA
 *
 * Author: Lingjian Kong
 *
 * Please make sure that the raw image satisfies the following requirement before
 * using the interface provided in this header file:
 *
 * 1. The bayered image must have CRBC pattern in the following format:
 * 
 * C R C R C R
 * B C B C B C
 * C R C R C R
 * B C B C B C
 * C R C R C R
 * B C B C B C
 * 
 * That is, the first element in the raw array must be C;
 * the second element in the raw array is must R, and so on.
 * That's being said, if your raw array has a RCCB pattern, the color will be messed up.
 * 
 * 2. Each pixel in the original bayered image must be 12 bits
 * which are stored in a 16 bits uint16_t structure.
 * 
 * 3. Information in C (clear) channel has already been compensated by setting proper gains for
 * R and B gain during image capture, so that we can treat C channel as G channel.
 * If this assumption is not valid, the color might be skewed.
 *
 * 4. CUDA kernels have been specifically optimize for image dimension of 
 * 1920 by 1208 captured by Motec cameras. If your raw image does not have this demension,
 * then the CUDA calls might fail.
 *
 */

#ifndef CUDA_DEBAYERING_H
#define CUDA_DEBAYERING_H

#include <stdint.h>
#include <stdio.h>

namespace brt {
namespace cuda {

/**
 * @brief Debayering in CUDA using bilinear interpolation.
 * @param bayeredImg Input bayered image.
 *     See top of this header file for specific requirement
 * @param debayeredImg Output debayered image.
 *     If downsample is set to true, then this debayeredImg pointer should point to an array with
 *     3 * (IMG_WIDTH / 2) * (IMG_HEIGHT / 2) * sizeof(uint8_t) bytes preallocated.
 *     If doDownsample is set to false,
 *     then this debayeredImg pointer should point to an array with
 *     3 * IMG_WIDTH * IMG_HEIGHT * sizeof(uint8_t) bytes preallocated.
 * @param outputBGR If set to true, the output debayered array will be in BGRBGRBGR...
 *     otherwise, the output debayered array will be in RGBRGBRGB...
 * @param doDownsample If set to ture, the output image will be downsampled to half the size
 *     of the original image AFTER debayering using biliear interpolation
 *     (which result in higher image quality than debayering-using-downsample).
 *     Pay special attention to the memory you want to allocate for output debayeredImg
 *     depending on whether you want to downsample or not.
 */
void debayerUsingBilinearInterpolation(const uint16_t* bayeredImg, uint8_t* debayeredImg, const bool outputBGR, const bool doDownsample);

/**
 * @brief Debayering in CUDA using downsample.
 *     *WARNING* you should generally prefer NOT to use this function since it does not have as good image quality
 *     as debayering using bilinear interpolation.
 *     Use this function only when speed is absolutely necessary and you are ok with sacrificing image quality.
 *     If you just want to get a debayered downsampled image in the end, simply
 *     use debayerWithBilinearInterpolation with doDownsample option enabled instead.
 * @param bayeredImg Input bayered image, see top of this header file for specific requirement
 *     this downsampledDebayeredImg pointer should point to an array with
 *     3 * (IMG_WIDTH / 2) * (IMG_HEIGHT / 2) * sizeof(uint8_t) bytes preallocated.
 * @param debayeredImg Output debayered image downsampled to half the size of the original image.
 * @param outputBGR If set to true, the output debayered array will be BGRBGRBGR...
 *     otherwise, the output debayered array will be RGBRGBRGB...
 */
void debayerUsingDownsample(const uint16_t* bayeredImg, uint8_t* downsampledDebayeredImg, const bool outputBGR);

} // end namespace brt
} // end namespace cuda

#endif // CUDA_DEBAYERING_H
