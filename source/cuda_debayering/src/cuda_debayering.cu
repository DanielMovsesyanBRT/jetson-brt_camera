#include <cuda_debayering.h>

# define IMG_WIDTH 1920
# define IMG_HEIGHT 1208
//# define IMG_HEIGHT 1080

//# define PIXEL_FIX(x)   ( ((x) >> 8) | (((x) & 0xF0) << 8) )
# define PIXEL_FIX(x)   (x)

// TODO - Lingjian : Cuda function call wrappers should go to
// a shared place when Embedded and CVML are merged.

// Error checking macro wrapper on every CUDA call.
// It will print a red message if a CUDA call fails.
#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=false)
{
    if (code != cudaSuccess) 
    {
        printf("\033[1;31m");
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        printf("\033[0m;");
        if (abort)
        {
            exit(code);
        }
    }
}

namespace brt {
namespace cuda {

// Downsample an debayered RGBRGBRGB (or BGRBGRBGR) image to half of its size.
__global__ void downsampleDebayeredImgCudaImpl(const uint8_t* debayeredImg, uint8_t* downsampledDebayeredImg)
{
    int x = 2 * ((blockIdx.x * blockDim.y) + threadIdx.y);
    int y = 2 * ((blockIdx.y * blockDim.x) + threadIdx.x);

    uint8_t output_c1, upper_left_c1, upper_right_c1, lower_left_c1, lower_right_c1;
    uint8_t output_c2, upper_left_c2, upper_right_c2, lower_left_c2, lower_right_c2;
    uint8_t output_c3, upper_left_c3, upper_right_c3, lower_left_c3, lower_right_c3;

    upper_left_c1 = debayeredImg[3 * (y * IMG_WIDTH + x)];
    upper_left_c2 = debayeredImg[3 * (y * IMG_WIDTH + x) + 1];
    upper_left_c3 = debayeredImg[3 * (y * IMG_WIDTH + x) + 2];

    upper_right_c1 = debayeredImg[3 * (y * IMG_WIDTH + (x+1))];
    upper_right_c2 = debayeredImg[3 * (y * IMG_WIDTH + (x+1)) + 1];
    upper_right_c3 = debayeredImg[3 * (y * IMG_WIDTH + (x+1)) + 2];

    lower_left_c1 = debayeredImg[3 * ((y+1) * IMG_WIDTH + x)];
    lower_left_c2 = debayeredImg[3 * ((y+1) * IMG_WIDTH + x) + 1];
    lower_left_c3 = debayeredImg[3 * ((y+1) * IMG_WIDTH + x) + 2];

    lower_right_c1 = debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1))];
    lower_right_c2 = debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1)) + 1];
    lower_right_c3 = debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1)) + 2];

    output_c1 = (uint8_t) ( ( (int)upper_left_c1 + (int)upper_right_c1 + (int)lower_left_c1 + (int)lower_right_c1 ) / 4 );
    output_c2 = (uint8_t) ( ( (int)upper_left_c2 + (int)upper_right_c2 + (int)lower_left_c2 + (int)lower_right_c2 ) / 4 );
    output_c3 = (uint8_t) ( ( (int)upper_left_c3 + (int)upper_right_c3 + (int)lower_left_c3 + (int)lower_right_c3 ) / 4 );

    downsampledDebayeredImg[3 * ((y/2) * (IMG_WIDTH/2) + (x/2))] = output_c1;
    downsampledDebayeredImg[3 * ((y/2) * (IMG_WIDTH/2) + (x/2)) + 1] = output_c2;
    downsampledDebayeredImg[3 * ((y/2) * (IMG_WIDTH/2) + (x/2)) + 2] = output_c3;
}

// Debayer an CRBC bayered image using bilinear interpolation
// and output debayered image in RGBRGBRGB (or BGRBGRBGR) in its
// original resolution.
__global__ void debayerUsingBilinearInterpolationCudaImpl(const uint16_t* bayeredImg, uint8_t* debayeredImg, const bool outputBGR)
{
    // The bayered image must have the following format (when expanded to 2D):
    //
    // C R C R C R
    // B C B C B C
    // C R C R C R
    // B C B C B C
    // C R C R C R
    // B C B C B C
    //
    // where upper left corner (i.e. the first element in the array is C,
    // and the second element in the array is R).
    //
    // Other format might work, but requires rewriting this kernel
    // Otherwise the color will be messed up.
    //
    // Also, each pixel in the original bayered image but be 12 bits
    // which stored in a 16 bit uint16_t structure.
    //
    // We will treat C channel as G channel in this kernel, because during image capture,
    // we have already set the proper gain for R and B.
    //
    // (x, y) is the coordinate how we will inspect the bayered pattern
    // Note that x and y are only even numbers, meaning that
    // in every kernel, we will perform bilinear interpolation for four pixels.
    // Therefore, for image size of 1920 * 1208, this kernel is called 960 * 604 times

    int x = 2 * ((blockIdx.x * blockDim.y) + threadIdx.y);
    int y = 2 * ((blockIdx.y * blockDim.x) + threadIdx.x);


    uint8_t b, g, r;


    /* Upper left: C */
    if (x == 0 && y == 0)
    {
        g = (uint8_t) ( PIXEL_FIX( bayeredImg[y * IMG_WIDTH + x]) >> 8 );

        r = (uint8_t) ( PIXEL_FIX( bayeredImg[y * IMG_WIDTH + (x+1)]) >> 8 );

        b = (uint8_t) ( PIXEL_FIX( bayeredImg[(y+1) * IMG_WIDTH + x]) >> 8 );

    } else if (x == 0)
    {
        g = (uint8_t) ( PIXEL_FIX( bayeredImg[y * IMG_WIDTH + x]) >> 8 );

        r = (uint8_t) ( PIXEL_FIX( bayeredImg[y * IMG_WIDTH + (x+1)]) >> 8 );

        b = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x])
                          ) >> 8
                        ) / 2
                      );

    } else if (y == 0)
    {
        g = (uint8_t) (PIXEL_FIX( bayeredImg[y * IMG_WIDTH + x]) >> 8 );

        r = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x-1)]) +
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)])
                          ) >> 8
                        ) / 2
                      );

        b = (uint8_t) (PIXEL_FIX( bayeredImg[(y+1) * IMG_WIDTH + x]) >> 8 );

    } else
    {
        g = (uint8_t) (PIXEL_FIX( bayeredImg[y * IMG_WIDTH + x]) >> 8 );

        r = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x-1)]) +
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)])
                          ) >> 8
                        ) / 2
                      );

        b = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x])
                          ) >> 8
                        ) / 2
                      );
    }

    if (outputBGR)
    {
        debayeredImg[3 * (y * IMG_WIDTH + x)] = b;
        debayeredImg[3 * (y * IMG_WIDTH + x) + 1] = g;
        debayeredImg[3 * (y * IMG_WIDTH + x) + 2] = r;
    } else
    {
        debayeredImg[3 * (y * IMG_WIDTH + x)] = r;
        debayeredImg[3 * (y * IMG_WIDTH + x) + 1] = g;
        debayeredImg[3 * (y * IMG_WIDTH + x) + 2] = b;
    }

    /* Upper right: R */
    if (x == IMG_WIDTH - 2 && y == 0)
    {
        r = (uint8_t) ( PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) >> 8 );

        g = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)])
                          ) >> 8
                        ) / 2
                      );

        b = (uint8_t) ( PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) >> 8 );

    } else if (y == 0)
    {
        r = (uint8_t) ( PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) >> 8 );

        g = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+2)]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)])
                          ) >> 8
                        ) / 3
                      );

        b = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+2)])
                          ) >> 8
                        ) / 2
                      );

    } else if (x == IMG_WIDTH - 2)
    {
        r = (uint8_t) ( PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) >> 8 );

        g = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + (x+1)]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)])
                          ) >> 8
                        ) / 3
                      );

        b = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x])
                          ) >> 8
                        ) / 2
                      );

    } else
    {
        r = (uint8_t) ( PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) >> 8 );

        g = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+2)]) +
                            (int)PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + (x+1)]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)])
                          ) >> 8
                        ) / 4
                      );

        b = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + (x+2)]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+2)])
                          ) >> 8
                        ) / 4
                      );
    }

    if (outputBGR)
    {
        debayeredImg[3 * (y * IMG_WIDTH + (x+1))] = b;
        debayeredImg[3 * (y * IMG_WIDTH + (x+1)) + 1] = g;
        debayeredImg[3 * (y * IMG_WIDTH + (x+1)) + 2] = r;
    } else
    {
        debayeredImg[3 * (y * IMG_WIDTH + (x+1))] = r;
        debayeredImg[3 * (y * IMG_WIDTH + (x+1)) + 1] = g;
        debayeredImg[3 * (y * IMG_WIDTH + (x+1)) + 2] = b;
    }


    /* Lower left: B */
    if (x == 0 && y == IMG_HEIGHT - 2)
    {
        b = (uint8_t) ( PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) >> 8 );

        r = (uint8_t) ( PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) >> 8 );

        g = (uint8_t) (
                        (
                          ( (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)])
                          ) >> 8
                        ) / 2
                      );
        
    } else if (x == 0)
    {
        b = (uint8_t) ( PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) >> 8 );

        r = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) +
                            (int)PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + (x+1)])
                          ) >> 8
                        ) / 2
                      );

        g = (uint8_t) (
                        (
                          ( (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]) +
                            (int)PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + x])
                          ) >> 8
                        ) / 3
                      );

    } else if (y == IMG_HEIGHT - 2)
    {
        b = (uint8_t) ( PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) >> 8 );

        r = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x-1)]) +
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)])
                          ) >> 8
                        ) / 2
                      );

        g = (uint8_t) (
                        (
                          ( (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x-1)])
                          ) >> 8
                        ) / 3
                      );

    } else
    {
        b = (uint8_t) ( PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) >> 8 );

        r = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x-1)]) +
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) +
                            (int)PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + (x-1)]) +
                            (int)PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + (x+1)])
                          ) >> 8
                        ) / 4
                      );

        g = (uint8_t) (
                        (
                          ( (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]) +
                            (int)PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x-1)])
                          ) >> 8
                        ) / 4
                      );
    }

    if (outputBGR)
    {
        debayeredImg[3 * ((y+1) * IMG_WIDTH + x)] = b;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + x) + 1] = g;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + x) + 2] = r;
    } else
    {
        debayeredImg[3 * ((y+1) * IMG_WIDTH + x)] = r;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + x) + 1] = g;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + x) + 2] = b;
    }

    /* Lower right: C */
    if (x == IMG_WIDTH - 2 && y == IMG_HEIGHT - 2)
    {
        g = (uint8_t) ( PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]) >> 8 );

        r = (uint8_t) ( PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) >> 8 );

        b = (uint8_t) ( PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) >> 8 );

    } else if (x == IMG_WIDTH - 2)
    {
        g = (uint8_t) ( PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]) >> 8 );

        r = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) +
                            (int)PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + (x+1)])
                          ) >> 8
                        ) / 2
                      );

        b = (uint8_t) ( PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) >> 8 );

    } else if (y == IMG_HEIGHT - 2)
    {
        g = (uint8_t) ( PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]) >> 8 );

        r = (uint8_t) ( PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)])  >> 8 );

        b = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+2)])
                          ) >> 8
                        ) / 2
                      );

    } else 
    {
        g = (uint8_t) ( PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]) >> 8 );

        r = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) +
                            (int)PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + (x+1)])
                          ) >> 8
                        ) / 2
                      );

        b = (uint8_t) (
                        (
                          (
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) +
                            (int)PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+2)])
                          ) >> 8
                        ) / 2
                      );
    }

    if (outputBGR)
    {
        debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1))] = b;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1)) + 1] = g;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1)) + 2] = r;
    } else
    {
        debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1))] = r;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1)) + 1] = g;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1)) + 2] = b;
    }
}

// Debayer an CRBC bayered image using bilinear interpolation
// and output debayered image in RGBRGBRGB (or BGRBGRBGR) in its
// original resolution.
__global__ void bilinearInterpolationDebayer16CudaImpl(const uint16_t* bayeredImg, uint16_t* debayeredImg, const bool outputBGR)
{
    // The bayered image must have the following format (when expanded to 2D):
    //
    // C R C R C R
    // B C B C B C
    // C R C R C R
    // B C B C B C
    // C R C R C R
    // B C B C B C
    //
    // where upper left corner (i.e. the first element in the array is C,
    // and the second element in the array is R).
    //
    // Other format might work, but requires rewriting this kernel
    // Otherwise the color will be messed up.
    //
    // Also, each pixel in the original bayered image but be 12 bits
    // which stored in a 16 bit uint16_t structure.
    //
    // We will treat C channel as G channel in this kernel, because during image capture,
    // we have already set the proper gain for R and B.
    //
    // (x, y) is the coordinate how we will inspect the bayered pattern
    // Note that x and y are only even numbers, meaning that
    // in every kernel, we will perform bilinear interpolation for four pixels.
    // Therefore, for image size of 1920 * 1208, this kernel is called 960 * 604 times

    int x = 2 * ((blockIdx.x * blockDim.y) + threadIdx.y);
    int y = 2 * ((blockIdx.y * blockDim.x) + threadIdx.x);


    uint16_t b, g, r;


    /* Upper left: C */
    if (x == 0 && y == 0)
    {
        g = PIXEL_FIX( bayeredImg[y * IMG_WIDTH + x]);
        r = PIXEL_FIX( bayeredImg[y * IMG_WIDTH + (x+1)]);
        b = PIXEL_FIX( bayeredImg[(y+1) * IMG_WIDTH + x]);
    }
    else if (x == 0)
    {
        g = PIXEL_FIX( bayeredImg[y * IMG_WIDTH + x]);
        r = PIXEL_FIX( bayeredImg[y * IMG_WIDTH + (x+1)]);
        b = (PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x])) / 2;
    }
    else if (y == 0)
    {
        g = PIXEL_FIX( bayeredImg[y * IMG_WIDTH + x]);
        r = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x-1)]) + PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)])) / 2;
        b = PIXEL_FIX( bayeredImg[(y+1) * IMG_WIDTH + x]);
    }
    else
    {
        g = PIXEL_FIX( bayeredImg[y * IMG_WIDTH + x]);
        r = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x-1)]) + PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)])) / 2;
        b = (PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x])) / 2;
    }

    if (outputBGR)
    {
        debayeredImg[3 * (y * IMG_WIDTH + x)] = b;
        debayeredImg[3 * (y * IMG_WIDTH + x) + 1] = g;
        debayeredImg[3 * (y * IMG_WIDTH + x) + 2] = r;
    } else
    {
        debayeredImg[3 * (y * IMG_WIDTH + x)] = r;
        debayeredImg[3 * (y * IMG_WIDTH + x) + 1] = g;
        debayeredImg[3 * (y * IMG_WIDTH + x) + 2] = b;
    }

    /* Upper right: R */
    if (x == IMG_WIDTH - 2 && y == 0)
    {
        r = PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]);
        g = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)])) / 2;
        b = PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]);
    }
    else if (y == 0)
    {
        r = PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]);
        g = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+2)]) +
                                      PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)])) / 3;

        b = (PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+2)])) / 2;
    }
    else if (x == IMG_WIDTH - 2)
    {
        r = PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]);
        g = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + (x+1)]) +
                                      PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)])) / 3;

        b = (PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x])) / 2;
    }
    else
    {
        r = PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]);
        g = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+2)]) +
                            PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + (x+1)]) +PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)])) / 4;

        b = (PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y-1) * IMG_WIDTH + (x+2)]) +
                            PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+2)])) / 4;
    }

    if (outputBGR)
    {
        debayeredImg[3 * (y * IMG_WIDTH + (x+1))] = b;
        debayeredImg[3 * (y * IMG_WIDTH + (x+1)) + 1] = g;
        debayeredImg[3 * (y * IMG_WIDTH + (x+1)) + 2] = r;
    }
    else
    {
        debayeredImg[3 * (y * IMG_WIDTH + (x+1))] = r;
        debayeredImg[3 * (y * IMG_WIDTH + (x+1)) + 1] = g;
        debayeredImg[3 * (y * IMG_WIDTH + (x+1)) + 2] = b;
    }

    /* Lower left: B */
    if (x == 0 && y == IMG_HEIGHT - 2)
    {
        b = PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]);
        r = PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]);
        g = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)])) / 2;
    }
    else if (x == 0)
    {
        b = PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]);
        r = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) + PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + (x+1)])) / 2;
        g = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]) +
                            PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + x])) / 3;
    }
    else if (y == IMG_HEIGHT - 2)
    {
        b = PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]);
        r = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x-1)]) + PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)])) / 2;
        g = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]) +
                            PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x-1)])) / 3;
    }
    else
    {
        b = PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]);
        r = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x-1)]) + PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) +
                            PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + (x-1)]) + PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + (x+1)])) / 4;

        g = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]) +
                            PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x-1)])) / 4;
    }

    if (outputBGR)
    {
        debayeredImg[3 * ((y+1) * IMG_WIDTH + x)] = b;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + x) + 1] = g;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + x) + 2] = r;
    }
    else
    {
        debayeredImg[3 * ((y+1) * IMG_WIDTH + x)] = r;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + x) + 1] = g;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + x) + 2] = b;
    }

    /* Lower right: C */
    if (x == IMG_WIDTH - 2 && y == IMG_HEIGHT - 2)
    {
        g = PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]);
        r = PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]);
        b = PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]);
    }
    else if (x == IMG_WIDTH - 2)
    {
        g = PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]);
        r = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) + PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + (x+1)])) / 2;
        b = PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]);
    }
    else if (y == IMG_HEIGHT - 2)
    {
        g = PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]);
        r = PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]);
        b = (PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+2)])) / 2;
    }
    else
    {
        g = PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+1)]);
        r = (PIXEL_FIX(bayeredImg[y * IMG_WIDTH + (x+1)]) + PIXEL_FIX(bayeredImg[(y+2) * IMG_WIDTH + (x+1)])) / 2;
        b = (PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + x]) + PIXEL_FIX(bayeredImg[(y+1) * IMG_WIDTH + (x+2)])) / 2;
    }

    if (outputBGR)
    {
        debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1))] = b;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1)) + 1] = g;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1)) + 2] = r;
    }
    else
    {
        debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1))] = r;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1)) + 1] = g;
        debayeredImg[3 * ((y+1) * IMG_WIDTH + (x+1)) + 2] = b;
    }
}

// Debayer an CRBC bayered image using downsampling
// and output the downsampled debayered image in
// RGBRGBRGB (or BGRBGRBGR)
__global__ void debayerUsingDownsampleCudaImpl(const uint16_t* bayeredImg, uint8_t* downsampledDebayeredImg, const bool outputBGR)
{
    // Each kernel calculated pixel x, y for the downsampled image of size (960, 604) (width by height)
    // blockDim.x is the width of downsampled image (960), so x is represented by threadIdx.x
    // gridDim.x is the height of downsampled image (604), so y is represented by blockIdx.x
    //
    // The bayered image must have the following format:
    //
    // C R C R C R
    // B C B C B C
    // C R C R C R
    // B C B C B C
    // C R C R C R
    // B C B C B C

    int x = threadIdx.x;
    int y = blockIdx.x;
    int idx = y * blockDim.x + x;

    uint8_t upper_left = (uint8_t) (bayeredImg[2 * y * IMG_WIDTH + (2 * x)] >> 8);
    uint8_t upper_right = (uint8_t) (bayeredImg[2 * y * IMG_WIDTH + 2 * x + 1] >> 8);
    uint8_t lower_left = (uint8_t) (bayeredImg[((2 * y) + 1) * IMG_WIDTH + (2 * x)] >> 8);
    uint8_t lower_right = (uint8_t) (bayeredImg[((2 * y) + 1) * IMG_WIDTH + (2 * x + 1)] >> 8);

    uint8_t b = lower_left;
    uint8_t g = (uint8_t) (((int)upper_left + (int)lower_right) / 2);
    uint8_t r = upper_right;

    if (outputBGR)
    {
        downsampledDebayeredImg[3 * idx] = b;
        downsampledDebayeredImg[3 * idx + 1] = g;
        downsampledDebayeredImg[3 * idx + 2] = r;
    } else
    {
        downsampledDebayeredImg[3 * idx] = r;
        downsampledDebayeredImg[3 * idx + 1] = g;
        downsampledDebayeredImg[3 * idx + 2] = b;
    }
}

void debayerUsingDownsample(const uint16_t* bayeredImage, uint8_t* debayedImage, const bool outputBGR)
{
    uint16_t* d_input_img;
    gpuCheck( cudaMalloc((uint16_t**)&d_input_img, IMG_WIDTH * IMG_HEIGHT * sizeof(uint16_t)) );
    gpuCheck( cudaMemcpy(d_input_img, bayeredImage, IMG_WIDTH * IMG_HEIGHT * sizeof(uint16_t), cudaMemcpyHostToDevice) );

    uint8_t* d_output_img;
    const size_t outputImgWidth = IMG_WIDTH / 2;
    const size_t outputImgHeight = IMG_HEIGHT / 2;
    gpuCheck( cudaMalloc((uint8_t**)&d_output_img, 3 * outputImgWidth * outputImgHeight * sizeof(uint8_t)) );

    debayerUsingDownsampleCudaImpl<<<outputImgHeight, outputImgWidth>>>(d_input_img, d_output_img, outputBGR);
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );

    gpuCheck( cudaMemcpy(debayedImage, d_output_img, 3 * outputImgWidth * outputImgHeight * sizeof(uint8_t), cudaMemcpyDeviceToHost) );

    // TODO - Lingjian & Daniel :
    // CUDA malloc/free should happen in class construtor/destructor.
    gpuCheck( cudaFree(d_input_img) );
    gpuCheck( cudaFree(d_output_img) );
}

void debayerUsingBilinearInterpolation(const uint16_t* bayeredImage, uint8_t* debayedImage, const bool outputBGR, const bool doDownsample)
{
    uint16_t* d_input_img;
    gpuCheck( cudaMalloc((uint16_t**)&d_input_img, IMG_WIDTH * IMG_HEIGHT * sizeof(uint16_t)) );
    gpuCheck( cudaMemcpy(d_input_img, bayeredImage, IMG_WIDTH * IMG_HEIGHT * sizeof(uint16_t), cudaMemcpyHostToDevice) );

    uint8_t* d_output_img_original_resolution;
    gpuCheck( cudaMalloc((uint8_t**)&d_output_img_original_resolution, 3 * IMG_WIDTH * IMG_HEIGHT * sizeof(uint8_t)) );

    dim3 threads(4, 64);
    dim3 grid((IMG_WIDTH / 2)/(threads.y), (IMG_HEIGHT / 2)/(threads.x));
    debayerUsingBilinearInterpolationCudaImpl<<<grid, threads>>>(d_input_img, d_output_img_original_resolution, outputBGR);
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );

    if (doDownsample)
    {
        uint8_t* d_output_img_half_resolution;
        gpuCheck( cudaMalloc((uint8_t**)&d_output_img_half_resolution, 3 * (IMG_WIDTH / 2) * (IMG_HEIGHT / 2) * sizeof(uint8_t)) );

        downsampleDebayeredImgCudaImpl<<<grid, threads>>>(d_output_img_original_resolution, d_output_img_half_resolution);
        gpuCheck( cudaPeekAtLastError() );
        gpuCheck( cudaDeviceSynchronize() );

        gpuCheck( cudaMemcpy(debayedImage, d_output_img_half_resolution, 3 * (IMG_WIDTH / 2) * (IMG_HEIGHT / 2) * sizeof(uint8_t), cudaMemcpyDeviceToHost) );

        gpuCheck( cudaFree(d_output_img_half_resolution) );
    } else
    {
        gpuCheck( cudaMemcpy(debayedImage, d_output_img_original_resolution, 3 * IMG_WIDTH * IMG_HEIGHT * sizeof(uint8_t), cudaMemcpyDeviceToHost) );
    }

    // TODO - Lingjian & Daniel :
    // CUDA malloc/free should happen in class construtor/destructor.
    gpuCheck( cudaFree(d_input_img) );
    gpuCheck( cudaFree(d_output_img_original_resolution) );
}

void bilinearInterpolationDebayer16(const uint16_t* bayeredImage, uint16_t* debayedImage, const bool outputBGR)
{
    uint16_t* d_input_img;
    gpuCheck( cudaMalloc(&d_input_img, IMG_WIDTH * IMG_HEIGHT * sizeof(uint16_t)) );
    gpuCheck( cudaMemcpy(d_input_img, bayeredImage, IMG_WIDTH * IMG_HEIGHT * sizeof(uint16_t), cudaMemcpyHostToDevice) );

    uint16_t* d_output_img_original_resolution;
    gpuCheck( cudaMalloc(&d_output_img_original_resolution, 3 * IMG_WIDTH * IMG_HEIGHT * sizeof(uint16_t)) );

    dim3 threads(4, 64);
    dim3 grid((IMG_WIDTH / 2)/(threads.y), (IMG_HEIGHT / 2)/(threads.x));
    bilinearInterpolationDebayer16CudaImpl<<<grid, threads>>>(d_input_img, d_output_img_original_resolution, outputBGR);
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );

    gpuCheck( cudaMemcpy(debayedImage, d_output_img_original_resolution, 3 * IMG_WIDTH * IMG_HEIGHT * sizeof(uint16_t), cudaMemcpyDeviceToHost) );

    // TODO - Lingjian & Daniel :
    // CUDA malloc/free should happen in class construtor/destructor.
    gpuCheck( cudaFree(d_input_img) );
    gpuCheck( cudaFree(d_output_img_original_resolution) );
}

} // end namespace cuda
} // end namespace brt
