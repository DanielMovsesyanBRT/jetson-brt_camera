/*
 * debayer.cu
 *
 *  Created on: Feb 14, 2020
 *      Author: daniel
 */

#include <cuda_2d_mem.hpp>
#include <cuda_mem.hpp>

#include "cuda_debayer.hpp"

#include <mutex>
#include <atomic>

namespace brt
{
namespace jupiter
{

/**
 * \struct RGB
 *
 * \brief <description goes here>
 */
struct RGB
{
  uint16_t                        _r;
  uint16_t                        _g;
  uint16_t                        _b;
};


/**
 * \struct LAB
 *
 * \brief <description goes here>
 */
struct LAB
{
  __device__ LAB() : _L(0), _a(0), _b(0) {}
  int32_t                         _L;
  int32_t                         _a;
  int32_t                         _b;

  __device__ void                 from(RGB& rgb)
  {
    _L = (rgb._r+rgb._r+rgb._r + rgb._b + rgb._g+rgb._g+rgb._g+rgb._g) >> 3;
    _a = 5 * (rgb._r+rgb._r + rgb._b - rgb._g-rgb._g-rgb._g);
    _b = 4 * (rgb._r+rgb._r + rgb._g+rgb._g+rgb._g - rgb._b-rgb._b-rgb._b-rgb._b);
  }
};

/**
 * \class Debayer_impl
 *
 * \brief <description goes here>
 */
class Debayer_impl
{
friend Debayer;
public:
Debayer_impl()
  : _thx(0),_thy(0)
  , _blkx(0),_blky(0)
  , _ref(0)
  { }

  virtual ~Debayer_impl() {}

          void                    init(size_t width,size_t height,size_t small_hits_size);
          image::RawRGBPtr        ahd(image::RawRGBPtr img);
          image::RawRGBPtr        mhc(image::RawRGBPtr img);
          image::RawRGBPtr        bilinear(image::RawRGBPtr img);
private:

  CudaPtr<uint16_t>               _raw;
  CudaPtr<RGB>                    _horiz;
  CudaPtr<RGB>                    _vert;
  CudaPtr<RGB>                    _result;

  CudaPtr<LAB>                    _hlab;
  CudaPtr<LAB>                    _vlab;

  CudaPtr<uint32_t>               _histogram;
  CudaPtr<uint32_t>               _histogram_max;
  CudaPtr<uint32_t>               _small_histogram;

  int                             _thx,_thy;
  int                             _blkx,_blky;

  std::atomic_uint32_t            _ref;
  std::mutex                      _mutex;
};

/**
 * \fn  green_interpolate
 *
 * @param  width : size_t 
 * @param  height :  size_t 
 * @param  raw :  uint16_t* 
 * @param  hr :  RGB* 
 * @param  vr :  RGB* 
 * @return  __global__ void
 * \brief <description goes here>
 */
__global__ void green_interpolate(size_t width, size_t height,
                                          uint16_t* raw,
                                          RGB* hr, RGB* vr)
{
  int origx = ((blockIdx.x * blockDim.x) + threadIdx.x) << 1;
  int origy = ((blockIdx.y * blockDim.y) + threadIdx.y) << 1;

  auto limit = [](int x,int a,int b)->int
  {
    if (a>b)
      return (x<b)?b:(x>a)?a:x;

    return (x<a)?a:(x>b)?b:x;
  };

  // C R
  // B C
  // (0,0) -> Clear
  int x = origx, y = origy;
  int io = x + y * width; // input offset

  vr[io]._g = hr[io]._g = raw[io];

  ////////////////////////////////////////////////
  // (1,0) -> Red
  x = origx + 1;
  y = origy;
  io = x + y * width; // input offset
  {
    int px[] = { (x > 0) ? raw[io - 1] : 0,                // x-1,y
                 (x < (width - 1)) ? raw[io + 1] : 0,      // x+1,y
                 (x > 1) ? raw[io - 2] : 0,                // x-2,y
                 (x < (width - 2)) ? raw[io + 2] : 0 };    // x+2,y

    int value =  ((( px[0] + raw[io] + px[1]) * 2) - px[2] - px[3]) >> 2;
    hr[io]._g = limit(value, px[0], px[1]);

    int py[] = { (y > 0) ? raw[io - width] : 0,                    // x,y-1
                 (y < (height - 1)) ? raw[io + width] : 0,         // x,y+1
                 (y > 1) ? raw[io - (2 * width)] : 0,              // x,y-2
                 (y < (height - 2))?raw[io + (2 * width)] : 0 };   // x,y+2

    value =  ((( py[0] + raw[io] + py[1]) * 2) - py[2] - py[3]) >> 2;
    vr[io]._g = limit(value, py[0], py[1]);
  }

  ////////////////////////////////////////////////
  // (0,1) -> Blue
  x = origx;
  y = origy + 1;
  io = x + y * width; // input offset
  {
    int px[] = { (x > 0) ? raw[io - 1] : 0,                // x-1,y
                 (x < (width - 1)) ? raw[io + 1] : 0,      // x+1,y
                 (x > 1) ? raw[io - 2] : 0,                // x-2,y
                 (x < (width - 2)) ? raw[io + 2] : 0 };    // x+2,y

    int value =  ((( px[0] + raw[io] + px[1]) * 2) - px[2] - px[3]) >> 2;
    hr[io]._g = limit(value, px[0], px[1]);

    int py[] = { (y > 0) ? raw[io - width] : 0,                    // x,y-1
                 (y < (height - 1)) ? raw[io + width] : 0,         // x,y+1
                 (y > 1) ? raw[io - (2 * width)] : 0,              // x,y-2
                 (y < (height - 2))?raw[io + (2 * width)] : 0 };   // x,y+2

    value =  ((( py[0] + raw[io] + py[1]) * 2) - py[2] - py[3]) >> 2;
    vr[io]._g = limit(value, py[0], py[1]);
  }
  ////////////////////////////////////////////////
  // (1,1) -> Clear
  x = origx + 1;
  y = origy + 1;
  io = x + y * width; // input offset

  vr[io]._g = hr[io]._g = raw[io];
}

/**
 * \fn  blue_red_interpolate
 *
 * @param  width : size_t 
 * @param  height :  size_t 
 * @param  raw :  uint16_t* 
 * @param  hr :  RGB* 
 * @param  vr :  RGB* 
 * @param  hl :  LAB* 
 * @param  vl : LAB* 
 * @return  __global__ void
 * \brief <description goes here>
 */
__global__ void blue_red_interpolate( size_t width, size_t height,
                                      uint16_t* raw,
                                      RGB* hr, RGB* vr,
                                      LAB* hl,LAB* vl)
{
  int origx = ((blockIdx.x * blockDim.x) + threadIdx.x) << 1;
  int origy = ((blockIdx.y * blockDim.y) + threadIdx.y) << 1;
  int z = blockIdx.z;

  RGB*  ip[2] = {hr, vr};
  LAB*  lb[2] = {hl, vl};

  auto limit = [](int x,int a,int b)->int
  {
    return (x<a)?a:(x>b)?b:x;
  };

  int sub[4][4] = { {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0} };

  int io = origx + origy * width;


  if (origx > 0)
  {
    if (origy > 0)
      sub[0][0] = raw[io-width-1] - ip[z][io-width-1]._g;

    sub[1][0] = raw[io-1] - ip[z][io-1]._g;
    sub[2][0] = raw[io+width-1] - ip[z][io+width-1]._g;

    if (origy < (height - 2))
      sub[3][0] = raw[io+2*width-1] - ip[z][io+2*width-1]._g;
  }

  if (origx < (width - 2))
  {
    if (origy > 0)
      sub[0][3] = raw[io-width+2] - ip[z][io-width+2]._g;

    sub[1][3] = raw[io+2] - ip[z][io+2]._g;
    sub[2][3] = raw[io+width+2] - ip[z][io+width+2]._g;

    if (origy < (height - 2))
      sub[3][3] = raw[io+2*width+2] - ip[z][io+2*width+2]._g;
  }

  if (origy > 0)
  {
    sub[0][1] = raw[io-width] - ip[z][io-width]._g;
    sub[0][2] = raw[io-width+1] - ip[z][io-width+1]._g;
  }

  if (origy < (height - 2))
  {
    sub[3][1] = raw[io+2*width] - ip[z][io+2*width]._g;
    sub[3][2] = raw[io+2*width+1] - ip[z][io+2*width+1]._g;
  }

  sub[1][1] = raw[io] - ip[z][io]._g;
  sub[1][2] = raw[io+1] - ip[z][io+1]._g;
  sub[2][1] = raw[io+width] - ip[z][io+width]._g;
  sub[2][2] = raw[io+width+1] - ip[z][io+width+1]._g;

  // C R
  // B C
  ////////////////////////////////////////////////
  // (0,0) -> ClearRead
  {
    ip[z][io]._r = limit(ip[z][io]._g + ((sub[1][0] + sub[1][2]) >> 1),0,(1<<16)-1);
    ip[z][io]._b = limit(ip[z][io]._g + ((sub[0][1] + sub[2][1]) >> 1),0,(1<<16)-1);

    lb[z][io].from(ip[z][io]);
  }

  ////////////////////////////////////////////////
  // (1,0) -> Red
  {
    ip[z][io+1]._r = raw[io+1];
    ip[z][io+1]._b = limit(ip[z][io+1]._g + ((sub[0][1] + sub[0][3] + sub[2][1] + sub[2][3]) >> 2),0,(1<<16)-1);

    lb[z][io+1].from(ip[z][io+1]);
  }

  ////////////////////////////////////////////////
  // (0,1) -> Blue
  {
    ip[z][io+width]._b = raw[io+width];

    ip[z][io+width]._r = limit(ip[z][io+width]._g + ((sub[1][0] + sub[1][2] + sub[3][0] + sub[3][2]) >> 2),0,(1<<16)-1);
    lb[z][io+width].from(ip[z][io+width]);
  }

  ////////////////////////////////////////////////
  // (1,1) -> ClearBlue
  {
    ip[z][io+width+1]._b = limit(ip[z][io+width+1]._g + ((sub[2][1] + sub[2][3]) >> 1),0,(1<<16)-1);
    ip[z][io+width+1]._r = limit(ip[z][io+width+1]._g + ((sub[1][2] + sub[3][2]) >> 1),0,(1<<16)-1);

    lb[z][io+width+1].from(ip[z][io+width+1]);
  }
}


/**
 * \fn  misguidance_color_artifacts
 *
 * @param  width : size_t 
 * @param  height :  size_t 
 * @param  rst :  RGB* 
 * @param  hr :  RGB* 
 * @param  vr : RGB* 
 * @param  hl :  LAB* 
 * @param  vl : LAB* 
 * @param  histogram :  uint32_t* 
 * @param  histogram_size : uint32_t 
 * @param  small_histogram :  uint32_t* 
 * @param  small_histogram_size :  uint32_t 
 * @return  __global__ void
 * \brief <description goes here>
 */
__global__ void misguidance_color_artifacts(size_t width, size_t height, RGB* rst,
                                            RGB* hr,RGB* vr,
                                            LAB* hl,LAB* vl,
                                            uint32_t* histogram,uint32_t histogram_size,
                                            uint32_t* small_histogram, uint32_t small_histogram_size)
{
  int x = ((blockIdx.x * blockDim.x) + threadIdx.x);
  int y = ((blockIdx.y * blockDim.y) + threadIdx.y);
  int io = x + y * width;

  auto sqr=[](int a)->int { return a*a; };

  int lv[2],lh[2],cv[2],ch[2];
  int hh = 0,hv = 0;


  LAB ph[] = { (x > 0) ? hl[io-1] : LAB(),                            // x-1,y
               (x < (width - 1)) ? hl[io+1] : LAB()};                 // x+1,y

  LAB pv[] = { (y > 0) ? vl[io-width] : LAB(),                        // x,y-1
               (y < (height - 1)) ? vl[io+width] : LAB()};            // x,y+1


  lh[0] = __sad(hl[io]._L,ph[0]._L,0);
  lh[1] = __sad(hl[io]._L,ph[1]._L,0);

  lv[0] = __sad(vl[io]._L,pv[0]._L,0);
  lv[1] = __sad(vl[io]._L,pv[1]._L,0);

  ch[0] = sqr(hl[io]._a - ph[0]._a) +
          sqr(hl[io]._b - ph[0]._b);

  ch[1] = sqr(hl[io]._a - ph[1]._a) +
          sqr(hl[io]._b - ph[1]._b);

  cv[0] = sqr(vl[io]._a - pv[0]._a) +
          sqr(vl[io]._b - pv[0]._b);

  cv[1] = sqr(vl[io]._a - pv[1]._a) +
          sqr(vl[io]._b - pv[1]._b);


  int h = (lh[0]>lh[1]) ? lh[0]: lh[1];
  int v = (lv[0]>lv[1]) ? lv[0]: lv[1];
  int eps_l = v < h ? v : h;

  h = (ch[0]>ch[1])?ch[0]:ch[1];
  v = (cv[0]>cv[1])?cv[0]:cv[1];
  int eps_c = v < h ? v : h;

  hh = ((lh[0]<=eps_l) * (ch[0]<= eps_c)) +
       ((lh[1]<=eps_l) * (ch[1]<= eps_c));

  hv = ((lv[0]<=eps_l) * (cv[0]<= eps_c)) +
       ((lv[1]<=eps_l) * (cv[1]<= eps_c));

  uint32_t r = 0,g = 0,b = 0;
  RGB *lf, *rt;

  lf = (hh > hv)? &hr[io] : &vr[io];
  rt = (hv > hh)? &vr[io] : &hr[io];

  r = rst[io]._r = (lf->_r + rt->_r) >> 1;
  g = rst[io]._g = (lf->_g + rt->_g) >> 1;
  b = rst[io]._b = (lf->_b + rt->_b) >> 1;

  uint32_t brightness = ((r+r+r+b+g+g+g+g) >> 3) * small_histogram_size >> 16;
  atomicAdd(&small_histogram[brightness % small_histogram_size], 1);

  brightness = ((r+r+r+b+g+g+g+g) >> 3) * histogram_size >> 16;
  atomicAdd(&histogram[brightness & ((1 << 16) - 1)], 1);
}


/**
 * \fn  cudaMax
 *
 * @param  org : uint32_t* 
 * @param  max : uint32_t* 
 * @return  __global__ void
 * \brief <description goes here>
 */
__global__ void cudaMax(uint32_t* org,uint32_t* max)
{
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  max[tid] = org[tid];

  auto step_size = 1;
  int number_of_threads = gridDim.x * blockDim.x;

  __syncthreads();

  while (number_of_threads > 0)
  {
    if (tid < number_of_threads)
    {
      const auto fst = tid * step_size * 2;
      const auto snd = fst + step_size;

      max[fst] = (max[fst] < max[snd]) ? max[snd] : max[fst];
    }

    step_size <<= 1;
    number_of_threads >>= 1;
  }
}


/**
 *
 *
 *   Malvar-He-Cutler Linear Image Demosaicking
 *
 *
 */

 #define MHC_KERNEL_RADIUS                (2)
 #define MHC_KERNEL_SIZE                  (MHC_KERNEL_RADIUS  * 2 + 1)

 __constant__ int16_t green[MHC_KERNEL_SIZE][MHC_KERNEL_SIZE] =
 { // Green at red or blue location
  { 0, 0,-1, 0, 0},
  { 0, 0, 2, 0, 0},
  {-1, 2, 4, 2,-1},
  { 0, 0, 2, 0, 0},
  { 0, 0,-1, 0, 0}
};

__constant__ int16_t red_blue_row[MHC_KERNEL_SIZE][MHC_KERNEL_SIZE] =
{ // Red at Green in red row multiplied by 2
  { 0, 0, 1, 0, 0},
  { 0,-2, 0,-2, 0},
  {-2, 8,10, 8,-2},
  { 0,-2, 0,-2, 0},
  { 0, 0, 1, 0, 0},
};

__constant__ int16_t red_blue_col[MHC_KERNEL_SIZE][MHC_KERNEL_SIZE] =
{ // Red at Green in blue row multiplied by 2
  { 0, 0,-2, 0, 0},
  { 0,-2, 8,-2, 0},
  { 1, 0,10, 0, 1},
  { 0,-2, 8,-2, 0},
  { 0, 0,-2, 0, 0},
};

__constant__ int16_t red_blue[MHC_KERNEL_SIZE][MHC_KERNEL_SIZE] =
{ // Red at blue  multiplied by 2
  { 0, 0,-3, 0, 0},
  { 0, 4, 0, 4, 0},
  {-3, 0,12, 0,-3},
  { 0, 4, 0, 4, 0},
  { 0, 0,-3, 0, 0},
};

/**
 * \fn  convolute
 *
 * @param  int16_t kernel[MHC_KERNEL_SIZE][MHC_KERNEL_SIZE]
 * @param  x0 :  size_t 
 * @param  y0 :  size_t 
 * @param  width :  size_t 
 * @param  height :  size_t 
 * @param  raw : uint16_t* 
 * @param  g_loc :  int 
 * @return  __device__ int
 * \brief <description goes here>
 */

 __device__ int convolute(int16_t kernel[MHC_KERNEL_SIZE][MHC_KERNEL_SIZE],
                          size_t x0, size_t y0, size_t width, size_t height, uint16_t* raw, int g_loc)
{
  int sum = 0;

  for (int i = -MHC_KERNEL_RADIUS; i <= MHC_KERNEL_RADIUS; i++)
  {
    for (int j = -MHC_KERNEL_RADIUS; j <= MHC_KERNEL_RADIUS; j++)
    {
      int x = x0+j, y=y0+i;
      if ((x >= 0) && (y >= 0) && 
          (x < (width - MHC_KERNEL_RADIUS)) && 
          (y < (height - MHC_KERNEL_RADIUS)))
      {
        sum += raw[g_loc + (j + i * width)] * kernel[i + MHC_KERNEL_RADIUS][j + MHC_KERNEL_RADIUS];
      }
    }
  }
  return sum;
}

/**
 * \fn  mhc_debayering
 *
 * @param  width : size_t 
 * @param  height :  size_t 
 * @param  raw : uint16_t* 
 * @param  result :  RGB* 
 * @param  histogram :  uint32_t* 
 * @param  histogram_size : uint32_t 
 * @param  small_histogram :  uint32_t* 
 * @param  small_histogram_size :  uint32_t 
 * @return  __global__ void
 * \brief <description goes here>
 */
__global__ void mhc_debayering(size_t width, size_t height,uint16_t* raw, RGB* result,
                                uint32_t* histogram,uint32_t histogram_size,
                                uint32_t* small_histogram, uint32_t small_histogram_size)
{
  enum ColorPos
  {
    eClearRed = 0, eRed = 1, eBlue = 2, eClearBlue = 3
  };

  auto position=[](int x, int y)->ColorPos
  {
    return static_cast<ColorPos>((x & 1) + (y & 1) * 2);
  };

  int x0 = ((blockIdx.x * blockDim.x) + threadIdx.x);
  int y0 = ((blockIdx.y * blockDim.y) + threadIdx.y);
  int g_loc = x0 + y0 * width; // input offset

  // convolution
  ColorPos  pos = position(x0, y0);
  int r,g,b;
  switch (pos)
  {
  case eClearRed:
    r = convolute(red_blue_row,x0,y0,width,height,raw,g_loc) >> 4;
    g = raw[g_loc];
    b = convolute(red_blue_col,x0,y0,width,height,raw,g_loc) >> 4;
    break;

  case eRed:
    r = raw[g_loc];
    g = convolute(green,x0,y0,width,height,raw,g_loc) >> 3;
    b = convolute(red_blue,x0,y0,width,height,raw,g_loc) >> 4;
    break;

  case eBlue:
    r = convolute(red_blue,x0,y0,width,height,raw,g_loc) >> 4;
    g = convolute(green,x0,y0,width,height,raw,g_loc) >> 3;
    b = raw[g_loc];
    break;

  case eClearBlue:  // coefficient Beta = 5/8 (or 10/16)
    r = convolute(red_blue_col,x0,y0,width,height,raw,g_loc) >> 4;
    g = raw[g_loc];
    b = convolute(red_blue_row,x0,y0,width,height,raw,g_loc) >> 4;
    break;
  }      
  
  result[g_loc]._r = (r <= 0xFFFF) ? r : 0xFFFF;
  result[g_loc]._g = (g <= 0xFFFF) ? g : 0xFFFF;
  result[g_loc]._b = (b <= 0xFFFF) ? b : 0xFFFF;
  
  uint32_t brightness = ((r+r+r+b+g+g+g+g) >> 3) * small_histogram_size >> 16;
  atomicAdd(&small_histogram[brightness % small_histogram_size], 1);

  brightness = ((r+r+r+b+g+g+g+g) >> 3) * histogram_size >> 16;
  atomicAdd(&histogram[brightness & ((1 << 16) - 1)], 1);
}


/**
 * \fn  bilinear_debayering
 *
 * @param  width : size_t 
 * @param  height :  size_t 
 * @param  raw : uint16_t* 
 * @param  result :  RGB* 
 * @param  histogram :  uint32_t* 
 * @param  histogram_size : uint32_t 
 * @param  small_histogram :  uint32_t* 
 * @param  small_histogram_size :  uint32_t 
 * \brief <description goes here>
 */
__global__ void bilinear_debayering(size_t width, size_t height,uint16_t* raw, RGB* result,
                                uint32_t* histogram,uint32_t histogram_size,
                                uint32_t* small_histogram, uint32_t small_histogram_size)
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
  auto value=[raw,width,height](int x,int y,int& count)->int
  {
    if (x<0 || y<0 || x>=width || y>=height)
      return 0;

    count++;
    return raw[x + y * width];
  };

  int x0 = ((blockIdx.x * blockDim.x) + threadIdx.x);
  int y0 = ((blockIdx.y * blockDim.y) + threadIdx.y);

  int r,g,b, count = 0;

  // Clear Red
  int x = (x0 << 1), y = (y0 << 1);
  int g_loc = x + y * width; // input offset

  r = (value(x - 1, y, count) + value(x + 1, y, count)) / count; count = 0;
  g = raw[x + y * width];
  b = (value(x, y - 1, count) + value(x, y + 1, count)) / count; count = 0;

  result[g_loc]._r = (r <= 0xFFFF) ? r : 0xFFFF;
  result[g_loc]._g = (g <= 0xFFFF) ? g : 0xFFFF;
  result[g_loc]._b = (b <= 0xFFFF) ? b : 0xFFFF;

  uint32_t brightness = ((r+r+r+b+g+g+g+g) >> 3) * small_histogram_size >> 16;
  atomicAdd(&small_histogram[brightness % small_histogram_size], 1);

  brightness = ((r+r+r+b+g+g+g+g) >> 3) * histogram_size >> 16;
  atomicAdd(&histogram[brightness & ((1 << 16) - 1)], 1);


  // Red
  x = (x0 << 1) + 1;
  y = (y0 << 1);
  g_loc = x + y * width; // input offset

  r = raw[x + y * width];
  g = (value(x - 1, y, count) + value(x + 1, y, count) + value(x, y - 1, count) + value(x, y + 1, count)) / count; count = 0;
  b = (value(x - 1, y - 1, count) + value(x + 1, y - 1, count) + value(x - 1, y + 1, count) + + value(x + 1, y + 1, count)) / count; count = 0;

  result[g_loc]._r = (r <= 0xFFFF) ? r : 0xFFFF;
  result[g_loc]._g = (g <= 0xFFFF) ? g : 0xFFFF;
  result[g_loc]._b = (b <= 0xFFFF) ? b : 0xFFFF;

  brightness = ((r+r+r+b+g+g+g+g) >> 3) * small_histogram_size >> 16;
  atomicAdd(&small_histogram[brightness % small_histogram_size], 1);

  brightness = ((r+r+r+b+g+g+g+g) >> 3) * histogram_size >> 16;
  atomicAdd(&histogram[brightness & ((1 << 16) - 1)], 1);

  // Blue
  x = (x0 << 1);
  y = (y0 << 1) + 1;
  g_loc = x + y * width; // input offset

  r = (value(x - 1, y - 1, count) + value(x + 1, y - 1, count) + value(x - 1, y + 1, count) + + value(x + 1, y + 1, count)) / count; count = 0;
  g = (value(x - 1, y, count) + value(x + 1, y, count) + value(x, y - 1, count) + value(x, y + 1, count)) / count; count = 0;
  b = raw[x + y * width];

  result[g_loc]._r = (r <= 0xFFFF) ? r : 0xFFFF;
  result[g_loc]._g = (g <= 0xFFFF) ? g : 0xFFFF;
  result[g_loc]._b = (b <= 0xFFFF) ? b : 0xFFFF;

  brightness = ((r+r+r+b+g+g+g+g) >> 3) * small_histogram_size >> 16;
  atomicAdd(&small_histogram[brightness % small_histogram_size], 1);

  brightness = ((r+r+r+b+g+g+g+g) >> 3) * histogram_size >> 16;
  atomicAdd(&histogram[brightness & ((1 << 16) - 1)], 1);

  // Clear Blue
  x = (x0 << 1) + 1;
  y = (y0 << 1) + 1;
  g_loc = x + y * width; // input offset

  r = (value(x, y - 1, count) + value(x, y + 1, count)) / count; count = 0;
  g = raw[x + y * width];
  b = (value(x - 1, y, count) + value(x + 1, y, count)) / count; count = 0;

  result[g_loc]._r = (r <= 0xFFFF) ? r : 0xFFFF;
  result[g_loc]._g = (g <= 0xFFFF) ? g : 0xFFFF;
  result[g_loc]._b = (b <= 0xFFFF) ? b : 0xFFFF;
 
  brightness = ((r+r+r+b+g+g+g+g) >> 3) * small_histogram_size >> 16;
  atomicAdd(&small_histogram[brightness % small_histogram_size], 1);

  brightness = ((r+r+r+b+g+g+g+g) >> 3) * histogram_size >> 16;
  atomicAdd(&histogram[brightness & ((1 << 16) - 1)], 1);
}



Debayer_impl* Debayer::_impl = nullptr;
/*
 * \\fn constructor Debayer::Debayer
 *
 * created on: Feb 14, 2020
 * author: daniel
 *
 */
Debayer::Debayer()
: _width(0)
, _height(0)
, _daType(daMHC)
{
  _impl = new Debayer_impl();
  _impl->_ref++;
}

/*
 * \\fn Debayer::~Debayer
 *
 * created on: Feb 14, 2020
 * author: daniel
 *
 */
Debayer::~Debayer()
{
  if (--_impl->_ref == 0)
    delete _impl;
}


/*
 * \\fn Debayer_impl::init
 *
 * created on: Feb 14, 2020
 * author: daniel
 *
 */
void Debayer_impl::init(size_t width,size_t height,size_t small_hits_size)
{
  _raw = CudaPtr<uint16_t>(width * height);  _raw.fill(0);

  _horiz = CudaPtr<RGB>(width * height); _horiz.fill(0);
  _vert = CudaPtr<RGB>(width * height); _vert.fill(0);
  _result = CudaPtr<RGB>(width * height); _result.fill(0);

  _hlab = CudaPtr<LAB>(width * height); _hlab.fill(0);
  _vlab = CudaPtr<LAB>(width * height); _vlab.fill(0);

  _histogram = CudaPtr<uint32_t>(1 << 16); _histogram.fill(0);
  _histogram_max = CudaPtr<uint32_t>(1 << 16); _histogram_max.fill(0);
  _small_histogram = CudaPtr<uint32_t>(small_hits_size); _small_histogram.fill(0);

  _thx = std::min(DEFAULT_NUMBER_OF_THREADS, (1 << __builtin_ctz(width)));
  if (_thx == 0)
    _thx = 1;

  _thy = std::min(DEFAULT_NUMBER_OF_THREADS, (1 << __builtin_ctz(height)));
  if (_thy == 0)
    _thy = 1;

  _blkx = width / _thx;
  if ((width % _thx) != 0)
    _blkx++;

  _blky = height / _thy;
  if ((height % _thy) != 0)
    _blky++;
}
/*
 * \\fn Debayer::init
 *
 * created on: Feb 14, 2020
 * author: daniel
 *
 */
bool Debayer::init(size_t width,size_t height,size_t small_hits_size)
{
  _impl->_mutex.lock();
  if (!_impl->_raw)
  {
    _width = width;
    _height = height;

    _impl->init(width, height, small_hits_size);
  }
  _impl->_mutex.unlock();
  return true;
}

/*
 * \\fn image::RawRGBPtr Debayer::ahd
 *
 * created on: Feb 14, 2020
 * author: daniel
 *
 */
image::RawRGBPtr Debayer_impl::ahd(image::RawRGBPtr img)
{
  if (!img)
    return image::RawRGBPtr();

  _mutex.lock();

  _histogram.fill(0);
  _histogram_max.fill(0);
  _small_histogram.fill(0);
  _raw.put((uint16_t*)img->bytes(),img->width() * img->height());

  dim3 threads(_thx >> 1,_thy >> 1);
  dim3 blocks(_blkx, _blky);
  dim3 blocks2(_blkx, _blky, 2);

  green_interpolate<<<blocks,threads>>>(img->width(), img->height(),_raw.ptr(), _horiz.ptr(), _vert.ptr());
  blue_red_interpolate<<<blocks2,threads>>>(img->width(), img->height(),_raw.ptr(), _horiz.ptr(), _vert.ptr(), _hlab.ptr(), _vlab.ptr());


  dim3 threads2(_thx,_thy);
  misguidance_color_artifacts<<<blocks,threads2>>>(img->width(), img->height(),
                      _result.ptr(), _horiz.ptr(),
                      _vert.ptr(), _hlab.ptr(), _vlab.ptr(),_histogram.ptr(), _histogram.size(),_small_histogram.ptr(), _small_histogram.size());

  size_t thx = 64;
  while (_histogram.size() < thx)
    thx >>= 1;

  cudaMax<<<_histogram.size() / thx, thx>>>(_histogram.ptr(), _histogram_max.ptr());

  image::RawRGBPtr result(new image::RawRGB(img->width(), img->height(), img->depth(), image::eRGB));
  _result.get((RGB*)result->bytes(), result->width() * result->height());

  image::HistPtr  full_hist(new image::Histogram);
  full_hist->_histogram.resize(_histogram.size());
  full_hist->_small_hist.resize(_small_histogram.size());

  _histogram.get((uint32_t*)full_hist->_histogram.data(), _histogram.size());
  _small_histogram.get(full_hist->_small_hist.data(), _small_histogram.size());
  _histogram_max.get(&full_hist->_max_value, 1);

  result->set_histogram(full_hist);

  _mutex.unlock();

  return result;
}


/**
 * \fn  Debayer_impl::mhc
 *
 * @param   img : image::RawRGBPtr
 * @return  image::RawRGBPtr
 * \brief <description goes here>
 */
image::RawRGBPtr Debayer_impl::mhc(image::RawRGBPtr img)
{
  if (!img)
    return image::RawRGBPtr();

  _mutex.lock();

  _histogram.fill(0);
  _histogram_max.fill(0);
  _small_histogram.fill(0);
  _raw.put((uint16_t*)img->bytes(),img->width() * img->height());

  dim3 threads(_thx, _thy);
  dim3 blocks(_blkx, _blky);

  mhc_debayering<<<blocks,threads>>>(img->width(), img->height(), _raw.ptr(), _result.ptr(),
                                          _histogram.ptr(), _histogram.size(), _small_histogram.ptr(), 
                                          _small_histogram.size());

  size_t thx = 64;
  while (_histogram.size() < thx)
    thx >>= 1;

  cudaMax<<<_histogram.size() / thx, thx>>>(_histogram.ptr(), _histogram_max.ptr());

  image::RawRGBPtr result(new image::RawRGB(img->width(), img->height(), img->depth(), image::eRGB));
  _result.get((RGB*)result->bytes(), result->width() * result->height());

  image::HistPtr  full_hist(new image::Histogram);
  full_hist->_histogram.resize(_histogram.size());
  full_hist->_small_hist.resize(_small_histogram.size());

  _histogram.get((uint32_t*)full_hist->_histogram.data(), _histogram.size());
  _small_histogram.get(full_hist->_small_hist.data(), _small_histogram.size());
  _histogram_max.get(&full_hist->_max_value, 1);

  result->set_histogram(full_hist);

  _mutex.unlock();

  return result;
}

/**
 * \fn  Debayer_impl::bilinear
 *
 * @param   img : image::RawRGBPtr
 * @return  image::RawRGBPtr
 * \brief <description goes here>
 */
image::RawRGBPtr Debayer_impl::bilinear(image::RawRGBPtr img)
{
  if (!img)
    return image::RawRGBPtr();

  _mutex.lock();

  _histogram.fill(0);
  _histogram_max.fill(0);
  _small_histogram.fill(0);
  _raw.put((uint16_t*)img->bytes(),img->width() * img->height());

  dim3 threads(_thx >> 1,_thy >> 1);
  dim3 blocks(_blkx, _blky);

  bilinear_debayering<<<blocks,threads>>>(img->width(), img->height(), _raw.ptr(), _result.ptr(),
      _histogram.ptr(), _histogram.size(), _small_histogram.ptr(),
      _small_histogram.size());

  size_t thx = 64;
  while (_histogram.size() < thx)
    thx >>= 1;

  cudaMax<<<_histogram.size() / thx, thx>>>(_histogram.ptr(), _histogram_max.ptr());

  image::RawRGBPtr result(new image::RawRGB(img->width(), img->height(), img->depth(), image::eRGB));
  _result.get((RGB*)result->bytes(), result->width() * result->height());

  image::HistPtr  full_hist(new image::Histogram);
  full_hist->_histogram.resize(_histogram.size());
  full_hist->_small_hist.resize(_small_histogram.size());

  _histogram.get((uint32_t*)full_hist->_histogram.data(), _histogram.size());
  _small_histogram.get(full_hist->_small_hist.data(), _small_histogram.size());
  _histogram_max.get(&full_hist->_max_value, 1);

  result->set_histogram(full_hist);
  _mutex.unlock();

  return result;
}

/**
 * \fn  Debayer::mhc
 *
 * @param   img : image::RawRGBPtr
 * @return  image::RawRGBPtr
 * \brief <description goes here>
 */
image::RawRGBPtr Debayer::mhc(image::RawRGBPtr img)
{
  return _impl->mhc(img);
}


/*
 * \\fn image::RawRGBPtr Debayer::ahd
 *
 * created on: Feb 14, 2020
 * author: daniel
 *
 */
image::RawRGBPtr Debayer::ahd(image::RawRGBPtr img)
{
  return _impl->ahd(img);
}

/**
 * \fn  Debayer::bilinear
 *
 * @param   img : image::RawRGBPtr
 * @return  image::RawRGBPtr
 * \brief <description goes here>
 */
image::RawRGBPtr Debayer::bilinear(image::RawRGBPtr img)
{
  return _impl->bilinear(img);
}


/*
 * \\fn void Debayer::consume
 *
 * created on: Feb 14, 2020
 * author: daniel
 *
 */
void Debayer::consume(image::ImageBox box)
{
  for (image::ImagePtr img : box)
  {
    image::RawRGBPtr result;
    switch (_daType)
    {
    case daBiLinear:
      result = _impl->bilinear(img->get_bits());
      break;
      
    case daAHD:
      result = _impl->ahd(img->get_bits());
      break;
    
    case daMHC:
      result = _impl->mhc(img->get_bits());
      break;
        
    default:
      return;
    }

    if (result)
      ImageProducer::consume(image::ImageBox(result).set_meta(*(img.get())));
  }
}


} /* namespace jupiter */
} /* namespace brt */
