//
// Created by Daniel Movsesyan on 2019-04-19.
//

#ifndef MOTEC_BU_UTILS_HPP
#define MOTEC_BU_UTILS_HPP


#include <string>
#include <memory>
#include <atomic>
#include <vector>
#include <algorithm>
#include <iostream>

#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define DEFAULT_FIFO_SIZE                   (32)

#define CUDA_CHECK(x)   cudaError_t ___err = (x); \
                        if (___err != cudaSuccess) \
                          std::cerr << " Function " #x " error:" << ___err << std::endl;

namespace brt {

namespace jupiter {

/**
 *
 */
template<typename T,int S = DEFAULT_FIFO_SIZE>
class FIFO
{
public:
  FIFO()
  : _write(0)
  , _read(0)
  {
    for (size_t index = 0;index < S;index++)
      _elem[index] = T(0);
  }

  virtual ~FIFO() {}

  /**
   *
   * @param elem
   */
  bool                            push(T elem)
  {
    if (available() == 0)
      return false;

    uint32_t wr = _write.load();
    _elem[wr] = elem;

    if (++wr >= S)
      wr = 0;

    _write.store(wr);
    return true;
  }

  /**
   *
   * @param elem
   * @return
   */
  bool                            pop(T& elem)
  {
    uint32_t wr = _write.load();
    uint32_t rd = _read.load();

    if (rd == wr)
      return false;

    elem = _elem[rd];
    if (++rd >= S)
      rd = 0;

    _read.store(rd);
    return true;
  }

  /**
   *
   * @return
   */
  uint32_t                        available() const
  {
    uint32_t wr = _write.load();
    uint32_t rd = _read.load();

    if (rd > wr)
      return rd - wr - 1;

    return S - wr + rd - 1;
  }

  /**
   *
   * @return
   */
  uint32_t                        num_elements() const
  {
    uint32_t wr = _write.load();
    uint32_t rd = _read.load();

    if (rd > wr)
      return S - rd + wr;

    return wr - rd;
  }

private:
  std::atomic_uint_fast32_t       _write;
  std::atomic_uint_fast32_t       _read;
  T                               _elem[S];
};

/*
 * \\class CudaPtr
 *
 * created on: Nov 25, 2019
 *
 */
template <class T>
class CudaPtr
{
private:

  /*
   * \\struct Data
   *
   * created on: Nov 25, 2019
   *
   */
  struct Data
  {
    Data(size_t size)
    : _size(size)
    , _data(nullptr)
    , _ref_cnt(1)
    , _valid(false)
    {
      if (cudaMalloc(&_data,size) == cudaSuccess)
        _valid = true;
    }

    ~Data()
    {
      if (_valid)
        cudaFree(_data);
    }

    void addref()
    { _ref_cnt++; }


    void release()
    {
      if (--_ref_cnt == 0)
        delete this;
    }


    size_t                          _size;
    void*                           _data;
    std::atomic_int_fast32_t        _ref_cnt;
    bool                            _valid;
  };

public:
  CudaPtr() : _data(nullptr) {}

  /*
   * \\fn Constructor CudaPtr
   *
   * created on: Nov 25, 2019
   * author: daniel
   *
   */
  CudaPtr(size_t size) : _data(new Data(size * sizeof(T)))
  {
    if (!_data->_valid)
    {
      delete _data;
      _data = nullptr;
    }
  }

  /*
   * \\fn Constructor CudaPtr
   *
   * created on: Nov 25, 2019
   * author: daniel
   *
   */
  CudaPtr(const CudaPtr& rval) : _data(rval._data)
  {
    if (_data != nullptr)
      _data->addref();
  }

  /*
   * \\fn Destructor ~CudaPtr
   *
   * created on: Nov 25, 2019
   * author: daniel
   *
   */
  ~CudaPtr()
  {
    if (_data != nullptr)
      _data->release();
  }

  /*
   * \\fn CudaPtr& operator=
   *
   * created on: Nov 25, 2019
   * author: daniel
   *
   */
  CudaPtr& operator=(const CudaPtr& rval)
  {
    if (_data != nullptr)
      _data->release();

    _data = rval._data;

    if (_data != nullptr)
      _data->addref();

    return *this;
  }

  /*
   * \\fn size_t size
   *
   * created on: Nov 25, 2019
   * author: daniel
   *
   */
  size_t size() const
  {
    if ((_data == nullptr) || !_data->_valid)
      return 0;

    return _data->_size / sizeof(T);
  }

  /*
   * \\fn operator bool
   *
   * created on: Nov 25, 2019
   * author: daniel
   *
   */
  operator bool()
  {
    return ((_data != nullptr) && (_data->_valid));
  }

  /*
   * \\fn bool put
   *
   * created on: Nov 25, 2019
   * author: daniel
   *
   */
  bool put(const T* data, size_t size)
  {
    if (_data == nullptr)
      _data = new Data(size * sizeof(T));

    else if (_data->_size != (size * sizeof(T)))
    {
      _data->release();
      _data = new Data(size * sizeof(T));
    }

    if (!_data->_valid)
    {
      delete _data;
      _data = nullptr;
      return false;
    }

    CUDA_CHECK(cudaMemcpy(_data->_data, data, size * sizeof(T), cudaMemcpyHostToDevice));
    return true;
  }

  /*
   * \\fn bool get
   *
   * created on: Nov 25, 2019
   * author: daniel
   *
   */
  size_t get(T* data, size_t size)
  {
    if ((_data == nullptr) || (!_data->_valid))
      return 0;

    size_t size_to_copy = std::min(size * sizeof(T),_data->_size);
    CUDA_CHECK(cudaMemcpy(data, _data->_data, size_to_copy, cudaMemcpyDeviceToHost));
    return size_to_copy / sizeof(T);
  }


  /*
   * \\fn const T* ptr
   *
   * created on: Nov 25, 2019
   * author: daniel
   *
   */
  const T* ptr() const
  {
    if ((_data == nullptr) || (!_data->_valid))
      return nullptr;

    return (T*)(_data->_data);
  }

  /*
   * \\fn T* ptr
   *
   * created on: Nov 25, 2019
   * author: daniel
   *
   */
  T* ptr()
  {
    if ((_data == nullptr) || (!_data->_valid))
      return nullptr;

    return (T*)(_data->_data);
  }


  /*
   * \\fn void fill
   *
   * created on: Nov 25, 2019
   * author: daniel
   *
   */
  void fill(uint8_t value)
  {
    if ((_data == nullptr) || (!_data->_valid))
      return;

    CUDA_CHECK(cudaMemset(_data->_data, value, _data->_size));
  }


  /*
   * \\fn void copy
   *
   * created on: Nov 25, 2019
   * author: daniel
   *
   */
  bool copy(CudaPtr<T>& dst)
  {
    if (!dst || (dst.size() != size()))
    {
      dst = CudaPtr<T>(size());
      if (!dst)
        return false;
    }

    CUDA_CHECK(cudaMemcpy(dst._data->_data, _data->_data, _data->_size, cudaMemcpyDeviceToDevice));
    return (___err == cudaSuccess);
  }

private:
  Data*                           _data;
};


/*
 * \\enum DisplayType
 *
 * created on: Nov 19, 2019
 *
 */
enum DisplayType
{
  eLocalDisplays = 1,
  eRemoteDisplay = 2,
  eAllDisplays = eLocalDisplays | eRemoteDisplay
};

/**
 *
 */
class Utils
{
public:
  Utils() {}
  virtual ~Utils() {}

  template<typename ... Args>
  static std::string              string_format( const std::string& format, Args ... arguments)
  {
    size_t size = snprintf( nullptr, 0, format.c_str(), arguments ... ) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), arguments ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
  }

  static const char*              stristr(const char* src,const char* dst,size_t len = (size_t)-1);
  static size_t                   stristr(const std::string& src,const char* dst,size_t len = (size_t)-1);
  static double                   frame_rate(const char* fr_string);

  static  std::vector<std::string>
                                  enumerate_displays(DisplayType = eAllDisplays);

};

} // jupiter
} // brt


#endif //MOTEC_BU_UTILS_HPP
