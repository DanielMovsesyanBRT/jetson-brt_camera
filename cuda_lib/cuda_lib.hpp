/**
 * 
 * 
 */

namespace brt 
{
namespace jupiter
{

/**
 * \class CudaLib
 *
 * \brief <description goes here>
 */
class CudaLib
{
private:
  CudaLib() {}
  virtual ~CudaLib() {}

public:
  static  CudaLib*                get() { return &_object; }
          bool                    init();

private:
  static  CudaLib                 _object;
};

} // jupiter
} // brt
