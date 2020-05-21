/**
 *
 * Author : Author
 * Created On : Wed May 20 2020 - ${TIME}
 * File : encoder.hpp
 *
 */

#include <image.hpp>
#include <cuda_color_convert.hpp>

namespace brt
{
namespace jupiter
{

/**
 * \class Encoder
 *
 * Inherited from :
 *             image :: ImageConsumer 
 * \brief <description goes here>
 */
class Encoder : public image::ImageConsumer
{
public:
  Encoder() {}
  virtual ~Encoder() {}

  virtual void                    consume(image::ImageBox);

private:
  Convert                         _converter;
};

} // namespace jupiter
} // namespace brt

