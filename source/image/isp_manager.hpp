/*
 * isp_manager.hpp
 *
 *  Created on: Nov 28, 2019
 *      Author: daniel
 */

#ifndef SOURCE_IMAGE_ISP_MANAGER_HPP_
#define SOURCE_IMAGE_ISP_MANAGER_HPP_

#include <vector>

namespace brt
{
namespace jupiter
{
namespace image
{

class ISP;
/*
 * \\class ISPManager
 *
 * created on: Nov 28, 2019
 *
 */
class ISPManager
{
public:
  ISPManager();
  virtual ~ISPManager();

          ISP*                    new_isp(bool group = false);
          void                    release();

private:
  std::vector<ISP*>               _isp_array;
};

} /* namespace image */
} /* namespace jupiter */
} /* namespace brt */

#endif /* SOURCE_IMAGE_ISP_MANAGER_HPP_ */
