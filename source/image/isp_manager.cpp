/*
 * isp_manager.cpp
 *
 *  Created on: Nov 28, 2019
 *      Author: daniel
 */

#include "isp_manager.hpp"
#include "isp.hpp"

namespace brt
{
namespace jupiter
{
namespace image
{

/*
 * \\fn ISPManager::ISPManager
 *
 * created on: Nov 28, 2019
 * author: daniel
 *
 */
ISPManager::ISPManager()
{
}

/*
 * \\fn ISPManager::~ISPManager
 *
 * created on: Nov 28, 2019
 * author: daniel
 *
 */
ISPManager::~ISPManager()
{
  release();
}

/*
 * \\fn ISP* ISPManager::new_isp
 *
 * created on: Nov 28, 2019
 * author: daniel
 *
 */
ISP* ISPManager::new_isp(bool group /*= false*/)
{
  _isp_array.push_back(new ISP(group));
  return _isp_array[_isp_array.size() - 1];
}

/*
 * \\fn ISPManager::release
 *
 * created on: Nov 28, 2019
 * author: daniel
 *
 */
void ISPManager::release()
{
  for (auto isp : _isp_array)
  {
    isp->stop();
    delete isp;
  }
  _isp_array.clear();
}


} /* namespace image */
} /* namespace jupiter */
} /* namespace brt */
