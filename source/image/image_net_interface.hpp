/**
 *
 * Author : Author
 * Created On : Thu May 21 2020 - ${TIME}
 * File : image_net_interface.hpp
 *
 */

#include "image_net.hpp"
#include <image.hpp>

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace brt
{
namespace jupiter
{


/**
 * \class ImageNetInterface
 *
 * Inherited from :
 *             iServerInterface 
 *             image :: ImageConsumer 
 * \brief <description goes here>
 */
class ImageNetInterface : public iServerInterface,
                          public image::PostImageConsumer
{
public:
  ImageNetInterface() : image::PostImageConsumer(1) {}
  virtual ~ImageNetInterface() {}

  virtual void                    on_receive(int socket, const uint8_t*, size_t);
  virtual void                    post_consume(image::ImageBox);

          void                    add_producer(image::ImageProducer*,const char* name);
private:
  /**
   * \struct client
   *
   * \brief <description goes here>
   */
  struct client
  {
    client() 
    : _socket(-1)
    , _request_flag(false)
    {}

    std::string                   _image_name;
    int                           _socket;
    bool                          _request_flag;
    std::vector<char>             _command_buffer;
  };

          void                    process_command(client &clnt);

private:
  std::unordered_set<std::string> _image_name_db;
  std::unordered_map<int,client>  _client_map;
  std::mutex                      _mutex;

  ImageNet                        _net_server;
};


} // namespace jupiter
} // namespace brt

