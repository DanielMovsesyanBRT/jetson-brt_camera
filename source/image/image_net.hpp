/**
 *
 * Author : Author
 * Created On : Thu May 21 2020 - ${TIME}
 * File : image_net.hpp
 *
 */

#include <vector> 
#include <unordered_set> 
#include <thread>
#include <mutex>
#include <memory>

#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h> 
#include <netinet/in.h> 


#define IMAGE_NET_TCP_PORT                  (4723)
#define IMAGE_NET_BACKLOG                   (10)

namespace brt
{
namespace jupiter
{
  
/**
 * \class iServerInterface
 *
 * \brief <description goes here>
 */
class iServerInterface
{
public:
  virtual ~iServerInterface() {}
  virtual void                    on_receive(int socket, const uint8_t*, size_t) = 0;
};

/**
 * \class ImageNet
 *
 * \brief <description goes here>
 */
class ImageNet
{
public:
  ImageNet();
  virtual ~ImageNet();

          void                    register_interface(iServerInterface*);
          void                    unregister_interface(iServerInterface*);
          bool                    init();

private:
  /**
   * \class client
   *
   * \brief <description goes here>
   */
  class client
  {
  public:
    client(ImageNet* owner);
    ~client() { if (_socket >= 0) ::close(_socket); }

    operator int() const { return _socket; }
    void                          receive();

  private:
    ImageNet*                       _owner;
    int                             _socket;
    struct sockaddr_in              _addr;
  };

  typedef std::shared_ptr<client>   client_ptr;

  void                            stop();
  int                             socket() const { return _server_socket; }

  void                            loop();

private:  
  int                             _server_socket;
  std::vector<client_ptr>         _clients;
  std::thread                     _main_thread;
  std::mutex                      _mutex;
  std::unordered_set<iServerInterface*>
                                  _interfeces;
  int                             _pipe[2];
};

} // namespace jupiter
} // namespace brt

