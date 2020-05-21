/**
 *
 * Author : Author
 * Created On : Thu May 21 2020 - ${TIME}
 * File : image_net.cpp
 *
 */


#include <stdlib.h> 
#include <string.h> 
#include <errno.h> 
#include <sys/wait.h> 
#include <fcntl.h> 

#include <iostream>
    
#include "image_net.hpp"


namespace brt
{
namespace jupiter
{

/**
 * \fn  constructor ImageNet::ImageNet
 *
 * \brief <description goes here>
 */
ImageNet::ImageNet()
: _server_socket(-1)
, _clients()
, _main_thread()
, _pipe{-1, -1}
{

}

/**
 * \fn  destructor ImageNet::~ImageNet
 *
 * \brief <description goes here>
 */
ImageNet::~ImageNet()
{
  stop();
}

/**
 * \fn  ImageNet::init
 *
 * @return  bool
 * \brief <description goes here>
 */
bool ImageNet::init()
{
  if (_main_thread.joinable() && (_server_socket != -1))
    return true; // nothing to do here

  stop();
  
  _server_socket = ::socket(AF_INET, SOCK_STREAM, 0);
  if (_server_socket == -1)
    return false;

  struct sockaddr_in    addr;
  addr.sin_family = AF_INET;
  addr.sin_port   = htons(IMAGE_NET_TCP_PORT);
  addr.sin_addr.s_addr = INADDR_ANY;
  memset(&addr.sin_zero, 0, sizeof(addr.sin_zero));

  if (bind(_server_socket, (struct sockaddr*)&addr, sizeof(addr)) == -1)
  {
    close(_server_socket);
    _server_socket = -1;
    return false;
  }

  if (listen(_server_socket, IMAGE_NET_BACKLOG) == -1)
  {
    close(_server_socket);
    _server_socket = -1;
    return false;
  }

  if (::pipe(_pipe) == -1)
  {
    close(_server_socket);
    _server_socket = -1;
    return false;
  }

  _main_thread = std::thread([](ImageNet *imnet)
  {
    imnet->loop();
  },this);

  return true;
}


/**
 * \fn  ImageNet::stop
 *
 * \brief <description goes here>
 */
void ImageNet::stop()
{
  if (_main_thread.joinable())
  {
    uint32_t event = 1;
    ::write(_pipe[1], &event, sizeof(event));

    _main_thread.join();
    _clients.clear();

    ::close(_server_socket);
    ::close(_pipe[0]);
    ::close(_pipe[1]);
  }
}

/**
 * \fn  ImageNet::loop
 *
 * \brief <description goes here>
 */
void ImageNet::loop()
{
  while(true)
  {
    fd_set rfds;
    int retval, max_fd = 0;

    FD_ZERO(&rfds);
    max_fd = (max_fd < _server_socket)?_server_socket : max_fd;
    FD_SET(_server_socket, &rfds);

    max_fd = (max_fd < _pipe[0])?_pipe[0] : max_fd;
    FD_SET(_server_socket, &rfds);

    for (auto clnt : _clients)
    {
      max_fd = (max_fd < *clnt)?*clnt : max_fd;
      FD_SET(*clnt, &rfds);
    }

    retval = select(max_fd + 1, &rfds, NULL, NULL, NULL);
    if (retval == -1)
      std::cerr << "error: select " << errno << std::endl;
    else if (retval > 0)
    {
      if (FD_ISSET(_pipe[0], &rfds))
      {
        uint32_t event;
        ::read(_pipe[0], &event, sizeof(event));

        //todo: receive message for now just exit
        break;
      }

      for (auto clnt : _clients)
      {
        if (FD_ISSET(*clnt, &rfds))
          clnt->receive();
      }

      if (FD_ISSET(_server_socket, &rfds))
      {
        client_ptr clnt(new client(this));
        if (clnt >= 0)
          _clients.push_back(clnt);
      }
    }
  }
}


/**
 * \fn  ImageNet::register_interface
 *
 * @param  iface : iServerInterface* 
 * \brief <description goes here>
 */
void ImageNet::register_interface(iServerInterface* iface)
{
  std::lock_guard<std::mutex> l(_mutex);
  _interfeces.insert(iface);
}

/**
 * \fn  ImageNet::unregister_interface
 *
 * @param  iface : iServerInterface* 
 * \brief <description goes here>
 */
void ImageNet::unregister_interface(iServerInterface* iface)
{
  std::lock_guard<std::mutex> l(_mutex);
  _interfeces.erase(iface);
}


/**
 * \fn  constructor ImageNet::client::client
 *
 * @param  owner : ImageNet* 
 * \brief <description goes here>
 */
ImageNet::client::client(ImageNet* owner)
: _owner(owner)
, _socket(-1)
, _addr()
{ 
  socklen_t sin_size = sizeof(_addr);
  _socket = accept(_owner->socket(), (struct sockaddr*)&_addr, &sin_size);
  if (_socket >= 0)
    ::fcntl(_socket, F_SETFL, O_NONBLOCK);
}

/**
 * \fn  ImageNet::client::receive
 *
 * \brief <description goes here>
 */
void ImageNet::client::receive()
{
  uint8_t  buffer[1024];
  int ret;

  while (true)
  {
    ret = recv(_socket, buffer, sizeof(buffer), 0);
    if (ret <= 0)
      break;

    std::lock_guard<std::mutex> l(_owner->_mutex);
    for (auto iface : _owner->_interfeces)
      iface->on_receive(_socket, buffer, ret);
  }
}

} // namespace jupiter
} // namespace brt






