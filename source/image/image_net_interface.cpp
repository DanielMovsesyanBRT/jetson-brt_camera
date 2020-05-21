/**
 *
 * Author : Author
 * Created On : Thu May 21 2020 - ${TIME}
 * File : image_net_interface.cpp
 *
 */

#include "image_net_interface.hpp"
#include <metadata.hpp>

#include <string.h>

namespace brt
{
namespace jupiter
{

/**
 * \fn  ImageNetInterface::on_receive
 *
 * @param  socket : int 
 * @param  buffer :  const uint8_t* 
 * @param  len :  size_t 
 * \brief <description goes here>
 */
void ImageNetInterface::on_receive(int socket, const uint8_t* buffer, size_t len)
{
  std::lock_guard<std::mutex> l(_mutex);
  client& clnt = _client_map[socket];
  clnt._socket = socket;

  while(len-- > 0)
  {
    if (*buffer == '\n')
      process_command(clnt);
    else if (*buffer != '\r')
      clnt._command_buffer.push_back(*buffer);
    
    buffer++;
  }
}

/**
 * \fn  ImageNetInterface::process_command
 *
 * @param  clnt : client &
 * \brief <description goes here>
 */
void ImageNetInterface::process_command(client &clnt)
{
  clnt._command_buffer.push_back((char)0);
  const char* buf = clnt._command_buffer.data();

  if (strstr(buf, "list") == buf)
  {
    // send the list of images
    std::string image_list;
    for (auto name : _image_name_db)
      image_list += name + '\n';

    ::send(clnt._socket, image_list.c_str(), image_list.size(), 0);
  }

  else if (strstr(buf, "request") == buf)
  {
    buf += 7; // request
    while ((*buf != '\0') && isspace(*buf))
      buf++;
    
    if (*buf != '\0')
      clnt._image_name = buf;

    clnt._request_flag = true;
  }

  clnt._command_buffer.clear();
}


/**
 * \fn  ImageNetInterface::post_consume
 *
 * @param   images : image::ImageBox
 * \brief <description goes here>
 */
void ImageNetInterface::post_consume(image::ImageBox images)
{
  for (auto img : images)
  {
    std::string name = img->get<std::string>("net_image");
    image::RawRGBPtr bits;
    int socket = -1;

    _mutex.lock();
    for (auto clnt : _client_map)
    {
      if (clnt.second._image_name == name)
      {
        if (!clnt.second._request_flag)
          break;

        bits = img->get_bits();
        if (!bits)
          break;

        clnt.second._request_flag = false;
        socket = clnt.first;
        break;        
      }
    }
    _mutex.unlock();

    if (bits && (socket >= 0))
    {
      ::send(socket, "image\n", 6, 0);
      
      uint32_t value = bits->width();
      ::send(socket, &value, sizeof(value), 0);

      value = bits->height();
      ::send(socket, &value, sizeof(value), 0);

      value = bits->depth();
      ::send(socket, &value, sizeof(value), 0);
      ::send(socket, bits->bytes(), bits->size(), 0);
    }
  }
}

/**
 * \fn  ImageNetInterface::add_producer
 *
 * @param   producer : image::ImageProducer*
 * @param  name : const char* 
 * \brief <description goes here>
 */
void ImageNetInterface::add_producer(image::ImageProducer* producer,const char* name)
{
  if (name == nullptr)
    return;
  
  _image_name_db.insert(std::string(name));
  producer->register_consumer(this,(name != nullptr) ? Metadata().set<std::string>("net_image",name) : Metadata() );

  _net_server.init();
  _net_server.register_interface(this);
}



} // namespace jupiter
} // namespace brt

