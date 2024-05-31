/**
 * @file device.h
 * @author synexens
 * @brief 
 * @version 0.0.4
 * @date 2022-04-24
 * 
 * Copyright(c) 2022 synexens.Co.Ltd. All rights reserved
 * 
 */
#ifndef LIBSYNEXENS3_DEVICE_H
#define LIBSYNEXENS3_DEVICE_H

#include <vector>
#include "macros.h"
#include "sensor.h"
#include"types.h"
#include "common/exception.h"

namespace SY3_NAMESPACE
{

  class SY3_EXPORT device
  {

  public:
  
    /**
     * \brief 获取sensor
     * \return
     */

    virtual const sensor *get_sensor(sy3_error &error) const = 0;

    /**
     * \brief 获取sensor 数量
     * \return
     */
    virtual size_t get_sensors_count(sy3_error &error) const = 0;

    /**
     * \brief 获取设备类型
     * \return
     */
    virtual const sy3_device_type get_type(sy3_error &error) const = 0;

    /**
     * \brief 获取设备所支持的流
     * \return
     */
    virtual const std::vector<sy3_stream> get_support_stream(sy3_error &error) const = 0;

    /**
     * \brief 获取设备支持的格式
     * \return
     */
    virtual const std::vector<sy3_format> get_support_format(sy3_error &error) const = 0;

    /**
     * \brief 获取设备指定数据流所支持的格式
     * \return
     */
    virtual const std::vector<sy3_format> get_support_format(sy3_stream stream,sy3_error &error) const = 0;

	virtual ~device() {};

  };

  /**
   * \brief 获取设备信息
   * \return
   */
  SY3_EXPORT const char *sy3_get_device_info(const device *device, sy3_camera_info info,sy3_error &error);


};

#endif