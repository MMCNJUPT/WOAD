/**
 * @file sensor.h
 * @author synexens
 * @brief 
 * @version 0.0.4
 * @date 2022-04-24
 * 
 * Copyright(c) 2022 synexens.Co.Ltd. All rights reserved
 * 
 */
#ifndef LIBSYNEXENS3_SENSOR_H
#define LIBSYNEXENS3_SENSOR_H

#include "option.h"

namespace SY3_NAMESPACE
{

	class sensor : public option_interface
	{
	public:
		/**
		 * \brief 
		 * \return
		 */
		virtual bool is_streaming(sy3_error &error) const = 0;

		virtual void write_calib_depth_320_240(uint8_t *data, uint16_t size) const= 0;

		virtual void write_calib_depth_640_480(uint8_t *data, uint16_t size) const= 0;
	
		virtual void write_device_sn(uint8_t sn[14])const =0;
	};

};
#endif
