/**
 * @file types.h
 * @author synexens
 * @brief 
 * @version 0.0.4
 * @date 2022-04-24
 * 
 * Copyright(c) 2022 synexens.Co.Ltd. All rights reserved
 * 
 */
#ifndef LIBSYNEXENS3_TYPES_H
#define LIBSYNEXENS3_TYPES_H

#include "macros.h"
#include <memory>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace SY3_NAMESPACE
{
	
	#define SY3_DEFAULT_TIMEOUT 15000


	template <class T>
	std::string hexify(const T &val)
	{
		static_assert((std::is_integral<T>::value), "hexify supports integral built-in types only");

		std::ostringstream oss;
		oss << std::setw(sizeof(T) * 2) << std::setfill('0') << std::uppercase << std::hex << val;
		return oss.str().c_str();
	}

	typedef enum sy3_error
	{
		SUCCESS = 0,
		INVALID_PID,
		INVALID_VID,
		DEVICE_NOT_FOUND,
		INVALID_FORMAT,
		INCONSISTENCY_RES,
		OPEN_FAILED,
		NOT_IMPLEMENTED,
		INVALID_INSTANCE,

	}sy3_error;

	SY3_EXPORT const char *sy3_error_to_string(sy3_error err);

	typedef enum sy3_device_type
	{
		DEVICE_CS30,
		DEVICE_CS20,
	} sy3_device_type;

	typedef enum sy3_camera_info
	{
		SY3_CAMERA_INFO_NAME,						  /**< Friendly name */
		SY3_CAMERA_INFO_SERIAL_NUMBER,				  /**< Device serial number */
		SY3_CAMERA_INFO_FIRMWARE_VERSION,			  /**< Primary firmware version */
		SY3_CAMERA_INFO_RECOMMENDED_FIRMWARE_VERSION, /**< Recommended firmware version */
		SY3_CAMERA_INFO_PRODUCT_ID,					  /**< Product ID as reported in the USB descriptor */
		SY3_CAMERA_INFO_COUNT						  /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
	} sy3_camera_info;

	const char *sy3_camera_info_to_string(sy3_camera_info info);

	typedef enum sy3_stream
	{
		SY3_STREAM_NONE,
		SY3_STREAM_RAW,
		SY3_STREAM_DEPTH=2,
		SY3_STREAM_RGB,
		SY3_STREAM_IR,
		SY3_STREAM_COUNT,

	} sy3_stream;

	SY3_EXPORT const char *sy3_stream_to_string(sy3_stream stream);

	typedef struct sy3_format {
		sy3_stream stream;
		int width;
		int height;
	}sy3_format;

	typedef struct sy3_intrinsics
	{
		int width;
		int height;
		float ppx;
		float ppy;
		float fx;
		float fy;
		float coeffs[5];
	} sy3_intrinsics;

	struct SY3_EXPORT stream_profile
	{

		stream_profile() {};
		virtual uint16_t  get_width()const = 0;
		virtual uint16_t  get_height()const = 0;
		virtual sy3_stream get_stream() const = 0;
		virtual sy3_intrinsics get_intrinsics()const = 0;
		virtual bool enable()const = 0;

	};

};

#endif