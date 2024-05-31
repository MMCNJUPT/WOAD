/**
 * @file option.h
 * @author synexens
 * @brief 
 * @version 0.0.4
 * @date 2022-04-24
 * 
 * Copyright(c) 2022 synexens.Co.Ltd. All rights reserved
 * 
 */
#ifndef LIBSYNEXENS3_OPTION_H
#define LIBSYNEXENS3_OPTION_H

#include "macros.h"
#include"types.h"

namespace SY3_NAMESPACE
{
    typedef enum sy3_option
    {
        SY3_OPTION_EXPOSURE,
		SY3_OPTION_EXPOSURE_RANGE,
		SY3_OPTION_DISTANCE_RANGE,
		SY3_OPTION_DEFAULT_DISTANCE_RANGE,
		SY3_OPTION_RGB_IMAGE_FLIP,
		SY3_OPTION_RGB_IMAGE_MIRROR,
		SY3_OPTION_TOF_IMAGE_FLIP,
		SY3_OPTION_TOF_IMAGE_MIRROR,
        SY3_OPTION_DEPTH_IMAGE_FILTER,
        SY3_OPTION_COUNT,
    } sy3_option;
	const char *sy3_option_to_string(sy3_option option);

    class option_interface
    {
	public:
        virtual int get_option(sy3_option option, uint16_t &value,sy3_error &error)const=0;
		virtual int get_option(sy3_option option, uint16_t &max, uint16_t &min,sy3_error &error)const = 0;
        virtual int set_option(sy3_option option, uint16_t value,sy3_error &error)const=0;
		virtual int set_option(sy3_option option, uint16_t max, uint16_t min,sy3_error &error)const = 0;
    };

}
#endif
