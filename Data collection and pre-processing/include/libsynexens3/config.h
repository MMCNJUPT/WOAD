/**
 * @file config.h
 * @author synexens
 * @brief 
 * @version 0.0.4
 * @date 2022-04-24
 * 
 * Copyright(c) 2022 synexens.Co.Ltd. All rights reserved
 * 
 */

#ifndef LIBSYNEXENS3_CONFIG_H
#define LIBSYNEXENS3_CONFIG_H

#include <string>
#include "macros.h"
#include "libsynexens3/types.h"

namespace SY3_NAMESPACE
{

	class SY3_EXPORT config
	{
	public:
		config(){};

		/**
		 * \brief 使用指定的分辨率开启流
		 * \param[in] stream 要启用的数据流类型
		 * \param[in] width  数据流图像的宽
		 * \param[in] height 数据流图像的高
		 * \return
		 */
		virtual void enable_stream(sy3_stream stream, uint16_t width, uint16_t height,sy3_error &error) = 0;

		/**
		 * \brief 禁用指定的数据流
		 * \param[in] stream 要禁用的数据流类型
		 * \return
		 */
		virtual void disable_stream(sy3_stream stream,sy3_error &error) = 0;

		/**
		 * \brief 禁用全部数据流
		 * \return
		 */
		virtual void disable_all_streams(sy3_error &error) = 0;

	
		virtual ~config() = default;

	};
}

#endif