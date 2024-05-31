/**
 * @file context.h
 * @author synexens
 * @brief
 * @version 0.0.4
 * @date 2022-04-24
 *
 * Copyright(c) 2022 synexens.Co.Ltd. All rights reserved
 *
 */
#ifndef LIBSYNEXENS3_CONTEXT_H
#define LIBSYNEXENS3_CONTEXT_H

#include <memory>
#include "macros.h"
#include "device-info.h"

namespace SY3_NAMESPACE
{

	class SY3_EXPORT context
	{

	public:

		/**
		 * \brief 获取设备信息
		 * \return
		 */
		virtual const device_info *get_device_info(sy3_error &error) const = 0;

		/**
		 * \brief 查询设备
		 * \return
		 */
		virtual device *query_device(sy3_error &error) const = 0;

		virtual ~context() = default;
	};

};

#endif