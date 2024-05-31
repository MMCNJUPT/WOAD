/**
 * @file libsynexens3.h
 * @author synexens
 * @brief
 * @version 0.0.4
 * @date 2022-04-24
 *
 * Copyright(c) 2022 synexens.Co.Ltd. All rights reserved
 *
 */
#ifndef LIBSYNEXENS3_H
#define LIBSYNEXENS3_H

#include "macros.h"
#include "context.h"
#include "device.h"
#include "types.h"
#include "sensor.h"
#include "pipeline.h"
#include "config.h"
#include "frame.h"
#include "process-engine.h"
#include "common/exception.h"

namespace SY3_NAMESPACE
{

	/**
	 * \brief 获取SDK版本
	 * \return
	 */
	SY3_EXPORT const char *sy3_get_version(sy3_error &error);

	/**
	 * \brief 创建Context
	 * \return
	 */
	SY3_EXPORT context *sy3_create_context(sy3_error &error);

	/**
	 * \brief 创建pipeline
	 * \return
	 */
	SY3_EXPORT pipeline *sy3_create_pipeline(const context *ctx,sy3_error &error);

	/**
	 * \brief 创建config
	 * \return
	 */
	SY3_EXPORT config *sy3_create_config(sy3_error &error);

	/**
	 * \brief 计算fov
	 * \param[out] to_fov fov
	 * \param[in] intrin 内参
	 * \return
	 */
	SY3_EXPORT void sy3_fov(float to_fov[2], const struct sy3_intrinsics *intrin,sy3_error &error);

	/**
	 * \brief 获取coloemap table
	 * \param[out] bar colormap table 数组
	 * \param[out] length  colormap table 数组长度
	 * \return
	 */
	SY3_EXPORT void sy3_get_colormap_table(unsigned char **bar, int &length,sy3_error &error);
}

#endif