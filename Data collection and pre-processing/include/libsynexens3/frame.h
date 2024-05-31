/**
 * @file frame.h
 * @author synexens
 * @brief 
 * @version 0.0.4
 * @date 2022-04-24
 * 
 * Copyright(c) 2022 synexens.Co.Ltd. All rights reserved
 * 
 */
#ifndef LIBSYNEXENS3_FRAME_H
#define LIBSYNEXENS3_FRAME_H

#include "macros.h"
#include "types.h"
#include "config.h"

namespace SY3_NAMESPACE
{

	class frame
	{

	public:
		/**
		 * \brief 获取帧图像的宽
		 * \return
		 */
		virtual const int get_width() = 0;

		/**
		 * \brief 获取帧图像的高
		 * \return
		 */
		virtual const int get_height() = 0;

		/**
		 * \brief 获取帧类型
		 * \return
		 */
		virtual const sy3_stream get_type() = 0;

		/**
		 * \brief 获取帧图像数组
		 * \return
		 */
		virtual void *get_data() = 0;

		/**
		 * \brief 获取帧配置
		 * \return
		 */
		virtual const stream_profile *get_profile() const = 0;

		/**
		 * \brief 保存帧数到本地，供调试使用
		 * \param[in] filenam 要保存的文件名
		 * \return
		 */
		virtual int dump(const char *filenam) = 0;

		virtual ~frame() = default;
	};

	class depth_frame : public frame
	{
	public:
		/**
		 * \brief 获取设备支持的格式
		 * \return
		 */
		virtual uint8_t *apply_colormap() = 0;
		virtual ~depth_frame() = default;
	};

	class ir_frame : public frame
	{
	public:
		virtual ~ir_frame() = default;
	};

	class rgb_frame : public frame
	{
	public:
		virtual ~rgb_frame() = default;
	};

	class raw_frame : public frame
	{
	public:
		virtual ~raw_frame() = default;
	};

	class points : public frame
	{
	public:
		points(){};

		/**
		 * \brief 获取点云数组
		 * \return
		 */
		virtual float *get_points() = 0;
		/**
		 * \brief 获取点云长度
		 * \return
		 */
		virtual int get_length() = 0;

	private:
		void *get_data() override { return nullptr; }
		const stream_profile *get_profile() const override { return nullptr; }
	};

	class frameset
	{

	public:
		frameset(){};

		/**
		 * \brief 获取depth帧
		 * \return
		 */
		virtual depth_frame *get_depth_frame() = 0;

		/**
		 * \brief 获取rgb帧
		 * \return
		 */
		virtual rgb_frame *get_rgb_frame() = 0;

		/**
		 * \brief 获取ir帧
		 * \return
		 */
		virtual ir_frame *get_ir_frame() = 0;

		/**
		 * \brief 获取raw帧
		 * \return
		 */
		virtual raw_frame *get_raw_frame() = 0;

		virtual ~frameset(){};
	};

}
#endif