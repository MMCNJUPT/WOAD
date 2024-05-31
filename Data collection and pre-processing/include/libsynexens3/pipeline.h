/**
 * @file pipeline.h
 * @author synexens
 * @brief 
 * @version 0.0.4
 * @date 2022-04-24
 * 
 * Copyright(c) 2022 synexens.Co.Ltd. All rights reserved
 * 
 */
#ifndef LIBSYNEXENS3_PIPELINE_H
#define LIBSYNEXENS3_PIPELINE_H
#include <mutex>

#include "macros.h"
#include "frame.h"
#include "context.h"
#include"process-engine.h"
#include"types.h"

namespace SY3_NAMESPACE
{

	class SY3_EXPORT pipeline
	{

	public:
		/**
		 * \brief 启动pipeline
		 * \param[in] cfg pipeline配置
		 * \return
		 */
		virtual void start(const config *cfg,sy3_error &error) = 0;


		virtual process_engine* get_process_engin(sy3_error &error)=0;

		/**
		 * \brief 停止pipeline
		 * \return
		 */
		virtual void stop(sy3_error &error) = 0;

		/**
		 * \brief 获取pipelinen帧集合
		 * \param[in] timeout_ms 超时
		 * \return frameset,需要手动释放内存
		 */
		virtual frameset *wait_for_frames(unsigned int timeout_ms,sy3_error &error) = 0;

		/**
		 * \brief 获取当前设备
		 * \return
		 */
		virtual const device *get_device(sy3_error &error) = 0;


		virtual ~pipeline() = default;
	};

};

#endif