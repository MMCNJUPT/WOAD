#ifndef LIBSYNEXENS3_PLATFORM_UVC_BASE_CAMERA_H
#define LIBSYNEXENS3_PLATFORM_UVC_BASE_CAMERA_H

#include "macros.h"
#include "platform/platform-interface.h"
#include "libuvc/libuvc.h"

namespace SY3_NAMESPACE
{

	namespace platform
	{
		class SY3_EXPORT uvc_base_camera : public platform_interface
		{

		private:
			int prepare_calib_depth_320_240();

			int prepare_calib_depth_640_480();

			int write_i2c(uint8_t slaver_id, uint16_t reg, uint8_t value);

		public:
			uvc_base_camera(uint16_t vid, uint16_t pid);
			int open_device() override;
			int start(onframe_callback *onframe) override;
			int stop() override;

			int set_format(request_format format_raw, request_format format_depth, request_format format_ir, request_format format_rgb) override;

			int get_calib_depth_320_240(ParamentsCalib &paraments_calib_depth_320_240) override;

			int get_calib_depth_640_480(ParamentsCalib &paraments_calib_depth_640_480) override;

			void write_calib_depth_320_240(uint8_t *data, uint16_t size) override;

			void write_calib_depth_640_480(uint8_t *data, uint16_t size)override;

			int set_exposure(uint16_t value) override;

			int get_exposure(uint16_t &value) override;

			int set_rgb_image_flip(uint16_t value) override;

			int set_rgb_image_mirror(uint16_t value) override;

			int set_tof_image_flip(uint16_t value) override;

			int set_tof_image_mirror(uint16_t value) override;

			int set_depth_image_mirror(uint16_t value) override;

			int set_depth_image_flip(uint16_t value) override;

			int set_depth_image_filter(uint16_t value) override;

			std::string get_fw_version() override;

			std::string get_device_sn() override;

			void write_device_sn(uint8_t sn[14])override;

			int get_filter_statu(uint16_t &value) override;

		protected:
			uvc_context_t *ctx_;
			uvc_device_t *dev_;
			uvc_device_handle_t *devh_;
			uvc_stream_ctrl_t ctrl_;
			uint8_t unit_id_;
		};
	}

};

#endif