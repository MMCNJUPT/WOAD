/**********************************************************
 *
 *    History:         
 *
 * 	 <time>  2022/07/06
 *
***********************************************************/

#ifndef LIBSYNEXENS3_PLATFORM_UVC_XU_DEF_H
#define LIBSYNEXENS3_PLATFORM_UVC_XU_DEF_H

#define CMD_TOOLS_CTRL_1  0x01
#define UVC_XU_SET_FORMAT 0xA1
#define UVC_XU_PREPARE_START 0xA2
#define UVC_XU_SET_EXPOSURE 0xA3
#define UVC_XU_SET_RGB_IMAGE_FLIP 0xA4
#define UVC_XU_SET_RGB_IMAGE_MIRROR 0xA5
#define UVC_XU_SET_TOF_IMAGE_FLIP 0xA6
#define UVC_XU_SET_TOF_IMAGE_MIRROR 0xA7
#define UVC_XU_SET_DEPTH_IMAGE_FLIP 0xA8
#define UVC_XU_SET_DEPTH_IMAGE_MIRROR 0xA9
#define UVC_XU_SET_DEPTH_IMAGE_FILTER 0xAA
#define UVC_XU_START_PREVIEW 0xB0

#define UVC_XU_GET_CALIB_DATA_DEPTH_320_240 0xC1
#define UVC_XU_GET_CALIB_DATA_DEPTH_640_480 0xC2
#define UVC_XU_GET_EXPOSURE 0xC3

#define UVC_XU_SET_CALIB_DATA_DEPTH_320_240 0xC4
#define UVC_XU_SET_CALIB_DATA_DEPTH_640_480 0xC5
#define UVC_XU_WRITE_I2C 0xC6

#define UVC_XU_GET_FIRMWARE_VERSION 0xC7
#define UVC_XU_GET_SERIAL_NUMBER 0xC8
#define UVC_XU_WRITE_SERIAL_NUMBER 0xC9
#define UVC_XU_GET_FILTER_STATU 0xCA

#endif