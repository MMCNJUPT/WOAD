#ifndef LIBSYNEXENS3_COMMON_CALIB_CONVERT_H
#define LIBSYNEXENS3_COMMON_CALIB_CONVERT_H

#include"paraments_calib.pb.h"
#include"reconstruction/utility.h"
#include "macros.h"

namespace SY3_NAMESPACE
{

	//inline void proto2struct(ParamentsCalib proto_calib, csapi::CameraCalibrationParam &paraments_calib) 
	//{
	//	paraments_calib.freq_lower = 80e6;
	//	paraments_calib.freq_high = 100e6;
	//	//tof camera intric
	//	paraments_calib.camera_sn = proto_calib.serialnumber();
	//	paraments_calib.tof_intric.fx = proto_calib.cameramatrix().fx();
	//	paraments_calib.tof_intric.fy = proto_calib.cameramatrix().fy();
	//	paraments_calib.tof_intric.ppx = proto_calib.cameramatrix().cx();
	//	paraments_calib.tof_intric.ppy = proto_calib.cameramatrix().cy();

	//	paraments_calib.tof_intric.coeffs[0] = proto_calib.distcoeffs().k1();
	//	paraments_calib.tof_intric.coeffs[1] = proto_calib.distcoeffs().k2();
	//	paraments_calib.tof_intric.coeffs[2] = proto_calib.distcoeffs().p1();
	//	paraments_calib.tof_intric.coeffs[3] = proto_calib.distcoeffs().p1();
	//	paraments_calib.tof_intric.coeffs[4] = proto_calib.distcoeffs().k3();

	//	//rgb camera intric
	//	if (proto_calib.otheroffset1_size()>0) {
	//		paraments_calib.rgb_intric.fx = proto_calib.otheroffset1(0);
	//		paraments_calib.rgb_intric.fy = proto_calib.otheroffset1(1);
	//		paraments_calib.rgb_intric.ppx = proto_calib.otheroffset1(2);
	//		paraments_calib.rgb_intric.ppy = proto_calib.otheroffset1(3);

	//		paraments_calib.rgb_intric.coeffs[0] = proto_calib.otheroffset1(4);
	//		paraments_calib.rgb_intric.coeffs[1] = proto_calib.otheroffset1(5);
	//		paraments_calib.rgb_intric.coeffs[2] = proto_calib.otheroffset1(6);
	//		paraments_calib.rgb_intric.coeffs[3] = proto_calib.otheroffset1(7);
	//		paraments_calib.rgb_intric.coeffs[4] = proto_calib.otheroffset1(8);
	//	}

	//

	//	//globaloffset
	//	paraments_calib.globaloffset_lower = proto_calib.globaloffset80();
	//	paraments_calib.globaloffset_high = proto_calib.globaloffset100();

	//	//wiggling 80  length = 39
	//	paraments_calib.wiggling_lut_lower_len = proto_calib.wigglingerror80_size();
	//	for (int i = 0; i < paraments_calib.wiggling_lut_lower_len; i++){
	//		paraments_calib.wiggling_lut_lower[i] = proto_calib.wigglingerror80(i);
	//	}
	//	//wiggling 100  length = 31
	//	paraments_calib.wiggling_lut_high_len = proto_calib.wigglingerror100_size();
	//	for (int i = 0; i < paraments_calib.wiggling_lut_high_len; i++){
	//		paraments_calib.wiggling_lut_high[i] = proto_calib.wigglingerror100(i);
	//	}
	//	//fppn 80  length = 6
	//	for (int i = 0; i < proto_calib.fitting80_size(); i++){
	//		paraments_calib.fppn_lower[i] = proto_calib.fitting80(i);
	//	}
	//	//fppn 100  length = 6
	//	for (int i = 0; i < proto_calib.fitting100_size(); i++){
	//		paraments_calib.fppn_high[i] = proto_calib.fitting100(i);
	//	}
	//	//temperature 80  length = 4
	//	for (int i = 0; i < proto_calib.tempoffset80_size(); i++){
	//		paraments_calib.temperature_coeffs_lower[i] = proto_calib.tempoffset80(i);
	//	}
	//	//temperature 100  length = 4
	//	for (int i = 0; i < proto_calib.tempoffset100_size(); i++){
	//		paraments_calib.temperature_coeffs_high[i] = proto_calib.tempoffset100(i);
	//	}
	//	//全局深度补偿
	//	paraments_calib.depthOffset = proto_calib.depthoffset();

	//	//fppn offset 80  length = 6 
	//	for (int i = 0; i < proto_calib.fitting80offset_size(); i++){
	//		paraments_calib.fppn_offset_lower[i] = proto_calib.fitting80offset(i);
	//	}
	//	//fppn offset 100  length = 6
	//	for (int i = 0; i < proto_calib.fitting100offset_size(); i++){
	//		paraments_calib.fppn_offset_high[i] = proto_calib.fitting100offset(i);
	//	}

	//	//tof2rgb_R 100  length = 9
	//	for (int i = 0; i < proto_calib.otheroffset2_size(); i++){
	//		paraments_calib.tof2rgb_R[i] = proto_calib.otheroffset2(i);
	//	}

	//	//tof2rgb_T 100  length = 3
	//	for (int i = 0; i < proto_calib.otheroffset3_size(); i++){
	//		paraments_calib.tof2rgb_T[i] = proto_calib.otheroffset3(i);
	//	}

	//	//otheroffset4 备用参数
	//	for (int i = 0; i < proto_calib.otheroffset4_size(); i++){
	//		//Sleep(1);
	//	}

	//	//otheroffset5 备用参数
	//	for (int i = 0; i < proto_calib.otheroffset4_size(); i++){
	//		//Sleep(1);
	//	}
	//	//频率80MHZ下  wiggling_lut_lower_len参数大于20则认为wiggling步长为50mm，否则为100mm
	//	//单频的另外考虑，因为单频40MHZ的只标定wiggling到半个周期。paraments_calib.freq_lower = 80e6 &&
	//	if ( paraments_calib.wiggling_lut_lower_len > 20)
	//		paraments_calib.wiggling_scale_factor = 0.02;
	//	else
	//		paraments_calib.wiggling_scale_factor = 0.01;
	//}
	inline void proto2struct(ParamentsCalib proto_calib, csapi::CameraCalibrationParam &paraments_calib)
	{
		paraments_calib.freq_lower = 80e6;
		paraments_calib.freq_high = 100e6;
		//tof camera intric
		paraments_calib.camera_sn = proto_calib.serialnumber();
		paraments_calib.tof_intric.fx = proto_calib.cameramatrix().fx();
		paraments_calib.tof_intric.fy = proto_calib.cameramatrix().fy();
		paraments_calib.tof_intric.ppx = proto_calib.cameramatrix().cx();
		paraments_calib.tof_intric.ppy = proto_calib.cameramatrix().cy();

		paraments_calib.tof_intric.coeffs[0] = proto_calib.distcoeffs().k1();
		paraments_calib.tof_intric.coeffs[1] = proto_calib.distcoeffs().k2();
		paraments_calib.tof_intric.coeffs[2] = proto_calib.distcoeffs().p1();
		paraments_calib.tof_intric.coeffs[3] = proto_calib.distcoeffs().p2();
		paraments_calib.tof_intric.coeffs[4] = proto_calib.distcoeffs().k3();

		//rgb camera intric
		if (proto_calib.otheroffset1_size() >=9) {
			paraments_calib.rgb_intric.fx = proto_calib.otheroffset1(0);
			paraments_calib.rgb_intric.fy = proto_calib.otheroffset1(1);
			paraments_calib.rgb_intric.ppx = proto_calib.otheroffset1(2);
			paraments_calib.rgb_intric.ppy = proto_calib.otheroffset1(3);

			paraments_calib.rgb_intric.coeffs[0] = proto_calib.otheroffset1(4);
			paraments_calib.rgb_intric.coeffs[1] = proto_calib.otheroffset1(5);
			paraments_calib.rgb_intric.coeffs[2] = proto_calib.otheroffset1(6);
			paraments_calib.rgb_intric.coeffs[3] = proto_calib.otheroffset1(7);
			paraments_calib.rgb_intric.coeffs[4] = proto_calib.otheroffset1(8);
	}

	//globaloffset
	paraments_calib.globaloffset_lower = proto_calib.globaloffset80();
	paraments_calib.globaloffset_high = proto_calib.globaloffset100();

	//wiggling 80  length = 39
	paraments_calib.wiggling_lut_lower_len = proto_calib.wigglingerror80_size();
	for (int i = 0; i < paraments_calib.wiggling_lut_lower_len; i++) {
		paraments_calib.wiggling_lut_lower[i] = proto_calib.wigglingerror80(i);
	}
	//wiggling 100  length = 31
	paraments_calib.wiggling_lut_high_len = proto_calib.wigglingerror100_size();
	for (int i = 0; i < paraments_calib.wiggling_lut_high_len; i++) {
		paraments_calib.wiggling_lut_high[i] = proto_calib.wigglingerror100(i);
	}
	//fppn 80  length = 6
	for (int i = 0; i < proto_calib.fitting80_size(); i++) {
		paraments_calib.fppn_lower[i] = proto_calib.fitting80(i);
	}
	//fppn 100  length = 6
	for (int i = 0; i < proto_calib.fitting100_size(); i++) {
		paraments_calib.fppn_high[i] = proto_calib.fitting100(i);
	}
	//temperature 80  length = 4
	for (int i = 0; i < proto_calib.tempoffset80_size(); i++) {
		paraments_calib.temperature_coeffs_lower[i] = proto_calib.tempoffset80(i);
	}
	//temperature 100  length = 4
	for (int i = 0; i < proto_calib.tempoffset100_size(); i++) {
		paraments_calib.temperature_coeffs_high[i] = proto_calib.tempoffset100(i);
	}
	//全局深度补偿
	paraments_calib.depthOffset = proto_calib.depthoffset();

	//fppn offset 80  length = 6 
	for (int i = 0; i < proto_calib.fitting80offset_size(); i++) {
		paraments_calib.fppn_offset_lower[i] = proto_calib.fitting80offset(i);
	}
	//fppn offset 100  length = 6
	for (int i = 0; i < proto_calib.fitting100offset_size(); i++) {
		paraments_calib.fppn_offset_high[i] = proto_calib.fitting100offset(i);
	}

	//tof2rgb_R 100  length = 9
	for (int i = 0; i < proto_calib.otheroffset2_size(); i++) {
		paraments_calib.tof2rgb_R[i] = proto_calib.otheroffset2(i);
	}

	//tof2rgb_T 100  length = 3
	for (int i = 0; i < proto_calib.otheroffset3_size(); i++) {
		paraments_calib.tof2rgb_T[i] = proto_calib.otheroffset3(i);
	}

	//otheroffset4 备用参数
	paraments_calib.other_offset_lut_4_len = proto_calib.otheroffset4_size();
	for (int i = 0; i < proto_calib.otheroffset4_size(); i++) {
		paraments_calib.other_offset_lut_4[i] = proto_calib.otheroffset4(i);
	}

	//otheroffset5 备用参数
	paraments_calib.other_offset_lut_5_len = proto_calib.otheroffset5_size();
	for (int i = 0; i < proto_calib.otheroffset5_size(); i++) {
		paraments_calib.other_offset_lut_5[i] = proto_calib.otheroffset5(i);
	}
	//频率80MHZ下  wiggling_lut_lower_len参数大于20则认为wiggling步长为50mm，否则为100mm
	//单频的另外考虑，因为单频40MHZ的只标定wiggling到半个周期。paraments_calib.freq_lower = 80e6 &&
	if (paraments_calib.wiggling_lut_lower_len > 20)
		paraments_calib.wiggling_scale_factor = 0.02;
	else
		paraments_calib.wiggling_scale_factor = 0.01;
}

} // namespace librealsense

#endif
