#ifndef LIBSYNEXENS3_PLATFORM_PLATFORM_H
#define LIBSYNEXENS3_PLATFORM_PLATFORM_H
#include<map>


#if defined(_MSC_VER)


#else

#endif

#include"platform-interface.h"


namespace SY3_NAMESPACE
{

    namespace platform
    {
        const uint16_t CS30D_VID = 0x2207;
        const uint16_t CS30D_PID = 0x0016;
		const uint16_t CS20_PID = 0x6666;
		const uint16_t CS20_VID = 0x2222;

        static const std::multimap<std::uint16_t, std::uint16_t> synexens_device_ids = {
            {CS30D_VID,CS30D_PID},
            {CS20_PID,CS20_VID},
        };

	    int verification_validity();

		int find_target_device_vid_pid(uint16_t &vid, uint16_t &pid);

        platform_interface* create_platform_device(uint16_t vid, uint16_t pid);

    };
};

#endif