#ifndef LIBSYNEXENS3_EXCEPTION_H
#define LIBSYNEXENS3_EXCEPTION_H

#include <exception>
#include <string>

#include "macros.h"

namespace SY3_NAMESPACE
{

	typedef enum sy3_exception_type
	{
		SY3_EXCEPTION_TYPE_UNKNOWN,
		SY3_EXCEPTION_TYPE_EXPIRED_VERSION, 
		SY3_EXCEPTION_TYPE_COUNT,
	} sy3_exception_type;

	class libsynexens_exception : public std::exception
	{
	public:
		const char *get_message() const noexcept { return _msg.c_str(); }

		sy3_exception_type get_exception_type() const noexcept { return _exception_type; }

		const char *what() const noexcept override { return _msg.c_str(); }

	protected:
		libsynexens_exception(const std::string &msg, sy3_exception_type exception_type) noexcept
			: _msg(msg), _exception_type(exception_type)
		{
		}

	private:
		std::string _msg;
		sy3_exception_type _exception_type;
	};

	class SY3_EXPORT recoverable_exception : public libsynexens_exception
	{
	public:
		recoverable_exception(const std::string &msg, sy3_exception_type exception_type) noexcept;
	};

} // namespace librealsense

#endif
