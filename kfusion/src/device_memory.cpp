#include <device_memory.hpp>
#include <safe_call.hpp>
#include <cassert>
#include <iostream>
#include <cstdlib>

#include <intrin.h>
#define CV_XADD(addr,delta) _InterlockedExchangeAdd((long volatile*)(addr), (delta))
///
namespace kf
{
	DeviceMemory::~DeviceMemory() { release(); }
	DeviceMemory& DeviceMemory::operator =(const DeviceMemory& other_arg)
	{
		if (this != &other_arg)
		{
			if (other_arg.refcount)
				CV_XADD(other_arg.refcount, 1);
			release();

			data_ = other_arg.data_;
			size_ = other_arg.size_;
			refcount = other_arg.refcount;
		}
		return *this;
	}
	void DeviceMemory::create(size_t s)
	{
		if (s > 0)
		{
			if (data_)
				release();
			size_ = s;
			cudaSafeCall(cudaMalloc(&data_, size_));
			refcount = new int;
			*refcount = 1;
		}
		else
			return;
	}
	//
	void DeviceMemory::release()
	{
		if (refcount && CV_XADD(refcount, -1) == 1)
		{
			// cv::fastFree(refcount);
			delete refcount;
			cudaSafeCall(cudaFree(data_));
		}
		data_ = 0;
		size_ = 0;
		refcount = 0;
	}

	void DeviceMemory::hostTodevice(const void *data, size_t size_)
	{
		create(size_);
		cudaSafeCall(cudaMemcpy(data_, data, size_, cudaMemcpyHostToDevice));
	}
	void DeviceMemory::deviceTohost(void *data)
	{
		cudaSafeCall(cudaMemcpy(data, data_, size_, cudaMemcpyDeviceToHost));
	}
	size_t DeviceMemory::memorySize() { return size_; }
	bool DeviceMemory::empty() { return !data_; }
	void* DeviceMemory::data() { return data_; }
}