#include <device_memory.hpp>
#include <safe_call.hpp>
#include <cassert>
#include <iostream>
#include <cstdlib>

#include <intrin.h>
#define CV_XADD(addr, delta) _InterlockedExchangeAdd((long volatile *)(addr), (delta))
///
namespace kf
{
	DeviceMemory::~DeviceMemory() { release(); }
	DeviceMemory &DeviceMemory::operator=(const DeviceMemory &other_)
	{
		if (this != &other_)
		{
			if (other_.refcount_)
				CV_XADD(other_.refcount_, 1);
			release();

			data_ = other_.data_;
			size_ = other_.size_;
			refcount_ = other_.refcount_;
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
			refcount_ = new int;
			*refcount_ = 1;
		}
		else
			return;
	}
	//
	void DeviceMemory::release()
	{
		if (refcount_ && CV_XADD(refcount_, -1) == 1)
		{
			// cv::fastFree(refcount);
			delete refcount_;
			cudaSafeCall(cudaFree(data_));
		}
		data_ = 0;
		size_ = 0;
		refcount_ = 0;
	}

	void DeviceMemory::upload(const void *v, size_t s)
	{
		create(s);
		cudaSafeCall(cudaMemcpy(data_, v, s, cudaMemcpyHostToDevice));
	}
	void DeviceMemory::download(void *v)
	{
		cudaSafeCall(cudaMemcpy(v, data_, size_, cudaMemcpyDeviceToHost));
	}
	size_t DeviceMemory::memorySize() { return size_; }
	bool DeviceMemory::empty() { return !data_; }
}