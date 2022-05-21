#pragma once
#include <device_memory.hpp>
#include <device_types.hpp>
namespace kf
{
    template <class T>
    class CudaArray : public DeviceMemory
    {
    public:
        enum
        {
            ele_len = sizeof(T)
        };
        CudaArray();

        CudaArray(int num);

        CudaArray(T *ptr, int num);

        CudaArray &operator=(const CudaArray &other);

        void create(int num);

        void release();

        void deviceTohost(T *data);

        void hostTodevice(const T *data, int num);

        T* data();

    };
    template <class T> inline CudaArray<T>::CudaArray(){};
    template <class T> inline CudaArray<T>::CudaArray(int num) { kf::DeviceMemory::create(num * ele_len); };
    template <class T> inline CudaArray<T>& CudaArray<T>::operator=(const CudaArray& other)
    { DeviceMemory::operator=(other); return *this; };
    template <class T> inline T* CudaArray<T>::data() {  return (T *)kf::DeviceMemory::data();  };
    template <class T> inline void CudaArray<T>::create(int num) {  kf::DeviceMemory::create(num * ele_len);  };
    template <class T> inline void CudaArray<T>::release(){  kf::DeviceMemory::release();  };
    template <class T> inline void CudaArray<T>::deviceTohost(T *v){  kf::DeviceMemory::deviceTohost(v);  };
    template <class T> inline void CudaArray<T>::hostTodevice(const T *v, int num){ kf::DeviceMemory::hostTodevice(v, ele_len *num);  };
}