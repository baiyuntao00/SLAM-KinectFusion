#pragma once
#include <device_memory.hpp>
#include <device_types.hpp>
namespace kf
{
    template <class T>
    class DeviceArray : public DeviceMemory
    {
    public:
        enum
        {
            elem_size = sizeof(T)
        };
        DeviceArray();

        DeviceArray(size_t num);

        DeviceArray(T *p, size_t num);

        DeviceArray &operator=(const DeviceArray &other);

        void create(size_t num);

        void release();

        void download(T *p);

        void upload(const T *p, size_t num);

        T *ptr();
        const T *ptr() const;

        operator T *();
        operator const T *() const;

        size_t size();
    };
    template <class T>
    inline DeviceArray<T>::DeviceArray(){};
    template <class T>
    inline DeviceArray<T>::DeviceArray(size_t num) { kf::DeviceMemory::create(num * elem_size); };
    template <class T>
    inline DeviceArray<T>::DeviceArray(T *v, size_t num) : DeviceMemory(v, num * elem_size) {}
    template <class T>
    inline DeviceArray<T> &DeviceArray<T>::operator=(const DeviceArray &other)
    {
        DeviceMemory::operator=(other);
        return *this;
    };
    template <class T>
    inline T *DeviceArray<T>::ptr() { return kf::DeviceMemory::ptr<T>(); };
    template <class T>
    inline const T *DeviceArray<T>::ptr() const { return kf::DeviceMemory::ptr<T>(); };
    template <class T>
    inline void DeviceArray<T>::create(size_t num) { kf::DeviceMemory::create(num * elem_size); };
    template <class T>
    inline void DeviceArray<T>::release() { kf::DeviceMemory::release(); };
    template <class T>
    inline void DeviceArray<T>::download(T *v) { kf::DeviceMemory::download(v); };
    template <class T>
    inline void DeviceArray<T>::upload(const T *v, size_t num) { kf::DeviceMemory::upload(v, num * elem_size); };
    template <class T>
    inline size_t DeviceArray<T>::size() { return kf::DeviceMemory::memorySize() / elem_size; }
}