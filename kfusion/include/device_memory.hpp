#pragma once
#include <opencv2/opencv.hpp>
namespace kf
{
    class DeviceMemory
    {
    public:
        DeviceMemory() : size_(0), data_(0), refcount_(0){};
        DeviceMemory::DeviceMemory(void *v, size_t s) : data_(v), size_(s), refcount_(0){}

        ~DeviceMemory();
        DeviceMemory &operator=(const DeviceMemory &other_);

        void create(size_t s);
        void release();

        void upload(const void *v, size_t s);
        void download(void *v);

        size_t memorySize();
        bool empty();

        template <class T>
        T *ptr(); 

        template <class T>
        const T *ptr() const;

        template <class U>
        operator cv::cuda::PtrSz<U>() const;

    private:
        size_t size_;
        void *data_;
        int *refcount_;
    };
    template <class T> inline T *kf::DeviceMemory::ptr() { return (T *)data_; }
    template <class T> inline const T *kf::DeviceMemory::ptr() const { return (const T *)data_; }
    template <class U> inline kf::DeviceMemory::operator cv::cuda::PtrSz<U>() const
    {
        cv::cuda::PtrSz<U> result;
        result.data = (U *)ptr<U>();
        result.size = size_ / sizeof(U);
        return result;
    }

}