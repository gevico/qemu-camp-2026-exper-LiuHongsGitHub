#ifndef _GPGPU_H
#define _GPGPU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 维度结构体 */
typedef struct {
    uint32_t x, y, z;
} dim3;

/* 内存句柄 */
typedef uint32_t gpgpuMemHandle;

/* 初始化/销毁 */
int gpgpuInit(void);
void gpgpuShutdown(void);

/* 内存管理 */
gpgpuMemHandle gpgpuMalloc(size_t size);
void gpgpuFree(gpgpuMemHandle handle);
int gpgpuMemcpyH2D(gpgpuMemHandle devPtr, const void *hostPtr, size_t size);
int gpgpuMemcpyD2H(void *hostPtr, gpgpuMemHandle devPtr, size_t size);

/* 直接从主机内存启动 kernel (内部处理 H2D 拷贝) */
int gpgpuLaunchKernel(gpgpuMemHandle kernelCode, 
                      gpgpuMemHandle kernelArgs,
                      dim3 gridDim, dim3 blockDim,
                      size_t sharedMemSize);

/* 同步 */
int gpgpuSync(void);

/* 直接访问 VRAM (通过 mmap) */
void *gpgpuGetVRAMPtr(gpgpuMemHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* _GPGPU_H */
