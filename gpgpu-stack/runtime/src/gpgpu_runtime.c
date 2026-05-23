/* /home/lh/code/gpgpu-stack/libgpgpu/src/gpgpu_runtime.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

#include "../include/gpgpu.h"

/* 从内核驱动复制的 ioctl 定义 */
#define GPGPU_IOC_MAGIC     'G'

struct gpgpu_dma_params {
    uint64_t host_addr;
    uint64_t dev_addr;
    uint32_t size;
};

struct gpgpu_launch_params {
    uint64_t kernel_addr;
    uint64_t args_addr;
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint32_t shared_mem_size;
};

#define GPGPU_IOC_ALLOC      _IOWR(GPGPU_IOC_MAGIC, 0x00, uint32_t)
#define GPGPU_IOC_FREE       _IOW(GPGPU_IOC_MAGIC, 0x01, uint32_t)
#define GPGPU_IOC_H2D        _IOW(GPGPU_IOC_MAGIC, 0x10, struct gpgpu_dma_params)
#define GPGPU_IOC_D2H        _IOW(GPGPU_IOC_MAGIC, 0x11, struct gpgpu_dma_params)
#define GPGPU_IOC_LAUNCH     _IOW(GPGPU_IOC_MAGIC, 0x20, struct gpgpu_launch_params)
#define GPGPU_IOC_SYNC       _IO(GPGPU_IOC_MAGIC, 0x30)

/* 内部状态 */
static int g_gpgpu_fd = -1;
static void *g_vram_base = NULL;
static size_t g_vram_size = 64 * 1024 * 1024; /* 64MB */

int gpgpuInit(void)
{
    g_gpgpu_fd = open("/dev/gpgpu", O_RDWR);
    if (g_gpgpu_fd < 0) {
        fprintf(stderr, "Failed to open /dev/gpgpu: %s\n", strerror(errno));
        return -1;
    }

    /* mmap VRAM for direct access */
    g_vram_base = mmap(NULL, g_vram_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                       g_gpgpu_fd, 0);
    if (g_vram_base == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap VRAM: %s\n", strerror(errno));
        close(g_gpgpu_fd);
        g_gpgpu_fd = -1;
        return -1;
    }

    return 0;
}

void gpgpuShutdown(void)
{
    if (g_vram_base && g_vram_base != MAP_FAILED) {
        munmap(g_vram_base, g_vram_size);
        g_vram_base = NULL;
    }
    if (g_gpgpu_fd >= 0) {
        close(g_gpgpu_fd);
        g_gpgpu_fd = -1;
    }
}

gpgpuMemHandle gpgpuMalloc(size_t size)
{
    if (g_gpgpu_fd < 0) {
        return (gpgpuMemHandle)-1;
    }
    
    uint32_t alloc_size = (uint32_t)size;
    if (ioctl(g_gpgpu_fd, GPGPU_IOC_ALLOC, &alloc_size) < 0) {
        return (gpgpuMemHandle)-1;
    }
    return alloc_size; /* offset */
}

void gpgpuFree(gpgpuMemHandle handle)
{
    ioctl(g_gpgpu_fd, GPGPU_IOC_FREE, &handle);
}

int gpgpuMemcpyH2D(gpgpuMemHandle devPtr, const void *hostPtr, size_t size)
{
    struct gpgpu_dma_params dma;
    dma.host_addr = (uint64_t)hostPtr;
    dma.dev_addr = (uint64_t)devPtr;
    dma.size = (uint32_t)size;
    return ioctl(g_gpgpu_fd, GPGPU_IOC_H2D, &dma);
}

int gpgpuMemcpyD2H(void *hostPtr, gpgpuMemHandle devPtr, size_t size)
{
    struct gpgpu_dma_params dma;
    dma.host_addr = (uint64_t)hostPtr;
    dma.dev_addr = (uint64_t)devPtr;
    dma.size = (uint32_t)size;
    return ioctl(g_gpgpu_fd, GPGPU_IOC_D2H, &dma);
}

int gpgpuLaunchKernel(gpgpuMemHandle kernelCode,
                      gpgpuMemHandle kernelArgs,
                      dim3 gridDim, dim3 blockDim,
                      size_t sharedMemSize)
{
    struct gpgpu_launch_params launch;
    launch.kernel_addr =  (uint64_t)kernelCode;
    launch.args_addr =  (uint64_t)kernelArgs;
    launch.grid_dim[0] = gridDim.x;
    launch.grid_dim[1] = gridDim.y;
    launch.grid_dim[2] = gridDim.z;
    launch.block_dim[0] = blockDim.x;
    launch.block_dim[1] = blockDim.y;
    launch.block_dim[2] = blockDim.z;
    launch.shared_mem_size = (uint32_t)sharedMemSize;
    
    return ioctl(g_gpgpu_fd, GPGPU_IOC_LAUNCH, &launch);
}

int gpgpuSync(void)
{
    return ioctl(g_gpgpu_fd, GPGPU_IOC_SYNC);
}

void *gpgpuGetVRAMPtr(gpgpuMemHandle handle)
{
    if (!g_vram_base) return NULL;
    return (uint8_t*)g_vram_base + handle;
}
