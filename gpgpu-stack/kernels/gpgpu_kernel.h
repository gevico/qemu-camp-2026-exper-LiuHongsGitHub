/*
 * gpgpu_kernel.h - GPGPU Kernel 编程模型定义
 *
 * 本头文件定义了 GPGPU kernel 的编程规范，包括：
 *   - 函数调用约定（入口参数寄存器）
 *   - 内置变量（threadIdx / blockIdx / blockDim / gridDim）
 *   - 线程退出协议
 *   - mhartid CSR 编码格式
 *   - VRAM 地址空间布局
 *
 * 用法：
 *   1. 手写 RV32IMF 汇编 .S 文件时参考本文件的寄存器约定
 *   2. 也可以在 C 文件中 include，使用宏定义计算线程索引
 */
#ifndef GPGPU_KERNEL_H
#define GPGPU_KERNEL_H

/*
 * ============================================================================
 * 1. Kernel 函数签名（__global__ 风格）
 * ============================================================================
 *
 * 每个 GPGPU kernel 是一段 RV32IMF 汇编代码，入口时：
 *
 *   a0 (x10) = kernel_args  — 指向 VRAM 中的参数结构体
 *                           结构体内容由 host 端 gpgpuLaunchKernel() 的
 *                           kernelArgs 参数决定，每个字段为 uint32_t
 *
 *   a1 (x11) = global_thread_id — 当前线程在 block 内的全局线性 ID
 *                           = thread_id_base + lane_index
 *                           = warp_id * 32 + lane_id
 *                           范围: [0, blockDim.x * blockDim.y * blockDim.z - 1]
 *
 * 示例 (vector_add.S):
 *   lw t0, 0(a0)        # t0 = A 基地址
 *   lw t1, 4(a0)        # t1 = B 基地址
 *   lw t2, 8(a0)        # t2 = C 基地址
 *   slli t3, a1, 2      # t3 = thread_id * 4 (float 偏移)
 *   add t0, t0, t3      # A[thread_id] 的地址
 *   ...
 */

 /* 入口参数寄存器 */
#define GPGPU_REG_ARGS          10    /* a0 (x10): kernel args pointer */
#define GPGPU_REG_THREAD_ID     11    /* a1 (x11): global thread id in block */

/* 退出码寄存器 */
#define GPGPU_REG_EXIT_CODE     17    /* a7 (x17): 退出码，1 = 正常退出 */

/* 退出码值 */
#define GPGPU_EXIT_SUCCESS      1     /* 正常退出 */

/*
 * ============================================================================
 * 2. 内置变量 (Built-in Variables)
 * ============================================================================
 *
 * 类 CUDA 的内置变量通过以下方式获取：
 *
 *   threadIdx.x / y / z  — 通过 mhartid CSR 解码，或通过 a1 计算
 *   blockIdx.x / y / z   — 通过 mhartid CSR 解码，或通过 CTRL 寄存器
 *   blockDim.x / y / z   — 通过 CTRL 寄存器读取
 *   gridDim.x / y / z    — 通过 CTRL 寄存器读取
 *
 * ┌──────────────────────────────────────────────────────────┐
 * │  方法 1 (推荐): 使用 mhartid CSR                         │
 * │                                                          │
 * │  csrr t0, mhartid     # 读 CSR 0xF14                     │
 * │  andi t1, t0, 0x1F    # threadIdx.x = t0 & 0x1F         │
 * │  srli t0, t0, 5                                             │
 * │  andi t2, t0, 0xFF   # warpId = (t0 >> 5) & 0xFF       │
 * │  srli t0, t0, 8                                             │
 * │  andi t3, t0, 0x7FFFF # blockIdx_linear = t0 >> 13      │
 * └──────────────────────────────────────────────────────────┘
 *
 * ┌──────────────────────────────────────────────────────────┐
 * │  方法 2: 使用 a1 (简化)                                   │
 * │                                                          │
 * │  a1 已包含全局线程 ID = warp_id * 32 + lane_id           │
 * │  适用于一维 block (blockDim.y == blockDim.z == 1)        │
 * │                                                          │
 * │  slli t0, a1, 2      # 偏移 = thread_id * 4             │
 * └──────────────────────────────────────────────────────────┘
 *
 * ┌──────────────────────────────────────────────────────────┐
 * │  方法 3: CTRL 寄存器 (内存映射)                           │
 * │                                                          │
 * │  注意: CTRL 寄存器当前仅支持指令取指，                    │
 * │  LW 暂不可用。未来版本将支持。                            │
 * │                                                          │
 * │  li t0, 0x80000000                                        │
 * │  lw t1, 0x00(t0)     # threadIdx.x                      │
 * │  lw t2, 0x10(t0)     # blockIdx.x                        │
 * │  lw t3, 0x20(t0)     # blockDim.x                       │
 * └──────────────────────────────────────────────────────────┘
 */
/* mhartid CSR 编码格式 */
#define GPGPU_CSR_MHARTID      0xF14

#define MHARTID_LANE_BITS      5
#define MHARTID_WARP_BITS      8
#define MHARTID_BLOCK_BITS     19

/* mhartid 位域提取宏 (C 语言) */
#define GPGPU_LANE_ID(mhartid)    ((mhartid) & 0x1F)
#define GPGPU_WARP_ID(mhartid)    (((mhartid) >> 5) & 0xFF)
#define GPGPU_BLOCK_ID(mhartid)   ((mhartid) >> 13)

/* CTRL 寄存器地址 (GPU 核心视角) */
#define GPGPU_CTRL_BASE           0x80000000
#define GPGPU_CTRL_THREAD_ID_X    (GPGPU_CTRL_BASE + 0x00)
#define GPGPU_CTRL_THREAD_ID_Y    (GPGPU_CTRL_BASE + 0x04)
#define GPGPU_CTRL_THREAD_ID_Z    (GPGPU_CTRL_BASE + 0x08)
#define GPGPU_CTRL_BLOCK_ID_X     (GPGPU_CTRL_BASE + 0x10)
#define GPGPU_CTRL_BLOCK_ID_Y     (GPGPU_CTRL_BASE + 0x14)
#define GPGPU_CTRL_BLOCK_ID_Z     (GPGPU_CTRL_BASE + 0x18)
#define GPGPU_CTRL_BLOCK_DIM_X    (GPGPU_CTRL_BASE + 0x20)
#define GPGPU_CTRL_BLOCK_DIM_Y    (GPGPU_CTRL_BASE + 0x24)
#define GPGPU_CTRL_BLOCK_DIM_Z    (GPGPU_CTRL_BASE + 0x28)
#define GPGPU_CTRL_GRID_DIM_X     (GPGPU_CTRL_BASE + 0x30)
#define GPGPU_CTRL_GRID_DIM_Y     (GPGPU_CTRL_BASE + 0x34)
#define GPGPU_CTRL_GRID_DIM_Z     (GPGPU_CTRL_BASE + 0x38)


/*
 * ============================================================================
 * 3. Grid / Block 维度约定
 * ============================================================================
 *
 *   gridDim  = (grid_dim_x, grid_dim_y, grid_dim_z)  — 由 host 端设定
 *   blockDim = (block_dim_x, block_dim_y, block_dim_z) — 由 host 端设定
 *
 *   threads_per_block = blockDim.x * blockDim.y * blockDim.z
 *   warps_per_block   = ceil(threads_per_block / 32)
 *
 *   全局线程 ID (跨 grid):
 *     global_id = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x
 *                 + blockIdx.x) * threads_per_block + local_id
 *     其中 local_id = a1 (block 内线性线程 ID)
 *
 *   Warp 分配:
 *     warp_id  = a1 / 32
 *     lane_id  = a1 % 32
 */
/* Warp 大小 */
#define GPGPU_WARP_SIZE         32


#endif // GPGPU_KERNEL_H