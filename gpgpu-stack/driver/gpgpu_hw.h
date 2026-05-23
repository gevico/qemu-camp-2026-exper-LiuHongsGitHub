/* SPDX-License-Identifier: GPL-2.0 */
/*
 * GPGPU Hardware Register Definitions
 *
 * These must match the QEMU device definitions in hw/gpgpu/gpgpu.h
 * Vendor ID: 0x1234  Device ID: 0x1337
 */

#ifndef _GPGPU_HW_H
#define _GPGPU_HW_H

/* PCI Device IDs */
#define GPGPU_VENDOR_ID         0x1234
#define GPGPU_DEVICE_ID         0x1337

/* BAR sizes */
#define GPGPU_CTRL_BAR_SIZE     (1UL * 1024 * 1024)   /* 1 MB */
#define GPGPU_VRAM_BAR_SIZE     (64UL * 1024 * 1024)  /* 64 MB */

/* BAR index */
#define GPGPU_BAR_CTRL          0   /* Control registers */
#define GPGPU_BAR_VRAM          2   /* Video RAM */

/* ===== Device Info Registers (RO) ===== */
#define GPGPU_REG_DEV_ID            0x0000
#define GPGPU_REG_DEV_VERSION       0x0004
#define GPGPU_REG_DEV_CAPS          0x0008
#define GPGPU_REG_VRAM_SIZE_LO      0x000C
#define GPGPU_REG_VRAM_SIZE_HI      0x0010

/* ===== Global Control Registers ===== */
#define GPGPU_REG_GLOBAL_CTRL       0x0100
#define GPGPU_REG_GLOBAL_STATUS     0x0104
#define GPGPU_REG_ERROR_STATUS      0x0108

/* GLOBAL_CTRL bits */
#define GPGPU_CTRL_ENABLE           (1 << 0)
#define GPGPU_CTRL_RESET            (1 << 1)

/* GLOBAL_STATUS bits */
#define GPGPU_STATUS_READY          (1 << 0)
#define GPGPU_STATUS_BUSY           (1 << 1)
#define GPGPU_STATUS_ERROR          (1 << 2)

/* ERROR_STATUS bits (write-1-to-clear) */
#define GPGPU_ERR_INVALID_CMD       (1 << 0)
#define GPGPU_ERR_VRAM_FAULT        (1 << 1)
#define GPGPU_ERR_KERNEL_FAULT      (1 << 2)
#define GPGPU_ERR_DMA_FAULT         (1 << 3)

/* ===== Interrupt Registers ===== */
#define GPGPU_REG_IRQ_ENABLE        0x0200
#define GPGPU_REG_IRQ_STATUS        0x0204
#define GPGPU_REG_IRQ_ACK           0x0208

/* IRQ bits */
#define GPGPU_IRQ_KERNEL_DONE       (1 << 0)
#define GPGPU_IRQ_DMA_DONE          (1 << 1)
#define GPGPU_IRQ_ERROR             (1 << 2)

/* ===== Kernel Dispatch Registers ===== */
#define GPGPU_REG_KERNEL_ADDR_LO    0x0300
#define GPGPU_REG_KERNEL_ADDR_HI    0x0304
#define GPGPU_REG_KERNEL_ARGS_LO    0x0308
#define GPGPU_REG_KERNEL_ARGS_HI    0x030C
#define GPGPU_REG_GRID_DIM_X        0x0310
#define GPGPU_REG_GRID_DIM_Y        0x0314
#define GPGPU_REG_GRID_DIM_Z        0x0318
#define GPGPU_REG_BLOCK_DIM_X       0x031C
#define GPGPU_REG_BLOCK_DIM_Y       0x0320
#define GPGPU_REG_BLOCK_DIM_Z       0x0324
#define GPGPU_REG_SHARED_MEM_SIZE   0x0328
#define GPGPU_REG_DISPATCH          0x0330

/* ===== DMA Engine Registers ===== */
#define GPGPU_REG_DMA_SRC_LO        0x0400
#define GPGPU_REG_DMA_SRC_HI        0x0404
#define GPGPU_REG_DMA_DST_LO        0x0408
#define GPGPU_REG_DMA_DST_HI        0x040C
#define GPGPU_REG_DMA_SIZE          0x0410
#define GPGPU_REG_DMA_CTRL          0x0414
#define GPGPU_REG_DMA_STATUS        0x0418

/* DMA_CTRL bits */
#define GPGPU_DMA_START             (1 << 0)
#define GPGPU_DMA_DIR_TO_VRAM       (0 << 1)
#define GPGPU_DMA_DIR_FROM_VRAM     (1 << 1)
#define GPGPU_DMA_IRQ_ENABLE        (1 << 2)

/* DMA_STATUS bits */
#define GPGPU_DMA_BUSY              (1 << 0)
#define GPGPU_DMA_COMPLETE          (1 << 1)
#define GPGPU_DMA_ERROR             (1 << 2)

/* ===== MSI-X Vectors ===== */
#define GPGPU_MSIX_VEC_KERNEL       0
#define GPGPU_MSIX_VEC_DMA          1
#define GPGPU_MSIX_VEC_ERROR        2
#define GPGPU_MSIX_VECTORS          4

#endif /* _GPGPU_HW_H */
