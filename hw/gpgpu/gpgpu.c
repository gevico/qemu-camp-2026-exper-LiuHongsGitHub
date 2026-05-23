/*
 * QEMU Educational GPGPU Device
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/units.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "qapi/error.h"
#include "hw/pci/pci.h"
#include "hw/pci/msi.h"
#include "hw/pci/msix.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"

#include "gpgpu.h"
#include "gpgpu_core.h"

static void gpgpu_reset(DeviceState *dev)
{
    GPGPUState *s = GPGPU(dev);

    s->global_ctrl = 0;
    s->global_status = GPGPU_STATUS_READY;
    s->error_status = 0;
    s->irq_enable = 0;
    s->irq_status = 0;
    memset(&s->kernel, 0, sizeof(s->kernel));
    memset(&s->dma, 0, sizeof(s->dma));
    memset(&s->simt, 0, sizeof(s->simt));
    timer_del(s->dma_timer);
    if (s->vram_ptr) {
        memset(s->vram_ptr, 0, s->vram_size);
    }
    pci_set_irq(PCI_DEVICE(s), 0);
}

/* TODO: Implement MMIO control register read */
static uint64_t gpgpu_ctrl_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = GPGPU(opaque);
    switch (addr)
    {
    case GPGPU_REG_DEV_ID:
        return GPGPU_DEV_ID_VALUE;
    case GPGPU_REG_DEV_VERSION:
        return GPGPU_DEV_VERSION_VALUE;
    case GPGPU_REG_DEV_CAPS:
        return 0;
    case GPGPU_REG_VRAM_SIZE_LO:
        return (uint32_t)(s->vram_size & 0xFFFFFFFF);
    case GPGPU_REG_VRAM_SIZE_HI:
        return (uint32_t)(s->vram_size >> 32);

    //global control/status registers
    case GPGPU_REG_GLOBAL_CTRL:
        return s->global_ctrl;
    case GPGPU_REG_GLOBAL_STATUS:
        return s->global_status;
    case GPGPU_REG_ERROR_STATUS:
        return s->error_status;

    //interrupt registers
    case GPGPU_REG_IRQ_ENABLE:
        return s->irq_enable;
    case GPGPU_REG_IRQ_STATUS:
        return s->irq_status;
    case GPGPU_REG_IRQ_ACK:
        return 0;
    /* ======== 内核分发 ======== */
    case GPGPU_REG_KERNEL_ADDR_LO:
        return (uint32_t)(s->kernel.kernel_addr & 0xFFFFFFFF);
    case GPGPU_REG_KERNEL_ADDR_HI:
        return (uint32_t)(s->kernel.kernel_addr >> 32);
    case GPGPU_REG_KERNEL_ARGS_LO:
        return (uint32_t)(s->kernel.kernel_args & 0xFFFFFFFF);
    case GPGPU_REG_KERNEL_ARGS_HI:
        return (uint32_t)(s->kernel.kernel_args >> 32);
    case GPGPU_REG_GRID_DIM_X:
        return s->kernel.grid_dim[0];
    case GPGPU_REG_GRID_DIM_Y:
        return s->kernel.grid_dim[1];
    case GPGPU_REG_GRID_DIM_Z:
        return s->kernel.grid_dim[2];
    case GPGPU_REG_BLOCK_DIM_X:
        return s->kernel.block_dim[0];
    case GPGPU_REG_BLOCK_DIM_Y:
        return s->kernel.block_dim[1];
    case GPGPU_REG_BLOCK_DIM_Z:
        return s->kernel.block_dim[2];
    case GPGPU_REG_SHARED_MEM_SIZE:
        return s->kernel.shared_mem_size;
    case GPGPU_REG_DISPATCH:
        return 0;  /* Write-only */

    /* ======== DMA ======== */
    case GPGPU_REG_DMA_SRC_LO:
        return (uint32_t)(s->dma.src_addr & 0xFFFFFFFF);
    case GPGPU_REG_DMA_SRC_HI:
        return (uint32_t)(s->dma.src_addr >> 32);
    case GPGPU_REG_DMA_DST_LO:
        return (uint32_t)(s->dma.dst_addr & 0xFFFFFFFF);
    case GPGPU_REG_DMA_DST_HI:
        return (uint32_t)(s->dma.dst_addr >> 32);
    case GPGPU_REG_DMA_SIZE:
        return s->dma.size;
    case GPGPU_REG_DMA_CTRL:
        return s->dma.ctrl;
    case GPGPU_REG_DMA_STATUS:
        return s->dma.status;

    /* ======== SIMT 线程上下文 (CTRL 设备) ======== */
    case GPGPU_REG_THREAD_ID_X:
        return s->simt.thread_id[0];
    case GPGPU_REG_THREAD_ID_Y:
        return s->simt.thread_id[1];
    case GPGPU_REG_THREAD_ID_Z:
        return s->simt.thread_id[2];
    case GPGPU_REG_BLOCK_ID_X:
        return s->simt.block_id[0];
    case GPGPU_REG_BLOCK_ID_Y:
        return s->simt.block_id[1];
    case GPGPU_REG_BLOCK_ID_Z:
        return s->simt.block_id[2];
    case GPGPU_REG_WARP_ID:
        return s->simt.warp_id;
    case GPGPU_REG_LANE_ID:
        return s->simt.lane_id;

    /* ======== 同步 ======== */
    case GPGPU_REG_BARRIER:
        return 0;  /* Write-only */
    case GPGPU_REG_THREAD_MASK:
        return s->simt.thread_mask;

    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "GPGPU: read to unimplemented register 0x%lx\n",
                      addr);
        return 0;
    }
}

static void gpgpu_dispatch_kernel(GPGPUState *s) {
    if (!(s->global_ctrl & GPGPU_CTRL_ENABLE) || (s->global_status & GPGPU_STATUS_BUSY)){
        s->error_status |= GPGPU_ERR_INVALID_CMD;
        return;
    }
    s->global_status |= GPGPU_STATUS_BUSY;
    gpgpu_core_exec_kernel(s);
    printf("[POST-KERNEL] vram_ptr at 0x80000:");
    for (int i = 0; i < 8; i++) {
        uint32_t val;
        memcpy(&val, s->vram_ptr + 0x80000 + i*4, 4);
        printf(" %08x", val);
    }
    printf("\n");
    s->global_status &= ~GPGPU_STATUS_BUSY;
    s->global_status |= GPGPU_STATUS_READY;
    s->irq_status |= GPGPU_IRQ_KERNEL_DONE;
    if (s->irq_enable & GPGPU_IRQ_KERNEL_DONE) {
        if (msix_enabled(PCI_DEVICE(s))) {
            msix_notify(PCI_DEVICE(s), GPGPU_MSIX_VEC_KERNEL);
        } else {
            pci_set_irq(PCI_DEVICE(s), 1);
        }
    }
    
}

/* TODO: Implement MMIO control register write */
static void gpgpu_ctrl_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = GPGPU(opaque);
    uint32_t offset = addr;
    uint32_t v = (uint32_t) val;
    switch (offset) {
        case GPGPU_REG_GLOBAL_CTRL:
            s->global_ctrl = v;
            if (v & GPGPU_CTRL_RESET) {
                s->global_ctrl &= ~GPGPU_CTRL_RESET;
                gpgpu_reset(DEVICE(s));
            }
            break;
        case GPGPU_REG_ERROR_STATUS:
            s->error_status &= ~v;
            break;
        case GPGPU_REG_IRQ_ENABLE:
            s->irq_enable = v;
            break;
        case GPGPU_REG_IRQ_ACK:
            s->irq_status &= ~v;
            if (s->irq_status == 0) {
                pci_set_irq(PCI_DEVICE(s), 0);
            }
            break;
        /* 内核分发 */
        case GPGPU_REG_KERNEL_ADDR_LO:
            s->kernel.kernel_addr = (s->kernel.kernel_addr & 0xFFFFFFFF00000000ULL) | v;
            break;
        case GPGPU_REG_KERNEL_ADDR_HI:
            s->kernel.kernel_addr = ((uint64_t)v << 32)|(s->kernel.kernel_addr & 0xFFFFFFFFULL);
            break;
        case GPGPU_REG_KERNEL_ARGS_LO:
            s->kernel.kernel_args = (s->kernel.kernel_args & 0xFFFFFFFF00000000ULL) | v;    
            break;
        case GPGPU_REG_KERNEL_ARGS_HI:
            s->kernel.kernel_args = ((uint64_t)v << 32) | (s->kernel.kernel_args & 0xFFFFFFFFULL);
            break;
        case GPGPU_REG_GRID_DIM_X:   s->kernel.grid_dim[0] = v; break;
        case GPGPU_REG_GRID_DIM_Y:   s->kernel.grid_dim[1] = v; break;
        case GPGPU_REG_GRID_DIM_Z:   s->kernel.grid_dim[2] = v; break;
        case GPGPU_REG_BLOCK_DIM_X:  s->kernel.block_dim[0] = v; break;
        case GPGPU_REG_BLOCK_DIM_Y:  s->kernel.block_dim[1] = v; break;
        case GPGPU_REG_BLOCK_DIM_Z:  s->kernel.block_dim[2] = v; break;
        case GPGPU_REG_SHARED_MEM_SIZE:
            s->kernel.shared_mem_size = v;
            break;
        case GPGPU_REG_DISPATCH:
            gpgpu_dispatch_kernel(s);
            break;
        //DMA
        case GPGPU_REG_DMA_SRC_LO:
            s->dma.src_addr = (s->dma.src_addr & 0xFFFFFFFF00000000ULL) | v;
            break;
        case GPGPU_REG_DMA_SRC_HI:
            s->dma.src_addr = ((uint64_t)v << 32) | (s->dma.src_addr & 0xFFFFFFFFULL);
            break;
        case GPGPU_REG_DMA_DST_LO:
            s->dma.dst_addr = (s->dma.dst_addr & 0xFFFFFFFF00000000ULL) | v;
            break;
        case GPGPU_REG_DMA_DST_HI:
            s->dma.dst_addr = ((uint64_t)v << 32) | (s->dma.dst_addr & 0xFFFFFFFFULL);
            break;
        case GPGPU_REG_DMA_SIZE:
            s->dma.size = v;
            break;
        case GPGPU_REG_DMA_CTRL:
            s->dma.ctrl = v;
            if (v&GPGPU_DMA_START) {
                s->dma.status = GPGPU_DMA_BUSY;
                AddressSpace *as = pci_device_iommu_address_space(PCI_DEVICE(s));
                if (s->dma.size > 0 && s->dma.size <= 16*1024*1024){
                    uint8_t *buf = g_malloc(s->dma.size);
                    address_space_read(as, s->dma.src_addr,MEMTXATTRS_UNSPECIFIED, buf, s->dma.size);
                    address_space_write(as, s->dma.dst_addr,MEMTXATTRS_UNSPECIFIED, buf, s->dma.size);
                    g_free(buf);
                } 
                timer_mod(s->dma_timer,qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) + 1);
            }
            break;
        case GPGPU_REG_THREAD_ID_X: s->simt.thread_id[0] = v; break;
        case GPGPU_REG_THREAD_ID_Y: s->simt.thread_id[1] = v; break;
        case GPGPU_REG_THREAD_ID_Z: s->simt.thread_id[2] = v; break;
        case GPGPU_REG_BLOCK_ID_X:  s->simt.block_id[0] = v; break;
        case GPGPU_REG_BLOCK_ID_Y:  s->simt.block_id[1] = v; break;
        case GPGPU_REG_BLOCK_ID_Z:  s->simt.block_id[2] = v; break;
        case GPGPU_REG_WARP_ID:     s->simt.warp_id = v; break;
        case GPGPU_REG_LANE_ID:     s->simt.lane_id = v; break;
        case GPGPU_REG_BARRIER:
            
            break;
        case GPGPU_REG_THREAD_MASK:
            s->simt.thread_mask = v;
            break;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,"GPGPU: write to unimplemented register 0x%lx\n", addr);
            break;
    }



}

static const MemoryRegionOps gpgpu_ctrl_ops = {
    .read = gpgpu_ctrl_read,
    .write = gpgpu_ctrl_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* TODO: Implement VRAM read */
static uint64_t gpgpu_vram_read(void *opaque, hwaddr addr, unsigned size)
{

    GPGPUState *s = GPGPU(opaque);

    uint64_t result = 0;
    if (addr +size<=s->vram_size){
        memcpy(&result,s->vram_ptr+addr,size);
    }
    // ★ 只追踪 C 数组区域 (hC=0x80000, size=1024)
    if (addr >= 0x80000 && addr < 0x80400) {
        printf("[VRAM_READ] addr=0x%lx size=%u result=0x%lx\n",
               (unsigned long)addr, size, (unsigned long)result);
    }
    return result;
}

/* TODO: Implement VRAM write */
static void gpgpu_vram_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
  
    GPGPUState *s = GPGPU(opaque);
    if (addr +size<=s->vram_size){
        memcpy(s->vram_ptr+addr,&val,size);
    }

}

static const MemoryRegionOps gpgpu_vram_ops = {
    .read = gpgpu_vram_read,
    .write = gpgpu_vram_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static uint64_t gpgpu_doorbell_read(void *opaque, hwaddr addr, unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)size;
    return 0;
}

static void gpgpu_doorbell_write(void *opaque, hwaddr addr, uint64_t val,
                                 unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)val;
    (void)size;
}

static const MemoryRegionOps gpgpu_doorbell_ops = {
    .read = gpgpu_doorbell_read,
    .write = gpgpu_doorbell_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* TODO: Implement DMA completion handler */
static void gpgpu_dma_complete(void *opaque)
{
    GPGPUState *s = GPGPU(opaque);
   
    s->dma.status &= ~GPGPU_DMA_BUSY;
    s->dma.status |= GPGPU_DMA_COMPLETE;
    s->dma.ctrl &= ~GPGPU_DMA_START;
    s->irq_status |= GPGPU_IRQ_DMA_DONE;
    
    /* 触发中断：优先 MSI-X，否则用 INTx */
     if (s->irq_enable & GPGPU_IRQ_DMA_DONE) {
        if (msix_enabled(PCI_DEVICE(s))) {
            msix_notify(PCI_DEVICE(s), GPGPU_MSIX_VEC_DMA);
        } else {
            pci_set_irq(PCI_DEVICE(s), 1);
        }
    }
}

/* TODO: Implement kernel completion handler */

static void gpgpu_realize(PCIDevice *pdev, Error **errp)
{
    GPGPUState *s = GPGPU(pdev);
    uint8_t *pci_conf = pdev->config;

    pci_config_set_interrupt_pin(pci_conf, 1);

    s->vram_ptr = g_malloc0(s->vram_size);
    if (!s->vram_ptr) {
        error_setg(errp, "GPGPU: failed to allocate VRAM");
        return;
    }

    /* BAR 0: control registers */
    memory_region_init_io(&s->ctrl_mmio, OBJECT(s), &gpgpu_ctrl_ops, s,
                          "gpgpu-ctrl", GPGPU_CTRL_BAR_SIZE);
    pci_register_bar(pdev, 0,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64,
                     &s->ctrl_mmio);

    /* BAR 2: VRAM */
    memory_region_init_io(&s->vram, OBJECT(s), &gpgpu_vram_ops, s,
                          "gpgpu-vram", s->vram_size);
    pci_register_bar(pdev, 2,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64 |
                     PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &s->vram);

    /* BAR 4: doorbell registers */
    memory_region_init_io(&s->doorbell_mmio, OBJECT(s), &gpgpu_doorbell_ops, s,
                          "gpgpu-doorbell", GPGPU_DOORBELL_BAR_SIZE);
    pci_register_bar(pdev, 4,
                     PCI_BASE_ADDRESS_SPACE_MEMORY,
                     &s->doorbell_mmio);

    if (msix_init(pdev, GPGPU_MSIX_VECTORS,
                  &s->ctrl_mmio, 0,0xFE000,
                  &s->ctrl_mmio, 0,0xFF000,
                  0, errp)) {
        g_free(s->vram_ptr);
        return;
    }

    // msi_init(pdev, 0, 1, true, false, errp);

    s->dma_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL, gpgpu_dma_complete, s);

    s->global_status = GPGPU_STATUS_READY;
}

static void gpgpu_exit(PCIDevice *pdev)
{
    GPGPUState *s = GPGPU(pdev);

    timer_free(s->dma_timer);
    g_free(s->vram_ptr);
    msix_uninit(pdev, &s->ctrl_mmio, &s->ctrl_mmio);
    msi_uninit(pdev);
}



static const Property gpgpu_properties[] = {
    DEFINE_PROP_UINT32("num_cus", GPGPUState, num_cus,
                       GPGPU_DEFAULT_NUM_CUS),
    DEFINE_PROP_UINT32("warps_per_cu", GPGPUState, warps_per_cu,
                       GPGPU_DEFAULT_WARPS_PER_CU),
    DEFINE_PROP_UINT32("warp_size", GPGPUState, warp_size,
                       GPGPU_DEFAULT_WARP_SIZE),
    DEFINE_PROP_UINT64("vram_size", GPGPUState, vram_size,
                       GPGPU_DEFAULT_VRAM_SIZE),
};

static const VMStateDescription vmstate_gpgpu = {
    .name = "gpgpu",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, GPGPUState),
        VMSTATE_UINT32(global_ctrl, GPGPUState),
        VMSTATE_UINT32(global_status, GPGPUState),
        VMSTATE_UINT32(error_status, GPGPUState),
        VMSTATE_UINT32(irq_enable, GPGPUState),
        VMSTATE_UINT32(irq_status, GPGPUState),
        VMSTATE_END_OF_LIST()
    }
};

static void gpgpu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);

    pc->realize = gpgpu_realize;
    pc->exit = gpgpu_exit;
    pc->vendor_id = GPGPU_VENDOR_ID;
    pc->device_id = GPGPU_DEVICE_ID;
    pc->revision = GPGPU_REVISION;
    pc->class_id = GPGPU_CLASS_CODE;

    device_class_set_legacy_reset(dc, gpgpu_reset);
    dc->desc = "Educational GPGPU Device";
    dc->vmsd = &vmstate_gpgpu;
    device_class_set_props(dc, gpgpu_properties);
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static const TypeInfo gpgpu_type_info = {
    .name          = TYPE_GPGPU,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(GPGPUState),
    .class_init    = gpgpu_class_init,
    .interfaces    = (InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { }
    },
};

static void gpgpu_register_types(void)
{
    type_register_static(&gpgpu_type_info);
}

type_init(gpgpu_register_types)
