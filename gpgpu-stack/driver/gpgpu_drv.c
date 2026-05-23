// SPDX-License-Identifier: GPL-2.0
/*
 * QEMU Educational GPGPU - Linux Kernel Driver
 *
 * A Linux kernel driver for the virtual GPGPU PCI device.
 * Provides char device interface (/dev/gpgpu) with ioctl and mmap support.
 *
 * Device: Vendor=0x1234, Device=0x1337
 * BAR0: Control MMIO registers (1MB)
 * BAR2: VRAM (64MB, mmap to userspace)
 * Interrupts: MSI-X (kernel done, DMA done, error)
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/pci.h>
#include <linux/msi.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <linux/mutex.h>
#include <linux/io.h>

#include "gpgpu_hw.h"

#define DRIVER_NAME    "gpgpu"
#define DEVICE_NAME    "gpgpu"
#define CLASS_NAME     "gpgpu"

/* IOCTL magic number and commands */
#define GPGPU_IOC_MAGIC     'G'

/* Memory allocation/free in VRAM */
#define GPGPU_IOC_ALLOC      _IOWR(GPGPU_IOC_MAGIC, 0x00, __u32)
#define GPGPU_IOC_FREE       _IOW(GPGPU_IOC_MAGIC, 0x01, __u32)

/* DMA operations: Host <-> Device (VRAM) */
struct gpgpu_dma_params {
    __u64 host_addr;    /* userspace virtual address */
    __u64 dev_addr;     /* offset in VRAM */
    __u32 size;
};
#define GPGPU_IOC_H2D       _IOW(GPGPU_IOC_MAGIC, 0x10, struct gpgpu_dma_params)
#define GPGPU_IOC_D2H       _IOW(GPGPU_IOC_MAGIC, 0x11, struct gpgpu_dma_params)

/* Kernel launch parameters */
struct gpgpu_launch_params {
    __u64 kernel_addr;      /* kernel code offset in VRAM */
    __u64 args_addr;        /* kernel args offset in VRAM */
    __u32 grid_dim[3];
    __u32 block_dim[3];
    __u32 shared_mem_size;
};
#define GPGPU_IOC_LAUNCH     _IOW(GPGPU_IOC_MAGIC, 0x20, struct gpgpu_launch_params)

/* Status query / synchronization */
#define GPGPU_IOC_SYNC       _IO(GPGPU_IOC_MAGIC, 0x30)

/* VRAM memory allocator simple bitmap-based */
#define GPGPU_MAX_ALLOCATIONS  256
struct vram_allocation {
    __u32 offset;    /* byte offset in VRAM */
    __u32 size;      /* size in bytes */
    bool used;
};

enum gpgpu_intr_type {
    GPGPU_INTR_NONE = 0,
    GPGPU_INTR_MSIX,
    GPGPU_INTR_INTX,
};

/* Per-device driver state */
struct gpgpu_device {
    /* PCI device info */
    struct pci_dev *pdev;

    /* MMIO mappings */
    void __iomem *ctrl_regs;       /* BAR0: control registers */
    void __iomem *vram;             /* BAR2: VRAM */
    resource_size_t vram_phys;      /* physical addr for mmap */
    resource_size_t vram_len;       /* VRAM length */

    /* Char device */
    dev_t devno;
    struct cdev cdev;
    struct device *dev;
    struct class *class;

    /* MSI-X interrupt vectors */
    int irq_kernel;
    int irq_dma;
    int irq_error;
    enum gpgpu_intr_type intr_type;
    /* Synchronization for kernel completion wait */
    struct mutex lock;
    wait_queue_head_t kernel_waitq;
    atomic_t kernel_done;
    atomic_t dma_done;

    /* VRAM allocator */
    struct vram_allocation alloc_table[GPGPU_MAX_ALLOCATIONS];
    struct mutex alloc_lock;
};

static int gpgpu_major;
static struct gpgpu_device *gpgpu_global_dev;

/* ============================================================
 *  Low-level register access helpers
 * ============================================================ */
static inline u32 gpgpu_read(struct gpgpu_device *gdev, u32 reg)
{
    return ioread32(gdev->ctrl_regs + reg);
}

static inline void gpgpu_write(struct gpgpu_device *gdev, u32 reg, u32 val)
{
    iowrite32(val, gdev->ctrl_regs + reg);
}

/* ============================================================
 *  MSI-X Interrupt Handlers
 * ============================================================ */
static irqreturn_t gpgpu_isr_kernel(int irq, void *dev_id)
{
    struct gpgpu_device *gdev = dev_id;

    /* Acknowledge IRQ */
    gpgpu_write(gdev, GPGPU_REG_IRQ_ACK, GPGPU_IRQ_KERNEL_DONE);

    dev_info(&gdev->pdev->dev, "Kernel execution complete\n");

    atomic_set(&gdev->kernel_done, 1);
    wake_up_interruptible(&gdev->kernel_waitq);

    return IRQ_HANDLED;
}

static irqreturn_t gpgpu_isr_dma(int irq, void *dev_id)
{
    struct gpgpu_device *gdev = dev_id;

    gpgpu_write(gdev, GPGPU_REG_IRQ_ACK, GPGPU_IRQ_DMA_DONE);

    atomic_set(&gdev->dma_done, 1);
    wake_up_interruptible(&gdev->kernel_waitq);

    dev_info(&gdev->pdev->dev, "DMA transfer complete\n");
    return IRQ_HANDLED;
}

static irqreturn_t gpgpu_isr_error(int irq, void *dev_id)
{
    struct gpgpu_device *gdev = dev_id;
    u32 err_status = gpgpu_read(gdev, GPGPU_REG_ERROR_STATUS);

    dev_err(&gdev->pdev->dev, "Device error! status=0x%x\n", err_status);

    /* Clear error by writing back */
    gpgpu_write(gdev, GPGPU_REG_ERROR_STATUS, err_status);
    gpgpu_write(gdev, GPGPU_REG_IRQ_ACK, GPGPU_IRQ_ERROR);

    return IRQ_HANDLED;
}
/* ============================================================
 *  INTx Unified Interrupt Handler (used when MSI-X is unavailable)
 * ============================================================ */
static irqreturn_t gpgpu_isr_intx(int irq, void *dev_id)
{
    struct gpgpu_device *gdev = dev_id;
    u32 irq_status;
    irqreturn_t ret = IRQ_NONE;

    /* Read interrupt status register to determine the source */
    irq_status = gpgpu_read(gdev, GPGPU_REG_IRQ_STATUS);
    
    if (irq_status & GPGPU_IRQ_KERNEL_DONE) {
        /* Acknowledge kernel done interrupt */
        gpgpu_write(gdev, GPGPU_REG_IRQ_ACK, GPGPU_IRQ_KERNEL_DONE);
        dev_info(&gdev->pdev->dev, "Kernel execution complete (INTx)\n");
        atomic_set(&gdev->kernel_done, 1);
        wake_up_interruptible(&gdev->kernel_waitq);
        ret = IRQ_HANDLED;
    }
    
    if (irq_status & GPGPU_IRQ_DMA_DONE) {
        /* Acknowledge DMA done interrupt */
        gpgpu_write(gdev, GPGPU_REG_IRQ_ACK, GPGPU_IRQ_DMA_DONE);
        atomic_set(&gdev->dma_done, 1);
        wake_up_interruptible(&gdev->kernel_waitq);
        dev_info(&gdev->pdev->dev, "DMA transfer complete (INTx)\n");
        ret = IRQ_HANDLED;
    }
    
    if (irq_status & GPGPU_IRQ_ERROR) {
        /* Acknowledge error interrupt */
        u32 err_status = gpgpu_read(gdev, GPGPU_REG_ERROR_STATUS);
        dev_err(&gdev->pdev->dev, "Device error! status=0x%x (INTx)\n", err_status);
        gpgpu_write(gdev, GPGPU_REG_ERROR_STATUS, err_status);
        gpgpu_write(gdev, GPGPU_REG_IRQ_ACK, GPGPU_IRQ_ERROR);
        ret = IRQ_HANDLED;
    }
    
    return ret;
}



/* ============================================================
 *  VRAM Simple Allocator (first-fit bitmap)
 * ============================================================ */
static int gpgpu_vram_alloc(struct gpgpu_device *gdev, u32 size, u32 *out_offset)
{
    int i;
    u32 aligned_size = (size + 0xFF) & ~0xFFU;  /* 256-byte align */

    mutex_lock(&gdev->alloc_lock);

    for (i = 0; i < GPGPU_MAX_ALLOCATIONS; i++) {
        if (!gdev->alloc_table[i].used) {
            gdev->alloc_table[i].offset = i * 256 * 1024;  /* 256KB slots */
            if (gdev->alloc_table[i].offset + aligned_size > gdev->vram_len) {
                mutex_unlock(&gdev->alloc_lock);
                return -ENOSPC;
            }
            gdev->alloc_table[i].size = aligned_size;
            gdev->alloc_table[i].used = true;
            *out_offset = gdev->alloc_table[i].offset;
            mutex_unlock(&gdev->alloc_lock);
            return 0;
        }
    }

    mutex_unlock(&gdev->alloc_lock);
    return -ENOMEM;
}

static int gpgpu_vram_free(struct gpgpu_device *gdev, u32 offset)
{
    int i;

    mutex_lock(&gdev->alloc_lock);
    for (i = 0; i < GPGPU_MAX_ALLOCATIONS; i++) {
        if (gdev->alloc_table[i].used && gdev->alloc_table[i].offset == offset) {
            gdev->alloc_table[i].used = false;
            /* Zero out the freed region */
            memset_io(gdev->vram + offset, 0, gdev->alloc_table[i].size);
            mutex_unlock(&gdev->alloc_lock);
            return 0;
        }
    }
    mutex_unlock(&gdev->alloc_lock);
    return -EINVAL;
}

/* ============================================================
 *  Char Device Operations: ioctl
 * ============================================================ */
static long gpgpu_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct gpgpu_device *gdev = filp->private_data;
    int ret = 0;

    mutex_lock(&gdev->lock);

    switch (cmd) {
    case GPGPU_IOC_ALLOC: {
        u32 size, offset;
        if (copy_from_user(&size, (void __user *)arg, sizeof(size))) {
            ret = -EFAULT;
            break;
        }
        ret = gpgpu_vram_alloc(gdev, size, &offset);
        if (!ret && copy_to_user((void __user *)arg, &offset, sizeof(offset)))
            ret = -EFAULT;
        break;
    }

    case GPGPU_IOC_FREE: {
        u32 offset;
        if (copy_from_user(&offset, (void __user *)arg, sizeof(offset))) {
            ret = -EFAULT;
            break;
        }
        ret = gpgpu_vram_free(gdev, offset);
        break;
    }

    case GPGPU_IOC_H2D: {
        struct gpgpu_dma_params dma;
        void __user *ubuf;

        if (copy_from_user(&dma, (void __user *)arg, sizeof(dma))) {
            ret = -EFAULT;
            break;
        }
        ubuf = (void __user *)(uintptr_t)dma.host_addr;

        /* Copy from userspace to VRAM via memcpy_fromio wrapper */
        /* We need to go through kernel buffer since we can't copy_from_user directly to io */
        {
            u8 *tmp = kmalloc(dma.size, GFP_KERNEL);
            if (!tmp) { ret = -ENOMEM; break; }
            if (copy_from_user(tmp, ubuf, dma.size)) {
                kfree(tmp); ret = -EFAULT; break;
            }
            memcpy_toio(gdev->vram + dma.dev_addr, tmp, dma.size);
            kfree(tmp);
        }
        break;
    }

    case GPGPU_IOC_D2H: {
        struct gpgpu_dma_params dma;
        void __user *ubuf;

        if (copy_from_user(&dma, (void __user *)arg, sizeof(dma))) {
            ret = -EFAULT;
            break;
        }
        ubuf = (void __user *)(uintptr_t)dma.host_addr;

        {
            u8 *tmp = kmalloc(dma.size, GFP_KERNEL);
            if (!tmp) { ret = -ENOMEM; break; }
            memcpy_fromio(tmp, gdev->vram + dma.dev_addr, dma.size);
            if (copy_to_user(ubuf, tmp, dma.size)) {
                kfree(tmp); ret = -EFAULT; break;
            }
            kfree(tmp);
        }
        break;
    }

    case GPGPU_IOC_LAUNCH: {
        struct gpgpu_launch_params launch;
        

        if (copy_from_user(&launch, (void __user *)arg, sizeof(launch))) {
            ret = -EFAULT;
            break;
        }

        /* Reset kernel-done flag */
        atomic_set(&gdev->kernel_done, 0);

        /* Write kernel address (lo/hi) */
        gpgpu_write(gdev, GPGPU_REG_KERNEL_ADDR_LO,
                    (u32)(launch.kernel_addr & 0xFFFFFFFF));
        gpgpu_write(gdev, GPGPU_REG_KERNEL_ADDR_HI,
                    (u32)(launch.kernel_addr >> 32));

        /* Write args address (lo/hi) */
        gpgpu_write(gdev, GPGPU_REG_KERNEL_ARGS_LO,
                    (u32)(launch.args_addr & 0xFFFFFFFF));
        gpgpu_write(gdev, GPGPU_REG_KERNEL_ARGS_HI,
                    (u32)(launch.args_addr >> 32));

        /* Grid dimensions */
        gpgpu_write(gdev, GPGPU_REG_GRID_DIM_X, launch.grid_dim[0]);
        gpgpu_write(gdev, GPGPU_REG_GRID_DIM_Y, launch.grid_dim[1]);
        gpgpu_write(gdev, GPGPU_REG_GRID_DIM_Z, launch.grid_dim[2]);

        /* Block dimensions */
        gpgpu_write(gdev, GPGPU_REG_BLOCK_DIM_X, launch.block_dim[0]);
        gpgpu_write(gdev, GPGPU_REG_BLOCK_DIM_Y, launch.block_dim[1]);
        gpgpu_write(gdev, GPGPU_REG_BLOCK_DIM_Z, launch.block_dim[2]);

        /* Shared memory size */
        gpgpu_write(gdev, GPGPU_REG_SHARED_MEM_SIZE, launch.shared_mem_size);

        /* Enable kernel-done interrupt */
        u32 irq_en = gpgpu_read(gdev, GPGPU_REG_IRQ_ENABLE);
        gpgpu_write(gdev, GPGPU_REG_IRQ_ENABLE, irq_en | GPGPU_IRQ_KERNEL_DONE);

        /* DISPATCH! Trigger kernel execution */
        gpgpu_write(gdev, GPGPU_REG_DISPATCH, 1);

        dev_info(&gdev->pdev->dev,
                 "Kernel launched grid=[%u,%u,%u] block=[%u,%u,%u]\n",
                 launch.grid_dim[0], launch.grid_dim[1], launch.grid_dim[2],
                 launch.block_dim[0], launch.block_dim[1], launch.block_dim[2]);
        break;
    }

    case GPGPU_IOC_SYNC:
        /* Wait until kernel execution is complete or signal pending */
        mutex_unlock(&gdev->lock);
        ret = wait_event_interruptible_timeout(
                gdev->kernel_waitq,
                atomic_read(&gdev->kernel_done),
                msecs_to_jiffies(10000));  /* 10 second timeout */
        mutex_lock(&gdev->lock);
        if (ret == 0)
            ret = -ETIMEDOUT;
        else if (ret > 0)
            ret = 0;
        break;

    default:
        dev_err(&gdev->pdev->dev, "Unknown ioctl: 0x%x\n", cmd);
        ret = -ENOTTY;
        break;
    }

    mutex_unlock(&gdev->lock);
    return ret;
}

/* ============================================================
 *  Char Device Operations: mmap (VRAM mapping to user space)
 * ============================================================ */
static int gpgpu_mmap(struct file *filp, struct vm_area_struct *vma)
{
    struct gpgpu_device *gdev = filp->private_data;
    unsigned long size = vma->vm_end - vma->vm_start;
    unsigned long pfn;

    dev_dbg(&gdev->pdev->dev, "mmap: size=%lu, offset=0x%lx\n",
            size, vma->vm_pgoff << PAGE_SHIFT);

    /* Bounds check */
    if (vma->vm_pgoff > (gdev->vram_len >> PAGE_SHIFT)) {
        dev_err(&gdev->pdev->dev, "mmap offset beyond VRAM\n");
        return -EINVAL;
    }

    if (size > gdev->vram_len - (vma->vm_pgoff << PAGE_SHIFT)) {
        dev_err(&gdev->pdev->dev, "mmap size exceeds VRAM\n");
        return -EINVAL;
    }

    if (gdev->vram_phys & ~PAGE_MASK) {
        dev_err(&gdev->pdev->dev, "VRAM physical address not page-aligned\n");
        return -EINVAL;
    }
    pfn = (gdev->vram_phys >> PAGE_SHIFT) + vma->vm_pgoff;


    /* Map VRAM as I/O memory (uncached) */
    if (remap_pfn_range(vma, vma->vm_start, pfn, size,
                        pgprot_noncached(vma->vm_page_prot))) {
        dev_err(&gdev->pdev->dev, "remap_pfn_range failed\n");
        return -EAGAIN;
    }

    return 0;
}

/* ============================================================
 *  Char Device File Operations
 * ============================================================ */
static int gpgpu_open(struct inode *inode, struct file *filp)
{
    struct gpgpu_device *gdev = container_of(inode->i_cdev,
                                              struct gpgpu_device, cdev);
    filp->private_data = gdev;
    return 0;
}

static int gpgpu_release(struct inode *inode, struct file *filp)
{
    filp->private_data = NULL;
    return 0;
}

static const struct file_operations gpgpu_fops = {
    .owner          = THIS_MODULE,
    .open           = gpgpu_open,
    .release        = gpgpu_release,
    .unlocked_ioctl = gpgpu_ioctl,
    .compat_ioctl   = compat_ptr_ioctl,
    .mmap           = gpgpu_mmap,
};

/* ============================================================
 *  PCI Probe - Called when device [1234:1337] is found
 * ============================================================ */
static int gpgpu_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    struct gpgpu_device *gdev;
    int ret;

    dev_info(&pdev->dev, "Probing GPGPU device [%04x:%04x]\n",
             pdev->vendor, pdev->device);

    /* Allocate driver state */
    gdev = kzalloc(sizeof(*gdev), GFP_KERNEL);
    if (!gdev)
        return -ENOMEM;
    gdev->pdev = pdev;

    mutex_init(&gdev->lock);
    mutex_init(&gdev->alloc_lock);
    init_waitqueue_head(&gdev->kernel_waitq);
    atomic_set(&gdev->kernel_done, 0);
    atomic_set(&gdev->dma_done, 0);

    /* Step 1: Enable PCI device with memory regions */
    ret = pcim_enable_device(pdev);
    if (ret) {
        dev_err(&pdev->dev, "pcim_enable_device failed: %d\n", ret);
        goto err_free;
    }

    /* Step 2: Enable bus mastering (required for DMA) */
    pci_set_master(pdev);

    ret = pcim_iomap_regions(pdev, BIT(GPGPU_BAR_CTRL), DRIVER_NAME);
    if (ret) {
        dev_err(&pdev->dev, "Failed to map BAR%d (ctrl regs)\n", GPGPU_BAR_CTRL);
        goto err_free;
    }
    gdev->ctrl_regs = pcim_iomap_table(pdev)[GPGPU_BAR_CTRL];

    /* Check BAR2 (VRAM) exists and is memory-mapped */
    if (!(pci_resource_flags(pdev, GPGPU_BAR_VRAM) & IORESOURCE_MEM)) {
        dev_err(&pdev->dev, "BAR%d not a memory region\n", GPGPU_BAR_VRAM);
        ret = -ENODEV;
        goto err_free;
    }

    gdev->vram_phys = pci_resource_start(pdev, GPGPU_BAR_VRAM);
    gdev->vram_len = pci_resource_len(pdev, GPGPU_BAR_VRAM);

    /* Map VRAM into kernel virtual address space */
    gdev->vram = pcim_iomap(pdev, GPGPU_BAR_VRAM, 0);
    if (!gdev->vram) {
        dev_err(&pdev->dev, "Failed to map BAR%d (VRAM)\n", GPGPU_BAR_VRAM);
        ret = -ENODEV;
        goto err_free;
    }

    /* Step 3: Allocate MSI-X interrupts */
    gdev->intr_type = GPGPU_INTR_NONE;
    ret = pci_alloc_irq_vectors(pdev, GPGPU_MSIX_VECTORS, GPGPU_MSIX_VECTORS,
                                 PCI_IRQ_MSIX);

    if (ret >= 0) {
        /* MSI-X 分配成功 */
        gdev->intr_type = GPGPU_INTR_MSIX;
        dev_info(&pdev->dev, "Using MSI-X interrupts\n");
        
        /* 获取 MSI-X 中断向量 */
        gdev->irq_kernel = pci_irq_vector(pdev, GPGPU_MSIX_VEC_KERNEL);
        gdev->irq_dma    = pci_irq_vector(pdev, GPGPU_MSIX_VEC_DMA);
        gdev->irq_error  = pci_irq_vector(pdev, GPGPU_MSIX_VEC_ERROR);
        
        /* 注册 MSI-X 中断处理函数 */
        ret = devm_request_irq(&pdev->dev, gdev->irq_kernel, gpgpu_isr_kernel,
                               IRQF_SHARED, "gpgpu-kernel", gdev);
        if (ret) {
            dev_err(&pdev->dev, "Failed to register kernel ISR: %d\n", ret);
            goto err_free_irq_vectors;
        }
        
        ret = devm_request_irq(&pdev->dev, gdev->irq_dma, gpgpu_isr_dma,
                               IRQF_SHARED, "gpgpu-dma", gdev);
        if (ret) {
            dev_err(&pdev->dev, "Failed to register DMA ISR: %d\n", ret);
            goto err_free_irq_vectors;
        }
        
        ret = devm_request_irq(&pdev->dev, gdev->irq_error,
                               gpgpu_isr_error, IRQF_SHARED, "gpgpu-error", gdev);
        if (ret) {
            dev_err(&pdev->dev, "Failed to register error ISR: %d\n", ret);
            goto err_free_irq_vectors;
        }


    } else {
        dev_warn(&pdev->dev, "MSI-X allocation failed (%d), falling back to INTx\n", ret);
        
        ret = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_LEGACY);
        if (ret < 0) {
            dev_err(&pdev->dev, "Failed to allocate INTx interrupt: %d\n", ret);
            goto err_free;
        }
        
        gdev->intr_type = GPGPU_INTR_INTX;
        gdev->irq_kernel = pci_irq_vector(pdev, 0);  /* INTx 使用向量 0 */
        gdev->irq_dma = -1;    /* INTx 模式下不使用独立 DMA 中断 */
        gdev->irq_error = -1;   /* INTx 模式下不使用独立 error 中断 */
        
        dev_info(&pdev->dev, "Using INTx interrupt (irq=%d)\n", gdev->irq_kernel);
        
        /* 启用 INTx（清除 DisINTx 位）*/
        pci_intx(pdev, 1);
        
        /* 注册统一的 INTx 中断处理函数 */
        ret = devm_request_irq(&pdev->dev, gdev->irq_kernel, gpgpu_isr_intx,
                               IRQF_SHARED, "gpgpu-intx", gdev);
        if (ret) {
            dev_err(&pdev->dev, "Failed to register INTx ISR: %d\n", ret);
            goto err_free_irq_vectors;
        }
    }


    

    /* Step 4: Allocate char device number */
    ret = alloc_chrdev_region(&gdev->devno, 0, 1, DEVICE_NAME);
    if (ret < 0) {
        dev_err(&pdev->dev, "alloc_chrdev_region failed: %d\n", ret);
        goto err_free;
    }
    gpgpu_major = MAJOR(gdev->devno);

    /* Initialize and add char device */
    cdev_init(&gdev->cdev, &gpgpu_fops);
    gdev->cdev.owner = THIS_MODULE;
    ret = cdev_add(&gdev->cdev, gdev->devno, 1);
    if (ret) {
        dev_err(&pdev->dev, "cdev_add failed: %d\n", ret);
        goto err_unregister_chrdev;
    }

    /* Create sysfs class and device node (/dev/gpgpu) */
    gdev->class = class_create(CLASS_NAME);
    if (IS_ERR(gdev->class)) {
        dev_err(&pdev->dev, "class_create failed\n");
        ret = PTR_ERR(gdev->class);
        goto err_cdev_del;
    }

    gdev->dev = device_create(gdev->class, &pdev->dev, gdev->devno,
                               NULL, DEVICE_NAME);
    if (IS_ERR(gdev->dev)) {
        dev_err(&pdev->dev, "device_create failed\n");
        ret = PTR_ERR(gdev->dev);
        goto err_class_destroy;
    }

    /* Save driver data */
    pci_set_drvdata(pdev, gdev);
    gpgpu_global_dev = gdev;

    /* Step 5: Enable the GPGPU device hardware */
    gpgpu_write(gdev, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_ENABLE);

    dev_info(&pdev->dev,
            "GPGPU initialized successfully:\n"
            "  Control regs @ %p, VRAM @ %pa (size %llu MB)\n"
            "  Device node: /dev/gpgpu (major=%d)\n"
            "  Interrupt type: %s\n",
            gdev->ctrl_regs, &gdev->vram_phys,
            (unsigned long long)(gdev->vram_len >> 20),
            gpgpu_major,
            gdev->intr_type == GPGPU_INTR_MSIX ? "MSI-X" : "INTx");


    return 0;

err_class_destroy:
    class_destroy(gdev->class);
err_cdev_del:
    cdev_del(&gdev->cdev);
err_unregister_chrdev:
    unregister_chrdev_region(gdev->devno, 1);
err_free_irq_vectors:
    pci_free_irq_vectors(pdev);
err_free:
    kfree(gdev);
    return ret;
}

/* ============================================================
 *  PCI Remove
 * ============================================================ */
static void gpgpu_remove(struct pci_dev *pdev)
{
    struct gpgpu_device *gdev = pci_get_drvdata(pdev);

    dev_info(&pdev->dev, "Removing GPGPU device\n");

    if (!gdev)
        return;

    /* Disable hardware */
    gpgpu_write(gdev, GPGPU_REG_GLOBAL_CTRL, 0);
    if (gdev->intr_type == GPGPU_INTR_INTX) {
        pci_intx(pdev, 0);  /* 禁用 INTx */
    }

    /* Destroy char device */
    device_destroy(gdev->class, gdev->devno);
    class_destroy(gdev->class);
    cdev_del(&gdev->cdev);
    unregister_chrdev_region(gdev->devno, 1);
    /* 清理中断向量 */
    pci_free_irq_vectors(pdev);
    /* Cleanup (pcim_ handles unmaps automatically) */
    gpgpu_global_dev = NULL;
    kfree(gdev);
}

/* ============================================================
 *  PCI Device ID Table & Driver Registration
 * ============================================================ */
static const struct pci_device_id gpgpu_pci_id_table[] = {
    { PCI_DEVICE(GPGPU_VENDOR_ID, GPGPU_DEVICE_ID) },
    { /* end: all zeroes */ }
};
MODULE_DEVICE_TABLE(pci, gpgpu_pci_id_table);

static struct pci_driver gpgpu_pci_driver = {
    .name       = DRIVER_NAME,
    .id_table   = gpgpu_pci_id_table,
    .probe      = gpgpu_probe,
    .remove     = gpgpu_remove,
};

module_pci_driver(gpgpu_pci_driver);

MODULE_AUTHOR("QEMU Camp 2026 GPGPU Team");
MODULE_DESCRIPTION("Linux kernel driver for QEMU Educational GPGPU device");
MODULE_LICENSE("GPL v2");
MODULE_VERSION("1.0");
