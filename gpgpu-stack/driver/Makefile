# SPDX-License-Identifier: GPL-2.0
#
# Makefile for GPGPU driver
#

# obj-$(CONFIG_GPGPU) += gpgpu.o

# gpgpu-y := gpgpu_drv.o
ARCH ?= riscv
CROSS_COMPILE ?= riscv64-linux-gnu-

KDIR ?= /home/lh/code/linux-riscv

obj-m += gpgpu.o
gpgpu-y := gpgpu_drv.o

all:
	make -C $(KDIR) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) modules

clean:
	make -C $(KDIR) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) clean
