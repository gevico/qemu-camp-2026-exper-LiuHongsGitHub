.section .text
.global vector_add_kernel
vector_add_kernel:
    /* 1. 从 args 加载 A, B, C 的基地址 */
    lw t0, 0(a0)
    lw t1, 4(a0)
    lw t2, 8(a0)
    /* 2. 计算偏移量: offset = thread_id * 4 (float = 4 bytes) */
    slli t3, a1, 2
    /* 3. 计算数组元素的地址 */
    add t0, t0, t3
    add t1, t1, t3
    add t2, t2, t3
    /* 4. 加载数据并执行加法 */
    flw f0, 0(t0)
    flw f1, 0(t1)
    fadd.s f2, f0, f1
    fsw f2, 0(t2)
  
    /* 5. 退出: 设置 a7 = 1，然后 ecall */
    li a7, 1
    ecall
