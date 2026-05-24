.section .text
.global matrix_mul_kernel
matrix_mul_kernel:
    lw  t0, 0(a0)       # t0 = d_A
    lw  t1, 4(a0)       # t1 = d_B
    lw  t2, 8(a0)       # t2 = d_C
    lw  t3, 12(a0)      # t3 = K
    lw  t4, 16(a0)      # t4 = N

    #=== row = threadId / N, col = threadId % N ===
    fcvt.s.w f4, a1, rne     # f4 = (float)threadId
    fcvt.s.w f5, t4, rne     # f5 = (float)N
    fdiv.s   f4, f4, f5, rne # f4 = threadId / N
    fcvt.w.s t5, f4, rtz     # t5 = row (向零截断=floor for正数)

    fcvt.s.w f6, t5, rne     # f6 = (float)row
    fmul.s   f6, f6, f5, rne # f6 = row * N
    fcvt.w.s t6, f6, rtz     # t6 = row * N
    sub      s1, a1, t6      # s1 = col = threadId - row*N

    #=== 累加器清零 ===
    fmv.s.x  f3, x0          # f3 = 0.0

    #=== A 的行偏移 ===
    fcvt.s.w f4, t5, rne     # f4 = (float)row
    fcvt.s.w f5, t3, rne     # f5 = (float)K
    fmul.s   f4, f4, f5, rne # f4 = row * K
    fcvt.w.s t5, f4, rtz     # t5 = row * K
    slli     t5, t5, 2       # t5 = row * K * 4
    add      t0, t0, t5      # t0 = &A[row][0]

    #=== B 的列偏移 ===
    slli     s1, s1, 2       # s1 = col * 4
    add      t1, t1, s1      # t1 = &B[0][col]

    #=== B 的行步长 ===
    slli     a2, t4, 2       # a2 = N * 4

    #=== 循环 ===
    li       t6, 0
loop:
    flw      f0, 0(t0)
    flw      f1, 0(t1)
    fmadd.s  f3, f0, f1, f3, rne
    addi     t0, t0, 4
    add      t1, t1, a2
    addi     t6, t6, 1
    blt      t6, t3, loop

    #=== 写回 ===
    slli     s1, a1, 2
    add      t2, t2, s1
    fsw      f3, 0(t2)

    li       a7, 1
    ecall
