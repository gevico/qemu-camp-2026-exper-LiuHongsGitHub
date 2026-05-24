.section .text
.global relu
relu:
    lw  t0, 0(a0)       # t0 = A
    lw  t1, 4(a0)       # t1 = B
    lw  t2, 8(a0)       # t2 = N
    slli t3, a1, 2       # t3 = threadId * 4
    bge  a1, t2, done
    add t4, t0, t3         #offset + threadId * 4
    flw fa0, 0(t4)      # fa0 = A[threadId]
    fmv.w.x fa1, zero  # fa1 = 0
    flt.s t5, fa1, fa0  
    beqz  t5, write_zero
    add t6, t1, t3
    fsw fa0, 0(t6) 
    j done
write_zero:
    add t6, t1, t3
    fmv.w.x fa1, zero
    fsw  fa1, 0(t6)
done:
    ebreak
