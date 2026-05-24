/*
 * Vector Add Kernel (RV32I assembly)
 * 
 * 功能: C[i] = A[i] + B[i]
 * 
 * 内置变量通过以下方式获取:
 *   - lane_id 存储在 mhartid (用于演示)
 * 
 * 实际 kernel 需要:
 *   - 从 args 地址读取参数 (A, B, C 的 VRAM 偏移)
 *   - 计算 thread_id
 *   - 执行加法
 *   - 写回结果
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../runtime/include/gpgpu.h"

/* 简单的 RV32I kernel 代码 (二进制形式) */
/* 这个 kernel 执行: C[thread_id] = A[thread_id] + B[thread_id] */

#define N 256

int main(void)
{
    int ret;
    float *A = NULL, *B = NULL, *C = NULL;
    gpgpuMemHandle hA = (gpgpuMemHandle)-1;
    gpgpuMemHandle hB = (gpgpuMemHandle)-1;
    gpgpuMemHandle hC = (gpgpuMemHandle)-1;
    gpgpuMemHandle hKernel = (gpgpuMemHandle)-1;
    gpgpuMemHandle hArgs = (gpgpuMemHandle)-1;
    dim3 gridDim = {1, 1, 1};
    dim3 blockDim = {256, 1, 1};

    printf("=== Vector Add Test ===\n");

    /* 1. 初始化 */
    ret = gpgpuInit();
    if (ret < 0) {
        fprintf(stderr, "gpgpuInit failed\n");
        return 1;
    }

    /* 2. 分配主机内存 */
    A = (float*)malloc(N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));
    if (!A || !B || !C) {
        fprintf(stderr, "Host malloc failed\n");
        goto cleanup;
    }

    /* 3. 初始化数据 */
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(i * 2);
    }

    /* 4. 分配设备内存 */
    hA = gpgpuMalloc(N * sizeof(float));
    hB = gpgpuMalloc(N * sizeof(float));
    hC = gpgpuMalloc(N * sizeof(float));
    if (hA == (gpgpuMemHandle)-1 || hB == (gpgpuMemHandle)-1 || hC == (gpgpuMemHandle)-1) {
        fprintf(stderr, "gpgpuMalloc failed for A/B/C\n");
        goto cleanup;
    }

    uint32_t kernel_args[3] = {hA, hB, hC};

    hArgs = gpgpuMalloc(sizeof(kernel_args));
    if (hArgs == (gpgpuMemHandle)-1) {
        fprintf(stderr, "gpgpuMalloc failed for args\n");
        goto cleanup;
    }

    /* 读取 kernel 代码 */
    FILE *f = fopen("vector_add.bin", "rb");
    if (!f) {
        fprintf(stderr, "Failed to open vector_add.bin\n");
        goto cleanup;
    }
    fseek(f, 0, SEEK_END);
    long kernel_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *kernel_code = (uint8_t*)malloc(kernel_size);
    if (!kernel_code) {
        fprintf(stderr, "malloc for kernel_code failed\n");
        fclose(f);
        goto cleanup;
    }

    size_t read_bytes = fread(kernel_code, 1, kernel_size, f);
    fclose(f);
    if (read_bytes != kernel_size) {
        fprintf(stderr, "Failed to read kernel binary (read %zu/%ld bytes)\n", read_bytes, kernel_size);
        free(kernel_code);
        goto cleanup;
    }

    hKernel = gpgpuMalloc(kernel_size);
    if (hKernel == (gpgpuMemHandle)-1) {
        fprintf(stderr, "gpgpuMalloc failed for kernel\n");
        free(kernel_code);
        goto cleanup;
    }

    /* 5. 拷贝数据到设备 */
    gpgpuMemcpyH2D(hA, A, N * sizeof(float));
    gpgpuMemcpyH2D(hB, B, N * sizeof(float));
    gpgpuMemcpyH2D(hKernel, kernel_code, kernel_size);
    free(kernel_code);
    kernel_code = NULL;

    gpgpuMemcpyH2D(hArgs, kernel_args, sizeof(kernel_args));

    /* 6. 启动 kernel */
    ret = gpgpuLaunchKernel(hKernel, hArgs, gridDim, blockDim, 0);
    if (ret < 0) {
        fprintf(stderr, "gpgpuLaunchKernel failed\n");
        goto cleanup;
    }

    /* 7. 等待完成 */
    gpgpuSync();

    /* 8. 拷贝结果回主机 */
    gpgpuMemcpyD2H(C, hC, N * sizeof(float));

    /* 9. 验证结果 */
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = A[i] + B[i];
        if (C[i] != expected) {
            printf("Error at %d: expected %f, got %f\n", i, expected, C[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("✓ Vector Add test PASSED\n");
    } else {
        printf("✗ Vector Add test FAILED (%d errors)\n", errors);
    }

cleanup:
    /* 10. 清理 */
    if (hA != (gpgpuMemHandle)-1) gpgpuFree(hA);
    if (hB != (gpgpuMemHandle)-1) gpgpuFree(hB);
    if (hC != (gpgpuMemHandle)-1) gpgpuFree(hC);
    if (hKernel != (gpgpuMemHandle)-1) gpgpuFree(hKernel);
    if (hArgs != (gpgpuMemHandle)-1) gpgpuFree(hArgs);
    free(A);
    free(B);
    free(C);
    gpgpuShutdown();

    return errors > 0 ? 1 : 0;
}
