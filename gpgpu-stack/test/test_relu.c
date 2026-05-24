#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../runtime/include/gpgpu.h"


int main(void)
{
    int ret;
    ret = gpgpuInit();
    if (ret < 0) {
        fprintf(stderr, "Failed to initialize GPGPU runtime\n");
        return 1;
    }
    gpgpuMemHandle hArgs = (gpgpuMemHandle)-1;
    gpgpuMemHandle hKernel = (gpgpuMemHandle)-1;
    gpgpuMemHandle d_A = (gpgpuMemHandle)-1;
    gpgpuMemHandle d_B  = (gpgpuMemHandle)-1;

    FILE *f = fopen("relu.bin", "rb");
    if (f == NULL) {
        fprintf(stderr, "Failed to open relu.bin\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long kernel_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *kernelCode =(uint8_t*) malloc(kernel_size);
    if (kernelCode == NULL) {
        fprintf(stderr, "Failed to allocate memory for kernel code\n");
        fclose(f);
        goto shutdown;
    }
    size_t read_bytes = fread(kernelCode, 1, kernel_size, f);
    fclose(f);
    if (read_bytes != kernel_size) {
        fprintf(stderr, "Failed to read kernel binary (read %zu/%ld bytes)\n", read_bytes, kernel_size);
        free(kernelCode);
        goto shutdown;
    }
    hKernel = gpgpuMalloc(kernel_size);
    if (hKernel == (gpgpuMemHandle)-1) {
        fprintf(stderr, "gpgpuMalloc failed for kernel\n");
        free(kernelCode);
        goto shutdown;
    }
    gpgpuMemcpyH2D(hKernel, kernelCode, kernel_size);
    free(kernelCode);

    int N = 64;
    float* A = (float*)malloc(N* sizeof(float));
    for (int i = 0; i < N; i++) {
        A[i] = (float) 2;
    }
    float* B = (float*)malloc(N  * sizeof(float));
    for (int i = 0; i < N; i++) {
        B[i] = (float) -1;
    }
    d_A =  gpgpuMalloc(N * sizeof(float));
    d_B =  gpgpuMalloc(N * sizeof(float));
    if (d_A == (gpgpuMemHandle)-1 || d_B == (gpgpuMemHandle)-1 ) {
        fprintf(stderr, "gpgpuMalloc failed for matrices\n");
        goto cleanup;
    }
    gpgpuMemcpyH2D(d_A,A,N * sizeof(float));
    gpgpuMemcpyH2D(d_B,B, N * sizeof(float)); 
    uint32_t kernel_args[3] = {d_A, d_B, N};
    hArgs = gpgpuMalloc(sizeof(kernel_args));
    if (hArgs == (gpgpuMemHandle)-1) {
        fprintf(stderr, "gpgpuMalloc failed for kernel arguments\n");
        goto cleanup;
       
    }
    gpgpuMemcpyH2D(hArgs, kernel_args, sizeof(kernel_args));
    ret = gpgpuLaunchKernel(hKernel, hArgs, (dim3){1,1,1}, (dim3){N,1,1}, 0);
    if (ret < 0) {
        fprintf(stderr, "Failed to launch kernel\n");
        goto cleanup;
    }
    gpgpuSync();
    gpgpuMemcpyD2H(B,d_B, N * sizeof(float));
     for (int i = 0; i < N; i++) {
        printf("relu result lane:%f ", B[i]);
    }
    printf("\n");
    int errors = 0;
    for (int i = 0; i <N; i++) {
        if (B[i] != 2.0f) {
            printf("Error at %d: expected 2.0, got %f\n", i, B[i]);
            errors++;
        }
    }
    if (errors == 0) printf("relu test PASSED\n");
    else printf("relu  test FAILED (%d errors)\n", errors);
cleanup:
    if (d_A != (gpgpuMemHandle)-1) gpgpuFree(d_A);
    if (d_B != (gpgpuMemHandle)-1) gpgpuFree(d_B);
    
    if (hArgs != (gpgpuMemHandle)-1) gpgpuFree(hArgs);
    if (hKernel != (gpgpuMemHandle)-1) gpgpuFree(hKernel);
shutdown:
    gpgpuShutdown();
    return 0;
}