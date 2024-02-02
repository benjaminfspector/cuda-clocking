#include "clocker.cuh"
#include "stdio.h"

#define BLOCK_DIM 256
#define N 131072

__global__ void add(int *a, int *b, int *c, TIMINGDETAIL_ARGS())
{
    INITTHREADTIMER();
    int tID = blockIdx.x * BLOCK_DIM + threadIdx.x;
    CLOCKRESET();
    int aa = a[tID];
    CLOCKPOINT(0, "Load A");
    int bb = b[tID];
    CLOCKPOINT(1, "Load B");
    int cc = bb;
    if (tID < N) {
        cc = aa + bb;
    }
    CLOCKPOINT(2, "Add");
    c[tID] = cc;
    CLOCKPOINT(3, "Write");

    __nanosleep(1000000);
    FINISHTHREADTIMER();
}

int main()
{
    constexpr dim3 block(BLOCK_DIM, 1, 1);
    constexpr dim3 grid(N / BLOCK_DIM, 1, 1);
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **) &dev_a, N*sizeof(int));
    cudaMalloc((void **) &dev_b, N*sizeof(int));
    cudaMalloc((void **) &dev_c, N*sizeof(int));
    // Fill Arrays
    for (int i = 0; i < N; i++) {
        a[i] = i,
        b[i] = 1;
    }
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    TimingData tdata(grid, block);
    add<<<grid,block>>>(dev_a, dev_b, dev_c, tdata.data);
    tdata.write(__FILE__);
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    printf("%s\n", __FILE__);
    for (int i = N-8; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    return 0;
}