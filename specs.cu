#include <cstdio>
#include <cuda_runtime.h>

int main()
{
    printf("CUDA SPECIFICATIONS\n");
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    printf("Device count: %d\n", deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("=== Device %d: %s ===\n", dev, prop.name);
        printf("  Compute capability: %d.%d\n",
               prop.major, prop.minor);
        printf("  Max threads per block: %d\n",
               prop.maxThreadsPerBlock);
        printf("  Max threads per multiprocessor: %d\n",
               prop.maxThreadsPerMultiProcessor);
        printf("  Number of SMs: %d\n",
               prop.multiProcessorCount);
        printf("  Max grid dimensions:  x=%d  y=%d  z=%d\n",
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf("  Warp size: %d\n",
               prop.warpSize);
        printf("\n");
    }
    return 0;
}
