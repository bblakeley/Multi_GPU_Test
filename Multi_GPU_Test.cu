// Multiple GPU version of cuFFT_check that uses multiple GPU's
// This program creates a real-valued 3D function sin(x)*cos(y)*cos(z) and then 
// takes the forward and inverse Fourier Transform, with the necessary scaling included. 
// The output of this process should match the input function

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// includes, project
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define NX 512
#define NY 512
#define NZ 512
#define NZ2 (NZ/2+1)
#define NN (NX*NY*NZ)
#define L (2*M_PI)
#define TX 8
#define TY 8
#define TZ 8

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
int idxClip(int idx, int idxMax){
    return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int stack, int width, int height, int depth){
    return idxClip(stack, depth) + idxClip(row, height)*depth + idxClip(col, width)*depth*height;
    // Note: using column-major indexing format
}

__global__ 
void initialize(double *f1, double *f2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    // if ((i >= NX) || (j >= NY) || (k >= NZ)) return;
    const int idx = flatten(i, j, k, NX, NY, NZ);

    // Initialize array
    f1[idx] = 0.5+0.5;
    f2[idx] = 2.5*2.0;

    return;
}

void initialize_singleGPU(double *f1, double *f2)
{
    // Launch CUDA kernel to initialize velocity field
    const dim3 blockSize(TX, TY, TZ);
    const dim3 gridSize(divUp(NX, TX), divUp(NY, TY), divUp(NZ, TZ));

    initialize<<<gridSize, blockSize>>>(f1, f2);

    return;
}

void initialize_multiGPU(const int GPUnum, double *f1, double *f2)
{
    int i, idx, NX_per_GPU;
    // Split data according to number of GPUs
    NX_per_GPU = NX/GPUnum;              // This is not a good solution long-term; needs more work for arbitrary grid sizes/nGPUs
    printf("   The number of divisions in the X-direction is %d\n", NX_per_GPU);

    // Launch CUDA kernel to initialize velocity field
    const dim3 blockSize(TX, TY, TZ);
    const dim3 gridSize(divUp(NX_per_GPU, TX), divUp(NY, TY), divUp(NZ, TZ));

    for (i = 0; i<GPUnum; ++i){
        cudaSetDevice(i);
        idx = i*NX_per_GPU*NY*NZ;                // sets the index value of the data to send to each gpu
        initialize<<<gridSize, blockSize>>>(&f1[idx], &f2[idx]);
    }

    return;
}

int main (void)
{
    int i, j, k, idx;

    // Declare variables
    double *u;
    double *u_fft;

    // Allocate memory for arrays
    cudaMallocManaged(&u, sizeof(double)*NN );
    cudaMallocManaged(&u_fft, sizeof(double)*NN );

    // Perform kernel calculation using only one GPU first:
    cudaSetDevice(0);

    initialize_singleGPU(u, u_fft);

    cudaDeviceSynchronize();

    double result1 = 0.0;
    for (i = 0; i < NX; ++i ){
        for (j = 0; j<NY; ++j){
            for (k = 0; k<NZ; ++k){
                idx = k + j*NZ + i*NY*NZ;
                result1 += u[idx] + u_fft[idx];
            }
        }
    }

    // Set GPU's to use and list device properties
    int nGPUs = 2, deviceNum[nGPUs];
    for(i = 0; i<nGPUs; ++i)
    {
        deviceNum[i] = i;

        cudaSetDevice(deviceNum[i]);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceNum[i]);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    initialize_multiGPU(nGPUs, u, u_fft);

    // Synchronize both GPUs in order to print reports
    for (i = 0; i<nGPUs; ++i){
        cudaSetDevice(deviceNum[i]);
        cudaDeviceSynchronize();
    }
    
    double result2 = 0.0;
    for (i = 0; i < NX; ++i ){
        for (j = 0; j<NY; ++j){
            for (k = 0; k<NZ; ++k){
                idx = k + j*NZ + i*NY*NZ;
                result2 += u[idx] + u_fft[idx];
            }
        }
    }

    printf("The value of f1 is %d, which should equal to 6*NX*NY*NZ, %d\n", (int)result1, NN + 5*NN);
    printf("The value of f2 is %d, which should equal to 6*NX*NY*NZ, %d\n", (int)result2, NN + 5*NN);

    cudaFree(u);
    cudaFree(u_fft);

    return 0;

}