#include <sys/time.h>
#include <stdio.h>

// time stamp function in seconds
double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_usec / 1000000 + tv.tv_sec;
}

// host side matrix addition
void h_addmat(float *A, float *B, float *C, int nx, int ny) {
    int i, j;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
            int index = i * ny + j;
            C[index] = A[index] + B[index];
        }
}


// device-side matrix addition
__global__ void f_addmat(float *A, float *B, float *C, int nx, int ny) {
    // kernel code might look something like this
    // but you may want to pad the matrices and index into them accordingly
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * nx + ix;
    if ((ix < nx) && (iy < ny))
        C[idx] = A[idx] + B[idx];
}

int main(int argc, char *argv[]) {
    // get program arguments
    if (argc != 3) {
        printf("Error: wrong number of args\n");
        exit(1);
    }


    int nx = atoi(argv[1]); // should check validity
    int ny = atoi(argv[2]); // should check validity

    int noElems = nx * ny;
    int bytes = noElems * sizeof(float);
    // but you may want to pad the matrices…

    // alloc memory host-side
    float *h_A = (float *) malloc(bytes);
    float *h_B = (float *) malloc(bytes);
    float *h_hC = (float *) malloc(bytes); // host result
    float *h_dC = (float *) malloc(bytes); // gpu result

    cudaHostRegister(h_A, bytes, 0);
    cudaHostRegister(h_B, bytes, 0);
    cudaHostRegister(h_dC, bytes, 0);

    // init matrices with random data

    int i, j, index;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
            index = i * ny + j;
            h_A[index] = (float) (i + j) / 3.0;
            h_B[index] = (float) 3.14 * (i + j);
        }


    // alloc memory dev-side
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, bytes);
    cudaMalloc((void **) &d_B, bytes);
    cudaMalloc((void **) &d_C, bytes);

    double timeStampA = getTimeStamp();
    //transfer data to dev
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    // note that the transfers would be twice as fast if h_A and h_B
    // matrices are pinned
    double timeStampB = getTimeStamp();

    // invoke Kernel
    dim3 block(1024, 1); // you will want to configure this
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    f_addmat << < grid, block >> > (d_A, d_B, d_C, nx, ny);
    cudaDeviceSynchronize();

    double timeStampC = getTimeStamp();
    //copy data back
    cudaMemcpy(h_dC, d_C, bytes, cudaMemcpyDeviceToHost);
    double timeStampD = getTimeStamp();

    // free GPU resources
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();

    // check result
    h_addmat(h_A, h_B, h_hC, nx, ny);

    bool correct = true;

    for (i = 0; i < nx * ny; i++)
        if (h_hC[i] != h_dC[i]) {
            correct = false;
            break;
        }

    if (!correct)
        printf("Error: Result Incorrect! \n");

    cudaHostUnregister(h_A);
    cudaHostUnregister(h_B);
    cudaHostUnregister(h_dC);


    // print out results
    printf("%.6f %.6f %.6f %.6f \n", timeStampD - timeStampA, timeStampB - timeStampA, timeStampC - timeStampB,
           timeStampD - timeStampC);
}
