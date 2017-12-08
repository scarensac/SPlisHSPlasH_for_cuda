
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DFSPH_cuda_basic.h"
#include <stdio.h>
#include "DFSPH_c_arrays_structure.h"

#define BLOCKSIZE 256
#define m_eps 1.0e-5



__global__ void DFSPH_updateVelocity_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	m_data.velFluid[i] += m_data.h * m_data.accFluid[i];
}



void cuda_updateVelocities(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_updateVelocity_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}








/*
	THE NEXT FUNCTIONS ARE FOR THE MEMORY ALLOCATION
*/

void allocate_c_array_struct_cuda_managed(SPH::DFSPHCData& data) {
	//cudaMallocManaged(&x, N * sizeof(float));
	//cudaMallocManaged(&y, N * sizeof(float));

	cudaMallocManaged(&(data.posBoundary), data.numBoundaryParticles * sizeof(Vector3d));
	cudaMallocManaged(&(data.velBoundary), data.numBoundaryParticles * sizeof(Vector3d));
	cudaMallocManaged(&(data.velBoundary), data.numBoundaryParticles * sizeof(Vector3d));
	cudaMallocManaged(&(data.boundaryPsi), data.numBoundaryParticles * sizeof(Real));


	//handle the fluid
	cudaMallocManaged(&(data.mass), data.numFluidParticles * sizeof(Real));
	cudaMallocManaged(&(data.posFluid), data.numFluidParticles * sizeof(Vector3d));
	cudaMallocManaged(&(data.velFluid), data.numFluidParticles * sizeof(Vector3d));
	cudaMallocManaged(&(data.accFluid), data.numFluidParticles * sizeof(Vector3d));
	cudaMallocManaged(&(data.numberOfNeighbourgs), data.numFluidParticles * 2 * sizeof(int));
	cudaMallocManaged(&(data.neighbourgs), data.numFluidParticles * 2 * MAX_NEIGHBOURS * sizeof(int));

	cudaMallocManaged(&(data.density), data.numFluidParticles * sizeof(Real));
	cudaMallocManaged(&(data.factor), data.numFluidParticles * sizeof(Real));
	cudaMallocManaged(&(data.kappa), data.numFluidParticles * sizeof(Real));
	cudaMallocManaged(&(data.kappaV), data.numFluidParticles * sizeof(Real));
	cudaMallocManaged(&(data.densityAdv), data.numFluidParticles * sizeof(Real));
}


void allocate_precomputed_kernel_managed(SPH::PrecomputedCubicKernelPerso& kernel){
	cudaMallocManaged(&(kernel.m_W), kernel.m_resolution * sizeof(Real));
	cudaMallocManaged(&(kernel.m_gradW), (kernel.m_resolution+1) * sizeof(Real));
}

/*
	AFTER THIS ARE ONLY THE TEST FUNCTION TO HAVE CUDA WORKING ...
*/


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
//*
__global__ void addKernel(Vector3d* vect)
{
	int i = threadIdx.x;
	vect[i].z = vect[i].x + vect[i].y;
}

__global__ void setVectkernel(Vector3d& vect)
{
	vect.x = 5;
	vect.y = 6;
	vect.z = 7;
}
//*/
int test_cuda()
{
	//DFSPHCData* data;

	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };
	//*
	Vector3d* vect;
	cudaMallocManaged(&vect, arraySize * sizeof(Vector3d));
	for (int i = 0; i < arraySize; ++i) {
		vect[i].x = a[i];
		vect[i].y = b[i];
	}
	//*/*

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}


	printf("macro val: %d, %d, %d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	for (int i = 0; i < arraySize; ++i) {
		c[i] = 0;
	}


	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, arraySize >> > (vect);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	printf("with vects {1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		(int)(vect[0].z), (int)(vect[1].z), (int)(vect[2].z), (int)(vect[3].z), (int)(vect[4].z));

	cudaFree(vect);



	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	/*
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	//*/

	printf("Finished test cuda\n");


	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
