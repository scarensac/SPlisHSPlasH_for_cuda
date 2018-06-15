
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DFSPH_cuda_basic.h"
#include <stdio.h>
#include "DFSPH_c_arrays_structure.h"
#include "cub.cuh"
#include <chrono>
#include "SPlisHSPlasH/Utilities/Timing.h"
#include <iostream>

#define BLOCKSIZE 256
#define m_eps 1.0e-5
#define CELL_ROW_LENGTH 256
#define CELL_COUNT CELL_ROW_LENGTH*CELL_ROW_LENGTH*CELL_ROW_LENGTH

#define USE_WARMSTART
#define USE_WARMSTART_V

#define BITSHIFT_INDEX_DYNAMIC_BODIES

#ifdef BITSHIFT_INDEX_DYNAMIC_BODIES
#define WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX(body_index,particle_index) WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX_BITSHIFT(body_index,particle_index)
#define READ_DYNAMIC_BODIES_PARTICLES_INDEX(neighbors_ptr,body_index,particle_index) READ_DYNAMIC_BODIES_PARTICLES_INDEX_BITSHIFT(neighbors_ptr,body_index,particle_index)
#else
#define WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX(body_index,particle_index) WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX_ADDITION(body_index,particle_index)
#define READ_DYNAMIC_BODIES_PARTICLES_INDEX(neighbors_ptr,body_index,particle_index) READ_DYNAMIC_BODIES_PARTICLES_INDEX_ADDITION(neighbors_ptr,body_index,particle_index)
#endif

//those defines are to create and read the dynamic bodies indexes
#define WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX_BITSHIFT(body_index,particle_index)  particle_index + (body_index << 0x10)
#define WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX_ADDITION(body_index,particle_index)  particle_index + (body_index * 1000000)

//WARNING his one declare the body/particle index by itself
//you just have to give it the variable name you want
#define READ_DYNAMIC_BODIES_PARTICLES_INDEX_BITSHIFT(neighbors_ptr, body_index,particle_index)  \
const unsigned int identifier = *neighbors_ptr++;\
const unsigned int particle_index = identifier & 0xFFFF;\
const unsigned int body_index = (identifier & ~0xFFFF) >> 0x10;

#define READ_DYNAMIC_BODIES_PARTICLES_INDEX_ADDITION(neighbors_ptr, body_index,particle_index)   \
const unsigned int identifier = *neighbors_ptr++;\
const unsigned int particle_index = identifier % (1000000);\
const unsigned int body_index=identifier / 1000000;


//using norton bitshift for the cells is slower than using a normal index, not that much though
//#define BITSHIFT_INDEX_NEIGHBORS_CELL


#ifdef BITSHIFT_INDEX_NEIGHBORS_CELL

#define USE_COMPLETE

__device__ void interleave_2_bits_magic_numbers(unsigned int& x) {
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
}
__device__ unsigned int compute_morton_magic_numbers(unsigned int x, unsigned int y, unsigned int z) {
	interleave_2_bits_magic_numbers(x);
	interleave_2_bits_magic_numbers(y);
	interleave_2_bits_magic_numbers(z);

	return x | (y << 1) | (z << 2);
}

#define COMPUTE_CELL_INDEX(x,y,z) compute_morton_magic_numbers(x,y,z)

#else
#define COMPUTE_CELL_INDEX(x,y,z) (x)+(y)*CELL_ROW_LENGTH+(z)*CELL_ROW_LENGTH*CELL_ROW_LENGTH
#endif



//those two variables are the identifiers that  link the ongle buffers to cuda
//cudaGraphicsResource_t vboRes_pos;
//cudaGraphicsResource_t vboRes_vel;

//easy function to check errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/*
//this is the bases for all kernels based function
__global__ void DFSPH__kernel(SPH::DFSPHCData m_data) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= m_data.numFluidParticles) { return; }

}
void cuda_(SPH::DFSPHCData& data) {
int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
DFSPH__kernel << <numBlocks, BLOCKSIZE >> > (data);

cudaError_t cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
exit(1598);
}
}
//*/

FUNCTION inline int* getNeighboursPtr(int * neighbourgs, int particle_id) {
	//	return neighbourgs + body_id*numFluidParticles*MAX_NEIGHBOURS + particle_id*MAX_NEIGHBOURS;
	return neighbourgs + particle_id*MAX_NEIGHBOURS;
}

FUNCTION inline unsigned int getNumberOfNeighbourgs(int* numberOfNeighbourgs, int particle_id, int body_id = 0) {
	//return numberOfNeighbourgs[body_id*numFluidParticles + particle_id]; 
	return numberOfNeighbourgs[particle_id * 3 + body_id];
}

__device__ void computeDensityChange(SPH::DFSPHCData& m_data, const unsigned int index) {
	unsigned int numNeighbors = m_data.fluid_data_cuda->getNumberOfNeighbourgs(index);
	// in case of particle deficiency do not perform a divergence solve
	if (numNeighbors < 20) {
		for (unsigned int pid = 1; pid < 3; pid++)
		{
			numNeighbors += m_data.fluid_data_cuda->getNumberOfNeighbourgs(index, pid);
		}
	}
	if (numNeighbors < 20) {
		m_data.fluid_data_cuda->densityAdv[index] = 0;
	}
	else {
		RealCuda densityAdv = 0;
		const Vector3d &xi = m_data.fluid_data_cuda->pos[index];
		const Vector3d &vi = m_data.fluid_data_cuda->vel[index];
		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		int* neighbors_ptr = m_data.fluid_data_cuda->getNeighboursPtr(index);
		int* end_ptr = neighbors_ptr + m_data.fluid_data_cuda->getNumberOfNeighbourgs(index);
		{
			SPH::UnifiedParticleSet& body = *(m_data.fluid_data_cuda);
			while (neighbors_ptr != end_ptr)
			{
				const unsigned int neighborIndex = *neighbors_ptr++;
				densityAdv += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_data.gradW(xi - body.pos[neighborIndex]));
			}
		}
		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
		{
			SPH::UnifiedParticleSet& body = *(m_data.boundaries_data_cuda);
			end_ptr += m_data.fluid_data_cuda->getNumberOfNeighbourgs(index, 1);
			while (neighbors_ptr != end_ptr)
			{
				const unsigned int neighborIndex = *neighbors_ptr++;
				densityAdv += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_data.gradW(xi - body.pos[neighborIndex]));
			}
		}

		//////////////////////////////////////////////////////////////////////////
		// Dynamic Bodies
		//////////////////////////////////////////////////////////////////////////
		end_ptr += m_data.fluid_data_cuda->getNumberOfNeighbourgs(index, 2);
		while (neighbors_ptr != end_ptr)
		{
			READ_DYNAMIC_BODIES_PARTICLES_INDEX(neighbors_ptr, bodyIndex, neighborIndex);
			SPH::UnifiedParticleSet& body = m_data.vector_dynamic_bodies_data_cuda[bodyIndex];
			densityAdv += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_data.gradW(xi - body.pos[neighborIndex]));
		}




		// only correct positive divergence
		m_data.fluid_data_cuda->densityAdv[index] = MAX_MACRO_CUDA(densityAdv, 0.0);
	}
}
template <bool warm_start> __device__ void divergenceSolveParticle(SPH::DFSPHCData& m_data, const unsigned int i) {
	Vector3d v_i = Vector3d(0, 0, 0);
	//////////////////////////////////////////////////////////////////////////
	// Evaluate rhs
	//////////////////////////////////////////////////////////////////////////
	const RealCuda ki = (warm_start) ? m_data.fluid_data_cuda->kappaV[i] : (m_data.fluid_data_cuda->densityAdv[i])*m_data.fluid_data_cuda->factor[i];

#ifdef USE_WARMSTART_V
	if (!warm_start) { m_data.fluid_data_cuda->kappaV[i] += ki; }
#endif

	const Vector3d &xi = m_data.fluid_data_cuda->pos[i];


	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	int* neighbors_ptr = m_data.fluid_data_cuda->getNeighboursPtr(i);
	int* end_ptr = neighbors_ptr + m_data.fluid_data_cuda->getNumberOfNeighbourgs(i);
	{
		SPH::UnifiedParticleSet& body = *(m_data.fluid_data_cuda);
		while (neighbors_ptr != end_ptr)
		{
			const unsigned int neighborIndex = *neighbors_ptr++;
			const RealCuda kSum = (ki + ((warm_start) ? body.kappaV[neighborIndex] : (body.densityAdv[neighborIndex])*body.factor[neighborIndex]));
			if (fabs(kSum) > m_eps)
			{
				// ki, kj already contain inverse density
				v_i += kSum *  body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	if (fabs(ki) > m_eps)
	{
		end_ptr += m_data.fluid_data_cuda->getNumberOfNeighbourgs(i, 1);
		SPH::UnifiedParticleSet& body = *(m_data.boundaries_data_cuda);
		while (neighbors_ptr != end_ptr)
		{
			const unsigned int neighborIndex = *neighbors_ptr++;
			const Vector3d delta = ki * body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);
			v_i += delta;// ki already contains inverse density
		}

	}

	//////////////////////////////////////////////////////////////////////////
	// Dynamic bodies
	//////////////////////////////////////////////////////////////////////////
	if (fabs(ki) > m_eps)
	{
		end_ptr += m_data.fluid_data_cuda->getNumberOfNeighbourgs(i, 2);
		while (neighbors_ptr != end_ptr)
		{
			READ_DYNAMIC_BODIES_PARTICLES_INDEX(neighbors_ptr, bodyIndex, neighborIndex);
			SPH::UnifiedParticleSet& body = m_data.vector_dynamic_bodies_data_cuda[bodyIndex];
			
			Vector3d delta = ki * body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);
			v_i += delta;// ki already contains inverse density

			//we apply the force to the body particle (no invH since it has been fatorized at the end)
			delta *= -m_data.fluid_data_cuda->mass[i];
			atomicAdd(&(body.F[neighborIndex].x), delta.x);
			atomicAdd(&(body.F[neighborIndex].y), delta.y);
			atomicAdd(&(body.F[neighborIndex].z), delta.z);
		}
	}

	m_data.fluid_data_cuda->vel[i] += v_i*m_data.h;
}

__device__ void computeDensityAdv(SPH::DFSPHCData& m_data, const unsigned int index) {
	const Vector3d xi = m_data.fluid_data_cuda->pos[index];
	const Vector3d vi = m_data.fluid_data_cuda->vel[index];
	RealCuda delta = 0;


	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	int* neighbors_ptr = m_data.fluid_data_cuda->getNeighboursPtr(index);
	int* end_ptr = neighbors_ptr + m_data.fluid_data_cuda->getNumberOfNeighbourgs(index);
	{
		SPH::UnifiedParticleSet& body = *(m_data.fluid_data_cuda);
		while (neighbors_ptr != end_ptr)
		{
			const unsigned int neighborIndex = *neighbors_ptr++;
			delta += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_data.gradW(xi - body.pos[neighborIndex]));
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	end_ptr += m_data.fluid_data_cuda->getNumberOfNeighbourgs(index, 1);
	{
		SPH::UnifiedParticleSet& body = *(m_data.boundaries_data_cuda);
		while (neighbors_ptr != end_ptr)
		{
			const unsigned int neighborIndex = *neighbors_ptr++;
			delta += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_data.gradW(xi - body.pos[neighborIndex]));
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Dynamic bodies
	//////////////////////////////////////////////////////////////////////////
	end_ptr += m_data.fluid_data_cuda->getNumberOfNeighbourgs(index, 2);
	while (neighbors_ptr != end_ptr)
	{
		READ_DYNAMIC_BODIES_PARTICLES_INDEX(neighbors_ptr, bodyIndex, neighborIndex);
		SPH::UnifiedParticleSet& body = m_data.vector_dynamic_bodies_data_cuda[bodyIndex];
		delta += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_data.gradW(xi - body.pos[neighborIndex]));
	}




	m_data.fluid_data_cuda->densityAdv[index] = MAX_MACRO_CUDA(m_data.fluid_data_cuda->density[index] + m_data.h_future*delta - m_data.density0, 0.0);
}

__device__ void computeDensityAdv(const unsigned int index, Vector3d* posFluid, Vector3d* velFluid, int* neighbourgs, int * numberOfNeighbourgs,
	RealCuda* mass, SPH::PrecomputedCubicKernelPerso m_kernel_precomp, RealCuda* boundaryPsi, Vector3d* posBoundary, Vector3d* velBoundary,
	SPH::UnifiedParticleSet* vector_dynamic_bodies_data_cuda, RealCuda* densityAdv, RealCuda* density, RealCuda h_future, RealCuda density0) {
	const Vector3d xi = posFluid[index];
	const Vector3d vi = velFluid[index];
	RealCuda delta = 0;

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	int* neighbors_ptr = getNeighboursPtr(neighbourgs, index);
	int* end_ptr = neighbors_ptr + getNumberOfNeighbourgs(numberOfNeighbourgs, index);
	while (neighbors_ptr != end_ptr)
	{
		const unsigned int neighborIndex = *neighbors_ptr++;
		delta += mass[neighborIndex] * (vi - velFluid[neighborIndex]).dot(m_kernel_precomp.gradW(xi - posFluid[neighborIndex]));
	}

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	end_ptr += getNumberOfNeighbourgs(numberOfNeighbourgs, index, 1);
	while (neighbors_ptr != end_ptr)
	{
		const unsigned int neighborIndex = *neighbors_ptr++;
		delta += boundaryPsi[neighborIndex] * (vi - velBoundary[neighborIndex]).dot(m_kernel_precomp.gradW(xi - posBoundary[neighborIndex]));
	}

	//////////////////////////////////////////////////////////////////////////
	// Dynamic bodies
	//////////////////////////////////////////////////////////////////////////
	end_ptr += getNumberOfNeighbourgs(numberOfNeighbourgs, index, 2);
	while (neighbors_ptr != end_ptr)
	{
		READ_DYNAMIC_BODIES_PARTICLES_INDEX(neighbors_ptr, bodyIndex, neighborIndex);
		SPH::UnifiedParticleSet& body = vector_dynamic_bodies_data_cuda[bodyIndex];
		delta += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_kernel_precomp.gradW(xi - body.pos[neighborIndex]));
	}




	densityAdv[index] = MAX_MACRO_CUDA(density[index] + h_future*delta - density0, 0.0);
}

template <bool warm_start> __device__ void pressureSolveParticle(SPH::DFSPHCData& m_data, const unsigned int i) {
	//////////////////////////////////////////////////////////////////////////
	// Evaluate rhs
	//////////////////////////////////////////////////////////////////////////
	const RealCuda ki = (warm_start) ? m_data.fluid_data_cuda->kappa[i] : (m_data.fluid_data_cuda->densityAdv[i])*m_data.fluid_data_cuda->factor[i];

#ifdef USE_WARMSTART
	if (!warm_start) { m_data.fluid_data_cuda->kappa[i] += ki; }
#endif


	Vector3d v_i = Vector3d(0, 0, 0);
	const Vector3d &xi = m_data.fluid_data_cuda->pos[i];

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	int* neighbors_ptr = m_data.fluid_data_cuda->getNeighboursPtr(i);
	int* end_ptr = neighbors_ptr + m_data.fluid_data_cuda->getNumberOfNeighbourgs(i);
	{
		SPH::UnifiedParticleSet& body = *(m_data.fluid_data_cuda);
		while (neighbors_ptr != end_ptr)
		{
			const unsigned int neighborIndex = *neighbors_ptr++;
			const RealCuda kSum = (ki + ((warm_start) ? body.kappa[neighborIndex] : (body.densityAdv[neighborIndex])*body.factor[neighborIndex]));
			if (fabs(kSum) > m_eps)
			{
				// ki, kj already contain inverse density
				v_i += kSum * body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	if (fabs(ki) > m_eps)
	{
		end_ptr += m_data.fluid_data_cuda->getNumberOfNeighbourgs(i, 1);
		SPH::UnifiedParticleSet& body = *(m_data.boundaries_data_cuda);
		while (neighbors_ptr != end_ptr)
		{
			const unsigned int neighborIndex = *neighbors_ptr++;
			const Vector3d delta = ki * body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);

			v_i += delta;// ki already contains inverse density
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Dynamic bodies
	//////////////////////////////////////////////////////////////////////////
	if (fabs(ki) > m_eps)
	{
		end_ptr += m_data.fluid_data_cuda->getNumberOfNeighbourgs(i, 2);
		while (neighbors_ptr != end_ptr)
		{
			READ_DYNAMIC_BODIES_PARTICLES_INDEX(neighbors_ptr, bodyIndex, neighborIndex);
			SPH::UnifiedParticleSet& body = m_data.vector_dynamic_bodies_data_cuda[bodyIndex];
			Vector3d delta = ki * body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);

			v_i += delta;// ki already contains inverse density

			//we apply the force to the body particle (no invH since it has been fatorized at the end)
			delta *= -m_data.fluid_data_cuda->mass[i];
			atomicAdd(&(body.F[neighborIndex].x), delta.x);
			atomicAdd(&(body.F[neighborIndex].y), delta.y);
			atomicAdd(&(body.F[neighborIndex].z), delta.z);
		}
	}



	// Directly update velocities instead of storing pressure accelerations
	m_data.fluid_data_cuda->vel[i] += v_i*m_data.h_future;
}

__global__ void DFSPH_divergence_warmstart_init_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	m_data.fluid_data_cuda->kappaV[i] = MAX_MACRO_CUDA(m_data.fluid_data_cuda->kappaV[i] * m_data.h_ratio_to_past / 2, -0.25);
	computeDensityChange(m_data, i);
}
void cuda_divergence_warmstart_init(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_divergence_warmstart_init_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_divergence_warmstart_init failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}

template<bool warmstart> __global__ void DFSPH_divergence_compute_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	if (warmstart) {
		if (m_data.fluid_data_cuda->densityAdv[i] > 0.0) {
			divergenceSolveParticle<warmstart>(m_data, i);
		}
	}
	else {
		divergenceSolveParticle<warmstart>(m_data, i);
	}

}
template<bool warmstart> void cuda_divergence_compute(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_divergence_compute_kernel<warmstart> << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_divergence_compute failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}
template void cuda_divergence_compute<true>(SPH::DFSPHCData& data);
template void cuda_divergence_compute<false>(SPH::DFSPHCData& data);

__global__ void DFSPH_divergence_init_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	{
		///TODO when doing this kernel I can actually fuse the code for all those computation to limit the number
		///of time I read the particles positions
		computeDensityChange(m_data, i);

		//I can actually make the factor and desity computation here
		{
			//////////////////////////////////////////////////////////////////////////
			// Compute gradient dp_i/dx_j * (1/k)  and dp_j/dx_j * (1/k)
			//////////////////////////////////////////////////////////////////////////
			const Vector3d &xi = m_data.fluid_data_cuda->pos[i];
			RealCuda sum_grad_p_k = 0;
			Vector3d grad_p_i;
			grad_p_i.setZero();

			RealCuda density = m_data.fluid_data_cuda->mass[i] * m_data.W_zero;

			//////////////////////////////////////////////////////////////////////////
			// Fluid
			//////////////////////////////////////////////////////////////////////////
			int* neighbors_ptr = m_data.fluid_data_cuda->getNeighboursPtr(i);
			int* end_ptr = neighbors_ptr + m_data.fluid_data_cuda->getNumberOfNeighbourgs(i);
			{
				SPH::UnifiedParticleSet& body = *(m_data.fluid_data_cuda);
				while (neighbors_ptr != end_ptr)
				{
					const unsigned int neighborIndex = *neighbors_ptr++;
					const Vector3d &xj = body.pos[neighborIndex];
					density += body.mass[neighborIndex] * m_data.W(xi - xj);
					const Vector3d grad_p_j = body.mass[neighborIndex] * m_data.gradW(xi - xj);
					sum_grad_p_k += grad_p_j.squaredNorm();
					grad_p_i += grad_p_j;
				}
			}

			//////////////////////////////////////////////////////////////////////////
			// Boundary
			//////////////////////////////////////////////////////////////////////////
			end_ptr += m_data.fluid_data_cuda->getNumberOfNeighbourgs(i, 1);
			{
				SPH::UnifiedParticleSet& body = *(m_data.boundaries_data_cuda);
				while (neighbors_ptr != end_ptr)
				{
					const unsigned int neighborIndex = *neighbors_ptr++;
					const Vector3d &xj = body.pos[neighborIndex];
					density += body.mass[neighborIndex] * m_data.W(xi - xj);
					const Vector3d grad_p_j = body.mass[neighborIndex] * m_data.gradW(xi - xj);
					sum_grad_p_k += grad_p_j.squaredNorm();
					grad_p_i += grad_p_j;
				}
			}

			//////////////////////////////////////////////////////////////////////////
			// Dynamic bodies
			//////////////////////////////////////////////////////////////////////////
			//*
			end_ptr += m_data.fluid_data_cuda->getNumberOfNeighbourgs(i, 2);
			while (neighbors_ptr != end_ptr)
			{
				READ_DYNAMIC_BODIES_PARTICLES_INDEX(neighbors_ptr, bodyIndex, neighborIndex);
				SPH::UnifiedParticleSet& body = m_data.vector_dynamic_bodies_data_cuda[bodyIndex];
				const Vector3d &xj = body.pos[neighborIndex];
				density += body.mass[neighborIndex] * m_data.W(xi - xj);
				const Vector3d grad_p_j = body.mass[neighborIndex] * m_data.gradW(xi - xj);
				sum_grad_p_k += grad_p_j.squaredNorm();
				grad_p_i += grad_p_j;
			}
			//*/


			sum_grad_p_k += grad_p_i.squaredNorm();

			//////////////////////////////////////////////////////////////////////////
			// Compute pressure stiffness denominator
			//////////////////////////////////////////////////////////////////////////
			m_data.fluid_data_cuda->factor[i] = (-m_data.invH / (MAX_MACRO_CUDA(sum_grad_p_k, m_eps)));
			m_data.fluid_data_cuda->density[i] = density;

		}

#ifdef USE_WARMSTART_V
		m_data.fluid_data_cuda->kappaV[i] = 0;
#endif
	}

}
void cuda_divergence_init(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_divergence_init_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_divergence_init failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}

__global__ void DFSPH_divergence_loop_end_kernel(SPH::DFSPHCData m_data, RealCuda* avg_density_err) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	computeDensityChange(m_data, i);
	//atomicAdd(avg_density_err, m_data.densityAdv[i]);
}
RealCuda cuda_divergence_loop_end(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	RealCuda* avg_density_err = NULL;
	if (avg_density_err == NULL) {
		cudaMalloc(&(avg_density_err), sizeof(RealCuda));
	}

	DFSPH_divergence_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data, avg_density_err);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_divergence_loop_end failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
	static void     *d_temp_storage = NULL;
	static size_t   temp_storage_bytes = 0;

	if (d_temp_storage == NULL) {
		cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, data.fluid_data->densityAdv, avg_density_err, data.numFluidParticles);
		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
	}
	// Run sum-reduction
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, data.fluid_data->densityAdv, avg_density_err, data.numFluidParticles);

	RealCuda result = 0;
	gpuErrchk(cudaMemcpy(&result, avg_density_err, sizeof(RealCuda), cudaMemcpyDeviceToHost));

	return result;
}

__global__ void DFSPH_viscosityXSPH_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	//I set the gravitation directly here to lover the number of kernels
	Vector3d ai = Vector3d(0, 0, 0);
	const Vector3d &xi = m_data.fluid_data_cuda->pos[i];
	const Vector3d &vi = m_data.fluid_data_cuda->vel[i];

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	int* neighbors_ptr = m_data.fluid_data_cuda->getNeighboursPtr(i);
	int* end_ptr = neighbors_ptr + m_data.fluid_data_cuda->getNumberOfNeighbourgs(i);
	{
		SPH::UnifiedParticleSet& body = *(m_data.fluid_data_cuda);
		while (neighbors_ptr != end_ptr)
		{
			const unsigned int neighborIndex = *neighbors_ptr++;

			// Viscosity
			ai -= m_data.invH * m_data.viscosity * (body.mass[neighborIndex] / body.density[neighborIndex]) *
				(vi - body.vel[neighborIndex]) * m_data.W(xi - body.pos[neighborIndex]);

		}
	}

	m_data.fluid_data_cuda->acc[i] = m_data.gravitation + ai;
}
void cuda_viscosityXSPH(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_viscosityXSPH_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_viscosityXSPH failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}

__global__ void DFSPH_CFL_kernel(SPH::DFSPHCData m_data, RealCuda* maxVel) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	for (unsigned int i = 0; i < m_data.numFluidParticles; i++)
	{
		const RealCuda velMag = (m_data.fluid_data_cuda->vel[i] + m_data.fluid_data_cuda->acc[i] * m_data.h).squaredNorm();
		if (velMag > *maxVel)
			*maxVel = velMag;
	}
}

__global__ void DFSPH_CFLVelSquaredNorm_kernel(SPH::DFSPHCData m_data, RealCuda* sqaredNorm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	sqaredNorm[i] = (m_data.fluid_data_cuda->vel[i] + m_data.fluid_data_cuda->acc[i] * m_data.h).squaredNorm();
}

__global__ void DFSPH_CFLAdvanced_kernel(SPH::DFSPHCData m_data, RealCuda *max, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ RealCuda cache[256];


	RealCuda temp = 0;
	while (index + offset < n) {
		int i = index + offset;
		const RealCuda velMag = (m_data.fluid_data_cuda->vel[i] + m_data.fluid_data_cuda->acc[i] * m_data.h).squaredNorm();
		temp = fmaxf(temp, velMag);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	unsigned int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			cache[threadIdx.x] = MAX_MACRO_CUDA(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		while (atomicCAS(mutex, 0, 1) != 0);  //lock
		*max = MAX_MACRO_CUDA(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}
void cuda_CFL(SPH::DFSPHCData& m_data, const RealCuda minTimeStepSize, RealCuda m_cflFactor, RealCuda m_cflMaxTimeStepSize) {

	//we compute the square norm

	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

	RealCuda* out_buff;
	cudaMalloc(&(out_buff), sizeof(RealCuda));

	if (true) {

		//cub version
		static RealCuda* temp_buff = NULL;
		if (temp_buff == NULL) {
			cudaMallocManaged(&(temp_buff), m_data.numFluidParticles * sizeof(RealCuda));
		}
		int numBlocks = (m_data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_CFLVelSquaredNorm_kernel << <numBlocks, BLOCKSIZE >> > (m_data, temp_buff);

		cudaError_t cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cuda_cfl squared norm failed: %d\n", (int)cudaStatus);
			exit(1598);
		}

		// Determine temporary device storage requirements
		static void     *d_temp_storage = NULL;
		static size_t   temp_storage_bytes = 0;
		if (d_temp_storage == NULL) {
			cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, temp_buff, out_buff, m_data.numFluidParticles);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
		}
		// Run max-reduction
		cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, temp_buff, out_buff, m_data.numFluidParticles);

	}
	else {
		//manual
		int *d_mutex;
		cudaMalloc((void**)&d_mutex, sizeof(int));
		cudaMemset(d_mutex, 0, sizeof(float));

		int numBlocks = (m_data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_CFLAdvanced_kernel << < numBlocks, BLOCKSIZE >> > (m_data, out_buff, d_mutex, m_data.numFluidParticles);

		cudaError_t cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cuda_cfl failed: %d\n", (int)cudaStatus);
			exit(1598);
		}
		cudaFree(d_mutex);
	}

	RealCuda maxVel;
	cudaMemcpy(&maxVel, out_buff, sizeof(RealCuda), cudaMemcpyDeviceToHost);
	cudaFree(out_buff);

	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

	RealCuda h = m_data.h;

	// Approximate max. time step size 		
	h = m_cflFactor * .4 * (2.0*m_data.particleRadius / (sqrt(maxVel)));

	h = min(h, m_cflMaxTimeStepSize);
	h = max(h, minTimeStepSize);

	m_data.updateTimeStep(h);//*/


	std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();



	float time_search = std::chrono::duration_cast<std::chrono::nanoseconds> (t1 - t0).count() / 1000000.0f;
	float time_comp = std::chrono::duration_cast<std::chrono::nanoseconds> (t2 - t1).count() / 1000000.0f;

	printf("Time to do cfl (search,comp): %f    %f\n", time_search, time_comp);
}

__global__ void DFSPH_update_vel_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	m_data.fluid_data_cuda->vel[i] += m_data.h * m_data.fluid_data_cuda->acc[i];

#ifdef USE_WARMSTART	
	//done here to have one less kernel
	m_data.fluid_data_cuda->kappa[i] = MAX_MACRO_CUDA(m_data.fluid_data_cuda->kappa[i] * m_data.h_ratio_to_past2, -0.5);
#endif
}
void cuda_update_vel(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_update_vel_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_update_vel failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}

template<bool warmstart> __global__ void DFSPH_pressure_compute_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	pressureSolveParticle<warmstart>(m_data, i);

}
template<bool warmstart> void cuda_pressure_compute(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_pressure_compute_kernel<warmstart> << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_pressure_compute failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}
template void cuda_pressure_compute<true>(SPH::DFSPHCData& data);
template void cuda_pressure_compute<false>(SPH::DFSPHCData& data);

__global__ void DFSPH_pressure_init_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	computeDensityAdv(m_data, i);

	m_data.fluid_data_cuda->factor[i] *= m_data.invH_future;
#ifdef USE_WARMSTART
	m_data.fluid_data_cuda->kappa[i] = 0;
#endif

}
void cuda_pressure_init(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_pressure_init_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_pressure_init failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}

__global__ void DFSPH_pressure_loop_end_kernel(SPH::DFSPHCData m_data, RealCuda* avg_density_err) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	computeDensityAdv(m_data, i);
	//atomicAdd(avg_density_err, m_data.densityAdv[i]);
}
//*
__global__ void DFSPH_pressure_loop_end_kernel(int numFluidParticles, Vector3d* posFluid, Vector3d* velFluid, int* neighbourgs, int * numberOfNeighbourgs,
	RealCuda* mass, SPH::PrecomputedCubicKernelPerso m_kernel_precomp, RealCuda* boundaryPsi, Vector3d* posBoundary, Vector3d* velBoundary,
	SPH::UnifiedParticleSet* vector_dynamic_bodies_data_cuda, RealCuda* densityAdv, RealCuda* density, RealCuda h_future, RealCuda density0) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numFluidParticles) { return; }

	computeDensityAdv(i, posFluid, velFluid, neighbourgs, numberOfNeighbourgs,
		mass, m_kernel_precomp, boundaryPsi, posBoundary, velBoundary,
		vector_dynamic_bodies_data_cuda, densityAdv, density, h_future, density0);
}//*/

RealCuda cuda_pressure_loop_end(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;

	std::chrono::steady_clock::time_point p0 = std::chrono::steady_clock::now();
	static RealCuda* avg_density_err = NULL;
	if (avg_density_err == NULL) {
		cudaMalloc(&(avg_density_err), sizeof(RealCuda));
	}

	DFSPH_pressure_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data, avg_density_err);

	/*
	///LOL the detailed implementation is slower so no need to even think about developping data
	DFSPH_pressure_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data.numFluidParticles, data.posFluid, data.velFluid,
	data.neighbourgs, data.numberOfNeighbourgs,
	data.mass, data.m_kernel_precomp, data.boundaryPsi, data.posBoundary, data.velBoundary,
	data.vector_dynamic_bodies_data_cuda, data.densityAdv, data.density, data.h_future, data.density0);
	//*/

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_pressure_loop_end failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	std::chrono::steady_clock::time_point p1 = std::chrono::steady_clock::now();
	static void     *d_temp_storage = NULL;
	static size_t   temp_storage_bytes = 0;

	if (d_temp_storage == NULL) {
		cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, data.fluid_data->densityAdv, avg_density_err, data.numFluidParticles);
		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
	}
	// Run sum-reduction
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, data.fluid_data->densityAdv, avg_density_err, data.numFluidParticles);


	RealCuda result = 0;
	gpuErrchk(cudaMemcpy(&result, avg_density_err, sizeof(RealCuda), cudaMemcpyDeviceToHost));


	std::chrono::steady_clock::time_point p2 = std::chrono::steady_clock::now();
	float time1 = std::chrono::duration_cast<std::chrono::nanoseconds> (p1 - p0).count() / 1000000.0f;
	float time2 = std::chrono::duration_cast<std::chrono::nanoseconds> (p2 - p1).count() / 1000000.0f;

	//std::cout << "pressure loop end details: " << time1 << "  " << time2 << std::endl;

	return result;
}

__global__ void DFSPH_update_pos_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	m_data.fluid_data_cuda->pos[i] += m_data.h * m_data.fluid_data_cuda->vel[i];
}
void cuda_update_pos(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_update_pos_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_update_pos failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}


int cuda_divergenceSolve(SPH::DFSPHCData& m_data, const unsigned int maxIter, const RealCuda maxError) {
	//////////////////////////////////////////////////////////////////////////
	// Init parameters
	//////////////////////////////////////////////////////////////////////////

	const RealCuda h = m_data.h;
	const int numParticles = m_data.numFluidParticles;
	const RealCuda density0 = m_data.density0;

#ifdef USE_WARMSTART_V
	cuda_divergence_warmstart_init(m_data);
	cuda_divergence_compute<true>(m_data);
#endif


	//////////////////////////////////////////////////////////////////////////
	// Compute velocity of density change
	//////////////////////////////////////////////////////////////////////////
	cuda_divergence_init(m_data);


	unsigned int m_iterationsV = 0;

	//////////////////////////////////////////////////////////////////////////
	// Start solver
	//////////////////////////////////////////////////////////////////////////

	// Maximal allowed density fluctuation
	// use maximal density error divided by time step size
	const RealCuda eta = maxError * 0.01 * density0 / h;  // maxError is given in percent

	RealCuda avg_density_err = 0.0;
	while (((avg_density_err > eta) || (m_iterationsV < 1)) && (m_iterationsV < maxIter))
	{

		//////////////////////////////////////////////////////////////////////////
		// Perform Jacobi iteration over all blocks
		//////////////////////////////////////////////////////////////////////////	
		cuda_divergence_compute<false>(m_data);

		avg_density_err = cuda_divergence_loop_end(m_data);

		avg_density_err /= numParticles;
		m_iterationsV++;
	}

	return m_iterationsV;
}
int cuda_pressureSolve(SPH::DFSPHCData& m_data, const unsigned int m_maxIterations, const RealCuda m_maxError) {
	const RealCuda density0 = m_data.density0;
	const int numParticles = (int)m_data.numFluidParticles;
	RealCuda avg_density_err = 0.0;


	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();


#ifdef USE_WARMSTART		
	cuda_pressure_compute<true>(m_data);
#endif


	std::chrono::steady_clock::time_point m1 = std::chrono::steady_clock::now();

	//////////////////////////////////////////////////////////////////////////
	// Compute rho_adv
	//////////////////////////////////////////////////////////////////////////
	cuda_pressure_init(m_data);


	std::chrono::steady_clock::time_point m2 = std::chrono::steady_clock::now();


	unsigned int m_iterations = 0;

	//////////////////////////////////////////////////////////////////////////
	// Start solver
	//////////////////////////////////////////////////////////////////////////

	// Maximal allowed density fluctuation
	const RealCuda eta = m_maxError * 0.01 * density0;  // maxError is given in percent

	float time_3_1 = 0;
	float time_3_2 = 0;
	while (((avg_density_err > eta) || (m_iterations < 2)) && (m_iterations < m_maxIterations))
	{
		std::chrono::steady_clock::time_point p0 = std::chrono::steady_clock::now();
		cuda_pressure_compute<false>(m_data);
		std::chrono::steady_clock::time_point p1 = std::chrono::steady_clock::now();
		avg_density_err = cuda_pressure_loop_end(m_data);
		std::chrono::steady_clock::time_point p2 = std::chrono::steady_clock::now();
		avg_density_err /= numParticles;

		m_iterations++;

		time_3_1 += std::chrono::duration_cast<std::chrono::nanoseconds> (p1 - p0).count() / 1000000.0f;
		time_3_2 += std::chrono::duration_cast<std::chrono::nanoseconds> (p2 - p1).count() / 1000000.0f;
	}
	/*
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	float time_1 = std::chrono::duration_cast<std::chrono::nanoseconds> (m1 - start).count() / 1000000.0f;
	float time_2 = std::chrono::duration_cast<std::chrono::nanoseconds> (m2 - m1).count() / 1000000.0f;
	float time_3 = std::chrono::duration_cast<std::chrono::nanoseconds> (end - m2).count() / 1000000.0f;

	std::cout << "detail pressure solve (iter total (warm init actual_comp (t1 t2))): " <<m_iterations <<"  "<< time_1 + time_2 +time_3 <<
	"  (" << time_1 << "  " << time_2<< "  "<< time_3 <<"("<< time_3_1<<" "<< time_3_2<<") )" << std::endl;

	//*/

	return m_iterations;

}


template<unsigned int grid_size, bool z_curve>
__global__ void DFSPH_computeGridIdx_kernel(Vector3d* in, unsigned int* out, RealCuda kernel_radius, unsigned int num_particles) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_particles) { return; }

	if (z_curve) {

	}
	else {
		//the +50 is an offset so that I don't use the border of the grid
		//it allosw me to be sure that I won't have particles outside of the grid
		//the main thing is that their domain has negative position values
		//that +10 prevent having any negative index by positioning the bounding area of the particles 
		//incide the area  described by our cells
		Vector3d pos = (in[i] / kernel_radius) + 50;
		out[i] = COMPUTE_CELL_INDEX((int)pos.x, (int)pos.y, (int)pos.z);
		//	(int)pos.x + ((int)pos.y)*CELL_ROW_LENGTH + ((int)pos.z)*grid_size*grid_size;
	}
}

__global__ void DFSPH_Histogram_kernel(unsigned int* in, unsigned int* out, unsigned int num_particles) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_particles) { return; }

	atomicAdd(&(out[in[i]]), 1);

}

__global__ void DFSPH_setBufferValueToItself_kernel(unsigned int* buff, unsigned int buff_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= buff_size) { return; }

	buff[i] = i;
}

__global__ void DFSPH_setVector3dBufferToZero_kernel(Vector3d* buff, unsigned int buff_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= buff_size) { return; }

	buff[i] = Vector3d(0, 0, 0);
}

__global__ void DFSPH_neighborsSearch_kernel(unsigned int numFluidParticles, RealCuda radius,
	SPH::UnifiedParticleSet* fluid_data,
	SPH::UnifiedParticleSet* boundaries_data,
	SPH::UnifiedParticleSet* vect_dynamic_bodies, int nb_dynamic_bodies) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numFluidParticles) { return; }

	RealCuda radius_sq = radius;
	Vector3d pos = fluid_data->pos[i];
	Vector3d pos_cell = (pos / radius_sq) + 50; //on that line the radius is not yet squared
	int x = (int)pos_cell.x;
	int y = (int)pos_cell.y;
	int z = (int)pos_cell.z;
	radius_sq *= radius_sq;

	unsigned int nb_neighbors_fluid = 0;
	unsigned int nb_neighbors_boundary = 0;
	unsigned int nb_neighbors_dynamic_objects = 0;
	int* cur_neighbor_ptr = fluid_data->neighbourgs + i*MAX_NEIGHBOURS;
	//int neighbors_fluid[MAX_NEIGHBOURS];//doing it with local buffer was not faster
	//int neighbors_boundary[MAX_NEIGHBOURS];

#ifdef USE_COMPLETE
	///this version uses the morton indexes
	//this needsto be recoded since the data structure changed



#else
	///this version uses  standart indexes

	//since this version use the std index to be able to iterate on 3 successive cells
	//I can do the -1 at the start on x.
	//one thing: it x=0 then we can only iterate 2 cells at a time
	unsigned int successive_cells_count = (x > 0) ? 3 : 2;
	x = (x > 0) ? x - 1 : x;

#define ITER_CELLS_FOR_BODY(input_body,code){\
		const SPH::UnifiedParticleSet& body = input_body;\
		for (int k = -1; k < 2; ++k) {\
			for (int m = -1; m < 2; ++m) {\
				unsigned int cur_cell_id = COMPUTE_CELL_INDEX(x, y + m, z + k);\
				unsigned int end = body.neighborsDataSet->cell_start_end[cur_cell_id + successive_cells_count];\
				for (unsigned int cur_particle = body.neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {\
					unsigned int j = body.neighborsDataSet->p_id_sorted[cur_particle];\
					if ((pos - body.pos[j]).squaredNorm() < radius_sq) {\
						code\
					}\
				}\
			}\
		}\
	}
	
	//fluid
	ITER_CELLS_FOR_BODY(*fluid_data, if (i != j) {*cur_neighbor_ptr++ = j;	nb_neighbors_fluid++;})
	 
	//boundaries
	ITER_CELLS_FOR_BODY(*boundaries_data, *cur_neighbor_ptr++ = j; nb_neighbors_boundary++; )

	
	if (vect_dynamic_bodies != NULL) {
		for (int id_body = 0; id_body < nb_dynamic_bodies; ++id_body) {
			ITER_CELLS_FOR_BODY(vect_dynamic_bodies[id_body], 
				*cur_neighbor_ptr++ = WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX(id_body, j); nb_neighbors_dynamic_objects++; )
		}
	}
#endif


	fluid_data->numberOfNeighbourgs[3 * i] = nb_neighbors_fluid;
	fluid_data->numberOfNeighbourgs[3 * i + 1] = nb_neighbors_boundary;
	fluid_data->numberOfNeighbourgs[3 * i + 2] = nb_neighbors_dynamic_objects;

	//memcpy((neighbors_buff + i*MAX_NEIGHBOURS*2), neighbors_fluid, sizeof(int)*nb_neighbors_fluid);
	//memcpy((neighbors_buff + i*MAX_NEIGHBOURS * 2 + MAX_NEIGHBOURS), neighbors_boundary, sizeof(int)*nb_neighbors_boundary);


}

void cuda_neighborsSearchInternal_sortParticlesId(Vector3d* pos, RealCuda kernel_radius, int numParticles, void **d_temp_storage_pair_sort,
	size_t   &temp_storage_bytes_pair_sort, unsigned int* cell_id, unsigned int* cell_id_sorted,
	unsigned int* p_id, unsigned int* p_id_sorted) {
	cudaError_t cudaStatus;

	/*
	//some test for the definition domain (it is just for debugging purposes)
	//check for negatives values
	for (int i = 0; i < numParticles; ++i) {
	Vector3d temp = (pos[i] / kernel_radius) + 2;
	if (temp.x <= 0 || temp.y <= 0 || temp.z <= 0 ) {
	fprintf(stderr, "negative coordinates: %d\n", (int)i);
	exit(1598);
	}
	}


	//find the bounding box of the particles
	Vector3d min = pos[0];
	Vector3d max = pos[0];
	for (int i = 0; i < numParticles; ++i) {

	if (pos[i].x < min.x) { min.x = pos[i].x; }
	if (pos[i].y < min.y) { min.y = pos[i].y; }
	if (pos[i].z < min.z) { min.z = pos[i].z; }

	if (pos[i].x > max.x) { max.x = pos[i].x; }
	if (pos[i].y > max.y) { max.y = pos[i].y; }
	if (pos[i].z > max.z) { max.z = pos[i].z; }

	}
	fprintf(stderr, "min: %f // %f // %f\n", min.x, min.y, min.z);
	fprintf(stderr, "max: %f // %f // %f\n", max.x, max.y, max.z);
	fprintf(stderr, "description: %f\n", CELL_ROW_LENGTH*kernel_radius);
	exit(1598);
	//*/
	int numBlocks = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;


	//compute the idx of the cell for each particles
	DFSPH_computeGridIdx_kernel<CELL_ROW_LENGTH, false> << <numBlocks, BLOCKSIZE >> > (pos, cell_id,
		kernel_radius, numParticles);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "idxs failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	//do the actual sort
	//first Determine temporary device storage requirements
	if ((*d_temp_storage_pair_sort) == NULL) {
		temp_storage_bytes_pair_sort = 0;
		cub::DeviceRadixSort::SortPairs(*d_temp_storage_pair_sort, temp_storage_bytes_pair_sort,
			cell_id, cell_id_sorted, p_id, p_id_sorted, numParticles);
		// Allocate temporary storage
		cudaMalloc(d_temp_storage_pair_sort, temp_storage_bytes_pair_sort);

	}

	// Run sorting operation
	cub::DeviceRadixSort::SortPairs(*d_temp_storage_pair_sort, temp_storage_bytes_pair_sort,
		cell_id, cell_id_sorted, p_id, p_id_sorted, numParticles);
	//*/


	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sort failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

}

void cuda_neighborsSearchInternal_computeCellStartEnd(int numParticles, unsigned int* cell_id_sorted,
	unsigned int* hist, void **d_temp_storage_cumul_hist, size_t   &temp_storage_bytes_cumul_hist, unsigned int* cell_start_end) {
	cudaError_t cudaStatus;
	int numBlocks = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;


	//Now we need to determine the start and end of each cell
	//init the histogram values. Maybe doing it wiith thrust fill is faster.
	//the doc is not realy clear
	cudaMemset(hist, 0, (CELL_COUNT + 1) * sizeof(unsigned int));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "histogram value reset failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	//compute the actual histogram (done here with atomic adds)
	DFSPH_Histogram_kernel << <numBlocks, BLOCKSIZE >> > (cell_id_sorted, hist, numParticles);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cerr << "histogram failed: " << (int)cudaStatus << std::endl;
		exit(1598);
	}

	//transformour histogram to a cumulative histogram to have  the start and end of each cell
	//note: the exlusive sum make so that each cell will contains it's start value

	if ((*d_temp_storage_cumul_hist) == NULL) {
		temp_storage_bytes_cumul_hist = 0;
		//get the necessary size
		cub::DeviceScan::ExclusiveSum(*d_temp_storage_cumul_hist, temp_storage_bytes_cumul_hist, hist, cell_start_end, (CELL_COUNT + 1));
		// Allocate temporary storage
		cudaMalloc(d_temp_storage_cumul_hist, temp_storage_bytes_cumul_hist);
	}
	// Run exclusive prefix sum
	cub::DeviceScan::ExclusiveSum(*d_temp_storage_cumul_hist, temp_storage_bytes_cumul_hist, hist, cell_start_end, (CELL_COUNT + 1));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cumulative histogram failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}



//this is the bases for all kernels based function
template<typename T>
__global__ void DFSPH_sortFromIndex_kernel(T* in, T* out, unsigned int* index, unsigned int nbElements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nbElements) { return; }

	out[i] = in[index[i]];
}


#include <sstream>
void cuda_sortData(SPH::UnifiedParticleSet& particleSet, SPH::NeighborsSearchDataSet& neighborsDataSet) {
	//*
	unsigned int numParticles = neighborsDataSet.numParticles;
	int numBlocks = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	unsigned int *p_id_sorted = neighborsDataSet.p_id_sorted;

	Vector3d* intermediate_buffer_v3d = NULL;
	cudaMallocManaged(&(intermediate_buffer_v3d), numParticles * sizeof(Vector3d));
	RealCuda* intermediate_buffer_real = NULL;
	cudaMalloc(&(intermediate_buffer_real), numParticles * sizeof(RealCuda));

	//pos
	DFSPH_sortFromIndex_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (particleSet.pos, intermediate_buffer_v3d, p_id_sorted, numParticles);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(particleSet.pos, intermediate_buffer_v3d, numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
	
	//vel
	DFSPH_sortFromIndex_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (particleSet.vel, intermediate_buffer_v3d, p_id_sorted, numParticles);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(particleSet.vel, intermediate_buffer_v3d, numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

	//mass
	DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.mass, intermediate_buffer_real, p_id_sorted, numParticles);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(particleSet.mass, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

	if (particleSet.velocity_impacted_by_fluid_solver) {
		//kappa
		DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.kappa, intermediate_buffer_real, p_id_sorted, numParticles);
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(particleSet.kappa, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

		//kappav
		DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.kappaV, intermediate_buffer_real, p_id_sorted, numParticles);
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(particleSet.kappaV, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
	}


	cudaFree(intermediate_buffer_v3d); intermediate_buffer_v3d = NULL;
	cudaFree(intermediate_buffer_real); intermediate_buffer_real = NULL;


	//now that everything is sorted we can set each particle index to itself
	gpuErrchk(cudaMemcpy(p_id_sorted, neighborsDataSet.p_id, numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

	std::cout << "particle set sorting done" << std::endl;
}




void cuda_neighborsSearch(SPH::DFSPHCData& data) {

	std::chrono::steady_clock::time_point begin_global = std::chrono::steady_clock::now();
	static unsigned int time_count = 0;
	float time_global;
	static float time_avg_global = 0;
	time_count++;

	cudaError_t cudaStatus;
	{

		float time;
		static float time_avg = 0;
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


		//first let's generate the cell start end for the dynamic bodies
		for (int i = 0; i < data.numDynamicBodies; ++i) {
			SPH::UnifiedParticleSet& body = data.vector_dynamic_bodies_data[i];
			body.initNeighborsSearchData(data.m_kernel_precomp.getRadius(), false);
		}
		

		//now update the cell start end of the fluid particles
		{

			//since it the init iter I'll sort both even if it's the boundaries
			static int step_count = 0;
			step_count++;

			data.fluid_data->initNeighborsSearchData(data.m_kernel_precomp.getRadius(), (step_count%25)==0);


			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "before neighbors search: %d\n", (int)cudaStatus);
				exit(1598);
			}


		}

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		time = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() / 1000000.0f;

		time_avg += time;
		//printf("Time to generate cell start end: %f ms   avg: %f ms \n", time, time_avg / time_count);
	}
	//and we can now do the actual search of the neaighbor for eahc fluid particle
	if (true)
	{
		float time;
		static float time_avg = 0;

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		//cuda way
		int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;


		DFSPH_neighborsSearch_kernel << <numBlocks, BLOCKSIZE >> > (data.numFluidParticles,
			data.m_kernel_precomp.getRadius(), 
			data.fluid_data_cuda, 
			data.boundaries_data_cuda,
			data.vector_dynamic_bodies_data_cuda, data.numDynamicBodies);



		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			std::cerr << "cuda neighbors search failed: " << (int)cudaStatus << std::endl;
			exit(1598);
		}

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		time = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() / 1000000.0f;

		time_avg += time;
		//printf("Time to generate neighbors buffers: %f ms   avg: %f ms \n", time, time_avg / time_count);

		/*
		//a simple check to know the max nbr of neighbors
		static int absolute_max = 0;
		int max = 0;

		static int absolute_max_d[3] = { 0 };
		int max_d[3] = { 0 };



		for (int j = 0; j < data.numFluidParticles; j++)
		{
		//check the global value
		int count_neighbors = 0;
		for (int k = 0; k < 2; ++k) {
		count_neighbors += data.getNumberOfNeighbourgs(j, k);
		}
		if (count_neighbors > max)max = count_neighbors;

		//chekc the max for each category
		for (unsigned int k = 0; k < 3; ++k) {
		if ((int)data.getNumberOfNeighbourgs(j,k) > max_d[k])max_d[k] = data.getNumberOfNeighbourgs(j,k);
		}

		}
		if (max>absolute_max)absolute_max = max;
		for (unsigned int k = 0; k < 3; ++k) {
		if (max_d[k]>absolute_max_d[k])absolute_max_d[k] = max_d[k];
		}
		printf("max nbr of neighbors %d  (%d) \n", absolute_max, max);
		printf("max nbr of neighbors %d  (%d)      absolute max  fluid // boundaries // bodies   %d // %d // %d\n",
		absolute_max, max, absolute_max_d[0], absolute_max_d[1], absolute_max_d[2]);
		//*/
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	time_global = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin_global).count() / 1000000.0f;

	time_avg_global += time_global;
	//printf("time taken by the neighbor function: %f ms   avg: %f ms \n", time_global, time_avg_global / time_count);
}



void cuda_initNeighborsSearchDataSet(SPH::UnifiedParticleSet& particleSet, SPH::NeighborsSearchDataSet& dataSet,
	RealCuda kernel_radius, bool sortBuffers){

	//com the id
	cuda_neighborsSearchInternal_sortParticlesId(particleSet.pos, kernel_radius, dataSet.numParticles,
		&dataSet.d_temp_storage_pair_sort, dataSet.temp_storage_bytes_pair_sort, dataSet.cell_id, dataSet.cell_id_sorted,
		dataSet.p_id, dataSet.p_id_sorted);

	//since it the init iter I'll sort both even if it's the boundaries
	if (sortBuffers) {
		cuda_sortData(particleSet, dataSet);
	}


	//and now I cna compute the start and end of each cell :)
	cuda_neighborsSearchInternal_computeCellStartEnd(dataSet.numParticles, dataSet.cell_id_sorted, dataSet.hist,
		&dataSet.d_temp_storage_cumul_hist, dataSet.temp_storage_bytes_cumul_hist, dataSet.cell_start_end);

}


void cuda_renderFluid(SPH::DFSPHCData& data) {
	cuda_opengl_renderParticleSet(*data.fluid_data->renderingData,data.numFluidParticles);
}



void cuda_renderBoundaries(SPH::DFSPHCData& data, bool renderWalls) {
	if (renderWalls) {
		cuda_opengl_renderParticleSet(*(data.boundaries_data->renderingData), data.boundaries_data->numParticles);
	}

	for (int i = 0; i < data.numDynamicBodies; ++i) {
		std::cout<<"test"<<i<<std::endl;
		SPH::UnifiedParticleSet& body= data.vector_dynamic_bodies_data[i];
		cuda_opengl_renderParticleSet(*body.renderingData, body.numParticles);
	}
}

/*
THE NEXT FUNCTIONS ARE FOR THE RENDERING
*/


void cuda_opengl_initParticleRendering(ParticleSetRenderingData& renderingData, unsigned int numParticles,
	Vector3d** pos, Vector3d** vel) {
	glGenVertexArrays(1, &renderingData.vaoFluid); // Créer le VAO
	glBindVertexArray(renderingData.vaoFluid); // Lier le VAO pour l'utiliser


	glGenBuffers(1, &renderingData.pos_buffer);
	// selectionne le buffer pour l'initialiser
	glBindBuffer(GL_ARRAY_BUFFER, renderingData.pos_buffer);
	// dimensionne le buffer actif sur array_buffer, l'alloue et l'initialise avec les positions des sommets de l'objet
	glBufferData(GL_ARRAY_BUFFER,
		/* length */	numParticles * sizeof(Vector3d),
		/* data */      NULL,
		/* usage */     GL_DYNAMIC_DRAW);
	//set it to the attribute
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FORMAT, GL_FALSE, 0, 0);

	glGenBuffers(1, &renderingData.vel_buffer);
	// selectionne le buffer pour l'initialiser
	glBindBuffer(GL_ARRAY_BUFFER, renderingData.vel_buffer);
	// dimensionne le buffer actif sur array_buffer, l'alloue et l'initialise avec les positions des sommets de l'objet
	glBufferData(GL_ARRAY_BUFFER,
		/* length */	numParticles * sizeof(Vector3d),
		/* data */      NULL,
		/* usage */     GL_DYNAMIC_DRAW);
	//set it to the attribute
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FORMAT, GL_FALSE, 0, 0);

	// nettoyage
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Registration with CUDA.
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&renderingData.pos, renderingData.pos_buffer, cudaGraphicsRegisterFlagsNone));
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&renderingData.vel, renderingData.vel_buffer, cudaGraphicsRegisterFlagsNone));

	//link the pos and vel buffer to cuda
	gpuErrchk(cudaGraphicsMapResources(1, &renderingData.pos, 0));
	gpuErrchk(cudaGraphicsMapResources(1, &renderingData.vel, 0));

	//set the openglbuffer for direct use in cuda
	Vector3d* vboPtr = NULL;
	size_t size = 0;

	// pos
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &size, renderingData.pos));//get cuda ptr
	*pos = vboPtr;

	// vel
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &size, renderingData.vel));//get cuda ptr
	*vel = vboPtr;

}

void cuda_opengl_renderParticleSet(ParticleSetRenderingData& renderingData, unsigned int numParticles) {

	//unlink the pos and vel buffer from cuda
	gpuErrchk(cudaGraphicsUnmapResources(1, &(renderingData.pos), 0));
	gpuErrchk(cudaGraphicsUnmapResources(1, &(renderingData.vel), 0));

	//Actual opengl rendering
	// link the vao
	glBindVertexArray(renderingData.vaoFluid);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//show it
	glDrawArrays(GL_POINTS, 0, numParticles);

	// unlink the vao
	glBindVertexArray(0);

	//link the pos and vel buffer to cuda
	gpuErrchk(cudaGraphicsMapResources(1, &renderingData.pos, 0));
	gpuErrchk(cudaGraphicsMapResources(1, &renderingData.vel, 0));

}






/*
THE NEXT FUNCTIONS ARE FOR THE MEMORY ALLOCATION
*/


void allocate_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& container) {

	//cudaMalloc(&(container.pos), container.numParticles * sizeof(Vector3d)); //use opengl buffer with cuda interop
	//cudaMalloc(&(container.vel), container.numParticles * sizeof(Vector3d)); //use opengl buffer with cuda interop
	cudaMalloc(&(container.mass), container.numParticles * sizeof(RealCuda));


	if (container.has_factor_computation) {
		//*
		cudaMallocManaged(&(container.numberOfNeighbourgs), container.numParticles * 3 * sizeof(int));
		cudaMalloc(&(container.neighbourgs), container.numParticles * MAX_NEIGHBOURS * sizeof(int));

		cudaMalloc(&(container.density), container.numParticles * sizeof(RealCuda));
		cudaMalloc(&(container.factor), container.numParticles * sizeof(RealCuda));
		cudaMalloc(&(container.densityAdv), container.numParticles * sizeof(RealCuda));
		
		if (container.velocity_impacted_by_fluid_solver) {
			cudaMalloc(&(container.acc), container.numParticles * sizeof(Vector3d));
			cudaMalloc(&(container.kappa), container.numParticles * sizeof(RealCuda));
			cudaMalloc(&(container.kappaV), container.numParticles * sizeof(RealCuda));
		}
		//*/
	}

	if (container.is_dynamic_object) {
		cudaMalloc(&(container.F), container.numParticles * sizeof(Vector3d));
	}

}

void load_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& container, Vector3d* pos, Vector3d* vel, RealCuda* mass) {
	gpuErrchk(cudaMemcpy(container.pos, pos, container.numParticles * sizeof(Vector3d), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(container.vel, vel, container.numParticles * sizeof(Vector3d), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(container.mass, mass, container.numParticles * sizeof(RealCuda), cudaMemcpyHostToDevice));

	if (container.is_dynamic_object) {
		int numBlocks = (container.numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_setVector3dBufferToZero_kernel << <numBlocks, BLOCKSIZE >> > (container.F, container.numParticles);
	}

	if (container.has_factor_computation) {
		
		if (container.velocity_impacted_by_fluid_solver) {
			gpuErrchk(cudaMemset(container.kappa, 0, container.numParticles * sizeof(RealCuda)));
			gpuErrchk(cudaMemset(container.kappaV, 0, container.numParticles * sizeof(RealCuda)));
		}
	}

}

void read_rigid_body_force_cuda(SPH::UnifiedParticleSet& container) {
	if (container.is_dynamic_object) {
		gpuErrchk(cudaMemcpy(container.F_cpu, container.F, container.numParticles * sizeof(Vector3d), cudaMemcpyDeviceToHost));
	}
}

void allocate_and_copy_UnifiedParticleSet_vector_cuda(SPH::UnifiedParticleSet** out_vector, SPH::UnifiedParticleSet* in_vector, int numSets) {
	//before to do anythng we need to make a copy of the data structure since
	//we will have to change the neighborsdataset from the cpu to the gpu
	//*
	SPH::UnifiedParticleSet* temp;
	temp = new SPH::UnifiedParticleSet[numSets];
	std::copy(in_vector, in_vector + numSets, temp);
	
	for (int i = 0; i < numSets; ++i) {
		SPH::UnifiedParticleSet& body = temp[i];
		gpuErrchk(cudaMalloc(&(body.neighborsDataSet), sizeof(SPH::NeighborsSearchDataSet)));

		gpuErrchk(cudaMemcpy(body.neighborsDataSet, in_vector[i].neighborsDataSet,
			sizeof(SPH::NeighborsSearchDataSet), cudaMemcpyHostToDevice));

	}
	//*/

	gpuErrchk(cudaMalloc(out_vector, numSets * sizeof(SPH::UnifiedParticleSet)));

	gpuErrchk(cudaMemcpy(*out_vector, temp,	numSets * sizeof(SPH::UnifiedParticleSet), cudaMemcpyHostToDevice));
}



void allocate_precomputed_kernel_managed(SPH::PrecomputedCubicKernelPerso& kernel, bool minimize_managed) {

	if (minimize_managed) {
		cudaMalloc(&(kernel.m_W), kernel.m_resolution * sizeof(RealCuda));
		cudaMalloc(&(kernel.m_gradW), (kernel.m_resolution + 1) * sizeof(RealCuda));
	}
	else {
		fprintf(stderr, "trying to use managed buffers for the kernels\n");
		exit(1256);
		//cudaMallocManaged(&(kernel.m_W), kernel.m_resolution * sizeof(RealCuda));
		//cudaMallocManaged(&(kernel.m_gradW), (kernel.m_resolution + 1) * sizeof(RealCuda));
	}
}


void init_precomputed_kernel_from_values(SPH::PrecomputedCubicKernelPerso& kernel, RealCuda* w, RealCuda* grad_W) {
	cudaError_t cudaStatus;
	//W
	cudaStatus = cudaMemcpy(kernel.m_W,
		w,
		kernel.m_resolution * sizeof(RealCuda),
		cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "precomputed initialization of W from data failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	//grad W
	cudaStatus = cudaMemcpy(kernel.m_gradW,
		grad_W,
		(kernel.m_resolution + 1) * sizeof(RealCuda),
		cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "precomputed initialization of grad W from data failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

}


void allocate_neighbors_search_data_set(SPH::NeighborsSearchDataSet& dataSet) {

	//allocatethe mme for fluid particles
	cudaMallocManaged(&(dataSet.cell_id), dataSet.numParticles * sizeof(unsigned int));
	cudaMallocManaged(&(dataSet.cell_id_sorted), dataSet.numParticles * sizeof(unsigned int));
	cudaMallocManaged(&(dataSet.local_id), dataSet.numParticles * sizeof(unsigned int));
	cudaMallocManaged(&(dataSet.p_id), dataSet.numParticles * sizeof(unsigned int));
	cudaMallocManaged(&(dataSet.p_id_sorted), dataSet.numParticles * sizeof(unsigned int));
	cudaMallocManaged(&(dataSet.cell_start_end), (CELL_COUNT + 1) * sizeof(unsigned int));
	cudaMallocManaged(&(dataSet.hist), (CELL_COUNT + 1) * sizeof(unsigned int));

	//init variables for cub calls
	dataSet.d_temp_storage_pair_sort = NULL;
	dataSet.temp_storage_bytes_pair_sort = 0;
	dataSet.d_temp_storage_cumul_hist = NULL;
	dataSet.temp_storage_bytes_cumul_hist = 0;

	//reset the particle id
	int numBlocks = (dataSet.numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_setBufferValueToItself_kernel << <numBlocks, BLOCKSIZE >> > (dataSet.p_id, dataSet.numParticles);
	DFSPH_setBufferValueToItself_kernel << <numBlocks, BLOCKSIZE >> > (dataSet.p_id_sorted, dataSet.numParticles);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "p_id init idxs failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	dataSet.internal_buffers_allocated = true;
}


void release_neighbors_search_data_set(SPH::NeighborsSearchDataSet& dataSet, bool keep_result_buffers) {
	//allocatethe mme for fluid particles
	cudaFree(dataSet.cell_id); dataSet.cell_id = NULL;
	cudaFree(dataSet.local_id); dataSet.local_id = NULL;
	cudaFree(dataSet.p_id); dataSet.p_id = NULL;
	cudaFree(dataSet.cell_id_sorted); dataSet.cell_id_sorted = NULL;
	cudaFree(dataSet.hist); dataSet.hist = NULL;

	//init variables for cub calls
	cudaFree(dataSet.d_temp_storage_pair_sort);
	dataSet.d_temp_storage_pair_sort = NULL;
	dataSet.temp_storage_bytes_pair_sort = 0;
	cudaFree(dataSet.d_temp_storage_cumul_hist);
	dataSet.d_temp_storage_cumul_hist = NULL;
	dataSet.temp_storage_bytes_cumul_hist = 0;

	dataSet.internal_buffers_allocated = false;

	if (!keep_result_buffers) {
		cudaFree(dataSet.p_id_sorted); dataSet.p_id_sorted = NULL;
		cudaFree(dataSet.cell_start_end); dataSet.cell_start_end = NULL;
	}
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
