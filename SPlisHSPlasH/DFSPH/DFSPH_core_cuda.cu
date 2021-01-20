#include "DFSPH_core_cuda.h"


#include <stdio.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <sstream>
#include <fstream>

#include "DFSPH_define_cuda.h"
#include "DFSPH_macro_cuda.h"
#include "DFSPH_static_variables_structure_cuda.h"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DFSPH_c_arrays_structure.h"
#include "cub.cuh"



#include <curand.h>
#include <curand_kernel.h>

#include "basic_kernels_cuda.cuh"

//#include "SPH_memory_storage_precomp_kernels.cuh"



////////////////////////////////////////////////////
/////////       constant memory kernel /////////////
////////////////////////////////////////////////////

#include "SPlisHSPlasH\BasicTypes.h"
#include <string>
#include <vector>

#include "SPlisHSPlasH\Vector.h"
#include "SPlisHSPlasH\Quaternion.h"

#include "DFSPH_define_c.h"
#include "cuda_runtime.h"

namespace CoreCuda
{
	__global__ void init_buffer_kernel(Vector3d* buff, unsigned int size, Vector3d val) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= size) { return; }

		buff[i] = val;
	}
}

//#include "SPH_memory_storage_precomp_kernels.cuh"
//*
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif//*/

#ifdef PRECOMPUTED_KERNELS_USE_CONSTANT_MEMORY

__constant__ RealCuda m_W[PRECOMPUTED_KERNELS_SAMPLE_COUNT];
__constant__ RealCuda m_gradW[PRECOMPUTED_KERNELS_SAMPLE_COUNT];
__constant__ RealCuda m_radius;
__constant__ RealCuda m_radius2;
__constant__ RealCuda m_invStepSize;





__device__  RealCuda get_constant_W_cuda(const SPH::Vector3d &r)
{
	RealCuda res = 0.0;
	const RealCuda r2 = r.squaredNorm();
	if (r2 <= m_radius2)
	{
		const RealCuda r = sqrt(r2);
		const unsigned int pos = (unsigned int)(r * m_invStepSize);
		res = m_W[pos];
	}
	return res;
}

//*
__device__  RealCuda get_constant_W_cuda(const RealCuda r)
{
	RealCuda res = 0.0;
	if (r <= m_radius)
	{
		const unsigned int pos = (unsigned int)(r * m_invStepSize);
		res = m_W[pos];
	}
	return res;
}
__device__  SPH::Vector3d get_constant_grad_W_cuda(const SPH::Vector3d &r)
{
	SPH::Vector3d res;
	const RealCuda r2 = r.squaredNorm();
	if (r2 <= m_radius2)
	{
		const RealCuda rl = sqrt(r2);
		const unsigned int pos = (unsigned int)(rl * m_invStepSize);
		res = m_gradW[pos] * r;
	}
	else
		res.setZero();

	return res;
}
//*/


#include "SPH_memory_storage_precomp_kernels.h"
#include "SPH_other_systems_cuda.h"
#include <iostream>

void writte_to_precomp_kernel(RealCuda* W_i, RealCuda* gradW_i, RealCuda radius, RealCuda radius2, RealCuda invStepSize) {
	cudaMemcpyToSymbol(m_W, W_i, sizeof(RealCuda) * PRECOMPUTED_KERNELS_SAMPLE_COUNT);
	read_last_error_cuda("test");
	cudaMemcpyToSymbol(m_gradW, gradW_i, sizeof(RealCuda) * PRECOMPUTED_KERNELS_SAMPLE_COUNT);
	read_last_error_cuda("test");
	cudaMemcpyToSymbol(m_radius, &radius, sizeof(RealCuda));
	read_last_error_cuda("test");
	cudaMemcpyToSymbol(m_radius2, &radius2, sizeof(RealCuda));
	read_last_error_cuda("test");
	cudaMemcpyToSymbol(m_invStepSize, &invStepSize, sizeof(RealCuda));
	read_last_error_cuda("test");
	cudaDeviceSynchronize();
	/*
	for (int i = 0; i < PRECOMPUTED_KERNELS_SAMPLE_COUNT; ++i) {
	std::cout << "kernel values: " << W_i[i] << "  " << gradW_i[i] << std::endl;
	}

	test_constant_mem_precomp_kernel_cuda();
	//*/
}
#include "DFSPH_macro_cuda.h"

__global__ void test_constant_mem_precomp_kernel_kernel(RealCuda* W, SPH::Vector3d* gradW, RealCuda* r, RealCuda* r2, RealCuda* invd) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= PRECOMPUTED_KERNELS_SAMPLE_COUNT) { return; }

	const RealCuda posX = 1.0 / m_invStepSize * (RealCuda)i;
	SPH::Vector3d distance = Vector3d(posX, 0.0, 0.0);

	W[i] = KERNEL_W("data", distance);
	gradW[i] = KERNEL_GRAD_W("data", distance);

	if (i == 0) {
		*r = m_radius;
		*r2 = m_radius2;
		*invd = m_invStepSize;
	}
}

void test_constant_mem_precomp_kernel_cuda() {
	RealCuda* W;
	SPH::Vector3d* gradW;
	RealCuda* r;
	RealCuda* r2;
	RealCuda* invd;

	cudaMallocManaged(&(W), sizeof(RealCuda) * PRECOMPUTED_KERNELS_SAMPLE_COUNT);
	cudaMallocManaged(&(gradW), sizeof(SPH::Vector3d) * PRECOMPUTED_KERNELS_SAMPLE_COUNT);
	cudaMallocManaged(&(r), sizeof(RealCuda));
	cudaMallocManaged(&(r2), sizeof(RealCuda));
	cudaMallocManaged(&(invd), sizeof(RealCuda));

	{//fluid
		int numBlocks = (PRECOMPUTED_KERNELS_SAMPLE_COUNT + BLOCKSIZE - 1) / BLOCKSIZE;
		test_constant_mem_precomp_kernel_kernel << <numBlocks, BLOCKSIZE >> > (W, gradW, r, r2, invd);
	}
	cudaDeviceSynchronize();

	for (int i = 0; i < PRECOMPUTED_KERNELS_SAMPLE_COUNT; ++i) {
		std::cout << "kernel values: " << W[i] << "  " << gradW[i].x << std::endl;
	}

	std::cout << "end_values: " << *r << "  " << *r2 << "  " << *invd << std::endl;

	CUDA_FREE_PTR(W);
	CUDA_FREE_PTR(gradW);
	CUDA_FREE_PTR(r);
	CUDA_FREE_PTR(r2);
	CUDA_FREE_PTR(invd);
}
#endif // !BLOCKER//see the macro_cuda_file_for an explanaitions







////////////////////////////////////////////////////
/////////       DIVERGENCE SOLVER      /////////////
////////////////////////////////////////////////////



template <bool ignore_when_no_fluid_near>
__global__ void DFSPH_divergence_warmstart_init_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (m_data.restriction_mode == 1) {
		if ((particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) &&
			(particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE_NEIGHBORS)) {
			return;
		}
	}else if (m_data.restriction_mode == 2) {
		if (i >= (m_data.count_active + m_data.count_active_neighbors)) { return; }
	}

	if (ignore_when_no_fluid_near) {
		if (particleSet->getNumberOfNeighbourgs(i) == 0) {
			return;
		}
	}

	particleSet->kappaV[i] = MAX_MACRO_CUDA(particleSet->kappaV[i] * m_data.h_ratio_to_past / 2, -0.5);
	//computeDensityChange(m_data, i);


	//I can actually make the factor and density computation here
	{
#ifndef STORE_PARTICLE_NEIGHBORS
		unsigned int numNeighbors = 0;
	#define computeDensityChange_additional numNeighbors++;
#else
	#define computeDensityChange_additional  
#endif // !STORE_PARTICLE_NEIGHBORS


		//////////////////////////////////////////////////////////////////////////
		// Compute gradient dp_i/dx_j * (1/k)  and dp_j/dx_j * (1/k)
		//////////////////////////////////////////////////////////////////////////
		const Vector3d &xi = particleSet->pos[i];
		const Vector3d &vi = particleSet->vel[i];
		RealCuda sum_grad_p_k = 0;
		Vector3d grad_p_i;
		grad_p_i.setZero();

		RealCuda density =  particleSet->getMass(i) * m_data.W_zero;
		RealCuda densityAdv = 0;

		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		ITER_NEIGHBORS_INIT(m_data,particleSet, i);

		ITER_NEIGHBORS_FLUID(m_data, particleSet,
			i,
			const Vector3d &xj = body.pos[neighborIndex];
		density += body.getMass(neighborIndex) * KERNEL_W(m_data,xi - xj);
		const Vector3d grad_p_j = body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - xj);
		sum_grad_p_k += grad_p_j.squaredNorm();
		grad_p_i += grad_p_j;
		densityAdv += (vi - body.vel[neighborIndex]).dot(grad_p_j);
		computeDensityChange_additional
		);


		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////

#ifdef BENDER2019_BOUNDARIES

		const Vector3d& xj = particleSet->X_rigids[i];
		const RealCuda mass = particleSet->V_rigids[i] * particleSet->density0;
		density += mass * KERNEL_W(m_data, xi - xj);
		const Vector3d grad_p_j = mass * KERNEL_GRAD_W(m_data, xi - xj);
		sum_grad_p_k += grad_p_j.squaredNorm();
		grad_p_i += grad_p_j;
		//No Vj for statics boundaries
		densityAdv += (vi).dot(grad_p_j);


#else

		ITER_NEIGHBORS_BOUNDARIES(m_data, particleSet,
			i,
			const Vector3d &xj = body.pos[neighborIndex];
		density += body.getMass(neighborIndex) * KERNEL_W(m_data,xi - xj);
		const Vector3d grad_p_j = body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - xj);
		sum_grad_p_k += grad_p_j.squaredNorm();
		grad_p_i += grad_p_j;
		densityAdv += (vi - body.vel[neighborIndex]).dot(grad_p_j);
		computeDensityChange_additional
		);

#endif

		//////////////////////////////////////////////////////////////////////////
		// Dynamic bodies
		//////////////////////////////////////////////////////////////////////////
		//*
		ITER_NEIGHBORS_SOLIDS(m_data, particleSet,
			i,
			const Vector3d &xj = body.pos[neighborIndex];
		density += body.getMass(neighborIndex) * KERNEL_W(m_data,xi - xj);
		const Vector3d grad_p_j = body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - xj);
		sum_grad_p_k += grad_p_j.squaredNorm();
		grad_p_i += grad_p_j;
		densityAdv += (vi - body.vel[neighborIndex]).dot(grad_p_j);
		computeDensityChange_additional
		);
		//*/


		sum_grad_p_k += grad_p_i.squaredNorm();

		//////////////////////////////////////////////////////////////////////////
		// Compute pressure stiffness denominator
		//////////////////////////////////////////////////////////////////////////
		particleSet->factor[i] = (-m_data.invH / (MAX_MACRO_CUDA(sum_grad_p_k, m_eps)));
		particleSet->density[i] = density;
		/*
		if ((particleSet->neighborsDataSet->cell_id[i] == TAG_ACTIVE_NEIGHBORS)) {
			printf("mass/height/density: %f %f %f\n", particleSet->mass[i], xi.y, density);
		}
		//*/

		//end the density adv computation
#ifdef STORE_PARTICLE_NEIGHBORS
		unsigned int numNeighbors = particleSet->getNumberOfNeighbourgs(i);
#endif //STORE_PARTICLE_NEIGHBORS
		// in case of particle deficiency do not perform a divergence solve
		if (numNeighbors < 20) {
			for (unsigned int pid = 1; pid < 3; pid++)
			{
				numNeighbors += particleSet->getNumberOfNeighbourgs(i, pid);
			}
		}
		if (numNeighbors < 20) {
			particleSet->densityAdv[i] = 0;
		}
		else {
			particleSet->densityAdv[i] = MAX_MACRO_CUDA(densityAdv, 0.0);

		}

		/*
		if (density < 998) {
			particleSet->color[i] = Vector3d(1, 0, 0);
			if (density < 995) {
				particleSet->color[i] = Vector3d(0, 1, 0);
				if (density < 990) {
					particleSet->color[i] = Vector3d(0, 0, 1);
				}
			}
		}
		//*/
	}

}

void cuda_divergence_warmstart_init(SPH::DFSPHCData& data) {
	{//fluid
		int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);
		DFSPH_divergence_warmstart_init_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
	}

	//*
	if (data.boundaries_data[0].has_factor_computation) {//boundaries
		int numBlocks = calculateNumBlocks(data.boundaries_data[0].numParticles);
		DFSPH_divergence_warmstart_init_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data[0].gpu_ptr);
	}
	//*/


	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_divergence_warmstart_init failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}


template <bool warm_start> __device__ void divergenceSolveParticle(SPH::DFSPHCData& m_data, SPH::UnifiedParticleSet* particleSet, const unsigned int i) {
	Vector3d v_i = Vector3d(0, 0, 0);
	//////////////////////////////////////////////////////////////////////////
	// Evaluate rhs
	//////////////////////////////////////////////////////////////////////////
	const RealCuda ki = (warm_start) ? particleSet->kappaV[i] : (particleSet->densityAdv[i])*particleSet->factor[i];

#ifdef USE_WARMSTART_V
	if (!warm_start) { particleSet->kappaV[i] += ki; }
#endif

	const Vector3d &xi = particleSet->pos[i];

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	ITER_NEIGHBORS_INIT(m_data, particleSet, i);

	ITER_NEIGHBORS_FLUID(m_data, particleSet,
		i,
		const RealCuda kSum = (ki + ((warm_start) ? body.kappaV[neighborIndex] : (body.densityAdv[neighborIndex])*body.factor[neighborIndex]));
	if (fabs(kSum) > m_eps)
	{
		// ki, kj already contain inverse density
		v_i += kSum *  body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]);
	}
	);

#ifdef USE_BOUNDARIES_DYNAMIC_PROPERTiES
	ITER_NEIGHBORS_BOUNDARIES(m_data, particleSet,
		i,
		const RealCuda kSum = (ki + ((warm_start) ? body.kappaV[neighborIndex] : (body.densityAdv[neighborIndex])*body.factor[neighborIndex]));
	if (fabs(kSum) > m_eps)
	{
		// ki, kj already contain inverse density
		v_i += kSum *  body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]);
	}
	);
#endif


	if (fabs(ki) > m_eps)
	{
		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
#ifndef USE_BOUNDARIES_DYNAMIC_PROPERTiES

#ifdef BENDER2019_BOUNDARIES
		const Vector3d& xj = particleSet->X_rigids[i];
		const RealCuda mass = particleSet->V_rigids[i] * particleSet->density0;
		const Vector3d delta = ki * mass * KERNEL_GRAD_W(m_data, xi - xj);
		v_i += delta;// ki already contains inverse density
#else
		ITER_NEIGHBORS_BOUNDARIES(m_data, particleSet,
			i,
			const Vector3d delta = ki * body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]);
		v_i += delta;// ki already contains inverse density
		);
#endif

#endif

		//////////////////////////////////////////////////////////////////////////
		// Dynamic bodies
		//////////////////////////////////////////////////////////////////////////

		ITER_NEIGHBORS_SOLIDS(m_data, particleSet,
			i,
			Vector3d delta = ki * body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]);
		v_i += delta;// ki already contains inverse density

					 //we apply the force to the body particle (no invH since it has been fatorized at the end)
		delta *= -particleSet->getMass(i);
		atomicAdd(&(body.F[neighborIndex].x), delta.x);
		atomicAdd(&(body.F[neighborIndex].y), delta.y);
		atomicAdd(&(body.F[neighborIndex].z), delta.z);
		);
	}

	particleSet->vel[i] += v_i*m_data.h;
}


//WARNING !!! this is not suposed to be called for the fluid this function is used for boundaries and object for witch 
//doing the velocity variation computation makes no sense but still need the accumulation of kappa for the warm start
__global__ void DFSPH_divergence_accumulate_kappaV_kernel(SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	const RealCuda ki = (particleSet->densityAdv[i])*particleSet->factor[i];
	particleSet->kappaV[i] += ki;
}


template<bool warmstart> __global__ void DFSPH_divergence_compute_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }


	if (m_data.restriction_mode == 1) {
		if ((particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE)) {
			return;
		}
	}else if (m_data.restriction_mode == 2) {
		if (i >= (m_data.count_active)) { return; }
	}

	if (warmstart) {
		if (particleSet->densityAdv[i] > 0.0) {
			divergenceSolveParticle<warmstart>(m_data, particleSet, i);
		}
	}
	else {
		divergenceSolveParticle<warmstart>(m_data, particleSet, i);
	}

}

template<bool warmstart> void cuda_divergence_compute(SPH::DFSPHCData& data) {
	{//fluid
		int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);
		DFSPH_divergence_compute_kernel<warmstart> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
	}

	//*

	if (data.boundaries_data[0].has_factor_computation) {//boundaries 
		if (!warmstart) {
			int numBlocks = calculateNumBlocks(data.boundaries_data[0].numParticles);
			DFSPH_divergence_accumulate_kappaV_kernel << <numBlocks, BLOCKSIZE >> > (data.boundaries_data[0].gpu_ptr);
		}
	}
	//*/

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_divergence_compute failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}
template void cuda_divergence_compute<true>(SPH::DFSPHCData& data);
template void cuda_divergence_compute<false>(SPH::DFSPHCData& data);



__device__ void computeDensityChange(const SPH::DFSPHCData& m_data, SPH::UnifiedParticleSet* particleSet, const unsigned int index) {
#ifdef STORE_PARTICLE_NEIGHBORS
#define computeDensityChange_additional  
	unsigned int numNeighbors = particleSet->getNumberOfNeighbourgs(index);
	// in case of particle deficiency do not perform a divergence solve
	if (numNeighbors < 20) {
		for (unsigned int pid = 1; pid < 3; pid++)
		{
			numNeighbors += particleSet->getNumberOfNeighbourgs(index, pid);
		}
	}
	if (numNeighbors < 20) {
		particleSet->densityAdv[index] = 0;
	}
	else 
#endif //STORE_PARTICLE_NEIGHBORS
	{
#ifndef STORE_PARTICLE_NEIGHBORS
		unsigned int numNeighbors = 0;
#define computeDensityChange_additional numNeighbors++;
#endif //STORE_PARTICLE_NEIGHBORS

		RealCuda densityAdv = 0;
		const Vector3d &xi = particleSet->pos[index];
		const Vector3d &vi = particleSet->vel[index];
		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		ITER_NEIGHBORS_INIT(m_data, particleSet, index);

		ITER_NEIGHBORS_FLUID(m_data, particleSet,
			index,
			densityAdv += body.getMass(neighborIndex) * (vi - body.vel[neighborIndex]).dot(KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]));
			computeDensityChange_additional
		);
		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
#ifdef BENDER2019_BOUNDARIES
		const Vector3d& xj = particleSet->X_rigids[index];
		const RealCuda mass = particleSet->V_rigids[index] * particleSet->density0;
		densityAdv += mass* (vi).dot(KERNEL_GRAD_W(m_data, xi - xj));
#else
		ITER_NEIGHBORS_BOUNDARIES(m_data, particleSet,
			index,
			densityAdv += body.getMass(neighborIndex) * (vi - body.vel[neighborIndex]).dot(KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]));
			computeDensityChange_additional
		);
#endif

		//////////////////////////////////////////////////////////////////////////
		// Dynamic Bodies
		//////////////////////////////////////////////////////////////////////////
		ITER_NEIGHBORS_SOLIDS(m_data, particleSet,
			index,
			densityAdv += body.getMass(neighborIndex) * (vi - body.vel[neighborIndex]).dot(KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]));
			computeDensityChange_additional
		);

#ifndef STORE_PARTICLE_NEIGHBORS
		if (numNeighbors < 20) {
			densityAdv = 0;
		}
#endif //STORE_PARTICLE_NEIGHBORS

		// only correct positive divergence
		particleSet->densityAdv[index] = MAX_MACRO_CUDA(densityAdv, 0.0);
	}
}



__global__ void DFSPH_divergence_init_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (m_data.restriction_mode == 1) {
		if ((particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) &&
			(particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE_NEIGHBORS)) {
			return;
		}
	}
	else if (m_data.restriction_mode == 2) {
		if (i >= (m_data.count_active + m_data.count_active_neighbors)) { return; }
	}

	{
#ifdef USE_WARMSTART_V
		particleSet->kappaV[i] = 0;
#endif

		///TODO when doing this kernel I can actually fuse the code for all those computation to limit the number
		///of time I read the particles positions
		computeDensityChange(m_data, particleSet, i);

#ifndef USE_WARMSTART_V
		//I can actually make the factor and desity computation here
		{
			//////////////////////////////////////////////////////////////////////////
			// Compute gradient dp_i/dx_j * (1/k)  and dp_j/dx_j * (1/k)
			//////////////////////////////////////////////////////////////////////////
			const Vector3d &xi = particleSet->pos[i];
			RealCuda sum_grad_p_k = 0;
			Vector3d grad_p_i;
			grad_p_i.setZero();

			RealCuda density = particleSet->getMass(i) * m_data.W_zero;

			//////////////////////////////////////////////////////////////////////////
			// Fluid
			//////////////////////////////////////////////////////////////////////////
			ITER_NEIGHBORS_INIT(m_data, particleSet, i);

			ITER_NEIGHBORS_FLUID(m_data, particleSet,
				i,
				const Vector3d &xj = body.pos[neighborIndex];
			density += body.getMass(neighborIndex) * KERNEL_W(m_data,xi - xj);
			const Vector3d grad_p_j = body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - xj);
			sum_grad_p_k += grad_p_j.squaredNorm();
			grad_p_i += grad_p_j;
			);

			//////////////////////////////////////////////////////////////////////////
			// Boundary
			//////////////////////////////////////////////////////////////////////////
			ITER_NEIGHBORS_BOUNDARIES(m_data, particleSet,
				i,
				const Vector3d &xj = body.pos[neighborIndex];
			density += body.getMass(neighborIndex) * KERNEL_W(m_data,xi - xj);
			const Vector3d grad_p_j = body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - xj);
			sum_grad_p_k += grad_p_j.squaredNorm();
			grad_p_i += grad_p_j;
			);

			//////////////////////////////////////////////////////////////////////////
			// Dynamic bodies
			//////////////////////////////////////////////////////////////////////////
			//*
			ITER_NEIGHBORS_SOLIDS(m_data, particleSet,
				i,
				const Vector3d &xj = body.pos[neighborIndex];
			density += body.getMass(neighborIndex) * KERNEL_W(m_data,xi - xj);
			const Vector3d grad_p_j = body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - xj);
			sum_grad_p_k += grad_p_j.squaredNorm();
			grad_p_i += grad_p_j;
			);
			//*/


			sum_grad_p_k += grad_p_i.squaredNorm();

			//////////////////////////////////////////////////////////////////////////
			// Compute pressure stiffness denominator
			//////////////////////////////////////////////////////////////////////////
			particleSet->factor[i] = (-m_data.invH / (MAX_MACRO_CUDA(sum_grad_p_k, m_eps)));
			particleSet->density[i] = density;

		}
#endif


	}

}

void cuda_divergence_init(SPH::DFSPHCData& data) {
	{//fluid
		int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);
		DFSPH_divergence_init_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
	}

	//*
	if (data.boundaries_data[0].has_factor_computation) {//boundaries 
		int numBlocks = calculateNumBlocks(data.boundaries_data[0].numParticles);
		DFSPH_divergence_init_kernel << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data[0].gpu_ptr);
	}
	//*/

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_divergence_init failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}


__global__ void DFSPH_divergence_loop_end_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet, RealCuda* avg_density_err) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (m_data.restriction_mode == 1) {
		if ((particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) &&
			(particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE_NEIGHBORS)) {
			return;
		}
	}
	else if (m_data.restriction_mode == 2) {
		if (i >= (m_data.count_active + m_data.count_active_neighbors)) { return; }
	}

	computeDensityChange(m_data, particleSet, i);
	//atomicAdd(avg_density_err, m_data.densityAdv[i]);
}

RealCuda cuda_divergence_loop_end(SPH::DFSPHCData& data) {
	RealCuda* avg_density_err = SVS_CU::get()->avg_density_err;

	{//fluid
		int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles );
		DFSPH_divergence_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr, avg_density_err);
	}

	//*
	if (data.boundaries_data[0].has_factor_computation) {//boundaries 
		int numBlocks = calculateNumBlocks(data.boundaries_data[0].numParticles );
		DFSPH_divergence_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data[0].gpu_ptr, avg_density_err);
	}
	//*/


	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_divergence_loop_end failed: %d\n", (int)cudaStatus);
		exit(1598);
	}


	// Run sum-reduction
	cub::DeviceReduce::Sum(data.fluid_data->d_temp_storage, data.fluid_data->temp_storage_bytes, data.fluid_data->densityAdv, avg_density_err, data.fluid_data[0].numParticles);
	gpuErrchk(cudaDeviceSynchronize());


	RealCuda result = 0;
	gpuErrchk(cudaMemcpy(&result, avg_density_err, sizeof(RealCuda), cudaMemcpyDeviceToHost));

	return result;
}


int cuda_divergenceSolve(SPH::DFSPHCData& m_data, const unsigned int maxIter, const RealCuda maxError) {
	//////////////////////////////////////////////////////////////////////////
	// Init parameters
	//////////////////////////////////////////////////////////////////////////

	const RealCuda h = m_data.h;
	int numParticles = m_data.fluid_data[0].numParticles;
	const RealCuda density0 = m_data.density0;
	if (m_data.true_particle_count >= 0) {
		numParticles = m_data.true_particle_count;
	}

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

#ifdef USE_WARMSTART_V
	cuda_divergence_warmstart_init(m_data);

	std::chrono::steady_clock::time_point m0 = std::chrono::steady_clock::now();
	cuda_divergence_compute<true>(m_data);
#endif

	std::chrono::steady_clock::time_point m1 = std::chrono::steady_clock::now();
	//////////////////////////////////////////////////////////////////////////
	// Compute velocity of density change
	//////////////////////////////////////////////////////////////////////////
	cuda_divergence_init(m_data);

	std::chrono::steady_clock::time_point m2 = std::chrono::steady_clock::now();

	unsigned int m_iterationsV = 0;

	//////////////////////////////////////////////////////////////////////////
	// Start solver
	//////////////////////////////////////////////////////////////////////////

	// Maximal allowed density fluctuation
	// use maximal density error divided by time step size
	const RealCuda eta = maxError * 0.01 * density0 / h;  // maxError is given in percent
	//std::cout << "divergence eta: " << eta << std::endl;

	float time_3_1 = 0;
	float time_3_2 = 0;
	RealCuda avg_density_err = 0.0;
	while (((avg_density_err > eta) || (m_iterationsV < 3)) && (m_iterationsV < maxIter))
	{

		//////////////////////////////////////////////////////////////////////////
		// Perform Jacobi iteration over all blocks
		//////////////////////////////////////////////////////////////////////////
		std::chrono::steady_clock::time_point p0 = std::chrono::steady_clock::now();
		cuda_divergence_compute<false>(m_data);
		std::chrono::steady_clock::time_point p1 = std::chrono::steady_clock::now();

		avg_density_err = cuda_divergence_loop_end(m_data);
		std::chrono::steady_clock::time_point p2 = std::chrono::steady_clock::now();

		avg_density_err /= numParticles;
		m_iterationsV++;

		time_3_1 += std::chrono::duration_cast<std::chrono::nanoseconds> (p1 - p0).count() / 1000000.0f;
		time_3_2 += std::chrono::duration_cast<std::chrono::nanoseconds> (p2 - p1).count() / 1000000.0f;
	}

	/*
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	float time_0 = std::chrono::duration_cast<std::chrono::nanoseconds> (m0 - start).count() / 1000000.0f;
	float time_1 = std::chrono::duration_cast<std::chrono::nanoseconds> (m1 - m0).count() / 1000000.0f;
	float time_2 = std::chrono::duration_cast<std::chrono::nanoseconds> (m2 - m1).count() / 1000000.0f;
	float time_3 = std::chrono::duration_cast<std::chrono::nanoseconds> (end - m2).count() / 1000000.0f;

	std::cout << "detail pressure solve (iter total (varible_comp warm_comp init actual_comp (t1 t2))): " << m_iterationsV << "  " << time_0+ time_1 + time_2 + time_3 <<
	"  (" << time_0 << "  " << time_1 << "  " << time_2 << "  " << time_3 << "(" << time_3_1 << " " << time_3_2 << ") )" << std::endl;

	//*/
	return m_iterationsV;
}

////////////////////////////////////////////////////
/////////          DENSITY SOLVER      /////////////
////////////////////////////////////////////////////

template <bool warm_start> __device__ void pressureSolveParticle(SPH::DFSPHCData& m_data, SPH::UnifiedParticleSet* particleSet, const unsigned int i) {
	//////////////////////////////////////////////////////////////////////////
	// Evaluate rhs
	//////////////////////////////////////////////////////////////////////////
	const RealCuda ki = (warm_start) ? particleSet->kappa[i] : (particleSet->densityAdv[i])*particleSet->factor[i];

#ifdef USE_WARMSTART
	//if (!warm_start) { particleSet->kappa[i] += ki; } //moved to the evaluation
#endif


	Vector3d v_i = Vector3d(0, 0, 0);
	const Vector3d &xi = particleSet->pos[i];

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	ITER_NEIGHBORS_INIT(m_data, particleSet, i);

	ITER_NEIGHBORS_FLUID(m_data, particleSet,
		i,
		const RealCuda kSum = (ki + ((warm_start) ? body.kappa[neighborIndex] : (body.densityAdv[neighborIndex])*body.factor[neighborIndex]));
	if (fabs(kSum) > m_eps)
	{
		// ki, kj already contain inverse density
		v_i += kSum * body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]);
	}
	);

#ifdef USE_BOUNDARIES_DYNAMIC_PROPERTiES
	ITER_NEIGHBORS_BOUNDARIES(m_data, particleSet,
		i,
		const RealCuda kSum = (ki + ((warm_start) ? body.kappa[neighborIndex] : (body.densityAdv[neighborIndex])*body.factor[neighborIndex]));
	if (fabs(kSum) > m_eps)
	{
		// ki, kj already contain inverse density
		v_i += kSum * body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]);
	}
	);
#endif

	if (fabs(ki) > m_eps)
	{
		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////

#ifndef USE_BOUNDARIES_DYNAMIC_PROPERTiES

#ifdef BENDER2019_BOUNDARIES
		const Vector3d& xj = particleSet->X_rigids[i];
		const RealCuda mass = particleSet->V_rigids[i] * particleSet->density0;

		v_i += ki * mass * KERNEL_GRAD_W(m_data, xi - xj);

#else
		ITER_NEIGHBORS_BOUNDARIES(m_data, particleSet,
			i,
			v_i += ki * body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]);
		);
#endif


#endif


		//////////////////////////////////////////////////////////////////////////
		// Dynamic bodies
		//////////////////////////////////////////////////////////////////////////
		ITER_NEIGHBORS_SOLIDS(m_data, particleSet,
			i,
			Vector3d delta = ki * body.getMass(neighborIndex) * KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]);
		v_i += delta;// ki already contains inverse density

					 //we apply the force to the body particle (no invH since it has been fatorized at the end)
		delta *= -particleSet->getMass(i);
		atomicAdd(&(body.F[neighborIndex].x), delta.x);
		atomicAdd(&(body.F[neighborIndex].y), delta.y);
		atomicAdd(&(body.F[neighborIndex].z), delta.z);
		);
	}

	// Directly update velocities instead of storing pressure accelerations
	particleSet->vel[i] += v_i*m_data.h_future;
}



//WARNING !!! this is not suposed to be called for the fluid this function is used for boundaries and object for witch 
//doing the velocity variation computation makes no sense but still need the accumulation of kappa for the warm start
__global__ void DFSPH_density_accumulate_kappa_kernel(SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	const RealCuda ki = (particleSet->densityAdv[i])*particleSet->factor[i];
	particleSet->kappa[i] += ki;
}

template<bool warmstart> __global__ void DFSPH_pressure_compute_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }
	
	if (m_data.restriction_mode == 1) {
		if ((particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE)) {
			return;
		}
	}
	else if (m_data.restriction_mode == 2) {
		if (i >= (m_data.count_active)) { return; }
	}

	pressureSolveParticle<warmstart>(m_data, particleSet, i);

}

template<bool warmstart> void cuda_pressure_compute(SPH::DFSPHCData& data) {
	{//fluid
		int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);
		DFSPH_pressure_compute_kernel<warmstart> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
	}

	if (data.boundaries_data[0].has_factor_computation) {//boundaries 
		if (!warmstart) {
			int numBlocks = calculateNumBlocks(data.boundaries_data[0].numParticles);
			DFSPH_density_accumulate_kappa_kernel << <numBlocks, BLOCKSIZE >> > (data.boundaries_data[0].gpu_ptr);
		}
	}

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_pressure_compute failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}
template void cuda_pressure_compute<true>(SPH::DFSPHCData& data);
template void cuda_pressure_compute<false>(SPH::DFSPHCData& data);


__global__ void DFSPH_pressure_compute_p1_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }


	{
		//////////////////////////////////////////////////////////////////////////
		// Compute gradient dp_i/dx_j * (1/k)  and dp_j/dx_j * (1/k)
		//////////////////////////////////////////////////////////////////////////
		const Vector3d& xi = particleSet->pos[i];
		const Vector3d& vi = particleSet->vel[i];

		RealCuda density = particleSet->getMass(i) * m_data.W_zero;

		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		ITER_NEIGHBORS_INIT(m_data, particleSet, i);

		ITER_NEIGHBORS_FLUID(m_data, particleSet,
			i,
			const Vector3d & xj = body.pos[neighborIndex];
		density += body.getMass(neighborIndex) * KERNEL_W(m_data, xi - xj);
			);


		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
		ITER_NEIGHBORS_BOUNDARIES(m_data, particleSet,
			i,
			const Vector3d & xj = body.pos[neighborIndex];
		density += body.getMass(neighborIndex) * KERNEL_W(m_data, xi - xj);
			);


		//////////////////////////////////////////////////////////////////////////
		// Dynamic bodies
		//////////////////////////////////////////////////////////////////////////
		//*
		ITER_NEIGHBORS_SOLIDS(m_data, particleSet,
			i,
			const Vector3d & xj = body.pos[neighborIndex];
		density += body.getMass(neighborIndex) * KERNEL_W(m_data, xi - xj);
			);
		//*/


		//////////////////////////////////////////////////////////////////////////
		// Compute pressure stiffness denominator
		//////////////////////////////////////////////////////////////////////////
		particleSet->density[i] = density;
		RealCuda C = density / m_data.density0 - 1;
		RealCuda lambda = 0;
		if (C > 0) {
			lambda = m_data.density0 * m_data.density0 * particleSet->factor[i] * C * m_data.h_future* m_data.h_future;
		}

		//let's store it in the density adv for now
		particleSet->densityAdv[i] = lambda;
		
	}

}

__global__ void DFSPH_pressure_compute_p2_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	Vector3d dx(0, 0, 0);
	const Vector3d& xi = particleSet->pos[i];
	RealCuda lambda_i = particleSet->densityAdv[i];
	{
		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		ITER_NEIGHBORS_INIT(m_data, particleSet, i);

		ITER_NEIGHBORS_FLUID(m_data, particleSet,
			i,
			const Vector3d & xj = body.pos[neighborIndex];
			RealCuda lambda_sum = lambda_i + body.densityAdv[neighborIndex];
			if (fabs(lambda_sum) > m_eps) 
			{
				dx += body.getMass(neighborIndex) * lambda_sum * KERNEL_GRAD_W(m_data, xi - xj);
			}
		);

		if (fabs(lambda_i) > m_eps) 
		{
			//////////////////////////////////////////////////////////////////////////
			// Boundary
			//////////////////////////////////////////////////////////////////////////
			ITER_NEIGHBORS_BOUNDARIES(m_data, particleSet,
				i,
				const Vector3d & xj = body.pos[neighborIndex];
				dx += body.getMass(neighborIndex) * lambda_i * KERNEL_GRAD_W(m_data, xi - xj);
			);


			//////////////////////////////////////////////////////////////////////////
			// Dynamic bodies
			//////////////////////////////////////////////////////////////////////////
			//*
			ITER_NEIGHBORS_SOLIDS(m_data, particleSet,
				i,
				const Vector3d & xj = body.pos[neighborIndex];
				dx += body.getMass(neighborIndex) * lambda_i * KERNEL_GRAD_W(m_data, xi - xj);
			);
			//*/
		}

	}
	dx /= m_data.density0;
	m_data.posBufferGroupedDynamicBodies[i]=dx;

}

__global__ void DFSPH_pressure_compute_p3_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	particleSet->pos[i] += data.posBufferGroupedDynamicBodies[i];

}

//this is the position based condition
//paper: Physics - Based Simulation of Ocean Scenes in Marine Simulator Visual System
void cuda_pressure_compute_v2(SPH::DFSPHCData& data) {
	//comp the necessary values
	{
		int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);
		DFSPH_pressure_compute_p1_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
		gpuErrchk(cudaDeviceSynchronize());
	}

	//comp the particle displacement
	{
		int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);
		DFSPH_pressure_compute_p2_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
		gpuErrchk(cudaDeviceSynchronize());
	}

	//affect the displacmeent
	{
		int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);
		DFSPH_pressure_compute_p3_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
		gpuErrchk(cudaDeviceSynchronize());
	}

	/*
	RealCuda displacement = 0;
	for (int i = 0; i < data.fluid_data->numParticles; ++i) {
		displacement += data.posBufferGroupedDynamicBodies[i].norm();
	}
	std::cout << "total displacmeent: " << displacement << std::endl;
	//*/

}

__device__ void computeDensityAdv(SPH::DFSPHCData& m_data, SPH::UnifiedParticleSet* particleSet, const unsigned int index) {
	const Vector3d xi = particleSet->pos[index];
	const Vector3d vi = particleSet->vel[index];
	RealCuda delta = 0;


	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	ITER_NEIGHBORS_INIT(m_data, particleSet, index);

	ITER_NEIGHBORS_FLUID(m_data, particleSet,
		index,
		delta += body.getMass(neighborIndex) * (vi - body.vel[neighborIndex]).dot(KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]));
	);

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////

#ifdef BENDER2019_BOUNDARIES
	const Vector3d& xj = particleSet->X_rigids[index];
	const RealCuda mass = particleSet->V_rigids[index] * particleSet->density0;

	delta += mass * (vi - xj).dot(KERNEL_GRAD_W(m_data, xi - xj));

#else
	
	ITER_NEIGHBORS_BOUNDARIES(m_data, particleSet,
		index,
		delta += body.getMass(neighborIndex) * (vi - body.vel[neighborIndex]).dot(KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]));
	);
#endif

	//////////////////////////////////////////////////////////////////////////
	// Dynamic bodies
	//////////////////////////////////////////////////////////////////////////
	ITER_NEIGHBORS_SOLIDS(m_data, particleSet,
		index,
		delta += body.getMass(neighborIndex) * (vi - body.vel[neighborIndex]).dot(KERNEL_GRAD_W(m_data,xi - body.pos[neighborIndex]));
	)

		particleSet->densityAdv[index] = MAX_MACRO_CUDA(particleSet->density[index] + m_data.h_future*delta - m_data.density0, 0.0);


#ifdef USE_WARMSTART
	particleSet->kappa[index] += (particleSet->densityAdv[index]) * particleSet->factor[index];
	//particleSet->kappa[index] = 0;
#endif
}


__global__ void DFSPH_pressure_init_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (m_data.restriction_mode == 1) {
		if ((particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) &&
			(particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE_NEIGHBORS)) {
			return;
		}
	}
	else if (m_data.restriction_mode == 2) {
		if (i >= (m_data.count_active + m_data.count_active_neighbors)) { return; }
	}

#ifdef USE_WARMSTART
	particleSet->kappa[i] = 0;
#endif

	particleSet->factor[i] *= m_data.invH_future;

	computeDensityAdv(m_data, particleSet, i);


}




void cuda_pressure_init(SPH::DFSPHCData& data) {
	{//fluid
		int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);
		DFSPH_pressure_init_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
	}

	if (data.boundaries_data[0].has_factor_computation) {//boundaries 
		int numBlocks = calculateNumBlocks(data.boundaries_data[0].numParticles);
		DFSPH_pressure_init_kernel << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data[0].gpu_ptr);
	}


	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_pressure_init failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}

__global__ void DFSPH_pressure_loop_end_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet, RealCuda* avg_density_err) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (m_data.restriction_mode == 1) {
		if ((particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) &&
			(particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE_NEIGHBORS)) {
			return;
		}
	}
	else if (m_data.restriction_mode == 2) {
		if (i >= (m_data.count_active + m_data.count_active_neighbors)) { return; }
	}

#ifdef USE_POSITION_BASED_DENSITY_CONTRAINT
	particleSet->densityAdv[i] = MAX_MACRO_CUDA(particleSet->density[i] - m_data.density0, 0.0);
#else
	computeDensityAdv(m_data, particleSet, i);
#endif 
	//atomicAdd(avg_density_err, m_data.densityAdv[i]);
}

RealCuda cuda_pressure_loop_end(SPH::DFSPHCData& data) {

	std::chrono::steady_clock::time_point p0 = std::chrono::steady_clock::now();

	RealCuda* avg_density_err = SVS_CU::get()->avg_density_err;

	{//fluid
		int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);
		DFSPH_pressure_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr, avg_density_err);
	}

	if (data.boundaries_data[0].has_factor_computation) {//boundaries 
		int numBlocks = calculateNumBlocks(data.boundaries_data[0].numParticles);
		DFSPH_pressure_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data[0].gpu_ptr, avg_density_err);
	}



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

	// Run sum-reduction
	cub::DeviceReduce::Sum(data.fluid_data->d_temp_storage, data.fluid_data->temp_storage_bytes, data.fluid_data->densityAdv, avg_density_err, data.fluid_data[0].numParticles);


	RealCuda result = 0;
	gpuErrchk(cudaMemcpy(&result, avg_density_err, sizeof(RealCuda), cudaMemcpyDeviceToHost));


	std::chrono::steady_clock::time_point p2 = std::chrono::steady_clock::now();
	float time1 = std::chrono::duration_cast<std::chrono::nanoseconds> (p1 - p0).count() / 1000000.0f;
	float time2 = std::chrono::duration_cast<std::chrono::nanoseconds> (p2 - p1).count() / 1000000.0f;

	//std::cout << "pressure loop end details: " << time1 << "  " << time2 << std::endl;

	return result;
}


int cuda_pressureSolve(SPH::DFSPHCData& m_data, const unsigned int m_maxIterations, const RealCuda m_maxError) {
	const RealCuda density0 = m_data.density0;
	int numParticles = (int)m_data.fluid_data[0].numParticles;
	if (m_data.true_particle_count >= 0) {
		numParticles = m_data.true_particle_count;
	}

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
	//std::cout << "density eta: " << eta << std::endl;
	static int count_step = 0; count_step++;

	float time_3_1 = 0;
	float time_3_2 = 0;
	while (((avg_density_err > eta) || (m_iterations < 2)) && (m_iterations < m_maxIterations))
	{

		std::chrono::steady_clock::time_point p0 = std::chrono::steady_clock::now();
#ifdef USE_POSITION_BASED_DENSITY_CONTRAINT
		cuda_pressure_compute_v2(m_data);
#else
		cuda_pressure_compute<false>(m_data);
#endif // DEBUG

		std::chrono::steady_clock::time_point p1 = std::chrono::steady_clock::now();
		avg_density_err = cuda_pressure_loop_end(m_data);
		std::chrono::steady_clock::time_point p2 = std::chrono::steady_clock::now();
		//std::cout << "total density err: " << avg_density_err << std::endl;
		avg_density_err /= numParticles;

		m_iterations++;

		time_3_1 += std::chrono::duration_cast<std::chrono::nanoseconds> (p1 - p0).count() / 1000000.0f;
		time_3_2 += std::chrono::duration_cast<std::chrono::nanoseconds> (p2 - p1).count() / 1000000.0f;



		//std::cout << "see density convergence: " << count_step << "  " << m_iterations << " " << avg_density_err << std::endl;
		
		//std::cout << "presure solve out: " << m_iterations << "   " << avg_density_err << std::endl;
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


// also prepare the normals for the adhesion force
__global__ void DFSPH_viscosityXSPH_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }


	if (m_data.restriction_mode == 1) {
		if ((particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE)) {
			return;
		}
	}
	else if (m_data.restriction_mode == 2) {
		if (i >= (m_data.count_active)) { return; }
	}

	//I set the gravitation directly here to lover the number of kernels
	Vector3d ai = Vector3d(0, 0, 0);
	Vector3d ni = Vector3d(0, 0, 0);
	const Vector3d &xi = particleSet->pos[i];
	const Vector3d &vi = particleSet->vel[i];

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
//*	

	ITER_NEIGHBORS_INIT(m_data, particleSet, i);

	//*
	ITER_NEIGHBORS_FLUID(m_data, particleSet,
		i,
		Vector3d xixj = xi - body.pos[neighborIndex];
	RealCuda mass_div_density = body.getMass(neighborIndex) / body.density[neighborIndex];
	ai -= m_data.invH * m_data.viscosity * (mass_div_density) * (vi - body.vel[neighborIndex]) * KERNEL_W(m_data,xixj);
	ni += mass_div_density * KERNEL_GRAD_W(m_data,xixj);
	)
		//*/
		/*
		//viscosity only
		ITER_NEIGHBORS_FLUID(
		i,
		ai -= m_data.invH * m_data.viscosity * (body.getMass(neighborIndex) / body.density[neighborIndex]) *
		(vi - body.vel[neighborIndex]) * KERNEL_W(m_data,xi - body.pos[neighborIndex]);

		)//*/

		particleSet->acc[i] = m_data.gravitation + ai;

	/*
	//I'm gona use the vector3D used for the agglomerated neigbor search to store the normals
	ni *= m_data.getKernelRadius();
	m_data.posBufferGroupedDynamicBodies[i] = ni;
	//*/
}


__global__ void DFSPH_applySurfaceAkinci2013SurfaceTension_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	//for more lisability of the code
	Vector3d* normals = m_data.posBufferGroupedDynamicBodies;
	RealCuda supportRadius = m_data.getKernelRadius();
	RealCuda k = m_data.getSurfaceTension();
	RealCuda density0 = m_data.density0;

	//I set the gravitation directly here to lover the number of kernels
	Vector3d ai = Vector3d(0, 0, 0);
	Vector3d ni = normals[i];
	RealCuda rhoi = particleSet->density[i];
	const Vector3d &xi = particleSet->pos[i];

	ITER_NEIGHBORS_INIT(m_data, particleSet, i);

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////

	ITER_NEIGHBORS_FLUID(m_data, particleSet,
		i,
		RealCuda K_ij = 2.0*density0 / (rhoi + body.density[neighborIndex]);

	Vector3d accel = Vector3d(0, 0, 0);


	// Cohesion force
	Vector3d xixj = xi - body.pos[neighborIndex];
	const Real length2 = xixj.squaredNorm();
	if (length2 > 1.0e-9)
	{
		xixj = ((Real) 1.0 / sqrt(length2)) * xixj;
		accel -= k * body.getMass(neighborIndex) * xixj * m_data.WCohesion(xixj);
	}

	// Curvature
	accel -= k * supportRadius* (ni - normals[neighborIndex]);

	ai += K_ij * accel;
	//*/
	);
	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////

#ifdef BENDER2019_BOUNDARIES

	const Vector3d& xj = particleSet->X_rigids[i];
	const RealCuda mass = particleSet->V_rigids[i] * particleSet->density0;
	
	Vector3d xixj = (xi - xj);
	const Real length2 = xixj.squaredNorm();
	if (length2 > 1.0e-9)
	{
		xixj = ((Real)1.0 / sqrt(length2)) * xixj;
		ai -= k * mass * xixj * m_data.WAdhesion(xixj);
	}

#else

	ITER_NEIGHBORS_BOUNDARIES(m_data, particleSet,
		i,
		// adhesion force
		Vector3d xixj = (xi - body.pos[neighborIndex]);
	const Real length2 = xixj.squaredNorm();
	if (length2 > 1.0e-9)
	{
		xixj = ((Real) 1.0 / sqrt(length2)) * xixj;
		ai -= k * body.getMass(neighborIndex) * xixj * m_data.WAdhesion(xixj);
	}
	);


#endif


	//////////////////////////////////////////////////////////////////////////
	// Dynamic Bodies
	//////////////////////////////////////////////////////////////////////////
	ITER_NEIGHBORS_SOLIDS(m_data, particleSet,
		i,
		// adhesion force
		Vector3d xixj = (xi - body.pos[neighborIndex]);
	const Real length2 = xixj.squaredNorm();
	if (length2 > 1.0e-9)
	{
		xixj = ((Real) 1.0 / sqrt(length2)) * xixj;
		ai -= k * body.getMass(neighborIndex) * xixj * m_data.WAdhesion(xixj);
	}
	);

	particleSet->acc[i] += ai;


}


void cuda_externalForces(SPH::DFSPHCData& data) {
	int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);
	DFSPH_viscosityXSPH_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_viscosityXSPH failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	//end the computations for the surface tension

	//DFSPH_applySurfaceAkinci2013SurfaceTension_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
	gpuErrchk(cudaDeviceSynchronize());
}




////////////////////////////////////////////////////
/////////         NEIGHBORS SEARCH     /////////////
////////////////////////////////////////////////////


__global__ void DFSPH_fill_aggregated_pos_buffer_kernel(SPH::DFSPHCData data, unsigned int num_particles) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_particles) { return; }

	if (data.is_fluid_aggregated) {
		if (i<data.fluid_data_cuda->numParticles) {

			//writte de pos
			data.posBufferGroupedDynamicBodies[i] = data.fluid_data_cuda->pos[i];

			return;
		}
	}

	//find the current dynamic body
	int count_particles_previous_bodies = (data.is_fluid_aggregated) ? data.fluid_data_cuda->numParticles : 0;
	int body_id = 0;
	while ((count_particles_previous_bodies + data.vector_dynamic_bodies_data_cuda[body_id].numParticles)<=i) {
		count_particles_previous_bodies += data.vector_dynamic_bodies_data_cuda[body_id].numParticles;
		body_id++;
	}

	//writte de pos
	data.posBufferGroupedDynamicBodies[i] = data.vector_dynamic_bodies_data_cuda[body_id].pos[i - count_particles_previous_bodies];
}




template<unsigned int grid_size, bool z_curve>
__global__ void DFSPH_computeGridIdx_kernel(Vector3d* in, unsigned int* out, RealCuda kernel_radius, unsigned int num_particles,
	Vector3i gridOffset) {

#ifndef INDEX_NEIGHBORS_CELL_FROM_STORAGE


	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//i *= 4;
	if (i >= num_particles) { return; }

	if (z_curve) {

	}
	else {
		//the offset is used to be able to use a small grid bu placing the simulation correctly inside it
		Vector3d pos = (in[i] / kernel_radius) + gridOffset;
		pos.toFloor();
		out[i] = COMPUTE_CELL_INDEX(pos.x, pos.y, pos.z);
		/*
		pos = (in[i + 1] / kernel_radius) + gridOffset;
		pos.toFloor();
		out[i + 1] = COMPUTE_CELL_INDEX(pos.x, pos.y, pos.z);

		pos = (in[i + 2] / kernel_radius) + gridOffset;
		pos.toFloor();
		out[i + 2] = COMPUTE_CELL_INDEX(pos.x, pos.y, pos.z);

		pos = (in[i + 3] / kernel_radius) + gridOffset;
		pos.toFloor();
		out[i + 3] = COMPUTE_CELL_INDEX(pos.x, pos.y, pos.z);
		//*/
	}


#endif // !INDEX_NEIGHBORS_CELL_FROM_STORAGE
}




void cuda_neighborsSearchInternal_sortParticlesId(Vector3d* pos, RealCuda kernel_radius, Vector3i gridOffset, int numParticles,
	void **d_temp_storage_pair_sort, size_t   &temp_storage_bytes_pair_sort,
	unsigned int* cell_id, unsigned int* cell_id_sorted,
	unsigned int* p_id, unsigned int* p_id_sorted) {
	cudaError_t cudaStatus;

#ifdef INDEX_NEIGHBORS_CELL_FROM_STORAGE
	exit(2563);
#endif // INDEX_NEIGHBORS_CELL_FROM_STORAGE


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
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	int numBlocks = calculateNumBlocks(numParticles);


	//compute the idx of the cell for each particles
	DFSPH_computeGridIdx_kernel<CELL_ROW_LENGTH, false> << <numBlocks, BLOCKSIZE >> > (pos, cell_id,
		kernel_radius, numParticles, gridOffset);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "idxs failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	//std::chrono::steady_clock::time_point middle = std::chrono::steady_clock::now();

	// Run sorting operation
	cub::DeviceRadixSort::SortPairs(*d_temp_storage_pair_sort, temp_storage_bytes_pair_sort,
		cell_id, cell_id_sorted, p_id, p_id_sorted, numParticles);
	//*/

	cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sort failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	/*
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	float time0;
	float time1;
	static float time_avg = 0;
	static int time_count = 0;
	time_count++;

	time0 = std::chrono::duration_cast<std::chrono::nanoseconds> (middle - begin).count() / 1000000.0f;
	time1 = std::chrono::duration_cast<std::chrono::nanoseconds> (end - middle).count() / 1000000.0f;

	time_avg += time0 + time1;
	printf("cuda_neighborsSearchInternal_sortParticlesId: %f ms (%f,%f)   avg: %f ms \n", time0 + time1, time0, time1, time_avg / time_count);
	//*/

}


void cuda_neighborsSearchInternal_computeCellStartEnd(int numParticles, unsigned int* cell_id_sorted,
	unsigned int* hist, void **d_temp_storage_cumul_hist, size_t   &temp_storage_bytes_cumul_hist, unsigned int* cell_start_end) {
	cudaError_t cudaStatus;
	int numBlocks = calculateNumBlocks(numParticles);


	//Now we need to determine the start and end of each cell
	//init the histogram values. Maybe doing it wiith thrust fill is faster.
	//the doc is not realy clear
	cudaMemset(hist, 0, (CELL_COUNT + 1) * sizeof(unsigned int));
	gpuErrchk(cudaDeviceSynchronize());

	//compute the actual histogram (done here with atomic adds)
	//*
	DFSPH_Histogram_kernel << <numBlocks, BLOCKSIZE >> > (cell_id_sorted, hist, numParticles);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cerr << "histogram failed: " << (int)cudaStatus << std::endl;
		exit(1598);
	}//*/

	//transformour histogram to a cumulative histogram to have  the start and end of each cell
	//note: the exlusive sum make so that each cell will contains it's start value
	// Run exclusive prefix sum
	cub::DeviceScan::ExclusiveSum(*d_temp_storage_cumul_hist, temp_storage_bytes_cumul_hist, hist, cell_start_end, (CELL_COUNT + 1));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cumulative histogram failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}



__global__ void DFSPH_computeGridIdx_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }


	//the offset is used to be able to use a small grid bu placing the simulation correctly inside it
	Vector3d pos = (particleSet->pos[i] / data.getKernelRadius()) + data.gridOffset;
	pos.toFloor();
	particleSet->neighborsDataSet->cell_id[i] = COMPUTE_CELL_INDEX(pos.x, pos.y, pos.z);
		
	//we can accumulate directly here
	particleSet->neighborsDataSet->cell_id_sorted[i] =
		atomicAdd(&(particleSet->neighborsDataSet->hist[particleSet->neighborsDataSet->cell_id[i]]), 1);

}


__global__ void DFSPH_CountingSortIds_kernel(SPH::UnifiedParticleSet* particleSet) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	int new_pos=particleSet->neighborsDataSet->cell_start_end[particleSet->neighborsDataSet->cell_id[i]] + particleSet->neighborsDataSet->cell_id_sorted[i];
	particleSet->neighborsDataSet->p_id_sorted[new_pos] = i;
}

void cuda_neighborsSearchInternal_sortParticlesId(SPH::UnifiedParticleSet& particleSet, SPH::NeighborsSearchDataSet& dataSet, SPH::DFSPHCData& data) {
	cudaError_t cudaStatus;

	//for (int i = 0; i < particleSet.numParticles; ++i)
	

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
	
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	
	//Now we need to determine the start and end of each cell
	//init the histogram values. Maybe doing it wiith thrust fill is faster.
	//the doc is not realy clear
	cudaMemset(particleSet.neighborsDataSet->hist, 0, (CELL_COUNT + 1) * sizeof(unsigned int));
	gpuErrchk(cudaDeviceSynchronize());

	std::chrono::steady_clock::time_point middle1 = std::chrono::steady_clock::now();

	int numBlocks = calculateNumBlocks(particleSet.numParticles);


	//compute the idx of the cell for each particles
	DFSPH_computeGridIdx_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet.gpu_ptr);
	gpuErrchk(cudaDeviceSynchronize());

	std::chrono::steady_clock::time_point middle2 = std::chrono::steady_clock::now();

	//transformour histogram to a cumulative histogram to have  the start and end of each cell
	//note: the exlusive sum make so that each cell will contains it's start value
	// Run exclusive prefix sum
	cub::DeviceScan::ExclusiveSum(particleSet.neighborsDataSet->d_temp_storage_cumul_hist, particleSet.neighborsDataSet->temp_storage_bytes_cumul_hist, 
		particleSet.neighborsDataSet->hist, particleSet.neighborsDataSet->cell_start_end, (CELL_COUNT + 1));
	gpuErrchk(cudaDeviceSynchronize());

	

	DFSPH_CountingSortIds_kernel << <numBlocks, BLOCKSIZE >> > (particleSet.gpu_ptr);
	gpuErrchk(cudaDeviceSynchronize());

	/*
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	float time0;
	float time1;
	float time2;
	static float time_avg = 0;
	static int time_count = 0;
	time_count++;

	time0 = std::chrono::duration_cast<std::chrono::nanoseconds> (middle1 - begin).count() / 1000000.0f;
	time1 = std::chrono::duration_cast<std::chrono::nanoseconds> (middle2 - middle1).count() / 1000000.0f;
	time2 = std::chrono::duration_cast<std::chrono::nanoseconds> (end - middle2).count() / 1000000.0f;

	time_avg += time0 + time1 + time2;
	printf("cuda_neighborsSearchInternal_sortParticlesId: %f ms (%f,%f,%f)   avg: %f ms \n", time0 + time1 + time2, time0, time1, time2, time_avg / time_count);
	//*/

}




void cuda_initNeighborsSearchDataSet(SPH::UnifiedParticleSet& particleSet, SPH::NeighborsSearchDataSet& dataSet,
	SPH::DFSPHCData& data, bool sortBuffers) {



	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	//com the id
	cuda_neighborsSearchInternal_sortParticlesId(particleSet, dataSet, data);

	//cuda_neighborsSearchInternal_sortParticlesId(particleSet.pos, data.getKernelRadius(), data.gridOffset, dataSet.numParticles,
	//	&dataSet.d_temp_storage_pair_sort, dataSet.temp_storage_bytes_pair_sort, dataSet.cell_id, dataSet.cell_id_sorted,
	//	dataSet.p_id, dataSet.p_id_sorted);
	std::chrono::steady_clock::time_point middle1 = std::chrono::steady_clock::now();

	//since it the init iter I'll sort both even if it's the boundaries
	if (sortBuffers) {
		cuda_sortData(particleSet, dataSet.p_id_sorted);
	}

	std::chrono::steady_clock::time_point middle2 = std::chrono::steady_clock::now();


	//and now I cna compute the start and end of each cell :)
	//cuda_neighborsSearchInternal_computeCellStartEnd(dataSet.numParticles, dataSet.cell_id_sorted, dataSet.hist,
	//	&dataSet.d_temp_storage_cumul_hist, dataSet.temp_storage_bytes_cumul_hist, dataSet.cell_start_end);





	/*
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	float time0;
	float time1;
	float time2;
	static float time_avg = 0;
	static int time_count = 0;
	time_count++;

	time0 = std::chrono::duration_cast<std::chrono::nanoseconds> (middle1 - begin).count() / 1000000.0f;
	time1 = std::chrono::duration_cast<std::chrono::nanoseconds> (middle2 - middle1).count() / 1000000.0f;
	time2 = std::chrono::duration_cast<std::chrono::nanoseconds> (end - middle2).count() / 1000000.0f;

	time_avg += time0 + time1 + time2;
	printf("Time to generate cell start end internal: %f ms (%f,%f,%f)   avg: %f ms \n", time0 + time1 + time2, time0, time1, time2, time_avg / time_count);
	//*/

}

void cuda_initNeighborsSearchDataSetGroupedDynamicBodies(SPH::DFSPHCData& data) {
	if (data.numDynamicBodies<1) {
		return;
	}

	SPH::NeighborsSearchDataSet& dataSet = *(data.neighborsDataSetGroupedDynamicBodies);


	//before anything I need to update the number of active particles
	int numParticles = (data.is_fluid_aggregated) ? data.fluid_data[0].numParticles : 0;
	for (int i = 0; i<data.numDynamicBodies; ++i) {
		numParticles += data.vector_dynamic_bodies_data[i].numParticles;
	}

	if (dataSet.numParticles != numParticles) {
		if (numParticles <= (int)dataSet.numParticlesMax) {
			dataSet.updateActiveParticleNumber(numParticles);
		}
		else {
			std::ostringstream oss;
			oss << "TODO::I need to add particles to the grouped data struct when the number of particle goes above the max" <<
				" current max: " << dataSet.numParticlesMax << "  number of particles: " << numParticles << std::endl;
			throw(oss.str());
		}
	}

	// now fill itr
	int numBlocks = calculateNumBlocks(dataSet.numParticles);
	DFSPH_fill_aggregated_pos_buffer_kernel << <numBlocks, BLOCKSIZE >> > (data, dataSet.numParticles);
	gpuErrchk(cudaDeviceSynchronize());

	//and now we can do the neighbor search
	//com the id
	cuda_neighborsSearchInternal_sortParticlesId(data.posBufferGroupedDynamicBodies, data.getKernelRadius(), data.gridOffset, dataSet.numParticles,
		&dataSet.d_temp_storage_pair_sort, dataSet.temp_storage_bytes_pair_sort, dataSet.cell_id, dataSet.cell_id_sorted,
		dataSet.p_id, dataSet.p_id_sorted);



	//and now I cna compute the start and end of each cell :)
	cuda_neighborsSearchInternal_computeCellStartEnd(dataSet.numParticles, dataSet.cell_id_sorted, dataSet.hist,
		&dataSet.d_temp_storage_cumul_hist, dataSet.temp_storage_bytes_cumul_hist, dataSet.cell_start_end);



}

template<typename T>
__global__ void DFSPH_sortFromIndex_kernel(T* in, T* out, unsigned int* index, unsigned int nbElements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nbElements) { return; }

	out[i] = in[index[i]];
}



void cuda_sortData(SPH::UnifiedParticleSet& particleSet, unsigned int * sort_id) {
	//*
	/*
	static std::chrono::steady_clock::time_point start;
	static std::chrono::steady_clock::time_point end;
	start = std::chrono::steady_clock::now();
	//*/
	unsigned int numParticles = particleSet.neighborsDataSet->numParticles;
	int numBlocks = calculateNumBlocks(numParticles);
	unsigned int *p_id_sorted = sort_id;

	Vector3d* intermediate_buffer_v3d = particleSet.neighborsDataSet->intermediate_buffer_v3d;
	RealCuda* intermediate_buffer_real = particleSet.neighborsDataSet->intermediate_buffer_real;

	//pos
	DFSPH_sortFromIndex_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (particleSet.pos, intermediate_buffer_v3d, p_id_sorted, numParticles);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(particleSet.pos, intermediate_buffer_v3d, numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

	//vel
	DFSPH_sortFromIndex_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (particleSet.vel, intermediate_buffer_v3d, p_id_sorted, numParticles);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(particleSet.vel, intermediate_buffer_v3d, numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

	//color
	if (particleSet.has_color_buffer) {
		DFSPH_sortFromIndex_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (particleSet.color, intermediate_buffer_v3d, p_id_sorted, numParticles);
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(particleSet.color, intermediate_buffer_v3d, numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
	}
	
	//mass
	DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.mass, intermediate_buffer_real, p_id_sorted, numParticles);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(particleSet.mass, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));


	if (particleSet.has_factor_computation) {
		//kappa
		DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.kappa, intermediate_buffer_real, p_id_sorted, numParticles);
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(particleSet.kappa, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

		//kappav
		DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.kappaV, intermediate_buffer_real, p_id_sorted, numParticles);
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(particleSet.kappaV, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
	}



	//now that everything is sorted we can set each particle index to itself
	gpuErrchk(cudaMemcpy(p_id_sorted, particleSet.neighborsDataSet->p_id, numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

	/*
	end = std::chrono::steady_clock::now();

	float time_iter = std::chrono::duration_cast<std::chrono::nanoseconds> (end-start).count() / 1000000.0f;
	static std::vector<float> timmings;

	timmings.push_back(time_iter);

	static int count = 0;
	count++;

	if (count==1300) {
		for (int i = 0; i < 1300; ++i) {
			std::cout << i << " " << timmings[i] << std::endl;
		}
	}//*/
}



__global__ void generateShuffleIndex_kernel(unsigned int *shuffle_index, unsigned int nbElements, curandState *state) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= 1) { return; }

	for (int j = 0; j < nbElements; ++j) {
		shuffle_index[j] = j;
	}



	curandState localState = *state;
	for (int j = 0; j < nbElements; ++j) {
		float x = curand_uniform(&localState);
		x *= nbElements;
		unsigned int idx = x;
		if (x < nbElements) {
			unsigned int temp = shuffle_index[idx];
			shuffle_index[idx] = shuffle_index[i];
			shuffle_index[i] = temp;
		}
	}
	*state = localState;
}


void cuda_shuffleData(SPH::UnifiedParticleSet& particleSet) {
	unsigned int numParticles = particleSet.numParticles;
	int numBlocks = calculateNumBlocks(numParticles);

	//create a random sorting index
	unsigned int* shuffle_index = SVS_CU::get()->shuffle_index;
	curandState *state = SVS_CU::get()->curand_state;
	if (shuffle_index == NULL) {
		cudaMallocManaged(&(SVS_CU::get()->shuffle_index), particleSet.numParticlesMax * sizeof(unsigned int));
		shuffle_index = SVS_CU::get()->shuffle_index;
		gpuErrchk(cudaDeviceSynchronize());
	}


	generateShuffleIndex_kernel << <1, 1 >> > (shuffle_index, numParticles, state);
	gpuErrchk(cudaDeviceSynchronize());


	unsigned int *p_id_sorted = shuffle_index;

	Vector3d* intermediate_buffer_v3d = particleSet.neighborsDataSet->intermediate_buffer_v3d;
	RealCuda* intermediate_buffer_real = particleSet.neighborsDataSet->intermediate_buffer_real;

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

	if (particleSet.has_factor_computation) {
		//kappa
		DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.kappa, intermediate_buffer_real, p_id_sorted, numParticles);
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(particleSet.kappa, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

		//kappav
		DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.kappaV, intermediate_buffer_real, p_id_sorted, numParticles);
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(particleSet.kappaV, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
	}



}






template <bool is_fluid_container>
__global__ void DFSPH_neighborsSearch_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }


	ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, particleSet, i);

	//this variable is need for the interleave but I'm not modifying all the macro for only a single case
#ifdef INTERLEAVE_NEIGHBORS
	int numParticles = particleSet->numParticles;
#endif

	unsigned int nb_neighbors_fluid = 0;
	unsigned int nb_neighbors_boundary = 0;
	unsigned int nb_neighbors_dynamic_objects =0;
	int* cur_neighbor_ptr= particleSet->getNeighboursPtr(i);
	//int neighbors_fluid[MAX_NEIGHBOURS];//doing it with local buffer was not faster
	//int neighbors_boundary[MAX_NEIGHBOURS];


	
	if (data.is_fluid_aggregated) {
		int neighbors_solids[MAX_NEIGHBOURS];

		//dynamic bodies
		if (data.numDynamicBodies >0) {

#ifdef GROUP_DYNAMIC_BODIES_NEIGHBORS_SEARCH
			ITER_NEIGHBORS_FROM_STRUCTURE(data.neighborsDataSetGroupedDynamicBodies_cuda, data.posBufferGroupedDynamicBodies,
				if (j<data.fluid_data_cuda->numParticles) {
					if (i != j) { WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, j);	nb_neighbors_fluid++; }
				}
				else {
					int body_id = 0; int count_particles_previous_bodies = data.fluid_data_cuda->numParticles;
					while ((count_particles_previous_bodies + data.vector_dynamic_bodies_data_cuda[body_id].numParticles)<=j) {
						count_particles_previous_bodies += data.vector_dynamic_bodies_data_cuda[body_id].numParticles;
						body_id++;
					}
					//*cur_neighbor_ptr++ = WRITE_DYNAMIC_BODIES_PARTICLES_INDEX(body_id, j-count_particles_previous_bodies);
					neighbors_solids[nb_neighbors_dynamic_objects] = WRITE_DYNAMIC_BODIES_PARTICLES_INDEX(body_id, j - count_particles_previous_bodies);
					nb_neighbors_dynamic_objects++;
				})
#else
			for (int id_body = 0; id_body < data.numDynamicBodies; ++id_body) {
				ITER_NEIGHBORS_FROM_STRUCTURE(data.vector_dynamic_bodies_data_cuda[id_body].neighborsDataSet, data.vector_dynamic_bodies_data_cuda[id_body].pos,
					*cur_neighbor_ptr++ = WRITE_DYNAMIC_BODIES_PARTICLES_INDEX(id_body, j); nb_neighbors_dynamic_objects++; )
			}
#endif

		}
		else {
			//fluid
			ITER_NEIGHBORS_FROM_STRUCTURE(data.fluid_data_cuda[0].neighborsDataSet, data.fluid_data_cuda[0].pos,
				if (i != j) { WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, j);	nb_neighbors_fluid++; });
		}

		//boundaries
#ifndef BENDER2019_BOUNDARIES
		ITER_NEIGHBORS_FROM_STRUCTURE(data.boundaries_data_cuda[0].neighborsDataSet, data.boundaries_data_cuda[0].pos,
			WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, j); nb_neighbors_boundary++; );
#endif

		//copy the dynamic bodies at the end
		for (int j = 0; j<nb_neighbors_dynamic_objects; ++j) {
			WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, neighbors_solids[j]);
		}

	}
	else {

		//uses the standart version
		//fluid
		if (is_fluid_container) {
#ifndef NEIGHBORS_RANGE_EXPLORATION
			ITER_NEIGHBORS_FROM_STRUCTURE(data.fluid_data_cuda[0].neighborsDataSet, data.fluid_data_cuda[0].pos,
				if (!is_fluid_container || i != j) { WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, j);	nb_neighbors_fluid++; });
#else
			//*
			if (true) {
				//here is a code that can explore chains of cells for hilber index
				bool found_none = true;
				int8_t count_analysed = 0;
				int last = -1;
				int min = 0;
				do {
					found_none = true;
					//get the minimum of the remainings
					int8_t k_min, m_min, n_min;
					for (int8_t k = -1; k < 2; ++k) {
						for (int8_t m = -1; m < 2; ++m) {
							for (int8_t n = -1; n < 2; ++n) {
								int id = COMPUTE_CELL_INDEX(x + n, y + k, z + m);
								if (id > last) {
									if (found_none) {
										min = id;
										found_none = false;
									}
									else if (id < min) {
										min = id;
									}
								}
							}
						}
					}

					if (found_none) {
						break;
					}
					last = min;

					unsigned int end = data.fluid_data_cuda->neighborsDataSet->cell_start_end[min + 1];
					for (unsigned int cur_particle = data.fluid_data_cuda->neighborsDataSet->cell_start_end[min]; cur_particle < end; ++cur_particle) {
						unsigned int j = data.fluid_data_cuda->neighborsDataSet->p_id_sorted[cur_particle];
						if ((pos - data.fluid_data_cuda->pos[j]).squaredNorm() < radius_sq) {
							if (!is_fluid_container || i != j) { WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, j);	nb_neighbors_fluid++; };
						}
					}

					count_analysed++;;

					//printf("count analysed // count contiguous: %i // %i\n", count_analysed, count_contiguous + 1);
				} while (count_analysed != 27);


					/*
					int count_found = 0;
					ITER_NEIGHBORS_FROM_STRUCTURE(data.fluid_data_cuda[0].neighborsDataSet, data.fluid_data_cuda[0].pos,
						if (!is_fluid_container || i != j) { count_found++; });

					if (nb_neighbors_fluid != count_found) {
						//printf("FLUID pb when storing neighbors old/new: %i // %i \n", count_found, nb_neighbors_fluid);

						int count_duplicate = 0;
						for (int k = 0; k < nb_neighbors_fluid; ++k) {
							int* neighbors_ptr = particleSet->getNeighboursPtr(i);
							if (neighbors_ptr[k*numParticles] == i) {
								printf("FLUID particle %i is inside it's own neighbors", i);
							}

							for (int l = k+1; l < nb_neighbors_fluid; ++l) {
								if (neighbors_ptr[k*numParticles] == neighbors_ptr[l*numParticles]) {
									count_duplicate++;
								}
							}
						}
						if (count_duplicate > 0) {
							printf("FLUID particle %i has duplicates inside it's neighboors, count: %i\n", i, count_duplicate);
						}
					}
					//*/
			}else {
				ITER_NEIGHBORS_FROM_STRUCTURE(data.fluid_data_cuda[0].neighborsDataSet, data.fluid_data_cuda[0].pos,
					if (!is_fluid_container || i != j) { WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, j);	nb_neighbors_fluid++; });

			}
#endif
		}

		//boundaries
#ifndef BENDER2019_BOUNDARIES

#ifndef NEIGHBORS_RANGE_EXPLORATION
		ITER_NEIGHBORS_FROM_STRUCTURE(data.boundaries_data_cuda[0].neighborsDataSet, data.boundaries_data_cuda[0].pos,
			if (is_fluid_container || i != j) { WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, j); nb_neighbors_boundary++; });
#else
		{
			//here is a code that can explore chains of cells for hilber index
			bool found_none = true;
			int8_t count_analysed = 0;
			int last = -1;
			int min = 0;
			do {
				found_none = true;
				//get the minimum of the remainings
				int8_t k_min, m_min, n_min;
				for (int8_t k = -1; k < 2; ++k) {
					for (int8_t m = -1; m < 2; ++m) {
						for (int8_t n = -1; n < 2; ++n) {
							int id = COMPUTE_CELL_INDEX(x + n, y + k, z + m);


							if (id > last) {
								if (found_none) {
									min = id;
									found_none = false;
								}
								else if (id < min) {
									min = id;
								}
							}
						}
					}
				}

				if (found_none) {
					break;
				}
				last = min;

				unsigned int end = data.boundaries_data_cuda->neighborsDataSet->cell_start_end[min + 1];
				for (unsigned int cur_particle = data.boundaries_data_cuda->neighborsDataSet->cell_start_end[min]; cur_particle < end; ++cur_particle) {
					unsigned int j = data.boundaries_data_cuda->neighborsDataSet->p_id_sorted[cur_particle];
					if ((pos - data.boundaries_data_cuda->pos[j]).squaredNorm() < radius_sq) {
						if (is_fluid_container || i != j) { WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, j); nb_neighbors_boundary++; };
					}
				}

				count_analysed ++;

				//printf("count analysed // count contiguous: %i // %i\n", count_analysed, count_contiguous + 1);
			} while (count_analysed != 27);

			/*
			//do a check
			int count_found = 0;
			ITER_NEIGHBORS_FROM_STRUCTURE(data.boundaries_data_cuda[0].neighborsDataSet, data.boundaries_data_cuda[0].pos,
				if (is_fluid_container || i != j) { count_found++; });

			if (nb_neighbors_boundary != count_found) {
				printf("BOUNDARIES pb when storing neighbors old/new: %i // %i \n", count_found, nb_neighbors_boundary);
			}
			//*/

		}

#endif

#endif

		if (data.numDynamicBodies > 0) {

#ifdef GROUP_DYNAMIC_BODIES_NEIGHBORS_SEARCH
			ITER_NEIGHBORS_FROM_STRUCTURE(data.neighborsDataSetGroupedDynamicBodies_cuda, data.posBufferGroupedDynamicBodies,
			{ int body_id = 0; int count_particles_previous_bodies = 0;
			while ((count_particles_previous_bodies + data.vector_dynamic_bodies_data_cuda[body_id].numParticles)<=j) {
				count_particles_previous_bodies += data.vector_dynamic_bodies_data_cuda[body_id].numParticles;
				body_id++;
			}
			int neighbor_idx= WRITE_DYNAMIC_BODIES_PARTICLES_INDEX(body_id, j - count_particles_previous_bodies);
			WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, neighbor_idx);
			nb_neighbors_dynamic_objects++; })
#else
			for (int id_body = 0; id_body < data.numDynamicBodies; ++id_body) {
				ITER_NEIGHBORS_FROM_STRUCTURE(data.vector_dynamic_bodies_data_cuda[id_body].neighborsDataSet, data.vector_dynamic_bodies_data_cuda[id_body].pos,
					{int neighbor_idx = WRITE_DYNAMIC_BODIES_PARTICLES_INDEX(id_body, j);
				    WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, neighbor_idx);
					nb_neighbors_dynamic_objects++;

					//printf("wrote a neighbor %i\n", nb_neighbors_boundary+nb_neighbors_fluid+nb_neighbors_dynamic_objects);
					});
			}
#endif

		}

	}

#ifdef SORT_NEIGHBORS
	//let's use an insertion sort
	//and for now i'll only sort the fluid and boundarries neighbors

	//the fluid
	int* neighbors_ptr = particleSet->getNeighboursPtr(i);
	for (int k = 1; k < nb_neighbors_fluid; ++k) {
		if (neighbors_ptr[k*numParticles] < neighbors_ptr[(k - 1)*numParticles]) {
			int stored_idx = neighbors_ptr[k*numParticles];
			int l = k;
			do {
				neighbors_ptr[l*numParticles] = neighbors_ptr[(l - 1)*numParticles];
				l--;
			} while ((l > 0) && (stored_idx < neighbors_ptr[(l - 1)*numParticles]));
			neighbors_ptr[l*numParticles] = stored_idx;
		}
	}

	//and the boundaries
	//*
	neighbors_ptr += nb_neighbors_fluid* numParticles;
	for (int k = 1; k < nb_neighbors_boundary; ++k) {
		if (neighbors_ptr[k*numParticles] < neighbors_ptr[(k - 1)*numParticles]) {
			int stored_idx = neighbors_ptr[k*numParticles];
			int l = k;
			do {
				neighbors_ptr[l*numParticles] = neighbors_ptr[(l - 1)*numParticles];
				l--;
			} while ((l > 0) && (stored_idx < neighbors_ptr[(l - 1)*numParticles]));
			neighbors_ptr[l*numParticles] = stored_idx;
		}
	}//*/

#endif // SORT_NEIGHBORS


	particleSet->numberOfNeighbourgs[3 * i] =  nb_neighbors_fluid;
	particleSet->numberOfNeighbourgs[3 * i + 1] = nb_neighbors_boundary;
	particleSet->numberOfNeighbourgs[3 * i + 2] = nb_neighbors_dynamic_objects;
	
	
	
	/*
	//simple splashless surface detection
	if (((nb_neighbors_fluid+nb_neighbors_boundary) < 35)&& (nb_neighbors_fluid + nb_neighbors_boundary) >15) {
		particleSet->color[i] = Vector3d(0, 1, 0);
	}
	//*/

	//memcpy((neighbors_buff + i*MAX_NEIGHBOURS*2), neighbors_fluid, sizeof(int)*nb_neighbors_fluid);
	//memcpy((neighbors_buff + i*MAX_NEIGHBOURS * 2 + MAX_NEIGHBOURS), neighbors_boundary, sizeof(int)*nb_neighbors_boundary);


}


///WARNING this version of the neighbor storage is special 
//it will fit what was given as parameter and as such may not correspond to the usual fluid/boudaries/solids
//the last parameter is so that more that 2 set can be given as parameters
//also when stored the stored index will obviously use the same compression method as the one used for the dynamic bodies in the noraml execution
__global__ void DFSPH_neighborsSearchSelf_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet_self,
	SPH::UnifiedParticleSet* particleSet_other,
	SPH::UnifiedParticleSet* particleSet_vect_additionals, int nb_additional_bodies, int iter_debug=-1) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet_self->numParticles) { return; }

	if (data.restriction_mode == 1) {
		int tag = particleSet_self->neighborsDataSet->cell_id[i];
		if (tag != TAG_ACTIVE && tag != TAG_ACTIVE_NEIGHBORS) {
			return;
		}
	}else if (data.restriction_mode == 2) {
		if (i >= (data.count_active + data.count_active_neighbors)) { return; }
	}

	ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, particleSet_self, i);

	//this variable is need for the interleave but I'm not modifying all the macro for only a single case
#ifdef INTERLEAVE_NEIGHBORS
	int numParticles = particleSet_self->numParticles;
#endif

	unsigned int nb_neighbors_fluid = 0;
	unsigned int nb_neighbors_boundary = 0;
	unsigned int nb_neighbors_dynamic_objects = 0;
	int* cur_neighbor_ptr = particleSet_self->getNeighboursPtr(i);
	//int neighbors_fluid[MAX_NEIGHBOURS];//doing it with local buffer was not faster
	//int neighbors_boundary[MAX_NEIGHBOURS];

	//uses the standart version
	//self

	ITER_NEIGHBORS_FROM_STRUCTURE(particleSet_self->neighborsDataSet, particleSet_self->pos,
		if (i != j) { WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, j);	nb_neighbors_fluid++; });

		
	//other
	ITER_NEIGHBORS_FROM_STRUCTURE(particleSet_other->neighborsDataSet, particleSet_other->pos,
		{ WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, j); nb_neighbors_boundary++; });


	//additional
	if (nb_additional_bodies > 0) {
		for (int id_body = 0; id_body < nb_additional_bodies; ++id_body) {
			ITER_NEIGHBORS_FROM_STRUCTURE(particleSet_vect_additionals[id_body].neighborsDataSet, particleSet_vect_additionals[id_body].pos,
				{ int neighbor_idx = WRITE_DYNAMIC_BODIES_PARTICLES_INDEX(id_body, j);
				WRITE_AND_ADVANCE_NEIGHBORS(cur_neighbor_ptr, neighbor_idx);
				nb_neighbors_dynamic_objects++;	});
		}
	}	

	particleSet_self->numberOfNeighbourgs[3 * i] = nb_neighbors_fluid;
	particleSet_self->numberOfNeighbourgs[3 * i + 1] = nb_neighbors_boundary;
	particleSet_self->numberOfNeighbourgs[3 * i + 2] = nb_neighbors_dynamic_objects;
}



__global__ void DFSPH_neighborsSearchBasic_kernel(unsigned int numFluidParticles, RealCuda radius,
	SPH::UnifiedParticleSet* fluid_data,
	SPH::UnifiedParticleSet* boundaries_data,
	SPH::UnifiedParticleSet* vect_dynamic_bodies, int nb_dynamic_bodies) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numFluidParticles) { return; }


	RealCuda radius_sq = radius;
	Vector3d pos = fluid_data->pos[i];
	radius_sq *= radius_sq;

	unsigned int nb_neighbors_fluid = 0;
	unsigned int nb_neighbors_boundary = 0;
	unsigned int nb_neighbors_dynamic_objects = 0;
	int* cur_neighbor_ptr = fluid_data->neighbourgs + i*MAX_NEIGHBOURS;

	for (int k = 0; k < fluid_data->numParticles; ++k) {
		if (i != k) {
			if ((fluid_data->pos[k] - pos).squaredNorm() < radius_sq) {
				*cur_neighbor_ptr++ = k;	nb_neighbors_fluid++;
			}
		}
	}

	/*
	for (int k = 0; k < boundaries_data->numParticles; ++k) {
	if ((boundaries_data->pos[k] - pos).squaredNorm() < radius_sq) {
	*cur_neighbor_ptr++ = k; nb_neighbors_boundary++;
	}
	}
	//*/

	/*
	for (int id_body = 0; id_body < nb_dynamic_bodies; ++id_body) {
	for (int k = 0; k < vect_dynamic_bodies[id_body].numParticles; ++k) {
	if ((vect_dynamic_bodies[id_body].pos[k] - pos).squaredNorm() < radius_sq) {
	*cur_neighbor_ptr++ = WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX(id_body, k); nb_neighbors_dynamic_objects++;
	}
	}
	}
	//*/


	fluid_data->numberOfNeighbourgs[3 * i] = nb_neighbors_fluid;
	fluid_data->numberOfNeighbourgs[3 * i + 1] = nb_neighbors_boundary;
	fluid_data->numberOfNeighbourgs[3 * i + 2] = nb_neighbors_dynamic_objects;

}




void cuda_neighborsSearch(SPH::DFSPHCData& data, bool need_sort) {

	static unsigned int* precomputedIndexPtr = NULL;

	if (precomputedIndexPtr == NULL) {
		precomputedIndexPtr = data.precomputedCellIndex;
	}
	data.precomputedCellIndex = precomputedIndexPtr;

	//std::chrono::steady_clock::time_point begin_global = std::chrono::steady_clock::now();
	static unsigned int time_count = 0;
	float time_global;
	static float time_avg_global = 0;
	time_count++;

	/*
	if (time_count<5) {
	cuda_shuffleData(data.fluid_data[0]);
	std::cout << "randomizing particle order" << std::endl;
	}
	//*/


	if (need_sort) {
		//std::cout<<"doing full neighbor search"<<std::endl;
	}

	bool old_fluid_aggregated = data.is_fluid_aggregated;
	cudaError_t cudaStatus;
	if (true) {
		if (need_sort&&data.is_fluid_aggregated) {
			data.is_fluid_aggregated = false;
		}


		//*
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		//*/

		//first let's generate the cell start end for the dynamic bodies
#ifdef GROUP_DYNAMIC_BODIES_NEIGHBORS_SEARCH
		cuda_initNeighborsSearchDataSetGroupedDynamicBodies(data);
#else
		for (int i = 0; i < data.numDynamicBodies; ++i) {
			SPH::UnifiedParticleSet& body = data.vector_dynamic_bodies_data[i];
			body.initNeighborsSearchData(data, false);
		}
#endif
		std::chrono::steady_clock::time_point middle = std::chrono::steady_clock::now();

		//no need to ever do it forthe boundaries since they don't ever move

		//now update the cell start end of the fluid particles
		if ((!data.is_fluid_aggregated) || data.numDynamicBodies<1) {

			//since it the init iter I'll sort both even if it's the boundaries
			static int step_count = 0;
			step_count++;

			data.fluid_data->initNeighborsSearchData(data, need_sort);


			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "before neighbors search: %d\n", (int)cudaStatus);
				exit(1598);
			}


		}

		/*

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		float time0;
		float time1;
		static float time_avg = 0;
		time0 = std::chrono::duration_cast<std::chrono::nanoseconds> (middle - begin).count() / 1000000.0f;
		time1 = std::chrono::duration_cast<std::chrono::nanoseconds> (end - middle).count() / 1000000.0f;

		time_avg += time0+time1;
		printf("Time to generate cell start end: %f ms (%f,%f)   avg: %f ms \n", time0+time1,time0,time1, time_avg / time_count);

		if (time_count > 150) {
		time_avg = 0;
		}
		//*/


	}
	//and we can now do the actual search of the neaighbor for eahc fluid particle
#ifdef STORE_PARTICLE_NEIGHBORS

	if (true)
	{
		//*
		float time;
		static float time_avg = 0;
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		//*/

		//cuda way
		int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);

		//*
		DFSPH_neighborsSearch_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr);

		//*
		if (data.boundaries_data->has_factor_computation) {
			DFSPH_neighborsSearch_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data_cuda);
		}
		gpuErrchk(cudaDeviceSynchronize());
		//*/

		//*/
		/*
		//this test show that even just computing the neighbors for the fluid particle
		//with a basic method take more time than building the whole structure
		DFSPH_neighborsSearchBasic_kernel << <numBlocks, BLOCKSIZE >> > (data.numFluidParticles,
		data.getKernelRadius(),
		data.fluid_data_cuda,
		data.boundaries_data_cuda,
		data.vector_dynamic_bodies_data_cuda, data.numDynamicBodies);
		//*/


		/*
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		time = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() / 1000000.0f;

		time_avg += time;
		printf("Time to generate neighbors buffers: %f ms   avg: %f ms \n", time, time_avg / time_count);

		if (time_count > 150) {
		time_avg = 0;
		time_count = 0;
		}
		//*/



		/*
		{
			{
			std::cout << "test: " << data.fluid_data->neighborsDataSet->cell_id_sorted[0] << "   " <<
				data.fluid_data->neighborsDataSet->cell_id_sorted[10] << "   " <<
				data.fluid_data->neighborsDataSet->cell_id_sorted[50] << "   " << std::endl;

			int count_valid = 0;
			for (int i = 0; i < data.fluid_data->numParticles; ++i) {
				if (data.fluid_data->neighborsDataSet->cell_id_sorted[i] != 0) {
					count_valid++;
				}
			}
			std::cout << "test2: " << count_valid << std::endl;
			}
		}
			//*/
		/*
		{
			//a simple check to know the max nbr of neighbors
			static int absolute_max = 0;
			int max = 0;

			static int absolute_max_d[3] = { 0 };
			int max_d[3] = { 0 };



			for (int j = 0; j < data.fluid_data->numParticles; j++)
			{

				//check the global value
				int count_neighbors = 0;
				for (int k = 0; k < 3; ++k) {
					count_neighbors += data.fluid_data->getNumberOfNeighbourgs(j, k);
				}
				if (count_neighbors > max) { 
					max = count_neighbors;
				}

				//chekc the max for each category
				for (unsigned int k = 0; k < 3; ++k) {
					if ((int)data.fluid_data->getNumberOfNeighbourgs(j,k) > max_d[k])max_d[k] = data.fluid_data->getNumberOfNeighbourgs(j,k);
				}

			}

			if (max>absolute_max)absolute_max = max;

			for (unsigned int k = 0; k < 3; ++k) {
				if (max_d[k]>absolute_max_d[k])absolute_max_d[k] = max_d[k];
			}

			printf("max nbr of neighbors %d  (%d) \n", absolute_max, max);
			printf("absolute??max  fluid // boundaries // bodies   %d // %d // %d  ?? %d // %d // %d\n",
			absolute_max_d[0], absolute_max_d[1], absolute_max_d[2], max_d[0], max_d[1], max_d[2]);
		}


		//*/
		/*
		{
		//another test ot be sure the contruction of the boundries neighbors works orrectly
		if (data.boundaries_data->has_factor_computation) {
		//a simple check to know the  nbr of neighbors of the first boundries particle

		int nb_neighbors[3] = { 0 };

		for (int k = 0; k < 3; ++k) {
		nb_neighbors[k] = data.boundaries_data->getNumberOfNeighbourgs(0, k);
		}

		printf(" nbr of neighbors %d     fluid // boundaries // bodies   %d // %d // %d\n",
		nb_neighbors[0] + nb_neighbors[1] + nb_neighbors[2], nb_neighbors[0], nb_neighbors[1], nb_neighbors[2]);
		}
		}

		//*/

		
	}
#endif //STORE_PARTICLE_NEIGHBORS

	//reactive the aggragation if we desactivated it because a sort was required
	if (need_sort&&old_fluid_aggregated) {
		data.is_fluid_aggregated = true;
	}

	data.precomputedCellIndex = NULL;

	/*
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	time_global = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin_global).count() / 1000000.0f;

	time_avg_global += time_global;
	printf("time taken by the neighbor function: %f ms   avg: %f ms \n", time_global, time_avg_global / time_count);
	//*/
}

void cuda_updateNeighborsStorage(SPH::DFSPHCData& data, SPH::UnifiedParticleSet& particleSet, int iter_debug) {
	int count_particles = particleSet.numParticles;
	if (data.restriction_mode == 2) {
		count_particles = (data.count_active + data.count_active_neighbors);
	}

	int numBlocks = calculateNumBlocks(count_particles);

	//*
	
	DFSPH_neighborsSearchSelf_kernel<< <numBlocks, BLOCKSIZE >> > (data, particleSet.gpu_ptr,
			data.boundaries_data->gpu_ptr, data.vector_dynamic_bodies_data_cuda, data.numDynamicBodies, iter_debug);

	
	gpuErrchk(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////
/////////             OTHERS           /////////////
////////////////////////////////////////////////////

__global__ void DFSPH_update_vel_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }


	if (m_data.restriction_mode == 1) {
		if ((particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE)) {
			return;
		}
	}
	else if (m_data.restriction_mode == 2) {
		if (i >= (m_data.count_active)) { return; }
	}

	particleSet->vel[i] += m_data.h * particleSet->acc[i];

#ifdef USE_WARMSTART	
	//done here to have one less kernel
	particleSet->kappa[i] = MAX_MACRO_CUDA(particleSet->kappa[i] * m_data.h_ratio_to_past2, -0.5);
	//this line is from the most recent impelmentation but it does not change anything
	//particleSet->kappa[i] = MAX_MACRO_CUDA(particleSet->kappa[i] * m_data.h_ratio_to_past2, -0.5 * m_data.density0*m_data.density0);
#endif

}




void cuda_update_vel(SPH::DFSPHCData& data) {
	int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);


#ifdef USE_WARMSTART
	/*
	Real count_capped = 0;

	for (int i = 0; i < (data.fluid_data[0].numParticles); ++i) {
		if ( data.fluid_data->kappa[i] * data.h_ratio_to_past2 < -0.5) {
			count_capped++;
		}
	}
	//*/
#endif


	DFSPH_update_vel_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_update_vel failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
	

#ifdef USE_WARMSTART
	/*
	RealCuda avg_warmstart = 0;
	for (int i = 0; i < (data.fluid_data[0].numParticles); ++i) {
		avg_warmstart += data.fluid_data->kappa[i];
		//std::cout << "pressure wamstart post clamping: " << i << " " << data.fluid_data->kappa[i] << std::endl;
	}
	avg_warmstart /= data.fluid_data[0].numParticles;
	//std::cout << "pressure wamstart avg: " << avg_warmstart << std::endl;

	static std::vector<Real> avgs_warmstart;
	static std::vector<int> counts_capped;
	avgs_warmstart.push_back(avg_warmstart);
	counts_capped.push_back(count_capped);
	int count = 1000;
	if (avgs_warmstart.size() == count) {
		for (int i = 0; i < avgs_warmstart.size(); i++) {
			std::cout << "pressure_wamstart_avg: " << i << " " << avgs_warmstart[i] << " " << counts_capped[i] << std::endl;
		}
		exit(0);
	}
	//*/
#endif
}


__global__ void DFSPH_update_pos_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (data.restriction_mode == 1) {
		if ((particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE)) {
			return;
		}
	}
	else if (data.restriction_mode == 2) {
		if (i >= (data.count_active)) { return; }
	}

	if (data.damp_borders) {
		/*
		RealCuda max_vel_sq = (data.particleRadius / 2.0f) / data.h;
		max_vel_sq *= max_vel_sq;
		RealCuda cur_vel_sq = particleSet->vel[i].squaredNorm();
		if (cur_vel_sq> max_vel_sq)
		{
		particleSet->vel[i] *= max_vel_sq / cur_vel_sq;
		}//*/

		RealCuda affected_distance_sq = data.particleRadius * 6;
		affected_distance_sq *= affected_distance_sq;

		for (int k = 0; k < data.damp_planes_count; ++k) {
			Vector3d plane = data.damp_planes[k];
			if ((particleSet->pos[i] * plane.abs() / plane.norm() - plane).squaredNorm() < affected_distance_sq) {
				if (data.damp_borders_steps_count>1) {
					RealCuda max_vel_sq = (data.particleRadius / 25.0f) / data.h;
					max_vel_sq *= max_vel_sq;
					RealCuda cur_vel_sq = particleSet->vel[i].squaredNorm();
					if (cur_vel_sq> max_vel_sq)
					{
						particleSet->vel[i] *= max_vel_sq / cur_vel_sq;
					}
					//if we triggered once no need to check for the other planes
					break;
				}
				else {
					particleSet->vel[i] *= 0.1;
				}
			}
		}
	}

	if (data.cancel_wave) {
		RealCuda affected_distance_sq = data.getKernelRadius();
		affected_distance_sq *= affected_distance_sq;
		for (int k = 0; k < 2; ++k) {
			Vector3i plane = data.cancel_wave_planes[k];
			if ((particleSet->pos[i] * plane.abs() / plane.norm() - plane).squaredNorm() < affected_distance_sq) {
				//particleSet->vel[i]=Vector3d(0,1,0);
			}
		}
		Vector3d axis = data.cancel_wave_planes[0].abs() / data.cancel_wave_planes[0].norm();
		if (particleSet->pos[i].y>data.cancel_wave_lowest_point) {
			if ((particleSet->pos[i].dot(axis))<(data.cancel_wave_planes[0].dot(axis))) {
				if ((particleSet->vel[i].dot(axis))<0) {
					particleSet->vel[i] -= particleSet->vel[i] * axis;
				}
			}

			if ((particleSet->pos[i].dot(axis))>(data.cancel_wave_planes[1].dot(axis))) {
				if ((particleSet->vel[i].dot(axis))>0) {
					particleSet->vel[i] -= particleSet->vel[i] * axis;
				}
			}
		}

	}

	if (data.restriction_mode == 1) {
		if (particleSet->pos[i].y > 1.95) {
			if (particleSet->vel[i].y > 0) {
				//particleSet->vel[i].y *= 0.25;
			}
		}
	}

	/*
	if (particleSet->density[i] < 900) {
		particleSet->color[i] = Vector3d(0, 1, 0);
	}
	//*/
	particleSet->pos[i] += data.h * particleSet->vel[i];
}



void cuda_update_pos(SPH::DFSPHCData& data) {


	int numBlocks = calculateNumBlocks(data.fluid_data[0].numParticles);
	DFSPH_update_pos_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);


	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_update_pos failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	if (data.damp_borders) {
		for (int k = 0; k < data.damp_planes_count; ++k) {
			Vector3d plane = data.damp_planes[k];
			//std::cout << "damping plane: " << plane.x << "  " << plane.y << "  " << plane.z << std::endl;
		}
		data.damp_borders_steps_count--;
		if (data.damp_borders_steps_count == 0) {
			data.damp_borders = false;
			data.damp_planes_count = 0;
		}
	}
	if (data.cancel_wave) {
		//*

		for (int k = 0; k < 2; ++k) {
			Vector3d plane = data.cancel_wave_planes[k];
			std::cout << "cancel wave plane: " << plane.x << "  " << plane.y << "  " << plane.z << std::endl;
		}
		data.cancel_wave_steps_count--;
		if (data.cancel_wave_steps_count == 0) {
			data.cancel_wave = false;
		}
		//*/
	}

}




__global__ void DFSPH_CFL_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet particleSet, RealCuda* maxVel) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.fluid_data[0].numParticles) { return; }

	for (unsigned int i = 0; i < m_data.fluid_data[0].numParticles; i++)
	{
		const RealCuda velMag = (particleSet.vel[i] + particleSet.acc[i] * m_data.h).squaredNorm();
		if (velMag > *maxVel)
			*maxVel = velMag;
	}
}

__global__ void DFSPH_CFLVelSquaredNorm_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet, RealCuda* sqaredNorm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	sqaredNorm[i] = (particleSet->vel[i] + particleSet->acc[i] * m_data.h).squaredNorm();
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
			cudaMallocManaged(&(temp_buff), m_data.fluid_data[0].numParticles * sizeof(RealCuda));
		}
		int numBlocks = calculateNumBlocks(m_data.fluid_data[0].numParticles);
		DFSPH_CFLVelSquaredNorm_kernel << <numBlocks, BLOCKSIZE >> > (m_data, m_data.fluid_data[0].gpu_ptr, temp_buff);

		cudaError_t cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cuda_cfl squared norm failed: %d\n", (int)cudaStatus);
			exit(1598);
		}

		// Determine temporary device storage requirements
		static void     *d_temp_storage = NULL;
		static size_t   temp_storage_bytes = 0;
		if (d_temp_storage == NULL) {
			cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, temp_buff, out_buff, m_data.fluid_data[0].numParticles);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
		}
		// Run max-reduction
		cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, temp_buff, out_buff, m_data.fluid_data[0].numParticles);

	}
	else {
		//manual
		int *d_mutex;
		cudaMalloc((void**)&d_mutex, sizeof(int));
		cudaMemset(d_mutex, 0, sizeof(float));

		int numBlocks = calculateNumBlocks(m_data.fluid_data[0].numParticles);
		DFSPH_CFLAdvanced_kernel << < numBlocks, BLOCKSIZE >> > (m_data, out_buff, d_mutex, m_data.fluid_data[0].numParticles);

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




//this is the bases for all kernels based function
//I also use that kernel to reset the force

__global__ void DFSPH_updateDynamicObjectParticles_kernel(int numParticles, Vector3d* pos, Vector3d* vel, Vector3d* pos0,
	Vector3d position, Vector3d velocity, Quaternion q, Vector3d angular_vel, Vector3d* F) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numParticles) { return; }

	//reset the force
	F[i] = Vector3d(0, 0, 0);

	//update location and velocity
	pos[i] = q.rotate(pos0[i]) + position;
	vel[i] = angular_vel.cross(pos[i] - position) + velocity;

}

void update_dynamicObject_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& particle_set) {


	if (particle_set.is_dynamic_object) {
		int numBlocks = calculateNumBlocks(particle_set.numParticles);


		//update the particle location and velocity
		DFSPH_updateDynamicObjectParticles_kernel << <numBlocks, BLOCKSIZE >> > (particle_set.numParticles,
			particle_set.pos, particle_set.vel, particle_set.pos0,
			particle_set.rigidBody_cpu->position, particle_set.rigidBody_cpu->velocity,
			particle_set.rigidBody_cpu->q, particle_set.rigidBody_cpu->angular_vel,
			particle_set.F);

		//also we can use that time to reset the force buffer
		//directly done in the other kernel
		//DFSPH_setVector3dBufferToZero_kernel << <numBlocks, BLOCKSIZE >> > (container.F, container.numParticles);

		cudaError_t cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "update_dynamicObject_UnifiedParticleSet_cuda failed: %d\n", (int)cudaStatus);
			exit(1369);
		}
	}


}



__global__ void compute_dynamic_body_particle_mass_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	Real delta = data.W_zero;

	RealCuda radius_sq = data.getKernelRadius();
	Vector3d pos = particleSet->pos[i];
	Vector3d pos_cell = (pos / radius_sq) + data.gridOffset; //on that line the radius is not yet squared
	pos_cell.toFloor();
	int x = pos_cell.x;
	int y = pos_cell.y;
	int z = pos_cell.z;
	radius_sq *= radius_sq;


	//since this version use the std index to be able to iterate on 3 successive cells
	//I can do the -1 at the start on x.
	//one thing: it x=0 then we can only iterate 2 cells at a time
	unsigned int successive_cells_count = (x > 0) ? 3 : 2;
	x = (x > 0) ? x - 1 : x;


	const SPH::UnifiedParticleSet& body = *particleSet;
	for (int k = -1; k < 2; ++k) {
		for (int m = -1; m < 2; ++m) {
			unsigned int cur_cell_id = COMPUTE_CELL_INDEX(x, y + k, z + m);
			unsigned int end = body.neighborsDataSet->cell_start_end[cur_cell_id + successive_cells_count];
			for (unsigned int cur_particle = body.neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {
				unsigned int j = body.neighborsDataSet->p_id_sorted[cur_particle];
				if ((pos - body.pos[j]).squaredNorm() < radius_sq) {
					if (i != j) { delta += KERNEL_W(data,pos - body.pos[j]); }
				}
			}
		}
	}


	const Real volume = 1.0 / delta;
	particleSet->mass[i] = particleSet->density0 * volume;
	particleSet->mass[i] = data.fluid_data_cuda->getMass(0);
}



__global__ void refine_dynamic_body_particle_mass_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	//the factor is due to the fact that we compensate only a part of the error (proportional to the importance of the mass in the density
	//particleSet->getMass(i) += (particleSet->getMass(i) * data.W_zero / particleSet->density[i])*(data.density0 - particleSet->density[i]) / (data.W_zero);
	particleSet->mass[i] += (0.3)*(data.density0 - particleSet->density[i]) / (data.W_zero);
}


__global__ void compute_boundaries_density_error_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, RealCuda* err, int* err_max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	RealCuda density = particleSet->getMass(i) * data.W_zero;

	//*
	ITER_NEIGHBORS_INIT(data, particleSet, i);

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	ITER_NEIGHBORS_BOUNDARIES(data, particleSet,
		i,
		density += body.getMass(neighborIndex) * KERNEL_W(data,particleSet->pos[i] - body.pos[neighborIndex]);
	);
	//*/
	/*
	RealCuda radius_sq = data.getKernelRadius();
	Vector3d pos_cell = (pos / radius_sq) + data.gridOffset; //on that line the radius is not yet squared
	pos_cell.toFloor();
	int x = pos_cell.x;
	int y = pos_cell.y;
	int z = pos_cell.z;
	radius_sq *= radius_sq;


	//since this version use the std index to be able to iterate on 3 successive cells
	//I can do the -1 at the start on x.
	//one thing: it x=0 then we can only iterate 2 cells at a time
	unsigned int successive_cells_count = (x > 0) ? 3 : 2;
	x = (x > 0) ? x - 1 : x;


	const SPH::UnifiedParticleSet& body = *particleSet;
	for (int k = -1; k < 2; ++k) {
		for (int m = -1; m < 2; ++m) {
			unsigned int cur_cell_id = COMPUTE_CELL_INDEX(x, y + k, z + m);
			unsigned int end = body.neighborsDataSet->cell_start_end[cur_cell_id + successive_cells_count];
			for (unsigned int cur_particle = body.neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {
				unsigned int j = body.neighborsDataSet->p_id_sorted[cur_particle];
				if ((pos - body.pos[j]).squaredNorm() < radius_sq) {
					if (i != j) { density += particleSet->getMass(j) * KERNEL_W(data,pos - body.pos[j]); }
				}
			}
		}
	}
	//*/

	particleSet->density[i] = density;
	const RealCuda error =  abs(data.density0-density);
	atomicAdd(err, error);
	atomicMax(err_max, (int)(error*10000));
}

void compute_UnifiedParticleSet_particles_mass_cuda(SPH::DFSPHCData& data, SPH::UnifiedParticleSet& container) {
	int numBlocks = calculateNumBlocks(container.numParticles);
	
	bool old_destructor_status = data.destructor_activated;
	data.destructor_activated = false;

	container.initNeighborsSearchData(data, false, false);
	//init the neighbors
	bool fluid_agg = data.is_fluid_aggregated;
	data.is_fluid_aggregated=false;
	//DFSPH_neighborsSearch_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data_cuda);
	data.is_fluid_aggregated = fluid_agg;

	//compute_dynamic_body_particle_mass_kernel << <numBlocks, BLOCKSIZE >> > (data, container.gpu_ptr);
	//gpuErrchk(cudaDeviceSynchronize());



	bool refine_masses = false;//This is a test using relaxed jacobi to calculate the true mass of the particle
	if (refine_masses) {


		RealCuda* err;
		int* err_max;
		cudaMallocManaged(&(err), sizeof(RealCuda));
		cudaMallocManaged(&(err_max), sizeof(int));
		*err = 0.0;
		*err_max = 0;

		//calc the error on the density
		compute_boundaries_density_error_kernel << <numBlocks, BLOCKSIZE >> > (data, container.gpu_ptr, err, err_max);
		gpuErrchk(cudaDeviceSynchronize());


		RealCuda target_error = data.density0 / 100.0*0.1;
		RealCuda target_error_max = data.density0 / 100.0*0.1;
		RealCuda avg_err = (*err) / container.numParticles;
		RealCuda err_max_float = (*err_max)/ 10000.0;
		*err = 0;
		*err_max = 0;
		std::cout << "current density error: " << avg_err << " // " << err_max_float << "  target error: " << target_error << std::endl;

		std::chrono::steady_clock::time_point begin= std::chrono::steady_clock::now();

		//and refine it
		//while (avg_err>(target_error)|| target_error_max>15)
		for (int i =0;i<100;++i )
		{
			//refine the values
			refine_dynamic_body_particle_mass_kernel << <numBlocks, BLOCKSIZE >> > (data, container.gpu_ptr);
			gpuErrchk(cudaDeviceSynchronize());

			//compute the new error
			compute_boundaries_density_error_kernel << <numBlocks, BLOCKSIZE >> > (data, container.gpu_ptr, err, err_max);
			gpuErrchk(cudaDeviceSynchronize());

			avg_err = (*err) / container.numParticles;
			err_max_float = (*err_max)/ 10000.0;
			*err = 0;
			*err_max = 0;
			//std::cout << "current density error: " << avg_err << " // " << err_max_float << "  target error: " << target_error << std::endl;
		}

		std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
		float time = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() / 1000000.0f;
		std::cout << "current density error: " << avg_err << " // " << err_max_float << "  computation_time: " << time << std::endl;


		if (true ) {
			std::string filename = "boundaries density adv.csv";
			std::remove(filename.c_str());
			std::ofstream myfile;
			myfile.open(filename, std::ios_base::app);
			if (myfile.is_open()) {
				for (int i = 0; i < data.boundaries_data->numParticles; ++i) {
					myfile << i << ", " << container.getNumberOfNeighbourgs(i, 0)
						<< ", " << container.getNumberOfNeighbourgs(i, 1)
						<< ", " << container.getNumberOfNeighbourgs(i, 2)
						<< ", " << container.density[i] << std::endl;
				}
				//myfile << total_time / (count_steps + 1) << ", " << m_iterations << ", " << m_iterationsV << std::endl;;
				myfile.close();
			}
			else {
				std::cout << "failed to open file: " << filename << "   reason: " << std::strerror(errno) << std::endl;
			}
		}

		CUDA_FREE_PTR(err);
		CUDA_FREE_PTR(err_max);
	}
	data.destructor_activated = old_destructor_status;
}


