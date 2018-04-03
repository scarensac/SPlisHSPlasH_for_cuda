
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DFSPH_cuda_basic.h"
#include <stdio.h>
#include "DFSPH_c_arrays_structure.h"
#include "cub.cuh"

#define BLOCKSIZE 256
#define m_eps 1.0e-5
#define CELL_ROW_LENGTH 256
#define CELL_COUNT CELL_ROW_LENGTH*CELL_ROW_LENGTH*CELL_ROW_LENGTH

#define USE_WARMSTART
#define USE_WARMSTART_V

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

__device__ void computeDensityChange(SPH::DFSPHCData& m_data, const unsigned int index) {
	unsigned int numNeighbors = m_data.getNumberOfNeighbourgs(index);
	// in case of particle deficiency do not perform a divergence solve
	if (numNeighbors < 20) {
		for (unsigned int pid = 1; pid < 2; pid++)
		{
			numNeighbors += m_data.getNumberOfNeighbourgs(index, pid);
		}
	}
	if (numNeighbors < 20) {
		m_data.densityAdv[index] = 0;
	}
	else {
		Real densityAdv = 0.0;
		const Vector3d &xi = m_data.posFluid[index];
		const Vector3d &vi = m_data.velFluid[index];
		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(index); j++)
		{
			const unsigned int neighborIndex = m_data.getNeighbour(index, j);
			densityAdv += m_data.mass[neighborIndex] * (vi - m_data.velFluid[neighborIndex]).dot(m_data.gradW(xi - m_data.posFluid[neighborIndex]));
		}

		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int pid = 1; pid < 2; pid++)
		{
			//numNeighbors += m_data.getNumberOfNeighbourgs(index, pid);
			for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(index, pid); j++)
			{
				const unsigned int neighborIndex = m_data.getNeighbour(index, j, pid);
				densityAdv += m_data.boundaryPsi[neighborIndex] * (vi - m_data.velBoundary[neighborIndex]).dot(m_data.gradW(xi - m_data.posBoundary[neighborIndex]));
			}
		}

		// only correct positive divergence
		m_data.densityAdv[index] = fmax(densityAdv, 0.0);
	}
}
template <bool warm_start> __device__ void divergenceSolveParticle(SPH::DFSPHCData& m_data, const unsigned int i) {
	Vector3d v_i = Vector3d(0, 0, 0);
	//////////////////////////////////////////////////////////////////////////
	// Evaluate rhs
	//////////////////////////////////////////////////////////////////////////
	const Real ki = (warm_start) ? m_data.kappaV[i] : (m_data.densityAdv[i])*m_data.factor[i];

#ifdef USE_WARMSTART_V
	if (!warm_start) { m_data.kappaV[i] += ki; }
#endif

	const Vector3d &xi = m_data.posFluid[i];


	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i); j++)
	{
		const unsigned int neighborIndex = m_data.getNeighbour(i, j);
		const Real kSum = (ki + ((warm_start) ? m_data.kappaV[neighborIndex] : (m_data.densityAdv[neighborIndex])*m_data.factor[neighborIndex]));
		if (fabs(kSum) > m_eps)
		{
			// ki, kj already contain inverse density
			v_i += kSum *  m_data.mass[neighborIndex] * m_data.gradW(xi - m_data.posFluid[neighborIndex]);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	if (fabs(ki) > m_eps)
	{
		for (unsigned int pid = 1; pid < 2; pid++)
		{
			for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i, pid); j++)
			{
				const unsigned int neighborIndex = m_data.getNeighbour(i, j, pid);
				///TODO fuse those lines
				const Vector3d delta = ki * m_data.boundaryPsi[neighborIndex] * m_data.gradW(xi - m_data.posBoundary[neighborIndex]);
				v_i += delta;// ki already contains inverse density

							 ///TODO reactivate this for objects see theoriginal sign to see the the actual sign
							 //m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * ki * grad_p_j;
			}
		}
	}

	m_data.velFluid[i] += v_i*m_data.h;
}
__device__ void computeDensityAdv(SPH::DFSPHCData& m_data, const unsigned int index) {
	const Vector3d &xi = m_data.posFluid[index];
	const Vector3d &vi = m_data.velFluid[index];
	Real delta = 0.0;

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(index); j++)
	{
		const unsigned int neighborIndex = m_data.getNeighbour(index, j);
		delta += m_data.mass[neighborIndex] * (vi - m_data.velFluid[neighborIndex]).dot(m_data.gradW(xi - m_data.posFluid[neighborIndex]));
	}

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int pid = 1; pid < 2; pid++)
	{
		for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(index, pid); j++)
		{
			const unsigned int neighborIndex = m_data.getNeighbour(index, j, pid);
			delta += m_data.boundaryPsi[neighborIndex] * (vi - m_data.velBoundary[neighborIndex]).dot(m_data.gradW(xi - m_data.posBoundary[neighborIndex]));
		}
	}
	m_data.densityAdv[index] = fmax(m_data.density[index] + m_data.h_future*delta - m_data.density0, 0.0);
}
template <bool warm_start> __device__ void pressureSolveParticle(SPH::DFSPHCData& m_data, const unsigned int i) {
	//////////////////////////////////////////////////////////////////////////
	// Evaluate rhs
	//////////////////////////////////////////////////////////////////////////
	const Real ki = (warm_start) ? m_data.kappa[i] : (m_data.densityAdv[i])*m_data.factor[i];

#ifdef USE_WARMSTART
	if (!warm_start) { m_data.kappa[i] += ki; }
#endif


	Vector3d v_i = Vector3d(0, 0, 0);
	const Vector3d &xi = m_data.posFluid[i];

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i); j++)
	{
		const unsigned int neighborIndex = m_data.getNeighbour(i, j);
		const Real kSum = (ki + ((warm_start) ? m_data.kappa[neighborIndex] : (m_data.densityAdv[neighborIndex])*m_data.factor[neighborIndex]));
		if (fabs(kSum) > m_eps)
		{
			// ki, kj already contain inverse density
			v_i += kSum * m_data.mass[neighborIndex] * m_data.gradW(xi - m_data.posFluid[neighborIndex]);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	if (fabs(ki) > m_eps)
	{
		for (unsigned int pid = 1; pid < 2; pid++)
		{
			for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i, pid); j++)
			{
				const unsigned int neighborIndex = m_data.getNeighbour(i, j, pid);
				const Vector3d delta = ki * m_data.boundaryPsi[neighborIndex] * m_data.gradW(xi - m_data.posBoundary[neighborIndex]);

				v_i += delta;// ki already contains inverse density

							 ///TODO reactivate the external forces check the original formula to be sure of the sign
							 //m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * ki * grad_p_j;
			}
		}
	}
	// Directly update velocities instead of storing pressure accelerations
	m_data.velFluid[i] += v_i*m_data.h_future;
}

__global__ void DFSPH_divergence_warmstart_init_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	m_data.kappaV[i] = 0.5*fmax(m_data.kappaV[i] * m_data.h_ratio_to_past, -0.5);
	computeDensityChange(m_data, i);
}
void cuda_divergence_warmstart_init(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_divergence_warmstart_init_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}

template<bool warmstart> __global__ void DFSPH_divergence_compute_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	if (warmstart) {
		if (m_data.densityAdv[i] > 0.0) {
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
		fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
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
			const Vector3d &xi = m_data.posFluid[i];
			Real sum_grad_p_k = 0.0;
			Vector3d grad_p_i;
			grad_p_i.setZero();

			Real density = m_data.mass[i] * m_data.W_zero;

			//////////////////////////////////////////////////////////////////////////
			// Fluid
			//////////////////////////////////////////////////////////////////////////
			for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i); j++)
			{
				const unsigned int neighborIndex = m_data.getNeighbour(i, j);
				const Vector3d &xj = m_data.posFluid[neighborIndex];
				density += m_data.mass[neighborIndex] * m_data.W(xi - xj);
				const Vector3d grad_p_j = m_data.mass[neighborIndex] * m_data.gradW(xi - xj);
				sum_grad_p_k += grad_p_j.squaredNorm();
				grad_p_i += grad_p_j;
			}

			//////////////////////////////////////////////////////////////////////////
			// Boundary
			//////////////////////////////////////////////////////////////////////////
			for (unsigned int pid = 1; pid < 2; pid++)
			{
				for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i, pid); j++)
				{
					const unsigned int neighborIndex = m_data.getNeighbour(i, j, pid);
					const Vector3d &xj = m_data.posBoundary[neighborIndex];
					density += m_data.boundaryPsi[neighborIndex] * m_data.W(xi - xj);
					const Vector3d grad_p_j = m_data.boundaryPsi[neighborIndex] * m_data.gradW(xi - xj);
					sum_grad_p_k += grad_p_j.squaredNorm();
					grad_p_i += grad_p_j;
				}
			}

			sum_grad_p_k += grad_p_i.squaredNorm();

			//////////////////////////////////////////////////////////////////////////
			// Compute pressure stiffness denominator
			//////////////////////////////////////////////////////////////////////////
			m_data.factor[i] = (-m_data.invH / (fmax(sum_grad_p_k, m_eps)));
			m_data.density[i] = density;

		}

#ifdef USE_WARMSTART_V
		m_data.kappaV[i] = 0.0;
#endif
	}

}
void cuda_divergence_init(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_divergence_init_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}

__global__ void DFSPH_divergence_loop_end_kernel(SPH::DFSPHCData m_data, Real* avg_density_err) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	computeDensityChange(m_data, i);
	atomicAdd(avg_density_err, m_data.densityAdv[i]);
}
Real cuda_divergence_loop_end(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	Real* avg_density_err;
	cudaMallocManaged(&(avg_density_err), sizeof(Real));
	*avg_density_err = 0.0;
	DFSPH_divergence_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data, avg_density_err);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	Real result = *avg_density_err;
	cudaFree(avg_density_err);
	return result;
}

__global__ void DFSPH_viscosityXSPH_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	//I set the gravitation directly here to lover the number of kernels
	Vector3d ai = Vector3d(0, 0, 0);
	const Vector3d &xi = m_data.posFluid[i];
	const Vector3d &vi = m_data.velFluid[i];

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i); j++)
	{
		const unsigned int neighborIndex = m_data.getNeighbour(i, j);

		// Viscosity
		ai -= m_data.invH * m_data.viscosity * (m_data.mass[neighborIndex] / m_data.density[neighborIndex]) *
			(vi - m_data.velFluid[neighborIndex]) * m_data.W(xi - m_data.posFluid[neighborIndex]);
	}

	m_data.accFluid[i] = m_data.gravitation + ai;
}
void cuda_viscosityXSPH(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_viscosityXSPH_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}

__global__ void DFSPH_CFL_kernel(SPH::DFSPHCData m_data, Real* maxVel) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	for (unsigned int i = 0; i < m_data.numFluidParticles; i++)
	{
		const Real velMag = (m_data.velFluid[i] + m_data.accFluid[i] * m_data.h).squaredNorm();
		if (velMag > *maxVel)
			*maxVel = velMag;
	}
}
void cuda_CFL(SPH::DFSPHCData& m_data, const Real minTimeStepSize, Real m_cflFactor, Real m_cflMaxTimeStepSize) {
	Real* out_buff;
	cudaMallocManaged(&(out_buff), sizeof(Real));
	*out_buff = 0.1;
	DFSPH_CFL_kernel << <1, 1 >> > (m_data, out_buff);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	Real maxVel = *out_buff;
	cudaFree(out_buff);


	Real h = m_data.h;

	// Approximate max. time step size 		
	h = m_cflFactor * .4 * (2.0*m_data.particleRadius / (sqrt(maxVel)));

	h = min(h, m_cflMaxTimeStepSize);
	h = max(h, minTimeStepSize);

	m_data.updateTimeStep(h);
}

__global__ void DFSPH_update_vel_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	m_data.velFluid[i] += m_data.h * m_data.accFluid[i];

#ifdef USE_WARMSTART	
	//done here to have one less kernel
	m_data.kappa[i] = fmax(m_data.kappa[i] * m_data.h_ratio_to_past2, -0.5);
#endif
}
void cuda_update_vel(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_update_vel_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
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
		fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}
template void cuda_pressure_compute<true>(SPH::DFSPHCData& data);
template void cuda_pressure_compute<false>(SPH::DFSPHCData& data);

__global__ void DFSPH_pressure_init_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	computeDensityAdv(m_data, i);

	m_data.factor[i] *= m_data.invH_future;
#ifdef USE_WARMSTART
	m_data.kappa[i] = 0.0;
#endif

}
void cuda_pressure_init(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_pressure_init_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}

__global__ void DFSPH_pressure_loop_end_kernel(SPH::DFSPHCData m_data, Real* avg_density_err) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	computeDensityAdv(m_data, i);
	atomicAdd(avg_density_err, m_data.densityAdv[i]);
}
Real cuda_pressure_loop_end(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	Real* avg_density_err;
	cudaMallocManaged(&(avg_density_err), sizeof(Real));
	*avg_density_err = 0.0;
	DFSPH_pressure_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data, avg_density_err);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	Real result = *avg_density_err;
	cudaFree(avg_density_err);
	return result;
}

__global__ void DFSPH_update_pos_kernel(SPH::DFSPHCData m_data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m_data.numFluidParticles) { return; }

	m_data.posFluid[i] += m_data.h * m_data.velFluid[i];
}
void cuda_update_pos(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_update_pos_kernel << <numBlocks, BLOCKSIZE >> > (data);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
		exit(1598);
	}
}


int cuda_divergenceSolve(SPH::DFSPHCData& m_data, const unsigned int maxIter, const Real maxError) {
	//////////////////////////////////////////////////////////////////////////
	// Init parameters
	//////////////////////////////////////////////////////////////////////////

	const Real h = m_data.h;
	const int numParticles = m_data.numFluidParticles;
	const Real density0 = m_data.density0;

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
	const Real eta = (1.0 / h) * maxError * 0.01 * density0;  // maxError is given in percent

	Real avg_density_err = 0.0;
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
int cuda_pressureSolve(SPH::DFSPHCData& m_data, const unsigned int m_maxIterations, const Real m_maxError) {
	const Real density0 = m_data.density0;
	const int numParticles = (int)m_data.numFluidParticles;
	Real avg_density_err = 0.0;

#ifdef USE_WARMSTART		
	cuda_pressure_compute<true>(m_data);
#endif


	//////////////////////////////////////////////////////////////////////////
	// Compute rho_adv
	//////////////////////////////////////////////////////////////////////////
	cuda_pressure_init(m_data);

	unsigned int m_iterations = 0;

	//////////////////////////////////////////////////////////////////////////
	// Start solver
	//////////////////////////////////////////////////////////////////////////

	// Maximal allowed density fluctuation
	const Real eta = m_maxError * 0.01 * density0;  // maxError is given in percent

	while (((avg_density_err > eta) || (m_iterations < 2)) && (m_iterations < m_maxIterations))
	{

		cuda_pressure_compute<false>(m_data);
		avg_density_err = cuda_pressure_loop_end(m_data);

		avg_density_err /= numParticles;

		m_iterations++;
	}
	return m_iterations;

}


template<unsigned int grid_size, bool z_curve>
__global__ void DFSPH_computeGridIdx_kernel(Vector3d* in, unsigned int* out, Real kernel_radius, unsigned int num_particles) {

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
		Vector3d pos = (in[i] / kernel_radius) +50;
		out[i] = (int)pos.x + ((int)pos.y)*grid_size + ((int)pos.z)*grid_size*grid_size;
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

template<unsigned int grid_size, bool z_curve>
__global__ void DFSPH_neighborsSearch_kernel(unsigned int numFluidParticles, Real radius, 
	Vector3d* posFluid, Vector3d* posBoundary, int* neighbors_buff, int* nb_neighbors_buff,
	unsigned int* p_id_sorted, unsigned int* cell_start_end, unsigned int* p_id_sorted_b, unsigned int* cell_start_end_b) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numFluidParticles) { return; }
	
	nb_neighbors_buff[i + numFluidParticles] = 0;
	nb_neighbors_buff[i] = 0;

	Real radius_sq = radius;
	Vector3d pos = posFluid[i];
	Vector3d pos_cell = (pos / radius_sq) + 50; //on that line the radius is not yet squared
	int x = (int)pos_cell.x;
	int y = (int)pos_cell.y;
	int z = (int)pos_cell.z;
	radius_sq *= radius_sq;

	unsigned int nb_neighbors_fluid = 0;
	unsigned int nb_neighbors_boundary = 0;
	//int neighbors_fluid[MAX_NEIGHBOURS];//doing it with local buffer was not faster
	//int neighbors_boundary[MAX_NEIGHBOURS];
	//now we iterate on the 9 cell block surronding the cell in which we have our particle
	for (int k = -1; k < 2; ++k) {
		for (int m = -1; m < 2; ++m) {
			//for (int l = -1; l < 2; ++l) {// I don't need to iter on x since the 3cells are successives: large gains
				//we iterate on the particles inside that cell
				unsigned int cur_cell_id = (x + -1) + (y + m)*grid_size + (z + k)*grid_size*grid_size;
				unsigned int end;
				//*
				//for the fluid particles
				end = cell_start_end[cur_cell_id + 3];
				for (unsigned int cur_particle = cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {
					unsigned int j = p_id_sorted[cur_particle];
					if (i != j) {
						if ((pos - posFluid[j]).squaredNorm() < radius_sq) {
							neighbors_buff[i*MAX_NEIGHBOURS + nb_neighbors_fluid] = j;
							//neighbors_fluid[nb_neighbors_fluid] = j;
							nb_neighbors_fluid++;
						}
					}
				}
				//*/
				//*
				//for the boundaries particles
				end = cell_start_end_b[cur_cell_id + 3];
				for (unsigned int cur_particle = cell_start_end_b[cur_cell_id]; cur_particle < end; ++cur_particle) {
					unsigned int j = p_id_sorted_b[cur_particle];
					if ((pos - posBoundary[j]).squaredNorm() < radius_sq) {
						neighbors_buff[1 * numFluidParticles*MAX_NEIGHBOURS + i*MAX_NEIGHBOURS + nb_neighbors_boundary] = j;
						//neighbors_boundary[nb_neighbors_boundary] = j;
						nb_neighbors_boundary++;
					}
				}
				//*/
			//}
		}
	}


	nb_neighbors_buff[i]=nb_neighbors_fluid;
	nb_neighbors_buff[i + numFluidParticles]=nb_neighbors_boundary;

	//memcpy((neighbors_buff + i*MAX_NEIGHBOURS), neighbors_fluid, sizeof(int)*nb_neighbors_fluid);
	//memcpy((neighbors_buff + numFluidParticles*MAX_NEIGHBOURS + i*MAX_NEIGHBOURS), neighbors_boundary, sizeof(int)*nb_neighbors_boundary);


}

void cuda_neighborsSearchInternal(Vector3d* pos, Real kernel_radius, int numParticles, void **d_temp_storage_pair_sort,
	size_t   &temp_storage_bytes_pair_sort, unsigned int* cell_id, unsigned int* cell_id_sorted,
	unsigned int* p_id, unsigned int* p_id_sorted, unsigned int* hist, void **d_temp_storage_cumul_hist,
	size_t   &temp_storage_bytes_cumul_hist, unsigned int* cell_start_end) {
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

	//reset the particle id
	int numBlocks = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	DFSPH_setBufferValueToItself_kernel << <numBlocks, BLOCKSIZE >> > (p_id, numParticles);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "p_id init idxs failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	/// now we will work on sorting the boundaries particles
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
		fprintf(stderr, "histogram failed: %d\n", (int)cudaStatus);
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

void cuda_neighborsSearch(SPH::DFSPHCData& data) {
	
	cudaError_t cudaStatus;

	static unsigned int* cell_id;
	static unsigned int* cell_id_sorted;
	static unsigned int* local_id;
	static unsigned int* p_id;
	static unsigned int* p_id_sorted;
	static unsigned int* cell_start_end;
	static unsigned int* hist;
	void *d_temp_storage_pair_sort = NULL;
	size_t temp_storage_bytes_pair_sort = 0;
	void *d_temp_storage_cumul_hist = NULL;
	size_t temp_storage_bytes_cumul_hist = 0;

	static unsigned int* cell_id_b;
	static unsigned int* cell_id_sorted_b;
	static unsigned int* local_id_b;
	static unsigned int* p_id_b;
	static unsigned int* p_id_sorted_b;
	static unsigned int* cell_start_end_b;
	static unsigned int* hist_b;
	void *d_temp_storage_pair_sort_b = NULL;
	size_t temp_storage_bytes_pair_sort_b = 0;
	void *d_temp_storage_cumul_hist_b = NULL;
	size_t temp_storage_bytes_cumul_hist_b = 0;


	static int* neighbors_buff;
	static int* nb_neighbors_buff;

	
	static bool first_time = true;
	if (first_time) {
		first_time = false;

		//allocatethe mme for fluid particles
		cudaMallocManaged(&(cell_id), data.numFluidParticles * sizeof(unsigned int));
		cudaMallocManaged(&(cell_id_sorted), data.numFluidParticles * sizeof(unsigned int));
		cudaMallocManaged(&(local_id), data.numFluidParticles * sizeof(unsigned int));
		cudaMallocManaged(&(p_id), data.numFluidParticles * sizeof(unsigned int));
		cudaMallocManaged(&(p_id_sorted), data.numFluidParticles * sizeof(unsigned int));
		cudaMallocManaged(&(cell_start_end), (CELL_COUNT + 1) * sizeof(unsigned int));
		cudaMallocManaged(&(hist), (CELL_COUNT + 1) * sizeof(unsigned int));
		
		//allocate memory for boundaries particles
		cudaMallocManaged(&(cell_id_b), data.numBoundaryParticles * sizeof(unsigned int));
		cudaMallocManaged(&(cell_id_sorted_b), data.numBoundaryParticles * sizeof(unsigned int));
		cudaMallocManaged(&(local_id_b), data.numBoundaryParticles * sizeof(unsigned int));
		cudaMallocManaged(&(p_id_b), data.numBoundaryParticles * sizeof(unsigned int));
		cudaMallocManaged(&(p_id_sorted_b), data.numBoundaryParticles * sizeof(unsigned int));
		cudaMallocManaged(&(cell_start_end_b), (CELL_COUNT + 1) * sizeof(unsigned int));
		cudaMallocManaged(&(hist_b), (CELL_COUNT + 1) * sizeof(unsigned int));
		
		cudaMallocManaged(&(nb_neighbors_buff), data.numFluidParticles * 2 * sizeof(int));
		cudaMallocManaged(&(neighbors_buff), data.numFluidParticles * 2 * MAX_NEIGHBOURS * sizeof(int));

		cudaMemset(nb_neighbors_buff, 0, data.numFluidParticles * 2 * sizeof(int));
		cudaMemset(neighbors_buff, 0, data.numFluidParticles * 2 * MAX_NEIGHBOURS * sizeof(int));


		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "neighbours storage init failed: %d\n", (int)cudaStatus);
			exit(1598);
		}

		
		
		//Compute the cell id and an array ith the id of the particles sorted
		//Also compute the start and end index of each cell
		cuda_neighborsSearchInternal(data.posBoundary, data.m_kernel_precomp.getRadius(), data.numBoundaryParticles,
			&d_temp_storage_pair_sort_b, temp_storage_bytes_pair_sort_b, cell_id_b, cell_id_sorted_b,
			p_id_b, p_id_sorted_b, hist_b, &d_temp_storage_cumul_hist_b, temp_storage_bytes_cumul_hist_b,
			cell_start_end_b);

	}
	/*
	unsigned int grid_size = CELL_ROW_LENGTH;
	for (int i = 0; i < data.numBoundaryParticles; ++i) {
		Vector3d pos = (data.posBoundary[i] / data.m_kernel_precomp.getRadius())-2;
		if (pos.x > grid_size || pos.y > grid_size || pos.z > grid_size) {
			fprintf(stderr, "the particle is outside of the possible indexes\n");
			exit(1256);
		}
		local_id_b[i] = (int)pos.x + ((int)pos.y)*grid_size + ((int)pos.z)*grid_size*grid_size;
	}//*/
	//*
	//fluid particles

	//Compute the cell id and an array ith the id of the particles sorted
	//Also compute the start and end index of each cell
	cuda_neighborsSearchInternal(data.posFluid, data.m_kernel_precomp.getRadius(), data.numFluidParticles,
		&d_temp_storage_pair_sort, temp_storage_bytes_pair_sort, cell_id, cell_id_sorted,
		p_id, p_id_sorted, hist, &d_temp_storage_cumul_hist, temp_storage_bytes_cumul_hist,
		cell_start_end);

	

	//*
	//basic version
	if (false) {
		double radius_sq = data.m_kernel_precomp.getRadius();
		radius_sq *= radius_sq;
		for (int i = 0; i < data.numFluidParticles; ++i) {
			//search the neigbors within the fluid particles
			for (int j = 0; j < data.numFluidParticles; ++j) {
				if (i != j) {
					if ((data.posFluid[i] - data.posFluid[j]).squaredNorm() < radius_sq) {
						neighbors_buff[i*MAX_NEIGHBOURS + nb_neighbors_buff[i]] = j;
						nb_neighbors_buff[i]++;
					}
				}
			}
			//search the neigbors within the boundary particles
			for (int j = 0; j < data.numBoundaryParticles; ++j) {
				if ((data.posFluid[i] - data.posBoundary[j]).squaredNorm() < radius_sq) {
					neighbors_buff[data.numFluidParticles*MAX_NEIGHBOURS + i*MAX_NEIGHBOURS + nb_neighbors_buff[i + data.numFluidParticles]] = j;
					nb_neighbors_buff[i + data.numFluidParticles]++;
				}
			}
		}
	}
	//*/
	//using our grid (no firther optimization however)
	if (false) {
		unsigned int grid_size = CELL_ROW_LENGTH;
		for (int i = 0; i < data.numFluidParticles; ++i) {
			nb_neighbors_buff[i + data.numFluidParticles]=0; 
			nb_neighbors_buff[i] = 0;

			Real radius_sq = data.m_kernel_precomp.getRadius();
			Vector3d pos = data.posFluid[i];
			Vector3d pos_cell = (pos / radius_sq) + 50; //on that line the radius is not yet squared
			int x = (int)pos_cell.x;
			int y = (int)pos_cell.y;
			int z = (int)pos_cell.z;
			radius_sq *= radius_sq;

			//now we iterate on the 9 cell block surronding the cell in which we have our particle
			for (int k = -1; k < 2; ++k) {
				for (int m = -1; m < 2; ++m) {
					for (int l = -1; l < 2; ++l) {
						//we iterate on the particles inside that cell
						unsigned int cur_cell_id = (x + k) + (y + m)*grid_size + (z + l)*grid_size*grid_size;
						unsigned int end;
						//*
						//for the fluid particles
						end = cell_start_end[cur_cell_id + 1];
						for (unsigned int cur_particle = cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {
							unsigned int j = p_id_sorted[cur_particle];
							if (i != j) {
								if ((pos - data.posFluid[j]).squaredNorm() < radius_sq) {
									neighbors_buff[i*MAX_NEIGHBOURS + nb_neighbors_buff[i]] = j;
									nb_neighbors_buff[i]++;
								}
							}
						}
						//*/
						//*
						//for the boundaries particles
						end = cell_start_end_b[cur_cell_id + 1];
						for (unsigned int cur_particle = cell_start_end_b[cur_cell_id]; cur_particle < end; ++cur_particle) {
							unsigned int j = p_id_sorted_b[cur_particle];
							if ((pos - data.posBoundary[j]).squaredNorm() < radius_sq) {
								neighbors_buff[1 * data.numFluidParticles*MAX_NEIGHBOURS + i*MAX_NEIGHBOURS + nb_neighbors_buff[i + data.numFluidParticles]] = j;
								nb_neighbors_buff[i + data.numFluidParticles]++;
							}
						}
						//*/
					}
				}
			}
		}
	}

	if (true) {

		float time;
		static float time_avg = 0;
		static unsigned int time_count = 0;
		cudaEvent_t start, stop;

		(cudaEventCreate(&start));
		(cudaEventCreate(&stop));
		(cudaEventRecord(start, 0));


		//cuda way
		int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_neighborsSearch_kernel<CELL_ROW_LENGTH, false> << <numBlocks, BLOCKSIZE >> > (data.numFluidParticles,
			data.m_kernel_precomp.getRadius(), data.posFluid, data.posBoundary, neighbors_buff, nb_neighbors_buff,
			p_id_sorted, cell_start_end, p_id_sorted_b,cell_start_end_b);



		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cuda neighbors search failed: %d\n", (int)cudaStatus);
			exit(1598);
		}

		(cudaEventRecord(stop, 0));
		(cudaEventSynchronize(stop));
		(cudaEventElapsedTime(&time, start, stop));

		time_avg += time;
		time_count++;
		printf("Time to generate: %f ms   avg: %f ms \n", time, time_avg/time_count);
	}

	for (int i = 0; i < data.numFluidParticles; ++i) {
		if (nb_neighbors_buff[i] != data.getNumberOfNeighbourgs(i)) {
			fprintf(stderr, "incoherent not the same number of neighbours: %d,  %d,  %d\n", i, nb_neighbors_buff[i], data.getNumberOfNeighbourgs(i));
			exit(1256);
		}
		if (nb_neighbors_buff[i + data.numFluidParticles] != data.getNumberOfNeighbourgs(i, 1)) {
			fprintf(stderr, "incoherent not the same number of neighbours boundary: %d,  %d,  %d\n", i, nb_neighbors_buff[i + data.numFluidParticles],
				data.getNumberOfNeighbourgs(i, 1));
			exit(1256);
		}
	}
	
}

/*
FUNCTION inline unsigned int getNeighbour(int particle_id, int neighbour_id, int body_id = 0) {
return neighbourgs[body_id*numFluidParticles*MAX_NEIGHBOURS + particle_id * MAX_NEIGHBOURS + neighbour_id];
}

FUNCTION inline unsigned int getNumberOfNeighbourgs(int particle_id, int body_id = 0) {
return numberOfNeighbourgs[body_id*numFluidParticles + particle_id];
}
*/


void cuda_renderFluid(SPH::DFSPHCData& data) {
	cuda_opengl_renderFluid(data);
}

#include <GL/glew.h>


void cuda_opengl_initFluidRendering(SPH::DFSPHCData& data) {
	glGenVertexArrays(1, &data.vao); // Créer le VAO
	glBindVertexArray(data.vao); // Lier le VAO pour l'utiliser


	glGenBuffers(1, &data.pos_buffer);
	// selectionne le buffer pour l'initialiser
	glBindBuffer(GL_ARRAY_BUFFER, data.pos_buffer);
	// dimensionne le buffer actif sur array_buffer, l'alloue et l'initialise avec les positions des sommets de l'objet
	glBufferData(GL_ARRAY_BUFFER,
		/* length */	data.numFluidParticles * sizeof(Vector3d),
		/* data */      &(data.posFluid[0]),
		/* usage */     GL_DYNAMIC_DRAW);
	//set it to the attribute
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 0, 0);

	glGenBuffers(1, &data.vel_buffer);
	// selectionne le buffer pour l'initialiser
	glBindBuffer(GL_ARRAY_BUFFER, data.vel_buffer);
	// dimensionne le buffer actif sur array_buffer, l'alloue et l'initialise avec les positions des sommets de l'objet
	glBufferData(GL_ARRAY_BUFFER,
		/* length */	data.numFluidParticles * sizeof(Vector3d),
		/* data */      &(data.velFluid[0]),
		/* usage */     GL_DYNAMIC_DRAW);
	//set it to the attribute
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, 0);

	// nettoyage
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);



}

void cuda_opengl_renderFluid(SPH::DFSPHCData& data) {
	glBindVertexArray(data.vao); // link the vao

	glBindBuffer(GL_ARRAY_BUFFER, data.pos_buffer);
	glBufferSubData(GL_ARRAY_BUFFER,
		0,
		data.numFluidParticles * sizeof(Vector3d),
		&(data.posFluid[0]));
	
	glBindBuffer(GL_ARRAY_BUFFER, data.vel_buffer);
	glBufferSubData(GL_ARRAY_BUFFER,
		0,
		data.numFluidParticles * sizeof(Vector3d),
		&(data.velFluid[0]));
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_POINTS, 0, data.numFluidParticles);

	glBindVertexArray(0); // unlink the vao
}










/*
	THE NEXT FUNCTIONS ARE FOR THE MEMORY ALLOCATION
*/

void allocate_c_array_struct_cuda_managed(SPH::DFSPHCData& data, bool minimize_managed) {

	cudaMallocManaged(&(data.posBoundary), data.numBoundaryParticles * sizeof(Vector3d));
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

void reset_c_array_struct_cuda_from_values(SPH::DFSPHCData& data, Vector3d* posBoundary, Vector3d* velBoundary, 
	Real* boundaryPsi, Vector3d* posFluid, Vector3d* velFluid, Real* mass) {
	
	fprintf(stderr, "start of reset values gpu: \n");

	cudaError_t cudaStatus;
	//boundaries
	gpuErrchk( cudaMemcpy(data.posBoundary, posBoundary, data.numBoundaryParticles * sizeof(Vector3d), cudaMemcpyHostToDevice));
	gpuErrchk( cudaMemcpy(data.velBoundary, velBoundary, data.numBoundaryParticles * sizeof(Vector3d), cudaMemcpyHostToDevice));
	gpuErrchk( cudaMemcpy(data.boundaryPsi, boundaryPsi, data.numBoundaryParticles * sizeof(Real), cudaMemcpyHostToDevice));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) { 
		fprintf(stderr, "init of boundaries particles from data failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	//fluid
	gpuErrchk(cudaMemcpy(data.posFluid, posFluid, data.numFluidParticles * sizeof(Vector3d), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(data.velFluid, velFluid, data.numFluidParticles * sizeof(Vector3d), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(data.mass, mass, data.numFluidParticles * sizeof(Real), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(data.accFluid, 0, data.numFluidParticles * sizeof(Vector3d)));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "init of fluid particles from data failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	//ohter values normaly only kappa and kappaV are necessary (ut this function is only called 
	//when reseting the fluid so it does not cost much to make sure everything is clean
	gpuErrchk(cudaMemset(data.density, 0, data.numFluidParticles * sizeof(Real)));
	gpuErrchk(cudaMemset(data.factor, 0, data.numFluidParticles * sizeof(Real)));
	gpuErrchk(cudaMemset(data.kappa, 0, data.numFluidParticles * sizeof(Real)));
	gpuErrchk(cudaMemset(data.kappaV, 0, data.numFluidParticles * sizeof(Real)));
	gpuErrchk(cudaMemset(data.densityAdv, 0, data.numFluidParticles * sizeof(Real)));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "init of fluid other values from data failed: %d\n", (int)cudaStatus);
		exit(1598);
	}


	fprintf(stderr, "end of reset gpu\n");
}


void allocate_precomputed_kernel_managed(SPH::PrecomputedCubicKernelPerso& kernel, bool minimize_managed) {
	
	if (minimize_managed) {
		cudaMalloc(&(kernel.m_W), kernel.m_resolution * sizeof(Real));
		cudaMalloc(&(kernel.m_gradW), (kernel.m_resolution + 1) * sizeof(Real));
	}
	else {
		cudaMallocManaged(&(kernel.m_W), kernel.m_resolution * sizeof(Real));
		cudaMallocManaged(&(kernel.m_gradW), (kernel.m_resolution + 1) * sizeof(Real));
	}
}


void init_precomputed_kernel_from_values(SPH::PrecomputedCubicKernelPerso& kernel, Real* w, Real* grad_W) {
	cudaError_t cudaStatus;
	//W
	cudaStatus = cudaMemcpy(kernel.m_W,
		w,
		kernel.m_resolution * sizeof(Real),
		cudaMemcpyHostToDevice);
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "precomputed initialization of W from data failed: %d\n", (int)cudaStatus);
		exit(1598);
	}

	//grad W
	cudaStatus = cudaMemcpy(kernel.m_gradW,
		grad_W,
		(kernel.m_resolution + 1) * sizeof(Real),
		cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "precomputed initialization of grad W from data failed: %d\n", (int)cudaStatus);
		exit(1598);
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
