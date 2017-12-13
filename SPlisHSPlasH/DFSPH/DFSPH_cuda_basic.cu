
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DFSPH_cuda_basic.h"
#include <stdio.h>
#include "DFSPH_c_arrays_structure.h"

#define BLOCKSIZE 256
#define m_eps 1.0e-5


#define USE_WARMSTART
#define USE_WARMSTART_V

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

__device__ void computeDensityChange(SPH::DFSPHCData& m_data,const unsigned int index) {
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
	computeDensityChange(m_data ,i);
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
		computeDensityChange(m_data,i);

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

	computeDensityChange(m_data,i);
	atomicAdd(avg_density_err, m_data.densityAdv[i]);
}
Real cuda_divergence_loop_end(SPH::DFSPHCData& data) {
	int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
	Real* avg_density_err;
	cudaMallocManaged(&(avg_density_err), sizeof(Real));
	*avg_density_err = 0.0;
	DFSPH_divergence_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data,avg_density_err);

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
