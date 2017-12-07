#include "DFSPH_cpu_c_arrays.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SimulationDataDFSPH.h"
#include <iostream>
#include "SPlisHSPlasH/Utilities/Timing.h"
#include "DFSPH_cuda_basic.h"

using namespace SPH;
using namespace std;

#define USE_WARMSTART
#define USE_WARMSTART_V

DFSPHCArrays::DFSPHCArrays(FluidModel *model) :
	TimeStep(model),
	m_simulationData(),
	m_data(model)
{
	m_simulationData.init(model);
	m_counter = 0;
	m_iterationsV = 0;
	m_enableDivergenceSolver = true;





}

DFSPHCArrays::~DFSPHCArrays(void)
{
}


void DFSPHCArrays::step()
{
	//test_cuda();
	


	//


	m_data.viscosity = m_viscosity->getViscosity();
	

	const unsigned int numParticles = m_model->numActiveParticles();


	performNeighborhoodSearch();



	//start of the c arrays
	m_data.loadDynamicData(m_model, m_simulationData);

	for (unsigned int i = 0; i < m_data.m_kernel_precomp.m_resolution; ++i) {
		checkReal("kernel values check:", FluidModel::PrecomputedCubicKernel::m_W[i], m_data.m_kernel_precomp.m_W[i]);
		checkReal("kernel grad values check:", FluidModel::PrecomputedCubicKernel::m_gradW[i], m_data.m_kernel_precomp.m_gradW[i]);
	}
	checkReal("kernel values w_0 check:", FluidModel::PrecomputedCubicKernel::m_W_zero, m_data.m_kernel_precomp.m_W_zero);

	for (int i = 0; i < m_data.numFluidParticles; ++i) {
		checkVector3("start check x:", m_model->getPosition(0, i), m_data.posFluid[i]);
		checkVector3("start check v:", m_model->getVelocity(0, i), m_data.velFluid[i]);
	}

	computeDensities();
	for (int i = 0; i < m_data.numFluidParticles; ++i) {
		checkReal("after density check:", m_model->getDensity(i), m_data.density[i]);
	}

	START_TIMING("computeDFSPHFactor");
	computeDFSPHFactor();
	STOP_TIMING_AVG;

	for (int i = 0; i < m_data.numFluidParticles; ++i) {
		checkReal("after factor check:", m_simulationData.getFactor(i), m_data.factor[i]);
	}


	if (m_enableDivergenceSolver)
	{
		START_TIMING("divergenceSolve");
		divergenceSolve();
		STOP_TIMING_AVG;
	}
	else
		m_iterationsV = 0;


	for (int i = 0; i < m_data.numFluidParticles; ++i) {
		checkVector3("after divergence check x:", m_model->getPosition(0, i), m_data.posFluid[i]);
		checkVector3("after divergence check v:", m_model->getVelocity(0, i), m_data.velFluid[i]);
	}


	// Compute accelerations: a(t)
	clearAccelerations();

	computeNonPressureForces();

	updateTimeStepSize();
	checkReal("step: h:", TimeManager::getCurrent()->getTimeStepSize(), m_data.h_future);

	updateVelocities(m_data.h);
	for (int i = 0; i < m_data.numFluidParticles; ++i) {
		checkVector3("after update vel check x:", m_model->getPosition(0, i), m_data.posFluid[i]);
		checkVector3("after update vel check v:", m_model->getVelocity(0, i), m_data.velFluid[i]);
	}


	START_TIMING("pressureSolve");
	pressureSolve();
	STOP_TIMING_AVG;
	for (int i = 0; i < m_data.numFluidParticles; ++i) {
		checkVector3("after pressure check x:", m_model->getPosition(0, i), m_data.posFluid[i]);
		checkVector3("after pressure check v:", m_model->getVelocity(0, i), m_data.velFluid[i]);
	}


	updatePositions(m_data.h);

	for (int i = 0; i < m_data.numFluidParticles; ++i) {
		checkVector3("after updatepos check x:", m_model->getPosition(0, i), m_data.posFluid[i]);
		checkVector3("after updatepos check v:", m_model->getVelocity(0, i), m_data.velFluid[i]);
	}

	emitParticles();

	m_data.h_past = m_data.h;
	m_data.h = m_data.h_future;


	// Compute new time	
	TimeManager::getCurrent()->setTime(TimeManager::getCurrent()->getTime() + m_data.h);


	std::cout << "step finished" << std::endl;
}

void DFSPHCArrays::computeDFSPHFactor()
{
	//////////////////////////////////////////////////////////////////////////
	// Init parameters
	//////////////////////////////////////////////////////////////////////////

	const Real h = m_data.h;
	const int numParticles = m_data.numFluidParticles;

	#pragma omp parallel default(shared)
	{
		//////////////////////////////////////////////////////////////////////////
		// Compute pressure stiffness denominator
		//////////////////////////////////////////////////////////////////////////

		#pragma omp for schedule(static)  
		for (int i = 0; i < numParticles; i++)
		{
			{
				//////////////////////////////////////////////////////////////////////////
				// Compute gradient dp_i/dx_j * (1/k)  and dp_j/dx_j * (1/k)
				//////////////////////////////////////////////////////////////////////////
				const Vector3d &xi = m_data.posFluid[i];
				Real sum_grad_p_k = 0.0;
				Vector3d grad_p_i;
				grad_p_i.setZero();

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i); j++)
				{
					const unsigned int neighborIndex = m_data.getNeighbour(i, j);
					const Vector3d &xj = m_data.posFluid[neighborIndex];
					const Vector3d grad_p_j = -m_data.mass[neighborIndex] * m_data.gradW(xi - xj);
					//checkVector3(" check gradpj:", grad_p_j2, grad_p_j);
					sum_grad_p_k += grad_p_j.squaredNorm();
					grad_p_i -= grad_p_j;
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
						const Vector3d grad_p_j = -m_data.boundaryPsi[neighborIndex] * m_data.gradW(xi - xj);
						sum_grad_p_k += grad_p_j.squaredNorm();
						grad_p_i -= grad_p_j;
					}
				}

				sum_grad_p_k += grad_p_i.squaredNorm();

				//////////////////////////////////////////////////////////////////////////
				// Compute pressure stiffness denominator
				//////////////////////////////////////////////////////////////////////////
				sum_grad_p_k = max(sum_grad_p_k, m_eps);
				m_data.factor[i] = -1.0 / (sum_grad_p_k);

				//m_data.factor[i] = (float)m_data.factor[i];
			}
			{
				//////////////////////////////////////////////////////////////////////////
				// Compute gradient dp_i/dx_j * (1/k)  and dp_j/dx_j * (1/k)
				//////////////////////////////////////////////////////////////////////////
				const Vector3r &xi = m_model->getPosition(0, i);
				Real sum_grad_p_k = 0.0;
				Vector3r grad_p_i;
				grad_p_i.setZero();

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
				{
					const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
					const Vector3r &xj = m_model->getPosition(0, neighborIndex);
					const Vector3r grad_p_j = -m_model->getMass(neighborIndex) * m_model->gradW(xi - xj);
					sum_grad_p_k += grad_p_j.squaredNorm();
					grad_p_i -= grad_p_j;
				}

				//////////////////////////////////////////////////////////////////////////
				// Boundary
				//////////////////////////////////////////////////////////////////////////
				for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
				{
					for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, i); j++)
					{
						const unsigned int neighborIndex = m_model->getNeighbor(pid, i, j);
						const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
						const Vector3r grad_p_j = -m_model->getBoundaryPsi(pid, neighborIndex) * m_model->gradW(xi - xj);
						sum_grad_p_k += grad_p_j.squaredNorm();
						grad_p_i -= grad_p_j;
					}
				}

				sum_grad_p_k += grad_p_i.squaredNorm();

				//////////////////////////////////////////////////////////////////////////
				// Compute pressure stiffness denominator
				//////////////////////////////////////////////////////////////////////////
				Real &factor = m_simulationData.getFactor(i);

				sum_grad_p_k = max(sum_grad_p_k, m_eps);
				factor = -1.0 / (sum_grad_p_k);
			}

			//checkReal("factor:", m_simulationData.getFactor(i), m_data.factor[i]);
			
		}
	}
}

void DFSPHCArrays::pressureSolve()
{
	const Real h = m_data.h;
	const Real h2 = h*h;
	const Real invH = 1.0 / h;
	const Real invH2 = 1.0/h2;
	const Real density0 = m_data.density0;
	const int numParticles = (int)m_data.numFluidParticles;
	Real avg_density_err = 0.0;

	

#ifdef USE_WARMSTART			
	#pragma omp parallel default(shared)
	{
		//////////////////////////////////////////////////////////////////////////
		// Divide by h^2, the time step size has been removed in 
		// the last step to make the stiffness value independent 
		// of the time step size
		//////////////////////////////////////////////////////////////////////////
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			{
				m_data.kappa[i] = max(m_data.kappa[i] *invH2, -0.5);
			}
			{
				m_simulationData.getKappa(i) = max(m_simulationData.getKappa(i)*invH2, -0.5);
			}
			//checkReal("pressureSolve: warm start: early kappa:", m_simulationData.getKappa(i), m_data.kappa[i]);

		}

		//////////////////////////////////////////////////////////////////////////
		// Predict v_adv with external velocities
		////////////////////////////////////////////////////////////////////////// 

		#pragma omp for schedule(static)  
		for (int i = 0; i < numParticles; i++)
		{
			{
				Vector3d &vel = m_data.velFluid[i];
				const Real ki = m_data.kappa[i];
				const Vector3d &xi = m_data.posFluid[i];

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				for (int j = 0; j < m_data.numberOfNeighbourgs[i]; j++)
				{
					const unsigned int neighborIndex = m_data.getNeighbour(i, j);
					const Real kj = m_data.kappa[neighborIndex];

					const Real kSum = (ki + kj);
					if (fabs(kSum) > m_eps)
					{
						const Vector3d &xj = m_data.posFluid[neighborIndex];
						const Vector3d grad_p_j = -m_data.mass[neighborIndex] * m_data.gradW(xi - xj);
						vel -= h * kSum * grad_p_j;					// ki, kj already contain inverse density
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
							const Vector3d &xj = m_data.posBoundary[neighborIndex];
							const Vector3d grad_p_j = -m_data.boundaryPsi[neighborIndex] * m_data.gradW(xi - xj);
							const Vector3d velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
							vel += velChange;

							//TODO reactivate this
							//m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * velChange * invH;
						}
					}
				}
			}
			{
				Vector3r &vel = m_model->getVelocity(0, i);
				const Real ki = m_simulationData.getKappa(i);
				const Vector3r &xi = m_model->getPosition(0, i);

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
				{
					const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
					const Real kj = m_simulationData.getKappa(neighborIndex);

					const Real kSum = (ki + kj);
					if (fabs(kSum) > m_eps)
					{
						const Vector3r &xj = m_model->getPosition(0, neighborIndex);
						const Vector3r grad_p_j = -m_model->getMass(neighborIndex) * m_model->gradW(xi - xj);
						vel -= h * kSum * grad_p_j;					// ki, kj already contain inverse density
					}
				}

				//////////////////////////////////////////////////////////////////////////
				// Boundary
				//////////////////////////////////////////////////////////////////////////
				if (fabs(ki) > m_eps)
				{
					for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
					{
						for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, i); j++)
						{
							const unsigned int neighborIndex = m_model->getNeighbor(pid, i, j);
							const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
							const Vector3r grad_p_j = -m_model->getBoundaryPsi(pid, neighborIndex) * m_model->gradW(xi - xj);
							const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
							vel += velChange;

							m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * velChange * invH;
						}
					}
				}
			}
			//checkVector3("pressureSolve: warm start: vel first loop:", m_model->getVelocity(0, i), m_data.velFluid[i]);
		}
	}
#endif
	

	//////////////////////////////////////////////////////////////////////////
	// Compute rho_adv
	//////////////////////////////////////////////////////////////////////////
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < numParticles; i++)
		{
			

			{
				computeDensityAdv(i, numParticles, h, density0);
				m_data.factor[i] *= invH2;
#ifdef USE_WARMSTART
				m_data.kappa[i] = 0.0;
#endif
			}
			{
				computeDensityAdv(i, numParticles, h, density0);
				m_simulationData.getFactor(i) *= invH2;
#ifdef USE_WARMSTART
				m_simulationData.getKappa(i) = 0.0;
#endif
			}
			//checkReal("pressureSolve: factor begin:", m_simulationData.getFactor(i), m_data.factor[i]);
		}
	}

	m_iterations = 0;

	//////////////////////////////////////////////////////////////////////////
	// Start solver
	//////////////////////////////////////////////////////////////////////////
	
	// Maximal allowed density fluctuation
	const Real eta = m_maxError * 0.01 * density0;  // maxError is given in percent
	
	while (((avg_density_err > eta) || (m_iterations < 2)) && (m_iterations < m_maxIterations))
	{
		

		#pragma omp parallel default(shared)
		{
			//////////////////////////////////////////////////////////////////////////
			// Compute pressure forces
			//////////////////////////////////////////////////////////////////////////
#pragma omp for schedule(static) 
			for (int i = 0; i < numParticles; i++)
			{

				//checkVector3("pressureSolve: vel start loop:", m_model->getVelocity(0, i), m_data.velFluid[i]);
				{
					//////////////////////////////////////////////////////////////////////////
					// Evaluate rhs
					//////////////////////////////////////////////////////////////////////////
					const Real b_i = m_data.densityAdv[i] - density0;
					const Real ki = b_i*m_data.factor[i];
#ifdef USE_WARMSTART
					m_data.kappa[i] += ki;
#endif

					Vector3d &v_i = m_data.velFluid[i];
					const Vector3d &xi = m_data.posFluid[i];

					//////////////////////////////////////////////////////////////////////////
					// Fluid
					//////////////////////////////////////////////////////////////////////////
					for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i); j++)
					{
						const unsigned int neighborIndex = m_data.getNeighbour(i, j);
						const Real b_j = m_data.densityAdv[neighborIndex] - density0;
						const Real kj = b_j*m_data.factor[neighborIndex];
						const Real kSum = (ki + kj);
						if (fabs(kSum) > m_eps)
						{
							const Vector3d &xj = m_data.posFluid[neighborIndex];
							const Vector3d grad_p_j = -m_data.mass[neighborIndex] * m_data.gradW(xi - xj);

							// Directly update velocities instead of storing pressure accelerations
							v_i -= h * kSum * grad_p_j;			// ki, kj already contain inverse density						
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
								const Vector3d &xj = m_data.posBoundary[neighborIndex];
								const Vector3d grad_p_j = -m_data.boundaryPsi[neighborIndex] * m_data.gradW(xi - xj);

								// Directly update velocities instead of storing pressure accelerations
								const Vector3d velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
								v_i += velChange;

								//TODO reactivate the external forces
								//m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * velChange * invH;
							}
						}
					}
				}
				{
					//////////////////////////////////////////////////////////////////////////
					// Evaluate rhs
					//////////////////////////////////////////////////////////////////////////
					const Real b_i = m_simulationData.getDensityAdv(i) - density0;
					const Real ki = b_i*m_simulationData.getFactor(i);
#ifdef USE_WARMSTART
					m_simulationData.getKappa(i) += ki;
#endif

					Vector3r &v_i = m_model->getVelocity(0, i);
					const Vector3r &xi = m_model->getPosition(0, i);

					//////////////////////////////////////////////////////////////////////////
					// Fluid
					//////////////////////////////////////////////////////////////////////////
					for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
					{
						const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
						const Real b_j = m_simulationData.getDensityAdv(neighborIndex) - density0;
						const Real kj = b_j*m_simulationData.getFactor(neighborIndex);
						const Real kSum = (ki + kj);
						if (fabs(kSum) > m_eps)
						{
							const Vector3r &xj = m_model->getPosition(0, neighborIndex);
							const Vector3r grad_p_j = -m_model->getMass(neighborIndex) * m_model->gradW(xi - xj);

							// Directly update velocities instead of storing pressure accelerations
							v_i -= h * kSum * grad_p_j;			// ki, kj already contain inverse density						
						}
					}

					//////////////////////////////////////////////////////////////////////////
					// Boundary
					//////////////////////////////////////////////////////////////////////////
					if (fabs(ki) > m_eps)
					{
						for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
						{
							for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, i); j++)
							{
								const unsigned int neighborIndex = m_model->getNeighbor(pid, i, j);
								const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
								const Vector3r grad_p_j = -m_model->getBoundaryPsi(pid, neighborIndex) * m_model->gradW(xi - xj);

								// Directly update velocities instead of storing pressure accelerations
								const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
								v_i += velChange;

								m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * velChange * invH;
							}
						}
					}
				}
				//checkVector3("pressureSolve: vel:", m_model->getVelocity(0,i) , m_data.velFluid[i]);
			}


			//////////////////////////////////////////////////////////////////////////
			// Update rho_adv and density error
			//////////////////////////////////////////////////////////////////////////
			avg_density_err = 0.0;
#pragma omp for reduction(+:avg_density_err) schedule(static) 
			for (int i = 0; i < numParticles; i++)
			{
				computeDensityAdv(i, numParticles, h, density0);

				avg_density_err += m_simulationData.getDensityAdv(i) - density0;

				//checkReal("pressure computation: avg_density_err checking density adv:", m_simulationData.getDensityAdv(i), m_data.densityAdv[i]);
			}
		}

		avg_density_err /= numParticles;

		m_iterations++;
	}


#ifdef USE_WARMSTART
	//////////////////////////////////////////////////////////////////////////
	// Multiply by h^2, the time step size has to be removed 
	// to make the stiffness value independent 
	// of the time step size
	//////////////////////////////////////////////////////////////////////////
	for (int i = 0; i < numParticles; i++) 
	{
		{
			m_data.kappa[i] *= h2;
		} 
		{
			m_simulationData.getKappa(i) *= h2;
		}
	}
		
#endif
}

void DFSPHCArrays::divergenceSolve()
{
	//////////////////////////////////////////////////////////////////////////
	// Init parameters
	//////////////////////////////////////////////////////////////////////////

	const Real h = m_data.h;
	const Real invH = 1.0 / h;
	const int numParticles = m_data.numFluidParticles;
	const unsigned int maxIter = m_maxIterationsV;
	const Real maxError = m_maxErrorV;
	const Real density0 = m_data.density0;


#ifdef USE_WARMSTART_V
	#pragma omp parallel default(shared)
	{
		//////////////////////////////////////////////////////////////////////////
		// Divide by h^2, the time step size has been removed in 
		// the last step to make the stiffness value independent 
		// of the time step size
		//////////////////////////////////////////////////////////////////////////
		#pragma omp for schedule(static)  
		for (int i = 0; i < numParticles; i++)
		{
			{
				m_data.kappaV[i] = 0.5*max(m_data.kappaV[i] *invH, -0.5);
				computeDensityChange(i, h, density0);
			}
			{
				m_simulationData.getKappaV(i) = 0.5*max(m_simulationData.getKappaV(i)*invH, -0.5);
				computeDensityChange(i, h, density0);
			}
			checkReal("divergence: warm start: computeDensityChange:", m_simulationData.getDensityAdv(i), m_data.densityAdv[i]);
			checkReal("divergence: warm start: kappaV first loop:", m_simulationData.getKappaV(i), m_data.kappaV[i]);
		}

		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			{
				if (m_data.densityAdv[i] > 0.0)
				{
					Vector3d &vel = m_data.velFluid[i];
					const Real ki = m_data.kappaV[i];
					const Vector3d &xi = m_data.posFluid[i];

					//////////////////////////////////////////////////////////////////////////
					// Fluid
					//////////////////////////////////////////////////////////////////////////
					for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i); j++)
					{
						const unsigned int neighborIndex = m_data.getNeighbour(i,j);
						const Real kj = m_data.kappaV[neighborIndex];

						const Real kSum = (ki + kj);
						if (fabs(kSum) > m_eps)
						{
							const Vector3d &xj = m_data.posFluid[neighborIndex];
							const Vector3d grad_p_j = -m_data.mass[neighborIndex] * m_data.gradW(xi - xj);
							vel -= h * kSum * grad_p_j;					// ki, kj already contain inverse density
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
								const Vector3d &xj = m_data.posBoundary[neighborIndex];
								const Vector3d grad_p_j = -m_data.boundaryPsi[neighborIndex] * m_data.gradW(xi - xj);

								const Vector3d velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
								vel += velChange;

								///TODO reactivate this to have the forces on rigid bodies
								//m_model->getForce(pid, neighborIndex) -= m_data.mass[i] * velChange * invH;
							}
						}
					}
				}
			}
			{
				if (m_simulationData.getDensityAdv(i) > 0.0)
				{
					Vector3r &vel = m_model->getVelocity(0, i);
					const Real ki = m_simulationData.getKappaV(i);
					const Vector3r &xi = m_model->getPosition(0, i);

					//////////////////////////////////////////////////////////////////////////
					// Fluid
					//////////////////////////////////////////////////////////////////////////
					for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
					{
						const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
						const Real kj = m_simulationData.getKappaV(neighborIndex);

						const Real kSum = (ki + kj);
						if (fabs(kSum) > m_eps)
						{
							const Vector3r &xj = m_model->getPosition(0, neighborIndex);
							const Vector3r grad_p_j = -m_model->getMass(neighborIndex) * m_model->gradW(xi - xj);
							vel -= h * kSum * grad_p_j;					// ki, kj already contain inverse density
						}
					}

					//////////////////////////////////////////////////////////////////////////
					// Boundary
					//////////////////////////////////////////////////////////////////////////
					if (fabs(ki) > m_eps)
					{
						for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
						{
							for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, i); j++)
							{
								const unsigned int neighborIndex = m_model->getNeighbor(pid, i, j);
								const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
								const Vector3r grad_p_j = -m_model->getBoundaryPsi(pid, neighborIndex) * m_model->gradW(xi - xj);

								const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
								vel += velChange;

								m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * velChange * invH;
							}
						}
					}
				}
			}
			//checkVector3("divergence: warm start: vel first loop:",m_model->getVelocity(0, i), m_data.velFluid[i]);
		}
		
	}
#endif


	//////////////////////////////////////////////////////////////////////////
	// Compute velocity of density change
	//////////////////////////////////////////////////////////////////////////
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			//*
			{
				computeDensityChange(i, h, density0);
				m_data.factor[i] *= invH;

#ifdef USE_WARMSTART_V
				m_data.kappaV[i] = 0.0;
#endif
			} 
			//*/
			{
				computeDensityChange(i, h, density0);
				m_simulationData.getFactor(i) *= invH;

#ifdef USE_WARMSTART_V
				m_simulationData.getKappaV(i) = 0.0;
#endif
			}
			//checkReal("divergence: init: computeDensityChange:", m_simulationData.getDensityAdv(i), m_data.densityAdv[i]);
		}
	}


	for (int i = 0; i < m_data.numFluidParticles; ++i) {
		checkVector3("divergecne middle check v:", m_model->getVelocity(0, i), m_data.velFluid[i]);
		checkReal("divergecne middle check desity adv:", m_simulationData.getDensityAdv(i), m_data.densityAdv[i]);
		checkReal("divergecne middle check desity adv:", m_simulationData.getFactor(i), m_data.factor[i]);
	}

	m_iterationsV = 0;

	//////////////////////////////////////////////////////////////////////////
	// Start solver
	//////////////////////////////////////////////////////////////////////////
	
	// Maximal allowed density fluctuation
	// use maximal density error divided by time step size
	const Real eta = (1.0/h) * maxError * 0.01 * density0;  // maxError is given in percent
	
	Real avg_density_err = 0.0;
	while (((avg_density_err > eta) || (m_iterationsV < 1)) && (m_iterationsV < maxIter))
	{
		avg_density_err = 0.0;
		
		//////////////////////////////////////////////////////////////////////////
		// Perform Jacobi iteration over all blocks
		//////////////////////////////////////////////////////////////////////////	
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				{
					Vector3d &v_i = m_data.velFluid[i];
					//////////////////////////////////////////////////////////////////////////
					// Evaluate rhs
					//////////////////////////////////////////////////////////////////////////
					const Real b_i = m_data.densityAdv[i];
					const Real ki = b_i*m_data.factor[i];
#ifdef USE_WARMSTART_V
					m_data.kappaV[i] += ki;
#endif
					const Vector3d &xi = m_data.posFluid[i];

					

					//////////////////////////////////////////////////////////////////////////
					// Fluid
					//////////////////////////////////////////////////////////////////////////
					for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i); j++)
					{
						const unsigned int neighborIndex = m_data.getNeighbour(i, j);
						const Real b_j = m_data.densityAdv[neighborIndex];
						const Real kj = b_j*m_data.factor[neighborIndex];
						const Real kSum = (ki + kj);
						if (fabs(kSum) > m_eps)
						{
							const Vector3d &xj = m_data.posFluid[neighborIndex];
							const Vector3d grad_p_j = -m_data.mass[neighborIndex] * m_data.gradW(xi - xj);
							v_i -= h * kSum * grad_p_j;			// ki, kj already contain inverse density
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
								const Vector3d &xj = m_data.posBoundary[neighborIndex];
								const Vector3d grad_p_j = -m_data.boundaryPsi[neighborIndex] * m_data.gradW(xi - xj);

								const Vector3d velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
								v_i += velChange;

								///TODO reactivate this for objects
								//m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * velChange * invH;
							}
						}
					}
				}
				{
					Vector3r &v_i = m_model->getVelocity(0, i);
					//////////////////////////////////////////////////////////////////////////
					// Evaluate rhs
					//////////////////////////////////////////////////////////////////////////
					const Real b_i = m_simulationData.getDensityAdv(i);
					const Real ki = b_i*m_simulationData.getFactor(i);
#ifdef USE_WARMSTART_V
					m_simulationData.getKappaV(i) += ki;
#endif
					const Vector3r &xi = m_model->getPosition(0, i);

					//////////////////////////////////////////////////////////////////////////
					// Fluid
					//////////////////////////////////////////////////////////////////////////
					for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
					{
						const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
						const Real b_j = m_simulationData.getDensityAdv(neighborIndex);
						const Real kj = b_j*m_simulationData.getFactor(neighborIndex);

						const Real kSum = (ki + kj);
						if (fabs(kSum) > m_eps)
						{
							const Vector3r &xj = m_model->getPosition(0, neighborIndex);
							const Vector3r grad_p_j = -m_model->getMass(neighborIndex) * m_model->gradW(xi - xj);
							v_i -= h * kSum * grad_p_j;			// ki, kj already contain inverse density
						}
					}

					//////////////////////////////////////////////////////////////////////////
					// Boundary
					//////////////////////////////////////////////////////////////////////////
					if (fabs(ki) > m_eps)
					{
						for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
						{
							for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, i); j++)
							{
								const unsigned int neighborIndex = m_model->getNeighbor(pid, i, j);
								const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
								const Vector3r grad_p_j = -m_model->getBoundaryPsi(pid, neighborIndex) * m_model->gradW(xi - xj);

								const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
								v_i += velChange;

								m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * velChange * invH;
							}
						}
					}
				}

				checkVector3("Divergence computation: v:", m_model->getVelocity(0, i), m_data.velFluid[i]);
				checkReal("Divergence computation : kappaV:", m_simulationData.getKappaV(i), m_data.kappaV[i]);
				
			}

			//////////////////////////////////////////////////////////////////////////
			// Update rho_adv and density error
			//////////////////////////////////////////////////////////////////////////
			#pragma omp for reduction(+:avg_density_err) schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				computeDensityChange(i, h, density0);
				avg_density_err += m_data.densityAdv[i];
				
				//checkReal("Divergence computation: avg_density_err checking density change:",m_simulationData.getDensityAdv(i), m_data.densityAdv[i]);
			}
		}	
	
		avg_density_err /= numParticles;
		m_iterationsV++;
	}

#ifdef USE_WARMSTART_V
	//////////////////////////////////////////////////////////////////////////
	// Multiply by h, the time step size has to be removed 
	// to make the stiffness value independent 
	// of the time step size
	//////////////////////////////////////////////////////////////////////////
	for (int i = 0; i < numParticles; i++) 
	{
		{
			m_data.kappaV[i] *= h;
		}
		{
			m_simulationData.getKappaV(i) *= h;
		}
		//checkReal("divergence: warm start: kappaV end loop:", m_simulationData.getKappaV(i), m_data.kappaV[i]);
	}
		
 #endif

	for (int i = 0; i < numParticles; i++)
	{
		{
			m_data.factor[i] *= h;
		}
		{
			m_simulationData.getFactor(i) *= h;
		}
		//checkReal("divergence: factor end loop:", m_simulationData.getFactor(i), m_data.factor[i]);
	}
}


void DFSPHCArrays::computeDensityAdv(const unsigned int index, const int numParticles, const Real h, const Real density0)
{
	{
		
		Real &densityAdv = m_data.densityAdv[index];
		const Real &density = m_data.density[index];
		const Vector3d &xi = m_data.posFluid[index];
		const Vector3d &vi = m_data.velFluid[index];
		Real delta = 0.0;
		//const Vector3r &vi = m_model->getVelocity(0, index);

		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(index); j++)
		{
			const unsigned int neighborIndex = m_data.getNeighbour(index, j);
			const Vector3d &xj = m_data.posFluid[neighborIndex];
			const Vector3d &vj = m_data.velFluid[neighborIndex];
			delta += m_data.mass[neighborIndex] * (vi - vj).dot(m_data.gradW(xi - xj));
		}

		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int pid = 1; pid < 2; pid++)
		{
			for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(index, pid); j++)
			{
				const unsigned int neighborIndex = m_data.getNeighbour(index, j, pid);
				const Vector3d &xj = m_data.posBoundary[neighborIndex];
				const Vector3d &vj = m_data.velBoundary[neighborIndex];
				delta += m_data.boundaryPsi[neighborIndex] * (vi - vj).dot(m_data.gradW(xi - xj));
			}
		}

		densityAdv = density + h*delta;
		densityAdv = max(densityAdv, density0);
	}
	{
		Real &densityAdv = m_simulationData.getDensityAdv(index);
		const Real &density = m_model->getDensity(index);
		const Vector3r &xi = m_model->getPosition(0, index);
		const Vector3r &vi = m_model->getVelocity(0, index);
		Real delta = 0.0;

		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, index); j++)
		{
			const unsigned int neighborIndex = m_model->getNeighbor(0, index, j);
			const Vector3r &xj = m_model->getPosition(0, neighborIndex);
			const Vector3r &vj = m_model->getVelocity(0, neighborIndex);
			delta += m_model->getMass(neighborIndex) * (vi - vj).dot(m_model->gradW(xi - xj));
		}

		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
		{
			for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, index); j++)
			{
				const unsigned int neighborIndex = m_model->getNeighbor(pid, index, j);
				const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
				const Vector3r &vj = m_model->getVelocity(pid, neighborIndex);
				delta += m_model->getBoundaryPsi(pid, neighborIndex) * (vi - vj).dot(m_model->gradW(xi - xj));
			}
		}

		densityAdv = density + h*delta;
		densityAdv = max(densityAdv, density0);
	}
	//checkReal("computeDensityAdv: densityadv:", m_simulationData.getDensityAdv(index), m_data.densityAdv[index]);
}

void DFSPHCArrays::computeDensityChange(const unsigned int index, const Real h, const Real density0)
{
	{
		Real &densityAdv = m_data.densityAdv[index];
		const Vector3d &xi = m_data.posFluid[index];
		const Vector3d &vi = m_data.velFluid[index];
		densityAdv = 0.0;
		unsigned int numNeighbors = m_data.getNumberOfNeighbourgs(index);

		//*
		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int j = 0; j < numNeighbors; j++)
		{
		const unsigned int neighborIndex = m_data.getNeighbour(index, j);
		const Vector3d &xj = m_data.posFluid[neighborIndex];
		const Vector3d &vj = m_data.velFluid[neighborIndex];
		
		//const Vector3r &vj2 = m_model->getVelocity(0, neighborIndex);
		//checkVector3("computeDensityChange vj:", vj2, vj);
		//checkVector3("computeDensityChange vi:", vector3dTo3r(vi), vi);
		//checkVector3("computeDensityChange vi-vj:", vi - vector3rTo3d(vj2), vi - vj);
		densityAdv += m_data.mass[neighborIndex] * (vi - vj).dot(m_data.gradW(xi - xj));
		}
		//*/
		
		//*
		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int pid = 1; pid < 2; pid++)
		{
			numNeighbors += m_data.getNumberOfNeighbourgs(index, pid);
			for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(index, pid); j++)
			{
				const unsigned int neighborIndex = m_data.getNeighbour(index, j, pid);
				const Vector3d &xj = m_data.posBoundary[neighborIndex];
				const Vector3d &vj = m_data.velBoundary[neighborIndex];
				densityAdv += m_data.boundaryPsi[neighborIndex] * (vi - vj).dot(m_data.gradW(xi - xj));
			}
		}
		//*/

		// only correct positive divergence
		densityAdv = max(densityAdv, 0.0);

		// in case of particle deficiency do not perform a divergence solve
		if (numNeighbors < 20)
			densityAdv = 0.0;
	}
	{
		Real &densityAdv = m_simulationData.getDensityAdv(index);
		const Vector3r &xi = m_model->getPosition(0, index);
		const Vector3r &vi = m_model->getVelocity(0, index);
		densityAdv = 0.0;
		unsigned int numNeighbors = m_model->numberOfNeighbors(0, index);

		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int j = 0; j < numNeighbors; j++)
		{
			const unsigned int neighborIndex = m_model->getNeighbor(0, index, j);
			const Vector3r &xj = m_model->getPosition(0, neighborIndex);
			const Vector3r &vj = m_model->getVelocity(0, neighborIndex);
			densityAdv += m_model->getMass(neighborIndex) * (vi - vj).dot(m_model->gradW(xi - xj));
		}

		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
		{
			numNeighbors += m_model->numberOfNeighbors(pid, index);
			for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, index); j++)
			{
				const unsigned int neighborIndex = m_model->getNeighbor(pid, index, j);
				const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
				const Vector3r &vj = m_model->getVelocity(pid, neighborIndex);
				densityAdv += m_model->getBoundaryPsi(pid, neighborIndex) * (vi - vj).dot(m_model->gradW(xi - xj));
			}
		}

		// only correct positive divergence
		densityAdv = max(densityAdv, 0.0);

		// in case of particle deficiency do not perform a divergence solve
		if (numNeighbors < 20)
			densityAdv = 0.0;

	}

	
}

void DFSPHCArrays::reset()
{
	TimeStep::reset();
	m_simulationData.reset();
	m_counter = 0;
	m_iterationsV = 0;
	m_data.reset(m_model);
}

void DFSPHCArrays::performNeighborhoodSearch()
{
	if (m_counter % 500 == 0)
	{
		m_model->performNeighborhoodSearchSort();
		m_simulationData.performNeighborhoodSearchSort();
		TimeStep::performNeighborhoodSearchSort();
		m_data.sortDynamicData(m_model);
	}
	m_counter++;

	TimeStep::performNeighborhoodSearch();
}

void DFSPHCArrays::emittedParticles(const unsigned int startIndex)
{
	m_simulationData.emittedParticles(startIndex);
	TimeStep::emittedParticles(startIndex);
}


void DFSPHCArrays::computeDensities()
{
	
	const unsigned int numParticles = m_data.numFluidParticles;

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			{
				Real &density = m_data.density[i];
				
				// Compute current density for particle i
				density = m_data.mass[i] * m_data.W_zero;
				const Vector3d &xi = m_data.posFluid[i];

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i); j++)
				{
					const unsigned int neighborIndex = m_data.getNeighbour(i, j);
					const Vector3d &xj = m_data.posFluid[neighborIndex];
					density += m_data.mass[neighborIndex] * m_data.W((xi - xj).norm());
				}
				//*
				//////////////////////////////////////////////////////////////////////////
				// Boundary
				//////////////////////////////////////////////////////////////////////////
				for (unsigned int pid = 1; pid < 2; pid++)
				{
					for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i, pid); j++)
					{
						const unsigned int neighborIndex = m_data.getNeighbour(i, j, pid);
						const Vector3d &xj = m_data.posBoundary[neighborIndex];
						density += m_data.boundaryPsi[neighborIndex] * m_data.W((xi - xj).norm());
					}
				}
				//*/
				//m_model->getDensity(i) = density;
			}
			{
				Real &density = m_model->getDensity(i);

				// Compute current density for particle i
				density = m_model->getMass(i) * m_model->W_zero();
				const Vector3r &xi = m_model->getPosition(0, i);
				
				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
				{
					const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
					const Vector3r &xj = m_model->getPosition(0, neighborIndex);
					density += m_model->getMass(neighborIndex) * m_model->W(xi - xj);
				}
				//*
				//////////////////////////////////////////////////////////////////////////
				// Boundary
				//////////////////////////////////////////////////////////////////////////
				for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
				{
					for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, i); j++)
					{
						const unsigned int neighborIndex = m_model->getNeighbor(pid, i, j);
						const Vector3r &xj = m_model->getPosition(pid, neighborIndex);

						// Boundary: Akinci2012
						density += m_model->getBoundaryPsi(pid, neighborIndex) * m_model->W(xi - xj);
					}
				}
				//*/
			}

			//checkReal("density:", m_model->getDensity(i), m_data.density[i]);
		}
	}
}

void DFSPHCArrays::clearAccelerations()
{
	const unsigned int count = m_data.numFluidParticles;
	const Vector3d &grav = m_data.gravitation;

	//checkVector3("clearAccelerations: check gravitation value:", m_model->getGravitation(), m_data.gravitation);

	for (unsigned int i = 0; i < count; i++)
	{
		{
			// Clear accelerations of existing particles
			if (m_data.mass[i] != 0.0)
			{
				m_data.accFluid[i] = grav;
			}
		}
		{
			// Clear accelerations of existing particles
			if (m_model->getMass(i) != 0.0)
			{
				m_model->getAcceleration(i) = m_model->getGravitation();
			}
		}
		//checkVector3("clearAccelerations: acc:", m_model->getAcceleration(i), m_data.accFluid[i]);
	}
}

void DFSPHCArrays::updateVelocities(double h) 
{
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)m_data.numFluidParticles; i++)
		{
			{
				m_data.velFluid[i] += h * m_data.accFluid[i];
			}
			{
				Vector3r &vel = m_model->getVelocity(0, i);
				vel += h * m_model->getAcceleration(i);
			}
			//checkVector3("updateVelocities:", m_model->getVelocity(0, i), m_data.velFluid[i]);
		}
		
	}
}

void DFSPHCArrays::updatePositions(double h)
{
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)m_data.numFluidParticles; i++)
		{
			{
				m_data.posFluid[i] += h * m_data.velFluid[i];
			}
			{
				Vector3r &xi = m_model->getPosition(0, i);
				const Vector3r &vi = m_model->getVelocity(0, i);
				xi += h * vi;
			}
			//checkVector3("updatePositions:", m_model->getPosition(0, i), m_data.posFluid[i]);
		}
	}
}

void DFSPHCArrays::computeNonPressureForces()
{
	START_TIMING("computeNonPressureForces");
	//computeSurfaceTension();
	//computeViscosity();
	//computeVorticity();
	//computeDragForce();
	//surfaceTension_Akinci2013();
	viscosity_XSPH();
	STOP_TIMING_AVG;

}

void DFSPHCArrays::updateTimeStepSizeCFL(const Real minTimeStepSize)
{
	TimeStep::updateTimeStepSizeCFL(minTimeStepSize);

	Real h = m_data.h;
	const Real radius = m_data.particleRadius;
	const Real diameter = 2.0*radius;
	const unsigned int numParticles = m_data.numFluidParticles;

	Real maxVel = 0.1;
	{
		// Approximate max. position change due to current velocities
		for (unsigned int i = 0; i < numParticles; i++)
		{
			const Vector3d &vel = m_data.velFluid[i];
			const Vector3d &accel = m_data.accFluid[i];
			const Real velMag = (vel + accel*h).squaredNorm();
			if (velMag > maxVel)
				maxVel = velMag;
		}

		// boundary particles
		///TODO place the code back here (see the timestep.cpp file for the code
		///but since it only consider dynamic objects and I don't have any I'll
		///simplify the code by removing for now
	}
	{
		Real maxVelOriginal = 0.1;
		// Approximate max. position change due to current velocities
		for (unsigned int i = 0; i < numParticles; i++)
		{
			const Vector3r &vel = m_model->getVelocity(0, i);
			const Vector3r &accel = m_model->getAcceleration(i);
			const Real velMag = (vel + accel*h).squaredNorm();
			if (velMag > maxVelOriginal)
				maxVelOriginal = velMag;
		}

		// boundary particles
		///TODO place the code back here (see the timestep.cpp file for the code
		///but since it only consider dynamic objects and I don't have any I'll
		///simplify the code by removing for now

		checkReal("updateTimeStepSizeCFL: maxVel:", maxVelOriginal, maxVel);
	}

	// Approximate max. time step size 		
	h = m_cflFactor * .4 * (diameter / (sqrt(maxVel)));

	h = min(h, m_cflMaxTimeStepSize);
	h = max(h, minTimeStepSize);

	m_data.h_future = h;

}

void DFSPHCArrays::viscosity_XSPH()
{
	const unsigned int numParticles = m_data.numFluidParticles;

	const Real h = m_data.h;
	const Real invH = (1.0 / h);

	// Compute viscosity forces (XSPH)
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			{
				const Vector3d &xi = m_data.posFluid[i];
				const Vector3d &vi = m_data.velFluid[i];
				Vector3d &ai = m_data.accFluid[i];
				const Real density_i = m_data.density[i];

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i); j++)
				{
					const unsigned int neighborIndex = m_data.getNeighbour(i, j);
					const Vector3d &xj = m_data.posFluid[neighborIndex];
					const Vector3d &vj = m_data.velFluid[neighborIndex];

					// Viscosity
					const Real density_j = m_data.density[neighborIndex];
					ai -= invH * m_data.viscosity * (m_data.mass[neighborIndex] / density_j) * (vi - vj) * m_data.W(xi - xj);
				}
			}
			{
				const Vector3r &xi = m_model->getPosition(0, i);
				const Vector3r &vi = m_model->getVelocity(0, i);
				Vector3r &ai = m_model->getAcceleration(i);
				const Real density_i = m_model->getDensity(i);

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
				{
					const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
					const Vector3r &xj = m_model->getPosition(0, neighborIndex);
					const Vector3r &vj = m_model->getVelocity(0, neighborIndex);

					// Viscosity
					const Real density_j = m_model->getDensity(neighborIndex);
					ai -= invH * m_viscosity->getViscosity() * (m_model->getMass(neighborIndex) / density_j) * (vi - vj) * m_model->W(xi - xj);
				}
			}
			//checkVector3("viscosityXSPH", m_model->getAcceleration(i), m_data.accFluid[i]);
		}
	}
}


void DFSPHCArrays::surfaceTension_Akinci2013()
{
	throw("go look for the code but there are multiples funtion to copy so ...");
}

void DFSPHCArrays::checkReal(std::string txt, Real old_v, Real new_v) {
	double error = std::abs(old_v - new_v);
	double trigger = 0.0; std::max(std::abs(old_v) * 1E-20, 0.0);
	if (error > trigger) {
	//if (old_v!=new_v) {
		ostringstream oss;
		oss << "(Real)" << txt << " old/ new: " << old_v << " / " << new_v <<
			" //// " << error << " / " << trigger << std::endl;
		std::cout << oss.str();
		exit(2679);
	}
}

void DFSPHCArrays::checkVector3(std::string txt, Vector3d old_v, Vector3d new_v) {

	Vector3d error = (old_v - new_v).toAbs();
	Vector3d trigger = Vector3d(0, 0, 0);// old_v.toAbs() * 1E-20;
	trigger.clampTo(0);
	if (error.x > trigger.x || error.y > trigger.y || error.z > trigger.z) {
		ostringstream oss;
		oss << "(Vector3) " << txt << " error/ trigger: " "(" << error.x << ", " << error.y << ", " << error.z << ")" << " / " <<
			"(" << trigger.x << ", " << trigger.y << ", " << trigger.z << ")" << " //// actual value old/new (x, y, z): " <<
			"(" << old_v.x << ", " << old_v.y << ", " << old_v.z << ")" << " / " <<
			"(" << new_v.x << ", " << new_v.y << ", " << new_v.z << ")" << std::endl;
		std::cout << oss.str();
		exit(2679);
	}

}


