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

DFSPHCData::DFSPHCData(FluidModel *model) {
	//just I dont hndle external objects for now
	if (model->m_particleObjects.size() > 2) {
		exit(2568);
	}
	
	m_kernel.setRadius(model->m_supportRadius);
	W_zero = m_kernel.W_zero();

	particleRadius = model->getParticleRadius();
	
	numBoundaryParticles = model->m_particleObjects[1]->numberOfParticles();
	numFluidParticles = model->numActiveParticles();
		
	if (true) {
		//initialisation on the GPU
		allocate_c_array_struct_cuda_managed((*this));
	}
	else {
		//initialisation on the CPU
		posBoundary = new Vector3d[numBoundaryParticles];
		velBoundary = new Vector3d[numBoundaryParticles];
		boundaryPsi = new Real[numBoundaryParticles];


		//handle the fluid
		mass = new Real[numFluidParticles];
		posFluid = new Vector3d[numFluidParticles];
		velFluid = new Vector3d[numFluidParticles];
		accFluid = new Vector3d[numFluidParticles];
		numberOfNeighbourgs = new int[numFluidParticles * 2];
		neighbourgs = new int[numFluidParticles * 2 * MAX_NEIGHBOURS];

		density = new Real[numFluidParticles];
		factor = new Real[numFluidParticles];
		kappa = new Real[numFluidParticles];
		kappaV = new Real[numFluidParticles];
		densityAdv = new Real[numFluidParticles];
	}





	reset(model);
}

void DFSPHCData::reset(FluidModel *model) {
	if (numFluidParticles != model->numActiveParticles()) {
		std::cout << "DFSPHCData::reset: fml the nbr of fluid particles has been modified" << std::endl;
		exit(3469);
	}
	if (numBoundaryParticles != model->m_particleObjects[1]->numberOfParticles()) {
		std::cout << "DFSPHCData::reset: fml the nbr of boundaries particles has been modified" << std::endl;
		exit(9657);
	}
	density0 = model->getDensity0();

	for (int i = 0; i < numBoundaryParticles; ++i) {
		FluidModel::RigidBodyParticleObject* particleObj = static_cast<FluidModel::RigidBodyParticleObject*>(model->m_particleObjects[1]);
		posBoundary[i] = vector3rTo3d(particleObj->m_x[i]);
		velBoundary[i] = vector3rTo3d(particleObj->m_v[i]);
		boundaryPsi[i] = particleObj->m_boundaryPsi[i];
	}

	for (int i = 0; i < numFluidParticles; ++i) {
		mass[i] = model->getMass(i);

		//clear the internal kappa and  kappa v
		kappa[i] = 0.0;
		kappaV[i] = 0.0;
	}
	
	h = TimeManager::getCurrent()->getTimeStepSize();
}


void DFSPHCData::loadDynamicData(FluidModel *model, const SimulationDataDFSPH& data) {

	if (model->numActiveParticles() != numFluidParticles ||
		model->getParticleRadius() != particleRadius ) {
		exit(1569);
	}

	//density and acc are not conserved between timesteps so no need to copy them
	for (int i = 0; i < numFluidParticles; ++i) {
		posFluid[i] = vector3rTo3d(model->getPosition(0, i));
		velFluid[i] = vector3rTo3d(model->getVelocity(0, i));
		kappa[i] = data.getKappa(i);
		kappaV[i] = data.getKappaV(i);
	}



	//load the neighbours
	for (int i = 0; i < (int)numFluidParticles; ++i)
	{
		numberOfNeighbourgs[i] = 0;
		numberOfNeighbourgs[i + numFluidParticles] = 0;

		for (unsigned int j = 0; j < model->numberOfNeighbors(0, i); j++)
		{
			neighbourgs[i*MAX_NEIGHBOURS + numberOfNeighbourgs[i]] = model->getNeighbor(0, i, j);
			numberOfNeighbourgs[i]++;
		}

		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
		if (model->numberOfPointSets() > 2) {
			exit(5976);
		}


		for (unsigned int pid = 1; pid < model->numberOfPointSets(); pid++)
		{
			const int offset = pid*numFluidParticles*MAX_NEIGHBOURS;
			for (unsigned int j = 0; j < model->numberOfNeighbors(pid, i); j++)
			{
				neighbourgs[offset + i * MAX_NEIGHBOURS + numberOfNeighbourgs[i + numFluidParticles*pid]] = model->getNeighbor(pid, i, j);
				numberOfNeighbourgs[i + numFluidParticles*pid]++;
			}
		}
	}

	for (int i = 0; i < (int)numFluidParticles * 2; ++i)
	{
		if (numberOfNeighbourgs[i] > MAX_NEIGHBOURS) {
			std::cout << "crossed the max neighbour: " << numberOfNeighbourgs[i] << std::endl;
		}
	}

	/*
	//effectively the maximum is around 50 for each side
	//so I'll put static buffers of 60
	int max_val_f = 0;
	int max_val_b = 0;
	for (int i = 0; i < (int)numFluidParticles*2; i+=2)
	{
	if (numberOfNeighbourgs[i ] > max_val_f) {
	max_val_f = numberOfNeighbourgs[i];
	}
	if (numberOfNeighbourgs[i + 1] > max_val_b) {
	max_val_b = numberOfNeighbourgs[i+numFluidParticles];
	}
	}

	std::cout << "max neighbours fluid/boundaries: " << max_val_f << " / " << max_val_b << std::endl;
	//*/
}


void DFSPHCData::readDynamicData(FluidModel *model, SimulationDataDFSPH& data) {

	if (model->numActiveParticles() != numFluidParticles) {
		exit(1569);
	}

	//density and acc are not conserved between timesteps so no need to copy them
	for (int i = 0; i < numFluidParticles; ++i) {
		model->getPosition(0, i)= vector3dTo3r(posFluid[i]);
		model->getVelocity(0, i)= vector3dTo3r(velFluid[i]);
		data.getKappa(i)= kappa[i];
		data.getKappaV(i)= kappaV[i];
	}
}

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

	m_data.viscosity = m_viscosity->getViscosity();
	
	const Real h = m_data.h;

	const unsigned int numParticles = m_model->numActiveParticles();


	performNeighborhoodSearch();

	//start of the c arrays
	m_data.loadDynamicData(m_model, m_simulationData);

	computeDensities();

	START_TIMING("computeDFSPHFactor");
	computeDFSPHFactor();
	STOP_TIMING_AVG;

	if (m_enableDivergenceSolver)
	{
		START_TIMING("divergenceSolve");
		divergenceSolve();
		STOP_TIMING_AVG;
	}
	else
		m_iterationsV = 0;


	// Compute accelerations: a(t)
	clearAccelerations();

	computeNonPressureForces();

	updateTimeStepSize();

	updateVelocities(h);
	


	START_TIMING("pressureSolve");
	pressureSolve();
	STOP_TIMING_AVG;

	std::cout << "pressure solve done" << std::endl;

	updatePositions(h);

	emitParticles();


	//start of the c arrays
	m_data.readDynamicData(m_model, m_simulationData);


	// Compute new time	

	TimeManager::getCurrent()->setTimeStepSize(m_data.h);
	TimeManager::getCurrent()->setTime(TimeManager::getCurrent()->getTime() + h);
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

			checkReal("factor:", m_simulationData.getFactor(i), m_data.factor[i]);
			
			//factor = m_data.factor[i];
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
			checkReal("pressureSolve: warm start: early kappa:", m_simulationData.getKappa(i), m_data.kappa[i]);

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
			checkVector3("pressureSolve: warm start: vel first loop:", m_model->getVelocity(0, i), m_data.velFluid[i]);
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
			checkReal("pressureSolve: factor begin:", m_simulationData.getFactor(i), m_data.factor[i]);
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

				checkVector3("pressureSolve: vel start loop:", m_model->getVelocity(0, i), m_data.velFluid[i]);
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
				checkVector3("pressureSolve: vel:", m_model->getVelocity(0,i) , m_data.velFluid[i]);
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

				checkReal("pressure computation: avg_density_err checking density adv:",
					m_simulationData.getDensityAdv(i), m_data.densityAdv[i]);
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
			checkVector3("divergence: warm start: vel first loop:",m_model->getVelocity(0, i), m_data.velFluid[i]);
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
		}
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
				
			}

			//////////////////////////////////////////////////////////////////////////
			// Update rho_adv and density error
			//////////////////////////////////////////////////////////////////////////
			#pragma omp for reduction(+:avg_density_err) schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				computeDensityChange(i, h, density0);
				avg_density_err += m_data.densityAdv[i];
				
				checkReal("Divergence computation: avg_density_err checking density change:",
					m_simulationData.getDensityAdv(i), m_data.densityAdv[i]);
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
		checkReal("divergence: warm start: kappaV end loop:", m_simulationData.getKappaV(i), m_data.kappaV[i]);
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
		checkReal("divergence: factor end loop:", m_simulationData.getFactor(i), m_data.factor[i]);
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
	checkReal("computeDensityAdv: densityadv:", m_simulationData.getDensityAdv(index), m_data.densityAdv[index]);
}

void DFSPHCArrays::computeDensityChange(const unsigned int index, const Real h, const Real density0)
{
	{
		Real &densityAdv = m_data.densityAdv[index];
		const Vector3d &xi = m_data.posFluid[index];
		const Vector3d &vi = m_data.velFluid[index];
		densityAdv = 0.0;
		unsigned int numNeighbors = m_data.getNumberOfNeighbourgs(index);
		

		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int j = 0; j < numNeighbors; j++)
		{
			/*
			const unsigned int neighborIndex = m_data.getNeighbour(index, j);
			const Vector3r &xj = m_data.posFluid[neighborIndex];
			const Vector3r &vj = m_data.velFluid[neighborIndex];
			densityAdv += m_data.mass[neighborIndex] * (vi - vj).dot(m_data.gradW(xi - xj));
			//*/
			const unsigned int neighborIndex = m_data.getNeighbour(index, j);
			const Vector3d &xj = m_data.posFluid[neighborIndex];
			const Vector3d &vj = m_data.velFluid[neighborIndex];
			densityAdv += m_data.mass[neighborIndex] * (vi - vj).dot(m_data.gradW(xi - xj));
		}

		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int pid = 1; pid < 2; pid++)
		{
			numNeighbors += m_data.getNumberOfNeighbourgs(index, pid);
			for (unsigned int j = 0; j <  m_data.getNumberOfNeighbourgs(index, pid); j++)
			{
				const unsigned int neighborIndex = m_data.getNeighbour(index, j, pid);
				const Vector3d &xj = m_data.posBoundary[neighborIndex];
				const Vector3d &vj = m_data.velBoundary[neighborIndex];
				densityAdv += m_data.boundaryPsi[neighborIndex] * (vi - vj).dot(m_data.gradW(xi - xj));
			}
		}


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

	checkReal("computeDensityChange:", m_simulationData.getDensityAdv(index), m_data.densityAdv[index]);
	
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
					density += m_data.mass[neighborIndex] * m_data.W(xi - xj);
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

						// Boundary: Akinci2012
						density += m_data.boundaryPsi[neighborIndex] * m_data.W(xi - xj);
					}
				}

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
			}

			checkReal("density:", m_model->getDensity(i), m_data.density[i]);
		}
	}
}

void DFSPHCArrays::clearAccelerations()
{
	const unsigned int count = m_data.numFluidParticles;
	const Vector3d &grav = m_data.gravitation;

	checkVector3("clearAccelerations: check gravitation value:", m_model->getGravitation(), m_data.gravitation);

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
		checkVector3("clearAccelerations: acc:", m_model->getAcceleration(i), m_data.accFluid[i]);
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
			checkVector3("updateVelocities:", m_model->getVelocity(0, i), m_data.velFluid[i]);
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
			checkVector3("updatePositions:", m_model->getPosition(0, i), m_data.posFluid[i]);
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
	
	m_data.h = h;

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
			checkVector3("viscosityXSPH", m_model->getAcceleration(i), m_data.accFluid[i]);
		}
	}
}


void DFSPHCArrays::surfaceTension_Akinci2013()
{
	throw("go look for the code but there are multiples funtion to copy so ...");
}

void DFSPHCArrays::checkReal(std::string txt, Real old_v, Real new_v) {
	double error = std::abs(old_v - new_v);
	double trigger = std::abs(old_v / 10000);
	if (error > trigger) {
		ostringstream oss;
		oss << txt <<" old/ new: " << old_v << " / " << new_v <<
			" //// " << error << " / " << trigger / 10000 << std::endl;
		std::cout << oss.str();
		exit(2679);
	}
}

void DFSPHCArrays::checkVector3(std::string txt, Vector3r old_v, Vector3r new_v) {

	double error = (old_v - new_v).norm();
	double trigger = std::abs(old_v.norm() / 10000);
	if (error > trigger) {
		ostringstream oss;
		oss << txt << " old/ new: " << old_v.norm() << " / " << new_v.norm() <<
			" //// " << error << " / " << trigger / 10000 << std::endl;
		std::cout << oss.str();
		exit(2679);
	}

}


