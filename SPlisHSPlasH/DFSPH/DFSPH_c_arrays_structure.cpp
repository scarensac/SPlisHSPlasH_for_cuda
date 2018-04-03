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

void PrecomputedCubicKernelPerso::setRadius(Real val)
{
	m_resolution = 10000;
	m_radius = val;
	m_radius2 = m_radius*m_radius;
	const Real stepSize = m_radius / (Real)m_resolution;
	m_invStepSize = 1.0 / stepSize;

	if (true) {
		Real* W_temp = new Real[m_resolution];
		Real* gradW_temp = new Real[m_resolution + 1];

		//init values
		CubicKernelPerso kernel;
		kernel.setRadius(val);
		
		for (unsigned int i = 0; i < m_resolution; ++i) {
			W_temp[i] = FluidModel::PrecomputedCubicKernel::m_W[i];
			gradW_temp[i] = FluidModel::PrecomputedCubicKernel::m_gradW[i];
		}

		gradW_temp[m_resolution] = 0.0;
		m_W_zero = kernel.W(0.0);

		
		allocate_precomputed_kernel_managed(*this, true);

		init_precomputed_kernel_from_values(*this, W_temp, gradW_temp);

		//clean
		delete[] W_temp;
		delete[] gradW_temp;
	}
	else {
		m_W = new Real[m_resolution];
		m_gradW = new Real[m_resolution + 1];
		
		//init values
		CubicKernelPerso kernel;
		kernel.setRadius(val);
		for (unsigned int i = 0; i < m_resolution; i++)
		{
			const Real posX = stepSize * (Real)i;		// Store kernel values in the middle of an interval
			m_W[i] = kernel.W(posX);
			kernel.setRadius(val);
			if (posX > 1.0e-9)
				m_gradW[i] = kernel.gradW(Vector3d(posX, 0.0, 0.0)).x / posX;
			else
				m_gradW[i] = 0.0;
		}
		m_gradW[m_resolution] = 0.0;
		m_W_zero = W(0.0);
	}
	
}


DFSPHCData::DFSPHCData(FluidModel *model) {
	//just I dont hndle external objects for now
	if (model->m_particleObjects.size() > 2) {
		exit(2568);
	}

	//m_kernel.setRadius(model->m_supportRadius);
	//W_zero = m_kernel.W_zero();
	m_kernel_precomp.setRadius(model->m_supportRadius);
	W_zero = m_kernel_precomp.W_zero();

	particleRadius = model->getParticleRadius();

	numBoundaryParticles = model->m_particleObjects[1]->numberOfParticles();
	numFluidParticles = model->numActiveParticles();

	if (true) {
		//initialisation on the GPU
		allocate_c_array_struct_cuda_managed((*this));

		//init the values
		reset(model);

		//init the rendering
		cuda_opengl_initFluidRendering(*this);
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


	//start of the c arrays
	//loadDynamicData(model, simulationData);

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

	h = TimeManager::getCurrent()->getTimeStepSize();
	h_future = h;
	h_past = h;
	h_ratio_to_past = 1.0;

	density0 = model->getDensity0();

	if (true) {
		// I need to transfer the data to c_typed buffers to use in .cu file
		Vector3d* posBoundary_temp= new Vector3d[numBoundaryParticles];
		Vector3d* velBoundary_temp = new Vector3d[numBoundaryParticles];
		Real* boundaryPsi_temp = new Real[numBoundaryParticles];

		for (int i = 0; i < numBoundaryParticles; ++i) {
			FluidModel::RigidBodyParticleObject* particleObj = static_cast<FluidModel::RigidBodyParticleObject*>(model->m_particleObjects[1]);
			posBoundary_temp[i] = vector3rTo3d(particleObj->m_x[i]);
			velBoundary_temp[i] = vector3rTo3d(particleObj->m_v[i]);
			boundaryPsi_temp[i] = particleObj->m_boundaryPsi[i];
		}

		Vector3d* posFluid_temp = new Vector3d[numFluidParticles];
		Vector3d* velFluid_temp = new Vector3d[numFluidParticles];
		Real* mass_temp = new Real[numFluidParticles];

		for (int i = 0; i < numFluidParticles; ++i) {
			posFluid_temp[i] = vector3rTo3d(model->getPosition(0, i));
			velFluid_temp[i] = vector3rTo3d(model->getVelocity(0, i));
			mass_temp[i] = model->getMass(i);
		}

		reset_c_array_struct_cuda_from_values(*this, posBoundary_temp, velBoundary_temp, boundaryPsi_temp,
			posFluid_temp, velFluid_temp, mass_temp);

		delete[] posBoundary_temp;
		delete[] velBoundary_temp;
		delete[] boundaryPsi_temp;
		delete[] posFluid_temp;
		delete[] velFluid_temp;
		delete[] mass_temp;

	}
	else {
		

		for (int i = 0; i < numBoundaryParticles; ++i) {
			FluidModel::RigidBodyParticleObject* particleObj = static_cast<FluidModel::RigidBodyParticleObject*>(model->m_particleObjects[1]);
			posBoundary[i] = vector3rTo3d(particleObj->m_x[i]);
			velBoundary[i] = vector3rTo3d(particleObj->m_v[i]);
			boundaryPsi[i] = particleObj->m_boundaryPsi[i];
		}

		for (int i = 0; i < numFluidParticles; ++i) {
			posFluid[i] = vector3rTo3d(model->getPosition(0, i));
			velFluid[i] = vector3rTo3d(model->getVelocity(0, i));
			mass[i] = model->getMass(i);

			//clear the internal kappa and  kappa v
			kappa[i] = 0.0;
			kappaV[i] = 0.0;
		}

	}



}


void DFSPHCData::loadDynamicData(FluidModel *model, const SimulationDataDFSPH& data) {

	if (model->numActiveParticles() != numFluidParticles ||
		model->getParticleRadius() != particleRadius) {
		exit(1569);
	}

	/*
	//normaly it's only needed on the first step
	for (int i = 0; i < numFluidParticles; ++i) {
		posFluid[i] = vector3rTo3d(model->getPosition(0, i));
		velFluid[i] = vector3rTo3d(model->getVelocity(0, i));
	}
	//*/
	//*
	//copy the data on the inital step
	static bool first_time = true;
	if (first_time) {
		for (int i = 0; i < numFluidParticles; ++i) {
			posFluid[i] = vector3rTo3d(model->getPosition(0, i));
			velFluid[i] = vector3rTo3d(model->getVelocity(0, i));
		}

		first_time = false;
	}

	//*/


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
		model->getPosition(0, i) = vector3dTo3r(posFluid[i]);
		model->getVelocity(0, i) = vector3dTo3r(velFluid[i]);
		
	}
}

void DFSPHCData::sortDynamicData(FluidModel *model) {

	if (model->numActiveParticles() != numFluidParticles) {
		exit(1569);
	}

	//*
	const unsigned int numPart = model->numActiveParticles();
	if (numPart == 0)
		return;

	auto const& d = model->getNeighborhoodSearch()->point_set(0);
	d.sort_field(&posFluid[0]);
	d.sort_field(&velFluid[0]);
	d.sort_field(&kappa[0]);
	d.sort_field(&kappaV[0]);


}
