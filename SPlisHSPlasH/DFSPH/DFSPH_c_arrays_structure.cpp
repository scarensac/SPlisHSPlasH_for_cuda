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

void PrecomputedCubicKernelPerso::setRadius(RealCuda val)
{
	m_resolution = 10000;
	m_radius = val;
	m_radius2 = m_radius*m_radius;
	const RealCuda stepSize = m_radius / (RealCuda)m_resolution;
	m_invStepSize = 1.0 / stepSize;

	if (true) {
		RealCuda* W_temp = new RealCuda[m_resolution];
		RealCuda* gradW_temp = new RealCuda[m_resolution + 1];

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
		m_W = new RealCuda[m_resolution];
		m_gradW = new RealCuda[m_resolution + 1];
		
		//init values
		CubicKernelPerso kernel;
		kernel.setRadius(val);
		for (unsigned int i = 0; i < m_resolution; i++)
		{
			const RealCuda posX = stepSize * (RealCuda)i;		// Store kernel values in the middle of an interval
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


NeighborsSearchDataSet::NeighborsSearchDataSet(unsigned int numParticles_i) :
	numParticles(numParticles_i)
{
	allocate_neighbors_search_data_set(*this);

}

NeighborsSearchDataSet::~NeighborsSearchDataSet() {
	release_neighbors_search_data_set(*this, false);
}

void NeighborsSearchDataSet::initData(DFSPHCData* data, bool is_boundaries) {
	//if the computation memory space was released to free memory space
	//we need to realocate it
	if (!internal_buffers_allocated) {
		//first clear everything even the result buffers
		release_neighbors_search_data_set(*this, false);

		//and realocate it
		allocate_neighbors_search_data_set(*this);
	}

	//do the actual init
	cuda_initNeighborsSearchDataSet(*data, *this, is_boundaries);
}

void NeighborsSearchDataSet::deleteComputationBuffer() {
	release_neighbors_search_data_set(*this, true);
}

RigidBodyContainer::RigidBodyContainer(int nbParticles)
{
	numParticles = nbParticles;


	renderingData = new ParticleSetRenderingData();
	cuda_opengl_initParticleRendering(*renderingData, numParticles, &pos, &vel);

	//initialisation on the GPU
	allocate_rigid_body_container_cuda((*this));

	F_cpu = new Vector3d[numParticles];
	
	neighborsDataSet = new NeighborsSearchDataSet(numParticles);

}

UnifiedParticleSet::UnifiedParticleSet() {
	numParticles=0;
	has_factor_computation = false;
	is_dynamic_object = false;
	velocity_impacted_by_fluid_solver = false;
	
	mass = NULL;
	density = NULL;
	pos = NULL;
	vel = NULL;
	neighborsDataSet = NULL;
	factor = NULL;
	densityAdv = NULL;
	numberOfNeighbourgs = NULL;
	neighbourgs = NULL;
	acc = NULL;
	kappa = NULL;
	kappaV = NULL;
	F = NULL;
	F_cpu = NULL;
	renderingDataFluid=NULL;
}

UnifiedParticleSet::UnifiedParticleSet(int nbParticles, bool has_factor_computation_i, bool is_dynamic_object_i,
	bool velocity_impacted_by_fluid_solver_i) 
	: UnifiedParticleSet()
{
	numParticles = nbParticles;
	has_factor_computation = has_factor_computation_i;
	is_dynamic_object = is_dynamic_object_i;
	velocity_impacted_by_fluid_solver = velocity_impacted_by_fluid_solver_i;

	//initialisation on the GPU
	allocate_UnifiedParticleSet_cuda((*this));


	neighborsDataSet = new NeighborsSearchDataSet(numParticles);
}

DFSPHCData::DFSPHCData(FluidModel *model) {

	m_kernel.setRadius(model->m_supportRadius);
	//W_zero = m_kernel.W_zero();
	m_kernel_precomp.setRadius(model->m_supportRadius);
	W_zero = m_kernel_precomp.W_zero();

	particleRadius = model->getParticleRadius();

	numBoundaryParticles = model->m_particleObjects[1]->numberOfParticles();
	numFluidParticles = model->numActiveParticles();

	if (true) {
		//initialisation on the GPU
		allocate_c_array_struct_cuda_managed((*this));

		//init the rendering
		renderingDataFluid = new ParticleSetRenderingData();
		cuda_opengl_initParticleRendering(*renderingDataFluid, numFluidParticles, &posFluid, &velFluid);

		renderingDataBoundaries = new ParticleSetRenderingData();
		cuda_opengl_initParticleRendering(*renderingDataBoundaries, numBoundaryParticles, &posBoundary, &velBoundary);
		
		//allocate the data set that are gonna be used for the neighbors search
		neighborsdataSetBoundaries = new NeighborsSearchDataSet(numBoundaryParticles);
		neighborsdataSetFluid = new NeighborsSearchDataSet(numFluidParticles);
		
		//allocate the data for the dynamic bodies
		numDynamicBodies = static_cast<int>(model->m_particleObjects.size() - 2);
		vector_dynamic_bodies_data = new RigidBodyContainer[numDynamicBodies];
		for (int i = 2; i < model->m_particleObjects.size(); ++i) {
			vector_dynamic_bodies_data[i-2]= RigidBodyContainer(model->m_particleObjects[i]->numberOfParticles());
		}

		allocate_dynamic_bodies_vector_cuda(*this);


		//init the values from the model
		reset(model);
		
		
	}
	else {
		//initialisation on the CPU
		posBoundary = new Vector3d[numBoundaryParticles];
		velBoundary = new Vector3d[numBoundaryParticles];
		boundaryPsi = new RealCuda[numBoundaryParticles];


		//handle the fluid
		mass = new RealCuda[numFluidParticles];
		posFluid = new Vector3d[numFluidParticles];
		velFluid = new Vector3d[numFluidParticles];
		accFluid = new Vector3d[numFluidParticles];
		numberOfNeighbourgs = new int[numFluidParticles * 2];
		neighbourgs = new int[numFluidParticles * 2 * MAX_NEIGHBOURS];

		density = new RealCuda[numFluidParticles];
		factor = new RealCuda[numFluidParticles];
		kappa = new RealCuda[numFluidParticles];
		kappaV = new RealCuda[numFluidParticles];
		densityAdv = new RealCuda[numFluidParticles];
		
		reset(model);
	}



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
		RealCuda* boundaryPsi_temp = new RealCuda[numBoundaryParticles];

		for (int i = 0; i < numBoundaryParticles; ++i) {
			FluidModel::RigidBodyParticleObject* particleObj = static_cast<FluidModel::RigidBodyParticleObject*>(model->m_particleObjects[1]);
			posBoundary_temp[i] = vector3rTo3d(particleObj->m_x[i]);
			velBoundary_temp[i] = vector3rTo3d(particleObj->m_v[i]);
			boundaryPsi_temp[i] = particleObj->m_boundaryPsi[i];
		}

		Vector3d* posFluid_temp = new Vector3d[numFluidParticles];
		Vector3d* velFluid_temp = new Vector3d[numFluidParticles];
		RealCuda* mass_temp = new RealCuda[numFluidParticles];

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

		//init the data set that are gonna be used for the neighbors search
		neighborsdataSetBoundaries->initData(this, true);
		neighborsdataSetBoundaries->deleteComputationBuffer();
		neighborsdataSetFluid->initData(this, false);


		//now initiate the data for the dynamic bodies the same way we did it for the boundaries
		readDynamicObjectsData(model);

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
	/*
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


void DFSPHCData::loadDynamicObjectsData(FluidModel *model) {
	//now initiate the data for the dynamic bodies the same way we did it for the boundaries
	// I need to transfer the data to c_typed buffers to use in .cu file
	for (int id = 2; id < model->m_particleObjects.size(); ++id) {
		FluidModel::RigidBodyParticleObject* particleObj = static_cast<FluidModel::RigidBodyParticleObject*>(model->m_particleObjects[id]);
		RigidBodyContainer& body = vector_dynamic_bodies_data[id - 2];

		Vector3d* pos_temp = new Vector3d[body.numParticles];
		Vector3d* vel_temp = new Vector3d[body.numParticles];
		RealCuda* psi_temp = new RealCuda[body.numParticles];

		for (int i = 0; i < body.numParticles; ++i) {
			pos_temp[i] = vector3rTo3d(particleObj->m_x[i]);
			vel_temp[i] = vector3rTo3d(particleObj->m_v[i]);
			psi_temp[i] = particleObj->m_boundaryPsi[i];
		}

		load_rigid_body_container_cuda(body,pos_temp, vel_temp, psi_temp);

		delete[] pos_temp;
		delete[] vel_temp;
		delete[] psi_temp;
	}
}

void DFSPHCData::readDynamicObjectsData(FluidModel *model) {
	//now initiate the data for the dynamic bodies the same way we did it for the boundaries
	// I need to transfer the data to c_typed buffers to use in .cu file
	for (int id = 2; id < model->m_particleObjects.size(); ++id) {
		RigidBodyContainer& body = vector_dynamic_bodies_data[id - 2];

		//convert gpu data to cpu
		read_rigid_body_force_cuda(body);

		for (int i = 0; i < body.numParticles; ++i) {
			model->getForce(id, i) = vector3dTo3r(body.F_cpu[i]);
		}
	}
}

