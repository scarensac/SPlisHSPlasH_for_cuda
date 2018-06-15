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



void NeighborsSearchDataSet::initData(UnifiedParticleSet* particleSet, RealCuda kernel_radius, bool sort_data) {
	//if the computation memory space was released to free memory space
	//we need to realocate it
	if (!internal_buffers_allocated) {
		//first clear everything even the result buffers
		release_neighbors_search_data_set(*this, false);

		//and realocate it
		allocate_neighbors_search_data_set(*this);
	}

	//do the actual init
	cuda_initNeighborsSearchDataSet(*particleSet,*this, kernel_radius, sort_data);
}

void NeighborsSearchDataSet::deleteComputationBuffer() {
	release_neighbors_search_data_set(*this, true);
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
	renderingData=NULL;
}

UnifiedParticleSet::UnifiedParticleSet(int nbParticles, bool has_factor_computation_i, bool velocity_impacted_by_fluid_solver_i,
	bool is_dynamic_object_i)
	: UnifiedParticleSet()
{
	numParticles = nbParticles;
	has_factor_computation = has_factor_computation_i;
	velocity_impacted_by_fluid_solver = velocity_impacted_by_fluid_solver_i;
	is_dynamic_object = is_dynamic_object_i;


	//init the rendering data
	renderingData = new ParticleSetRenderingData();
	cuda_opengl_initParticleRendering(*renderingData, numParticles, &pos, &vel);

	//initialisation on the GPU
	allocate_UnifiedParticleSet_cuda((*this));


	neighborsDataSet = new NeighborsSearchDataSet(numParticles);
}

void UnifiedParticleSet::transferForcesToCPU() {
	if (is_dynamic_object) {
		//init the buffer if it's NULL
		if (F_cpu == NULL) {
			F_cpu = new Vector3d[numParticles];
		}

		read_rigid_body_force_cuda(*this);
	}
}

void UnifiedParticleSet::initNeighborsSearchData(RealCuda kernel_radius, bool sort_data, bool delete_computation_data) {
	neighborsDataSet->initData(this, kernel_radius, sort_data);

	if (delete_computation_data) {
		neighborsDataSet->deleteComputationBuffer();
	}
}

template<class T>
void UnifiedParticleSet::reset(T* particleObj) {
	Vector3d* pos_temp = new Vector3d[numParticles];
	Vector3d* vel_temp = new Vector3d[numParticles];
	RealCuda* mass_temp = new RealCuda[numParticles];

	if (dynamic_cast<FluidModel::RigidBodyParticleObject*>(particleObj) != NULL) {
		FluidModel::RigidBodyParticleObject* obj = reinterpret_cast<FluidModel::RigidBodyParticleObject*>(particleObj);
		for (int i = 0; i < numParticles; ++i) {
			pos_temp[i] = vector3rTo3d(obj->m_x[i]);
			vel_temp[i] = vector3rTo3d(obj->m_v[i]);
			mass_temp[i] = obj->m_boundaryPsi[i];
		}
	}
	else {
		FluidModel *model = reinterpret_cast<FluidModel*>(particleObj);
		for (int i = 0; i < numParticles; ++i) {
			pos_temp[i] = vector3rTo3d(model->getPosition(0, i));
			vel_temp[i] = vector3rTo3d(model->getVelocity(0, i));
			mass_temp[i] = model->getMass(i);
		}
	}

	load_UnifiedParticleSet_cuda(*this, pos_temp, vel_temp, mass_temp);

	delete[] pos_temp;
	delete[] vel_temp;
	delete[] mass_temp;
}

DFSPHCData::DFSPHCData(FluidModel *model) {

	m_kernel.setRadius(model->m_supportRadius);
	//W_zero = m_kernel.W_zero();
	m_kernel_precomp.setRadius(model->m_supportRadius);
	W_zero = m_kernel_precomp.W_zero();

	particleRadius = model->getParticleRadius();

	numFluidParticles = model->numActiveParticles();

	if (true) {
		//unified particles for the boundaries
		boundaries_data = new UnifiedParticleSet[1];
		boundaries_data[0]= UnifiedParticleSet(model->m_particleObjects[1]->numberOfParticles(), false, false, false);
		allocate_and_copy_UnifiedParticleSet_vector_cuda(&boundaries_data_cuda, boundaries_data, 1);

		//unified particles for the fluid
		fluid_data = new UnifiedParticleSet[1];
		fluid_data[0] = UnifiedParticleSet(model->numActiveParticles(), true, true, false);
		allocate_and_copy_UnifiedParticleSet_vector_cuda(&fluid_data_cuda, fluid_data, 1);

		//allocate the data for the dynamic bodies
		numDynamicBodies = static_cast<int>(model->m_particleObjects.size() - 2);
		vector_dynamic_bodies_data = new UnifiedParticleSet[numDynamicBodies];
		for (int i = 2; i < model->m_particleObjects.size(); ++i) {
			vector_dynamic_bodies_data[i-2]= UnifiedParticleSet(model->m_particleObjects[i]->numberOfParticles(),false,false,true);
		}

		allocate_and_copy_UnifiedParticleSet_vector_cuda(&vector_dynamic_bodies_data_cuda, vector_dynamic_bodies_data, numDynamicBodies);


		//init the values from the model
		reset(model);
		
		
	}



}

void DFSPHCData::reset(FluidModel *model) {
	if (numFluidParticles != model->numActiveParticles()) {
		std::cout << "DFSPHCData::reset: fml the nbr of fluid particles has been modified" << std::endl;
		exit(3469);
	}
	if (boundaries_data->numParticles != model->m_particleObjects[1]->numberOfParticles()) {
		std::cout << "DFSPHCData::reset: fml the nbr of boundaries particles has been modified" << std::endl;
		exit(9657);
	}

	h = TimeManager::getCurrent()->getTimeStepSize();
	h_future = h;
	h_past = h;
	h_ratio_to_past = 1.0;

	density0 = model->getDensity0();

	if (true) {
		//load the data for the fluid
		fluid_data->reset<FluidModel>(model);
		//init the boundaries neighbor searchs
		fluid_data->initNeighborsSearchData(this->m_kernel_precomp.getRadius(), true, false);
		
		//load the data for the boundaries
		FluidModel::RigidBodyParticleObject* particleObj = static_cast<FluidModel::RigidBodyParticleObject*>(model->m_particleObjects[1]);
		boundaries_data->reset<FluidModel::RigidBodyParticleObject>(particleObj);
		//init the boundaries neighbor searchs
		boundaries_data->initNeighborsSearchData(this->m_kernel_precomp.getRadius(), true, true);
		
		//now initiate the data for the dynamic bodies 
		loadDynamicObjectsData(model);
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

	/*
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
	//*/
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
		model->getPosition(0, i) = vector3dTo3r(fluid_data->pos[i]);
		model->getVelocity(0, i) = vector3dTo3r(fluid_data->vel[i]);
		
	}
}




template<typename T>
void DFSPHCData::loadObjectData(UnifiedParticleSet& body, T* particleObj) {

	Vector3d* pos_temp = new Vector3d[body.numParticles];
	Vector3d* vel_temp = new Vector3d[body.numParticles];
	RealCuda* psi_temp = new RealCuda[body.numParticles];

	for (int i = 0; i < body.numParticles; ++i) {
		pos_temp[i] = vector3rTo3d(particleObj->m_x[i]);
		vel_temp[i] = vector3rTo3d(particleObj->m_v[i]);
		psi_temp[i] = particleObj->m_boundaryPsi[i];
	}

	load_UnifiedParticleSet_cuda(body, pos_temp, vel_temp, psi_temp);

	delete[] pos_temp;
	delete[] vel_temp;
	delete[] psi_temp;
}

void DFSPHCData::loadDynamicObjectsData(FluidModel *model) {
	//now initiate the data for the dynamic bodies the same way we did it for the boundaries
	// I need to transfer the data to c_typed buffers to use in .cu file
	for (int id = 2; id < model->m_particleObjects.size(); ++id) {
		FluidModel::RigidBodyParticleObject* particleObj = static_cast<FluidModel::RigidBodyParticleObject*>(model->m_particleObjects[id]);
		UnifiedParticleSet& body = vector_dynamic_bodies_data[id - 2];

		loadObjectData<FluidModel::RigidBodyParticleObject>(body, particleObj);

		///the reason I don't do the neighbor search here is that I would not be able to use it
		///to sort the data since I copy them at every time step (so it would be useless)
	}
}

void DFSPHCData::readDynamicObjectsData(FluidModel *model) {
	//now initiate the data for the dynamic bodies the same way we did it for the boundaries
	// I need to transfer the data to c_typed buffers to use in .cu file
	for (int id = 2; id < model->m_particleObjects.size(); ++id) {
		UnifiedParticleSet& body = vector_dynamic_bodies_data[id - 2];

		//convert gpu data to cpu
		body.transferForcesToCPU();

		for (int i = 0; i < body.numParticles; ++i) {
			model->getForce(id, i) = vector3dTo3r(body.F_cpu[i]);
		}
	}
}

