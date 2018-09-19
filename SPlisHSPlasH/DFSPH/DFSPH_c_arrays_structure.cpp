#include "DFSPH_cpu_c_arrays.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SimulationDataDFSPH.h"
#include <iostream>
#include "SPlisHSPlasH/Utilities/Timing.h"
#include "DFSPH_cuda_basic.h"
#include <fstream>

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
	releaseDataOnDestruction = false;

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
	pos0 = NULL;
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


UnifiedParticleSet::~UnifiedParticleSet() {
	//*
	if (releaseDataOnDestruction) {
		std::cout << "destroying the an unifiedDataSet with numParticles: " << numParticles<< std::endl;

		//firsst free the neighbors
		delete neighborsDataSet;

		//release the cuda buffers
		release_UnifiedParticleSet_cuda((*this));

		//delete the rendering data and the buffer using the cuda interop
		cuda_opengl_releaseParticleRendering(*renderingData);
		delete renderingData;
	}
	//*/
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

template<class T>
void UnifiedParticleSet::updateDynamicBodiesParticles(T* particleObj) {
	if (is_dynamic_object) {
		if (dynamic_cast<FluidModel::RigidBodyParticleObject*>(particleObj) != NULL) {
			FluidModel::RigidBodyParticleObject* obj = reinterpret_cast<FluidModel::RigidBodyParticleObject*>(particleObj);
			Vector3d position;
			Vector3d velocity;
			Quaternion q;
			Vector3d angular_vel;
		

			position = vector3rTo3d(obj->m_rigidBody->getPosition());
			velocity = vector3rTo3d(obj->m_rigidBody->getVelocity());;
			angular_vel = vector3rTo3d(obj->m_rigidBody->getAngularVelocity());;

			RealCuda rotation_matrix[9];
			Matrix3rToArray(obj->m_rigidBody->getRotation(), rotation_matrix);
			q = Quaternion(rotation_matrix);
		
			updateDynamicBodiesParticles(position, velocity, q, angular_vel);
		}
	}

}

void UnifiedParticleSet::updateDynamicBodiesParticles(Vector3d position, Vector3d velocity, Quaternion q, Vector3d angular_vel) {
	if (is_dynamic_object) {
		update_dynamicObject_UnifiedParticleSet_cuda(*this,position, velocity, q, angular_vel);
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
			if (is_dynamic_object) {
				pos_temp[i] = vector3rTo3d(obj->m_x0[i]);
			}
			else {
				pos_temp[i] = vector3rTo3d(obj->m_x[i]);
			}
			vel_temp[i] = vector3rTo3d(obj->m_v[i]);
			mass_temp[i] = obj->m_boundaryPsi[i];
		}
	
		load_UnifiedParticleSet_cuda(*this, pos_temp, vel_temp, mass_temp);

		updateDynamicBodiesParticles<FluidModel::RigidBodyParticleObject>(obj);
	}
	else {
		FluidModel *model = reinterpret_cast<FluidModel*>(particleObj);
		for (int i = 0; i < numParticles; ++i) {
			pos_temp[i] = vector3rTo3d(model->getPosition(0, i));
			vel_temp[i] = vector3rTo3d(model->getVelocity(0, i));
			mass_temp[i] = model->getMass(i);
		}
	
		load_UnifiedParticleSet_cuda(*this, pos_temp, vel_temp, mass_temp);

	}

	
	delete[] pos_temp;
	delete[] vel_temp;
	delete[] mass_temp;
	
	
}


void UnifiedParticleSet::reset(std::ifstream& file_data, bool velocities_in_file, bool load_velocities) {
	Vector3d* pos_temp = new Vector3d[numParticles];
	Vector3d* vel_temp = new Vector3d[numParticles];
	RealCuda* mass_temp = new RealCuda[numParticles];


	for (int i = 0; i < numParticles; ++i) {
		RealCuda mass;
		Vector3d pos;
		Vector3d vel = Vector3d(0, 0, 0);

		file_data >> mass;
		file_data >> pos.x;
		file_data >> pos.y;
		file_data >> pos.z;
		if (velocities_in_file) {
			file_data >> vel.x;
			file_data >> vel.y;
			file_data >> vel.z;
			if (!load_velocities) {
				vel = Vector3d(0, 0, 0);
			}
		}

		mass_temp[i] = mass;
		pos_temp[i] = pos;

		vel_temp[i] = vel;
	}

	load_UnifiedParticleSet_cuda(*this, pos_temp, vel_temp, mass_temp);

	delete[] pos_temp;
	delete[] vel_temp;
	delete[] mass_temp;

}

void UnifiedParticleSet::load_from_file(std::string file_path) {
	//the first thing I need to do is to clear everything
}

DFSPHCData::DFSPHCData(FluidModel *model) {

	destructor_activated = true;

	m_kernel.setRadius(model->m_supportRadius);
	//W_zero = m_kernel.W_zero();
	m_kernel_precomp.setRadius(model->m_supportRadius);
	W_zero = m_kernel_precomp.W_zero();

	particleRadius = model->getParticleRadius();

	numFluidParticles = model->numActiveParticles();

	if (true) {
		//unified particles for the boundaries
		boundaries_data = new UnifiedParticleSet[1];
		boundaries_data[0] = UnifiedParticleSet(model->m_particleObjects[1]->numberOfParticles(), false, false, false);
		boundaries_data[0].releaseDataOnDestruction = true;
		allocate_and_copy_UnifiedParticleSet_vector_cuda(&boundaries_data_cuda, boundaries_data, 1);


		//unified particles for the fluid
		fluid_data = new UnifiedParticleSet[1];
		fluid_data[0] = UnifiedParticleSet(model->numActiveParticles(), true, true, false);
		fluid_data[0].releaseDataOnDestruction = true;
		allocate_and_copy_UnifiedParticleSet_vector_cuda(&fluid_data_cuda, fluid_data, 1);

		//allocate the data for the dynamic bodies
		numDynamicBodies = static_cast<int>(model->m_particleObjects.size() - 2);
		vector_dynamic_bodies_data = new UnifiedParticleSet[numDynamicBodies];
		for (int i = 2; i < model->m_particleObjects.size(); ++i) {
			vector_dynamic_bodies_data[i - 2] = UnifiedParticleSet(model->m_particleObjects[i]->numberOfParticles(), false, false, true);
			vector_dynamic_bodies_data[i - 2].releaseDataOnDestruction = true;
		}

		allocate_and_copy_UnifiedParticleSet_vector_cuda(&vector_dynamic_bodies_data_cuda, vector_dynamic_bodies_data, numDynamicBodies);


		//init the values from the model
		reset(model);

	}
}

DFSPHCData::~DFSPHCData() {
	if (destructor_activated) {
		std::cout << "destroying the data structure" << std::endl;

		//release the riigid bodies
		release_UnifiedParticleSet_vector_cuda(&vector_dynamic_bodies_data_cuda, numDynamicBodies);
		delete[] vector_dynamic_bodies_data;
	
		//release the fluid
		release_UnifiedParticleSet_vector_cuda(&fluid_data_cuda, 1);
		delete[] fluid_data;

		//release the boundaries
		release_UnifiedParticleSet_vector_cuda(&boundaries_data_cuda, 1);
		delete[] boundaries_data;

		//the last thing the needs to be done is to clear the kernel (in the case of the precomputed one
		///TODO clear the kernel
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
		//I need to update the ptrs on the cuda version because for the boudaries I clear the intermediary buffer to fre some memory
		update_neighborsSearchBuffers_UnifiedParticleSet_vector_cuda(&boundaries_data_cuda, boundaries_data, 1);
		
		//now initiate the data for the dynamic bodies the same way we did it for the boundaries
		// I need to transfer the data to c_typed buffers to use in .cu file
		for (int id = 2; id < model->m_particleObjects.size(); ++id) {
			FluidModel::RigidBodyParticleObject* particleObjDynamic = static_cast<FluidModel::RigidBodyParticleObject*>(model->m_particleObjects[id]);
			UnifiedParticleSet& body = vector_dynamic_bodies_data[id - 2];

			body.reset<FluidModel::RigidBodyParticleObject>(particleObjDynamic);

			///the reason I don't do the neighbor search here is that I would not be able to use it
			///to sort the data since I copy them at every time step (so it would be useless)
		}
	}

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



void DFSPHCData::loadDynamicObjectsData(FluidModel *model) {
	//now initiate the data for the dynamic bodies the same way we did it for the boundaries
	// I need to transfer the data to c_typed buffers to use in .cu file
	for (int id = 2; id < model->m_particleObjects.size(); ++id) {
		FluidModel::RigidBodyParticleObject* particleObj = static_cast<FluidModel::RigidBodyParticleObject*>(model->m_particleObjects[id]);
		UnifiedParticleSet& body = vector_dynamic_bodies_data[id - 2];

		//loadObjectData<FluidModel::RigidBodyParticleObject>(body, particleObj);
		body.updateDynamicBodiesParticles<FluidModel::RigidBodyParticleObject>(particleObj);

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

void DFSPHCData::write_fluid_to_file(bool save_velocities) {
	std::cout << "saving fluid start: " << save_velocities << std::endl;

	

	UnifiedParticleSet* set_cpu;
	set_cpu = fluid_data;

	
	Vector3d* pos_temp = new Vector3d[set_cpu->numParticles];
	Vector3d* vel_temp = new Vector3d[set_cpu->numParticles];
	RealCuda* mass_temp = new RealCuda[set_cpu->numParticles];


	read_UnifiedParticleSet_cuda(*set_cpu, pos_temp, vel_temp, mass_temp);
	


	std::ostringstream oss;
	oss << set_cpu->numParticles <<" "<<save_velocities<<std::endl;

	for (int i = 0; i < set_cpu->numParticles; ++i) {
		oss << mass_temp[i] << " ";
		oss << pos_temp[i].x << " " << pos_temp[i].y << " " << pos_temp[i].z << " ";
		if (save_velocities) {
			oss<< vel_temp[i].x << " " << vel_temp[i].y << " " << vel_temp[i].z << " ";
		}
		oss << std::endl;
	}


	delete[] pos_temp;
	delete[] vel_temp;
	delete[] mass_temp;
	
	ofstream myfile;
	myfile.open("fluid_file.txt");
	myfile << oss.str();
	myfile.close();


	std::cout << "saving fluid end: " << save_velocities << std::endl;
}

void DFSPHCData::read_fluid_from_file(bool load_velocities) {
	std::cout << "loading fluid start: " << load_velocities << std::endl;

	//first we clear the fluid data
	clear_fluid_data();

	ifstream myfile;
	myfile.open("fluid_file.txt");
	//read the first line
	int numParticles;
	bool velocities_in_file;
	myfile >> numParticles;
	myfile >> velocities_in_file;

	//now we recreate the fluid structure witht he right number of particles
	fluid_data = new UnifiedParticleSet[1];
	fluid_data[0] = UnifiedParticleSet(numParticles, true, true, false);
	fluid_data[0].releaseDataOnDestruction = true;
	allocate_and_copy_UnifiedParticleSet_vector_cuda(&fluid_data_cuda, fluid_data, 1);



	UnifiedParticleSet* set_cpu;
	set_cpu = fluid_data;



	if (load_velocities && !velocities_in_file) {
		std::cout << "loading fluid: velocities data not present in the file, setting the velociities to 0" << std::endl;
	}

	//reset position, mass and potentially velocities from file
	set_cpu->reset(myfile, velocities_in_file, load_velocities);
	

	//init the boundaries neighbor searchs
	set_cpu->initNeighborsSearchData(this->m_kernel_precomp.getRadius(), true, false);



	std::cout << "loading fluid end" << std::endl;
}

void DFSPHCData::clear_fluid_data() {
	release_UnifiedParticleSet_vector_cuda(&fluid_data_cuda, 1);
	delete[] fluid_data;
}