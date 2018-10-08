#include "DFSPH_c_arrays_structure.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SimulationDataDFSPH.h"
#include <iostream>
#include "SPlisHSPlasH/Utilities/Timing.h"
#include "DFSPH_cuda_basic.h"
#include <fstream>
#include <stdio.h>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <thread>

using namespace SPH;
using namespace std;

const std::string fluid_files_folder = "./";

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


NeighborsSearchDataSet::NeighborsSearchDataSet(unsigned int numParticles_i, unsigned int numParticlesMax_i) :
	numParticles(numParticles_i),numParticlesMax(numParticlesMax_i)
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
	init();
}

void UnifiedParticleSet::init() {
	releaseDataOnDestruction = false;

	numParticles = 0;
	numParticlesMax = 0;
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
	rigidBody_cpu = NULL;
	renderingData = NULL;
	gpu_ptr = NULL;
	d_temp_storage = NULL;
	temp_storage_bytes = 0;
}

UnifiedParticleSet::UnifiedParticleSet(int nbParticles, bool has_factor_computation_i, bool velocity_impacted_by_fluid_solver_i,
	bool is_dynamic_object_i)
	: UnifiedParticleSet()
{
	init( nbParticles,  has_factor_computation_i,  velocity_impacted_by_fluid_solver_i, is_dynamic_object_i);
}

void UnifiedParticleSet::init(int nbParticles, bool has_factor_computation_i, bool velocity_impacted_by_fluid_solver_i,
	bool is_dynamic_object_i)
{
	//set default values
	init();

	//now use the actual parameters
	numParticles = nbParticles;
	numParticlesMax = numParticles;
	has_factor_computation = has_factor_computation_i;
	velocity_impacted_by_fluid_solver = velocity_impacted_by_fluid_solver_i;
	is_dynamic_object = is_dynamic_object_i;
	
	//if I have a fluid them I allocate more than the actual nbtr of particles so I can use more in the future
	if (!is_dynamic_object) {
		numParticlesMax *= 1.3;
	}


	//init the rendering data
	renderingData = new ParticleSetRenderingData();
	cuda_opengl_initParticleRendering(*renderingData, numParticlesMax, &pos, &vel);


	//allocate the neighbor structure
	neighborsDataSet = new NeighborsSearchDataSet(numParticles, numParticlesMax);

	//allocate the structure containing the rigidBody information
	if (is_dynamic_object) {
		rigidBody_cpu = new DynamicBody();
	}

	//initialisation on the GPU
	allocate_UnifiedParticleSet_cuda((*this));
}


UnifiedParticleSet::~UnifiedParticleSet() {
	//*
	if (releaseDataOnDestruction) {
		std::cout << "destroying the an unifiedDataSet with numParticles: " << numParticles<< std::endl;

		clear();
		
	}
	//*/
}

void UnifiedParticleSet::clear() {
	//firsst free the neighbors
	delete neighborsDataSet;

	//release the cuda buffers
	release_UnifiedParticleSet_cuda((*this));

	//delete the cpu buffers if there are some
	if (is_dynamic_object) {
		if (F_cpu != NULL) {
			delete[] F_cpu; F_cpu = NULL;
		}
	}

	//delete the rendering data and the buffer using the cuda interop
	if (renderingData != NULL) {
		cuda_opengl_releaseParticleRendering(*renderingData);
		delete renderingData; renderingData = NULL;
	}

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
			q.fromEulerMatrix(rotation_matrix);
		
			updateDynamicBodiesParticles(position, velocity, q, angular_vel);
		}
	}

}

void UnifiedParticleSet::updateDynamicBodiesParticles(Vector3d position, Vector3d velocity, Quaternion q, Vector3d angular_vel) {
	if (is_dynamic_object) {
		rigidBody_cpu->position = position;
		rigidBody_cpu->velocity = velocity;
		rigidBody_cpu->q = q;
		rigidBody_cpu->angular_vel = angular_vel;

		updateDynamicBodiesParticles();
	}
}

void UnifiedParticleSet::updateDynamicBodiesParticles() {
	if (is_dynamic_object) {
		update_dynamicObject_UnifiedParticleSet_cuda(*this);
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
		
		density0= model->getDensity0();
		m_V= model->getMass(0) /density0;
		
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


void UnifiedParticleSet::reset(Vector3d* pos_temp ,Vector3d* vel_temp ,RealCuda* mass_temp) {

	load_UnifiedParticleSet_cuda(*this, pos_temp, vel_temp, mass_temp);

	if (is_dynamic_object) {
		updateDynamicBodiesParticles();
	}

}

void UnifiedParticleSet::write_to_file(std::string file_path) {
	std::cout << "saving UnifiedParticleSet start: " << numParticles<<"    to: "<< file_path<< std::endl;

	Vector3d* pos_temp = new Vector3d[numParticles];
	Vector3d* vel_temp = new Vector3d[numParticles];
	RealCuda* mass_temp = new RealCuda[numParticles];
	Vector3d* pos0_temp = pos_temp;
	DynamicBody rb_simulation_state;
	

	read_UnifiedParticleSet_cuda(*this, pos_temp, vel_temp, mass_temp, pos0_temp);
	
	
	//write the first line
	std::ostringstream oss;
	oss << numParticles << " " << has_factor_computation << " " << velocity_impacted_by_fluid_solver << " " << is_dynamic_object << std::endl;


	//need to store the dynamic body information
	//Vector3d position, Vector3d velocity, Quaternion q, Vector3d angular_vel
	if (is_dynamic_object) {
		oss << rigidBody_cpu->position.x << " " << rigidBody_cpu->position.y << " " << rigidBody_cpu->position.z << " ";
		oss << rigidBody_cpu->velocity.x << " " << rigidBody_cpu->velocity.y << " " << rigidBody_cpu->velocity.z << " ";
		oss << rigidBody_cpu->q.v.x << " " << rigidBody_cpu->q.v.y << " " << rigidBody_cpu->q.v.z << " " << rigidBody_cpu->q.s << " ";
		oss << rigidBody_cpu->angular_vel.x << " " << rigidBody_cpu->angular_vel.y << " " << rigidBody_cpu->angular_vel.z << " " << std::endl;
	}

	//write the particle information
	for (int i = 0; i <numParticles; ++i) {
		oss << mass_temp[i] << " ";
		oss << pos_temp[i].x << " " << pos_temp[i].y << " " << pos_temp[i].z << " ";
		oss << vel_temp[i].x << " " << vel_temp[i].y << " " << vel_temp[i].z << " ";
		oss << std::endl;
	}

	delete[] pos_temp;
	delete[] vel_temp;
	delete[] mass_temp;


	ofstream myfile;
	myfile.open(file_path, std::ios_base::app);
	if (myfile.is_open()) {
		myfile << oss.str();
		myfile.close();
	}
	else {
		std::cout << "failed to open file: " << file_path << "   reason: "<< std::strerror(errno)<<std::endl;
	}
}

void UnifiedParticleSet::load_from_file(std::string file_path, bool load_velocities) {
	std::cout << "Loading UnifiedParticleSet start: " << file_path << std::endl;
	
	//first we clear all the data structure
	clear();

	ifstream myfile;
	myfile.open(file_path);
	//read the first line
	myfile >> numParticles;
	myfile >> has_factor_computation;
	myfile >> velocity_impacted_by_fluid_solver;
	myfile >> is_dynamic_object;

	//now we can initialise the structure
	//using the attribute to store the parameter before init shoudl not cause any probelm because
	//the call to the function will make a copy
	init(numParticles, has_factor_computation, velocity_impacted_by_fluid_solver, is_dynamic_object);
	releaseDataOnDestruction = true;
	

	//read the dynamic bodies information
	if (is_dynamic_object) {
		myfile >> rigidBody_cpu->position.x; myfile >> rigidBody_cpu->position.y; myfile >> rigidBody_cpu->position.z;
		myfile >> rigidBody_cpu->velocity.x; myfile >> rigidBody_cpu->velocity.y; myfile >> rigidBody_cpu->velocity.z;
		myfile >> rigidBody_cpu->q.v.x; myfile >> rigidBody_cpu->q.v.y; myfile >> rigidBody_cpu->q.v.z; myfile >> rigidBody_cpu->q.s;
		myfile >> rigidBody_cpu->angular_vel.x; myfile >> rigidBody_cpu->angular_vel.y; myfile >> rigidBody_cpu->angular_vel.z;

		if (!load_velocities) {
			rigidBody_cpu->velocity = Vector3d(0, 0, 0);
		}
	}


	//now we can read the particles informations
	Vector3d* pos_temp = new Vector3d[numParticles];
	Vector3d* vel_temp = new Vector3d[numParticles];
	RealCuda* mass_temp = new RealCuda[numParticles];


	for (int i = 0; i < numParticles; ++i) {
		RealCuda mass;
		Vector3d pos;
		Vector3d vel = Vector3d(0, 0, 0);

		myfile >> mass;
		myfile >> pos.x; myfile >> pos.y; myfile >> pos.z;
		myfile >> vel.x; myfile >> vel.y; myfile >> vel.z;

		if (!load_velocities) {
			vel = Vector3d(0, 0, 0);
		}

		mass_temp[i] = mass;
		pos_temp[i] = pos;
		vel_temp[i] = vel;
	}

	reset(pos_temp, vel_temp, mass_temp);

	delete[] pos_temp;
	delete[] vel_temp;
	delete[] mass_temp;

}

void UnifiedParticleSet::write_forces_to_file(std::string file_path) {
	if (!is_dynamic_object) {
		return;
	}

	//write the first line
	std::ostringstream oss;
	oss << numParticles  << std::endl;

	Vector3d* pos0_temp = new Vector3d[numParticles];
	read_UnifiedParticleSet_cuda(*this, NULL, NULL, NULL, pos0_temp);


	//write the particle information
	for (int i = 0; i <numParticles; ++i) {
		oss << pos0_temp[i].x << " " << pos0_temp[i].y << " " << pos0_temp[i].z << " ";
		oss << F_cpu[i].x << " " << F_cpu[i].y << " " << F_cpu[i].z << " ";
		oss << std::endl;
	}

	delete[] pos0_temp;
	

	ofstream myfile;
	myfile.open(file_path, std::ios_base::app);
	if (myfile.is_open()) {
		myfile << oss.str();
		myfile.close();
	}
	else {
		std::cout << "failed to open file: " << file_path << "   reason: " << std::strerror(errno) << std::endl;
	}
}


void UnifiedParticleSet::add_particles(std::vector<Vector3d> pos, std::vector<Vector3d> vel) {
	if (numParticles + pos.size() > numParticlesMax) {
		std::cout << "UnifiedParticleSet::add_particles  exceeded the allocated space." << std::endl;
		return;
	}

	add_particles_cuda(*this, (int)pos.size(), pos.data(), vel.data());

}

DFSPHCData::DFSPHCData() {
	destructor_activated = false;;
	W_zero=0;
	density0=0;
	particleRadius=0;
	viscosity=0;


	h = 0.001;
	invH = 1.0 / h;
	invH2 = 1.0 / (h*h);
	updateTimeStep(h);
	onSimulationStepEnd();

	fluid_data = NULL;
	fluid_data_cuda = NULL;

	boundaries_data = NULL;
	boundaries_data_cuda = NULL;


	vector_dynamic_bodies_data = NULL;
	vector_dynamic_bodies_data_cuda = NULL;
	numDynamicBodies=0;
}

DFSPHCData::DFSPHCData(FluidModel *model): DFSPHCData()
{

	destructor_activated = true;

	m_kernel.setRadius(model->m_supportRadius);
	//W_zero = m_kernel.W_zero();
	m_kernel_precomp.setRadius(model->m_supportRadius);
	W_zero = m_kernel_precomp.W_zero();

	std::cout << "support radius: " << model->m_supportRadius << std::endl;

	particleRadius = model->getParticleRadius();

	
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


	if (fluid_data->numParticles != model->numActiveParticles()) {
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

	if (model->numActiveParticles() != fluid_data->numParticles) {
		exit(1569);
	}

	//density and acc are not conserved between timesteps so no need to copy them
	for (int i = 0; i <  fluid_data->numParticles; ++i) {
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


void DFSPHCData::write_fluid_to_file() {
	std::cout << "saving fluid start: " << std::endl;

	std::string file_name = fluid_files_folder + "fluid_file.txt";
	std::remove(file_name.c_str());

	fluid_data[0].write_to_file(file_name);

	std::cout << "saving fluid end: " << std::endl;
}

void DFSPHCData::read_fluid_from_file(bool load_velocities) {
	std::cout << "loading fluid start: " << load_velocities << std::endl;
	//reset the data strucure
	clear_fluid_data();
	fluid_data = new UnifiedParticleSet[1];

	//read the data to cpu pointer
	std::string file_name = fluid_files_folder + "fluid_file.txt";
	fluid_data[0].load_from_file(file_name, load_velocities);

	//init gpu struct
	allocate_and_copy_UnifiedParticleSet_vector_cuda(&fluid_data_cuda, fluid_data, 1);

	//init the boundaries neighbor searchs
	//fluid_data[0].initNeighborsSearchData(this->m_kernel_precomp.getRadius(), true, false);

	std::cout << "loading fluid end" << std::endl;
}

void DFSPHCData::write_boundaries_to_file() {
	std::cout << "saving boundaries start: " << std::endl;


	std::string file_name = fluid_files_folder + "boundaries_file.txt";
	std::remove(file_name.c_str());

	boundaries_data[0].write_to_file(file_name);

	std::cout << "saving boundaries end: " << std::endl;
}

void DFSPHCData::read_boundaries_from_file(bool load_velocities) {
	std::cout << "loading boundaries start: " << load_velocities << std::endl;
	//reset the data strucure
	clear_boundaries_data();
	boundaries_data = new UnifiedParticleSet[1];

	//read the data to cpu pointer

	std::string file_name = fluid_files_folder + "boundaries_file.txt";
	boundaries_data[0].load_from_file(file_name, load_velocities);

	//init gpu struct
	allocate_and_copy_UnifiedParticleSet_vector_cuda(&boundaries_data_cuda, boundaries_data, 1);

	//init the boundaries neighbor searchs
	boundaries_data[0].initNeighborsSearchData(this->m_kernel_precomp.getRadius(), true, false);

	std::cout << "loading boundaries end" << std::endl;
}

void DFSPHCData::write_solids_to_file() {
	std::cout << "saving solids start: " << std::endl;


	std::string file_name = fluid_files_folder + "solids_file.txt";
	std::remove(file_name.c_str());

	//writte the number of solid to the main file
	ofstream myfile;
	myfile.open(file_name);
	myfile << numDynamicBodies;
	myfile.close();


	//writte each solid
	for (int i = 0; i < numDynamicBodies; ++i) {
		std::ostringstream oss;
		oss << "solid_file_" << i << ".txt";

		std::remove(oss.str().c_str());
		vector_dynamic_bodies_data[i].write_to_file(oss.str());
	}


	std::cout << "saving solids end: " << std::endl;
}

void DFSPHCData::read_solids_from_file(bool load_velocities) {
	std::cout << "loading solids start: " << load_velocities << std::endl;
	//clear the gpu strucure
	clear_solids_data();

	//read the number of solids and allocate the vector

	std::string file_name = fluid_files_folder + "solids_file.txt";
	ifstream myfile;
	myfile.open(file_name);
	myfile >> numDynamicBodies;
	myfile.close();
	vector_dynamic_bodies_data = new UnifiedParticleSet[numDynamicBodies];

	//read the data for each solid
	for (int i = 0; i < numDynamicBodies; ++i) {
		std::ostringstream oss;
		oss << "solid_file_" << i << ".txt";
		vector_dynamic_bodies_data[i].load_from_file(oss.str(), load_velocities);
	}

	//init gpu struct
	allocate_and_copy_UnifiedParticleSet_vector_cuda(&vector_dynamic_bodies_data_cuda, vector_dynamic_bodies_data, numDynamicBodies);


	std::cout << "loading solids end" << std::endl;
}

void DFSPHCData::clear_fluid_data() {
	if (fluid_data_cuda != NULL) {
		release_UnifiedParticleSet_vector_cuda(&fluid_data_cuda, 1);
	}

	if (fluid_data != NULL) {
		delete[] fluid_data; fluid_data = NULL;
	}
}

void DFSPHCData::clear_boundaries_data() {
	if (boundaries_data_cuda != NULL) {
		release_UnifiedParticleSet_vector_cuda(&boundaries_data_cuda, 1);
	}

	if (boundaries_data != NULL) {
		delete[] boundaries_data; boundaries_data = NULL;
	}
}

void DFSPHCData::clear_solids_data() {
	if (vector_dynamic_bodies_data_cuda != NULL) {
		release_UnifiedParticleSet_vector_cuda(&vector_dynamic_bodies_data_cuda, numDynamicBodies);
	}

	if (vector_dynamic_bodies_data != NULL) {
		delete[] vector_dynamic_bodies_data; vector_dynamic_bodies_data = NULL;
	}
}


void DFSPHCData::update_solids_to_file() {
	std::cout << "saving solids forces: " << std::endl;

	std::remove("solids_file.txt");


	//writte each solid
	for (int i = 0; i < numDynamicBodies; ++i) {
		std::ostringstream oss;
		oss << "solid_forces_file_" << i << ".txt";

		std::remove(oss.str().c_str());
		vector_dynamic_bodies_data[i].write_forces_to_file(oss.str());
	}

	//writte the number of solid to the main file
	ofstream myfile;
	myfile.open("solids_file.txt");
	myfile << numDynamicBodies;
	myfile.close();

}

void DFSPHCData::update_solids_from_file() {
	ifstream myfile;

	//so first I need to wait until the correct infomation is here
	for (;;) {
		myfile.open("update_solids.txt");
		
		//we can continue if the file has successfully been opened
		if (myfile.is_open()) {
			break;
		}

		//we need to wait if the file opening failed (it means the file is not yet on the disk
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	std::cout << "file_opened !!!" << std::endl;

	//read the data for each solid
	for (int i = 0; i < numDynamicBodies; ++i) {
		Vector3d position;
		Vector3d velocity;
		Quaternion q;
		Vector3d angular_vel;

		myfile >> position.x; myfile >> position.y; myfile >> position.z;
		myfile >> velocity.x; myfile >> velocity.y; myfile >> velocity.z;
		myfile >> q.v.x; myfile >> q.v.y; myfile >> q.v.z; myfile >> q.s;
		myfile >> angular_vel.x; myfile >> angular_vel.y; myfile >> angular_vel.z;

		vector_dynamic_bodies_data[i].updateDynamicBodiesParticles(position,velocity,q,angular_vel);
	}

	myfile.close();

	//finaly we remove the file from the disk now that we have read it
	std::remove("update_solids.txt");
}

void DFSPHCData::update_solids(std::vector<DynamicBody> vect_new_info) {
	if (vect_new_info.size() != numDynamicBodies) {
		std::cout << "number of existing rigid bodies do not fit the input data" << std::endl;
		exit(1256);
	}

	for (int i = 0; i < numDynamicBodies; ++i) {
		DynamicBody& body = vect_new_info[i];
		vector_dynamic_bodies_data[i].updateDynamicBodiesParticles(body.position, body.velocity, body.q, body.angular_vel);
	}

}
