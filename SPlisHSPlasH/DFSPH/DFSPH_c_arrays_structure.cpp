#include "DFSPH_c_arrays_structure.h"

#ifdef SPLISHSPLASH_FRAMEWORK
#include "SPlisHSPlasH/TimeManager.h"
#include "SimulationDataDFSPH.h"
#endif //SPLISHSPLASH_FRAMEWORK
#include "SPlisHSPlasH/SPHKernels.h"
#include "DFSPH_cuda_basic.h"
#include "DFSPH_define_cuda.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <thread>
#include <sstream>
#include "Utilities/FileSystem.h"


using namespace SPH;
using namespace std;





NeighborsSearchDataSet::NeighborsSearchDataSet() 
{
	numParticles=0;
	numParticlesMax = 0;
	cell_id=NULL;
	cell_id_sorted = NULL;
	local_id = NULL;
	p_id = NULL;
	p_id_sorted = NULL;
	cell_start_end = NULL;
	hist = NULL;

	d_temp_storage_pair_sort = NULL;
	temp_storage_bytes_pair_sort = 0;
	d_temp_storage_cumul_hist = NULL;
	temp_storage_bytes_cumul_hist = 0;

	internal_buffers_allocated=false;

	intermediate_buffer_v3d = NULL;
	intermediate_buffer_real = NULL;

    gpu_ptr=NULL;

}

NeighborsSearchDataSet::NeighborsSearchDataSet(unsigned int numParticles_i, unsigned int numParticlesMax_i) : NeighborsSearchDataSet()	
{
	static int count_sets = 0;
	set_id=count_sets++;

	numParticles = numParticles_i;
	numParticlesMax = numParticlesMax_i;

	allocate_neighbors_search_data_set(*this);

}

NeighborsSearchDataSet::~NeighborsSearchDataSet() {
	release_neighbors_search_data_set(*this, false);
}



void NeighborsSearchDataSet::initData(UnifiedParticleSet* particleSet, SPH::DFSPHCData& data, bool sort_data) {
	//if the computation memory space was released to free memory space
	//we need to realocate it
	if (!internal_buffers_allocated) {
		//first clear everything even the result buffers
		release_neighbors_search_data_set(*this, false);

		//and realocate it
		allocate_neighbors_search_data_set(*this);

		std::cout << "allocating neighbors done" << std::endl;
	}

	//do the actual init
    cuda_initNeighborsSearchDataSet(*particleSet,*this, data, sort_data);
}

void NeighborsSearchDataSet::deleteComputationBuffer() {
    release_neighbors_search_data_set(*this, true);
}


void NeighborsSearchDataSet::updateActiveParticleNumber(unsigned int val) {

    numParticles = val;
    if (gpu_ptr!=NULL){
        update_active_particle_number_cuda<SPH::NeighborsSearchDataSet>(*this);
    }
}

void NeighborsSearchDataSet::changeMaxParticleNumber(int numParticlesMax_i){
    change_max_particle_number(*this,numParticlesMax_i);
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
	color = NULL;
	has_color_buffer = false;
	mass_flow = NULL;

    density0=1000;//basic value
}

UnifiedParticleSet::UnifiedParticleSet(int nbParticles, bool has_factor_computation_i, bool velocity_impacted_by_fluid_solver_i,
	bool is_dynamic_object_i)
	: UnifiedParticleSet()
{
	init( nbParticles,  has_factor_computation_i,  velocity_impacted_by_fluid_solver_i, is_dynamic_object_i, velocity_impacted_by_fluid_solver_i);
}

void UnifiedParticleSet::init(int nbParticles, bool has_factor_computation_i, bool velocity_impacted_by_fluid_solver_i,
	bool is_dynamic_object_i, bool need_color_buffer)
{
	//set default values
	init();

	//now use the actual parameters
	numParticles = nbParticles;
	numParticlesMax = numParticles;
	has_factor_computation = has_factor_computation_i;
	velocity_impacted_by_fluid_solver = velocity_impacted_by_fluid_solver_i;
	is_dynamic_object = is_dynamic_object_i;
	has_color_buffer=need_color_buffer;

	//if I have a fluid them I allocate more than the actual nbtr of particles so I can use more in the future
	if (velocity_impacted_by_fluid_solver) {
		numParticlesMax *= 1.3;
	}


	//init the rendering data
	renderingData = new ParticleSetRenderingData();
	cuda_opengl_initParticleRendering(*renderingData, numParticlesMax, &pos, &vel, has_color_buffer, &color);
	if (has_color_buffer) {
		resetColor();
	}

	//allocate the neighbor structure
	neighborsDataSet = new NeighborsSearchDataSet(numParticles, numParticlesMax);

	//allocate the structure containing the rigidBody information
	if (is_dynamic_object) {
		rigidBody_cpu = new DynamicBody();
	}

	//initialisation on the GPU
	allocate_UnifiedParticleSet_cuda((*this));
}


UnifiedParticleSet::UnifiedParticleSet(UnifiedParticleSet* other) {
	//allocate the memory space
	init(other->numParticles, other->has_factor_computation, other->velocity_impacted_by_fluid_solver, other->is_dynamic_object, other->has_color_buffer);

	//copy the data
	copy_UnifiedParticleSet_cuda(*this, *other);
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

	read_last_error_cuda("middle 1  ");

	//firsst free the neighbors
	delete neighborsDataSet;


	read_last_error_cuda("middle 2  ");

	//release the cuda buffers
	release_UnifiedParticleSet_cuda((*this));



	//delete the rendering data and the buffer using the cuda interop
    if (renderingData != NULL) {
        //the rendering data and the pos/vel buffer
        cuda_opengl_releaseParticleRendering(*renderingData);release_UnifiedParticleSet_cuda((*this));
		

		
		delete renderingData; renderingData = NULL;
	}



}

void UnifiedParticleSet::computeParticlesMass(DFSPHCData* data){
    if (!velocity_impacted_by_fluid_solver){
        compute_UnifiedParticleSet_particles_mass_cuda(*data,*this);
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

void UnifiedParticleSet::initNeighborsSearchData(SPH::DFSPHCData& data, bool sort_data, bool delete_computation_data) {
    neighborsDataSet->initData(this, data, sort_data);

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
		
		int NbrLoadedParticles = numParticles;
		for (int i = 0; i < numParticles; ++i) {

#ifdef OCEAN_BOUNDARIES_PROTOTYPE
			/*
			if (vector3rTo3d(model->getPosition(0, i)).x > (-2.0 + 4*model->getSupportRadius())) {
				NbrLoadedParticles--;
				continue;
			}
			//*/
#endif

			pos_temp[i] = vector3rTo3d(model->getPosition(0, i));
			vel_temp[i] = vector3rTo3d(model->getVelocity(0, i));
			mass_temp[i] = model->getMass(i);
		}
	
		std::cout << "nbr of particles loaded: "<< NbrLoadedParticles << "    from nbrModelParticles: "<<model->getNumActiveParticles0()<<std::endl;

		updateActiveParticleNumber(NbrLoadedParticles);

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

void UnifiedParticleSet::load_from_file(std::string file_path, bool load_velocities, Vector3d *min_o, Vector3d *max_o, bool positions_limitations) {
    std::cout << "UnifiedParticleSet::load_from_file start: " << file_path << std::endl;
	
	//first we clear all the data structure
	clear();

	ifstream myfile;
	myfile.open(file_path);
    if(!myfile.is_open()){
        std::cout<<"trying to read from unexisting file: "<<file_path<<std::endl;
        exit(256);
    }

	//read the first line
	myfile >> numParticles;
	myfile >> has_factor_computation;
	myfile >> velocity_impacted_by_fluid_solver;
	myfile >> is_dynamic_object;


	//now we can initialise the structure
	//using the attribute to store the parameter before init shoudl not cause any probelm because
	//the call to the function will make a copy
	init(numParticles, has_factor_computation, velocity_impacted_by_fluid_solver, is_dynamic_object, true);// velocity_impacted_by_fluid_solver);
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


    Vector3d min=Vector3d(1000);
    Vector3d max=Vector3d(-1000);
	int NbrLoadedParticles = numParticles;
    for (int i = 0; i < NbrLoadedParticles; ++i) {
		RealCuda mass;
		Vector3d pos;
		Vector3d vel = Vector3d(0, 0, 0);

		myfile >> mass;
		myfile >> pos.x; myfile >> pos.y; myfile >> pos.z;
		myfile >> vel.x; myfile >> vel.y; myfile >> vel.z;

		if (!load_velocities) {
			vel = Vector3d(0, 0, 0);
        }

        //this is just used to lowerthe floorrelative the the rb simulation
        if(!is_dynamic_object&&!velocity_impacted_by_fluid_solver){
            //pos.y-=0.2;
        }

#ifdef OCEAN_BOUNDARIES_PROTOTYPE
		/*
		float height = 1;
		if ((positions_limitations)&&
			(velocity_impacted_by_fluid_solver)&&
			//(((pos.x > (-2.0 + 8 * 0.1)) && (pos.x < (2.0 - 8 * 0.1))) || (pos.y > height))
			//(((pos.z > (-0.7 + 8 * 0.1)) && (pos.z < (0.7 - 8 * 0.1))) || (pos.y > height))
			((pos.y > height))
			){
			NbrLoadedParticles--;
			i--;
			continue;
		}
		//*/
#endif
		
        mass_temp[i] = mass;
        pos_temp[i] = pos;
        vel_temp[i] = vel;

        min.toMin(pos);
        max.toMax(pos);



	}

	reset(pos_temp, vel_temp, mass_temp);

	std::cout << "nbr of particles loaded: " << NbrLoadedParticles << "    from nbrModelParticles: " << numParticles << std::endl;

	updateActiveParticleNumber(NbrLoadedParticles);

	delete[] pos_temp;
	delete[] vel_temp;
	delete[] mass_temp;

    updateDynamicBodiesParticles();

    //the min and max do not work for dynamic bodies (well for the dynamic bodies they are the min and max of the local coordinates
    if (min_o!=NULL){
        *min_o=min;
    }
    if (max_o!=NULL){
        *max_o=max;
    }


    std::cout << "UnifiedParticleSet::load_from_file end: " << file_path << std::endl;

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



void UnifiedParticleSet::updateActiveParticleNumber(unsigned int val) {

    numParticles = val;
    if (gpu_ptr!=NULL){
        update_active_particle_number_cuda<SPH::UnifiedParticleSet>(*this);
    }

    neighborsDataSet->updateActiveParticleNumber(val);
}


void UnifiedParticleSet::changeMaxParticleNumber(int numParticlesMax_i){
    change_max_particle_number(*this,numParticlesMax_i);


    neighborsDataSet->changeMaxParticleNumber(numParticlesMax_i);
}


void UnifiedParticleSet::add_particles(std::vector<Vector3d> pos, std::vector<Vector3d> vel) {
	if (numParticles + pos.size() > numParticlesMax) {
		std::cout << "UnifiedParticleSet::add_particles  exceeded the allocated space." << std::endl;
		return;
	}

	add_particles_cuda(*this, (int)pos.size(), pos.data(), vel.data());

}

void UnifiedParticleSet::zeroVelocities() {
    set_buffer_to_value<Vector3d>(vel,Vector3d(0,0,0),numParticles);

    if(is_dynamic_object){
        //if it's a dynamic body we actualy need to set the rb vel and angular vel to zero
        //on top of setting the buffer to zero
        rigidBody_cpu->angular_vel=Vector3d(0,0,0);
        rigidBody_cpu->velocity=Vector3d(0,0,0);

        //don't need the heavy computation simply setting the vel buffer to zero is fine
        //updateDynamicBodiesParticles();
    }
}


void UnifiedParticleSet::getMinMaxNaive(Vector3d& min, Vector3d& max) {
	get_UnifiedParticleSet_min_max_naive_cuda(*this, min, max);
}

void UnifiedParticleSet::loadBender2019BoundariesFromCPU(RealCuda* V_rigids_i, Vector3d* X_rigids_i) {
	load_bender2019_boundaries_from_cpu(*this, V_rigids_i, X_rigids_i);

}

void UnifiedParticleSet::resetColor() {
	cuda_reset_color(this);
}

DFSPHCData::DFSPHCData() {



	destructor_activated = false;;
	W_zero=0;
	density0=0;
	particleRadius=0;
	viscosity=0;
    m_surfaceTension=0.05;
    gridOffset=Vector3i(50);
	dynamicWindowTotalDisplacement = Vector3d(0, 0, 0);


	h = 0.003;
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
    neighborsDataSetGroupedDynamicBodies=NULL;
    neighborsDataSetGroupedDynamicBodies_cuda=NULL;
    posBufferGroupedDynamicBodies=NULL;
    is_fluid_aggregated=false;


    damp_borders=false;
    damp_borders_steps_count = 0;
    cancel_wave=false;
    cancel_wave_steps_count = 0;

	bmin=NULL;
	bmax=NULL;
	damp_planes = NULL;
	damp_planes_count = 0;

	allocate_DFSPHCData_base_cuda(*this);

	std::ostringstream oss;
#ifdef SPLISHSPLASH_FRAMEWORK
	oss << FileSystem::get_folder_path("data", 5);
	oss<<  "save_folder/";
#else
	oss << FileSystem::get_folder_path("configuration_data", 5);
	oss << "fluid_data/cdp_char_reduced/";
	//oss << "fluid_data/cdp_char/";
	//oss << "fluid_data/test_ball_object/";
	//oss << "fluid_data/test_box_object/";
	//oss << "fluid_data/test_objects/";
#endif

	fluid_files_folder = oss.str();
	std::cout << "detected fluid file folder: " << fluid_files_folder << std::endl;
}

DFSPHCData::DFSPHCData(FluidModel *model): DFSPHCData()
{


#ifdef SPLISHSPLASH_FRAMEWORK
    particleRadius = model->getParticleRadius();
	RealCuda supportRadius = model->m_supportRadius;
#else
    particleRadius = 0.025;
	RealCuda supportRadius = particleRadius * 4;
#endif //SPLISHSPLASH_FRAMEWORK

#ifdef PRECOMPUTED_KERNELS 
	m_kernel_precomp.setRadius(supportRadius);
	W_zero = m_kernel_precomp.W_zero();
#else
	m_kernel.setRadius(supportRadius);
    W_zero = m_kernel.W_zero();
#endif


    m_kernel_adhesion.setRadius(supportRadius);
    m_kernel_cohesion.setRadius(supportRadius);




    std::cout << "particle radius and suport radius: " << particleRadius<<"   "<< supportRadius  << std::endl;

#ifdef SPLISHSPLASH_FRAMEWORK
	if (model != NULL) {
		//unified particles for the boundaries
		bool compute_boundaries_pressure = false;
#ifdef	COMPUTE_BOUNDARIES_DYNAMIC_PROPERTiES
		compute_boundaries_pressure = true;
		std::cout << "COMPUTE_BOUNDARIES_DYNAMIC_PROPERTiES" << std::endl;
#ifdef USE_BOUNDARIES_DYNAMIC_PROPERTiES
		std::cout << "USE_BOUNDARIES_DYNAMIC_PROPERTiES" << std::endl;

#endif
#endif
		boundaries_data = new UnifiedParticleSet[1];
		boundaries_data[0] = UnifiedParticleSet(model->m_particleObjects[1]->numberOfParticles(), compute_boundaries_pressure, false, false);
		boundaries_data[0].releaseDataOnDestruction = true;
		allocate_and_copy_UnifiedParticleSet_vector_cuda(&boundaries_data_cuda, boundaries_data, 1);

		//allocate the data for the dynamic bodies
		numDynamicBodies = static_cast<int>(model->m_particleObjects.size() - 2);
		vector_dynamic_bodies_data = new UnifiedParticleSet[numDynamicBodies];
		for (int i = 2; i < model->m_particleObjects.size(); ++i) {
			vector_dynamic_bodies_data[i - 2] = UnifiedParticleSet(model->m_particleObjects[i]->numberOfParticles(), false, false, true);
			vector_dynamic_bodies_data[i - 2].releaseDataOnDestruction = true;
		}

		allocate_and_copy_UnifiedParticleSet_vector_cuda(&vector_dynamic_bodies_data_cuda, vector_dynamic_bodies_data, numDynamicBodies);



		//unified particles for the fluid
		fluid_data = new UnifiedParticleSet[1];
		fluid_data[0] = UnifiedParticleSet(model->numActiveParticles(), true, true, false);
		fluid_data[0].releaseDataOnDestruction = true;
		allocate_and_copy_UnifiedParticleSet_vector_cuda(&fluid_data_cuda, fluid_data, 1);


#ifdef GROUP_DYNAMIC_BODIES_NEIGHBORS_SEARCH
		allocate_grouped_neighbors_struct_cuda(*this);
#endif
	}

#else
    //in that cas ewe need to do a load from file but it will be handled late anyway
#endif //SPLISHSPLASH_FRAMEWORK

    //init the values from the model
    //it is called in the DFSOH_CUDA constructor anyway
	//reset(model);

	read_last_error_cuda("check for errors end creation  ");

	std::cout << "here" << std::endl;
	destructor_activated = true;
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
        m_kernel_precomp.freeMemory();

        free_DFSPHCData_base_cuda(*this);

#ifdef GROUP_DYNAMIC_BODIES_NEIGHBORS_SEARCH
        free_grouped_neighbors_struct_cuda(*this);
#endif
    }
}



void DFSPHCData::reset(FluidModel *model) {
	bool old_destructor_status = destructor_activated;
	destructor_activated = false;

#ifdef SPLISHSPLASH_FRAMEWORK
	if (fluid_data->numParticles != model->numActiveParticles()) {
		std::cout << "DFSPHCData::reset: fml the nbr of fluid particles has been modified but I'll try" << std::endl;
		//exit(3469);
	}
	if (boundaries_data->numParticles != model->m_particleObjects[1]->numberOfParticles()) {
		std::cout << "DFSPHCData::reset: fml the nbr of boundaries particles has been modified" << std::endl;
		exit(9657);
	}
	
	h = TimeManager::getCurrent()->getTimeStepSize();
    density0 = model->getDensity0();

	std::cout << "initialising with timestep // density " << h << " // " << density0 << std::endl;
#else
    h = 0.001;
    density0 = 1000;
#endif //SPLISHSPLASH_FRAMEWORK

	h_future = h;
	h_past = h;
	h_ratio_to_past = 1.0;


#ifdef SPLISHSPLASH_FRAMEWORK

	dynamicWindowTotalDisplacement = Vector3d(0, 0, 0);



    //load the data for the boundaries
    FluidModel::RigidBodyParticleObject* particleObj = static_cast<FluidModel::RigidBodyParticleObject*>(model->m_particleObjects[1]);
    boundaries_data->reset<FluidModel::RigidBodyParticleObject>(particleObj);
	initGridOffset();
	

    //init the boundaries neighbor searchs
    boundaries_data->initNeighborsSearchData(*this, true, false);
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
	


	//load the data for the fluid
	fluid_data->reset<FluidModel>(model);
	//init the boundaries neighbor searchs
	fluid_data->initNeighborsSearchData(*this, true, false);

	//redo the rigid bodies mass computation
	computeRigidBodiesParticlesMass();

#endif //SPLISHSPLASH_FRAMEWORK

	destructor_activated = old_destructor_status;
}

void DFSPHCData::initGridOffset(bool readInput, Vector3d min, Vector3d max) {

	if (!readInput) {
		boundaries_data->getMinMaxNaive(min, max);
	}

	boundingBoxMin = min;
	boundingBoxMax = max;

	Vector3i required_grid_size = ((max - min) / getKernelRadius()).toFloor();

	std::cout << "detected min(x y z): " << min.x << " " << min.y << " " << min.z <<
		"     max:(x y z): " << max.x << " " << max.y << " " << max.z <<
		"     required grid size:(x y z): " << required_grid_size.x << " " << required_grid_size.y << " " << required_grid_size.z <<
		std::endl;

	//check if the simulation is actually possible
	Vector3i temp = required_grid_size;
	temp.toMin(CELL_ROW_LENGTH-2);

	if (temp != required_grid_size) {
		temp = Vector3i(CELL_ROW_LENGTH-2);
		std::cout << "DFSPHCData::initGridOffset grid size insufficient for the desired simulation for at least one axe" << std::endl;
		std::cout << "required grid size:(x y z): " << required_grid_size.toString() << std::endl;
		std::cout << "maximum grid size:(x y z):  " << temp.toString() << std::endl;
		exit(198);
	}

	//a second check to see if we are close to the limit
	temp = required_grid_size;
	temp.toMin(CELL_ROW_LENGTH - 10);

	if (temp != required_grid_size) {
		temp = Vector3i(CELL_ROW_LENGTH - 2);
		std::cout << "DFSPHCData::initGridOffset the required grid size is close to the maximum possible, augmenting the grid size might be recommended" << std::endl;
		std::cout << "required grid size:(x y z): " << required_grid_size.toString() << std::endl;
		std::cout << "maximum grid size:(x y z):  " << temp.toString() << std::endl;
	}

	//compute the offset
	//just take the id of the min minus 1 (the minus 2 is just to be safe ad be compatible when the dynamic area is used)
	//of course you need to take the negative of the result...
	gridOffset = ((min / getKernelRadius()).toFloor() - 2)*-1;

	std::cout << "new grid offset(x y z): " << gridOffset.x << " " << gridOffset.y << " " << gridOffset.z << std::endl;
}

void DFSPHCData::updateTimeStep(RealCuda h_fut) {
	h_future = h_fut;
	invH_future = 1.0 / h_future;
	invH2_future = 1.0 / (h_future * h_future);
	h_ratio_to_past = h / h_future;
	h_ratio_to_past2 = (h * h) / (h_future * h_future);
}

void DFSPHCData::onSimulationStepEnd() {
	h_past = h;
	invH_past = invH;
	invH2_past = invH2;
	h = h_future;
	invH = invH_future;
	invH2 = invH2_future;

	runAdvancedRendering(Vector3d(3, 3, 3), Vector3d(0, 0, 0));

}

void DFSPHCData::readDynamicData(FluidModel *model, SimulationDataDFSPH& data) {

#ifdef SPLISHSPLASH_FRAMEWORK
	if (model->numActiveParticles() != fluid_data->numParticles) {
		std::cout << "DFSPHCData::readDynamicData the nbr of particles inside the model and on the GPU do not match (model // gpu)" <<
			model->numActiveParticles() << "  //   " << fluid_data->numParticles << std::endl;
		exit(1569);
	}
	

	Vector3d* pos = new Vector3d[fluid_data->numParticles];
	Vector3d* vel = new Vector3d[fluid_data->numParticles];
	read_UnifiedParticleSet_cuda(*fluid_data, pos, vel, NULL, NULL);

	
	//density and acc are not conserved between timesteps so no need to copy them
	for (int i = 0; i <  fluid_data->numParticles; ++i) {
		model->getPosition(0, i) = vector3dTo3r(pos[i]);
		model->getVelocity(0, i) = vector3dTo3r(vel[i]);
		
	}

	delete[] pos;
	delete[] vel;
#else
    throw("DFSPHCData::readDynamicDatamust not be used outside of the SPLISHSPLASH framework");
#endif //SPLISHSPLASH_FRAMEWORK

}



void DFSPHCData::loadDynamicObjectsData(FluidModel *model) {


#ifdef SPLISHSPLASH_FRAMEWORK
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
#else
    //for anything outside of the splishsplash framework I'll supose the dynamic bodies data have been updated before
    for (int i=0;i<numDynamicBodies;++i){
        UnifiedParticleSet& body =vector_dynamic_bodies_data[i];
        body.updateDynamicBodiesParticles();
    }
#endif //SPLISHSPLASH_FRAMEWORK
}

void DFSPHCData::readDynamicObjectsData(FluidModel *model) {

#ifdef SPLISHSPLASH_FRAMEWORK
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
#endif //SPLISHSPLASH_FRAMEWORK
}


void DFSPHCData::write_fluid_to_file() {


	std::cout << "saving fluid start: " << std::endl;

	std::string file_name = fluid_files_folder + "fluid_file.txt";
	std::remove(file_name.c_str());

	fluid_data[0].write_to_file(file_name);

	

	std::cout << "saving fluid end: " << std::endl;
}

void DFSPHCData::read_fluid_from_file(bool load_velocities) {

	destructor_activated = false;

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

	



	//this damping is only needed if the simulation frequencies of the save nd load are not the same
	//since I now export it it is fine
	//damp_borders = true;
	//damp_borders_steps_count = 5;
	//add_border_to_damp_planes_cuda(*this);


    //allocate the grouped neighbor struct
#ifdef GROUP_DYNAMIC_BODIES_NEIGHBORS_SEARCH
    allocate_grouped_neighbors_struct_cuda(*this);
#endif

    std::cout << "loading fluid end" << std::endl;


	destructor_activated = true;
}

void DFSPHCData::write_boundaries_to_file() {
	std::cout << "saving boundaries start: " << std::endl;


	std::string file_name = fluid_files_folder + "boundaries_file.txt";
	std::remove(file_name.c_str());

	boundaries_data[0].write_to_file(file_name);

	std::cout << "saving boundaries end: " << std::endl;
}

void DFSPHCData::read_boundaries_from_file(bool load_velocities) {

	destructor_activated = false;

	std::cout << "loading boundaries start: " << load_velocities << std::endl;
	//reset the data strucure
	clear_boundaries_data();
	boundaries_data = new UnifiedParticleSet[1];

	//read the data to cpu pointer

	std::string file_name = fluid_files_folder + "boundaries_file.txt";
    Vector3d min,max;
    boundaries_data[0].load_from_file(file_name, load_velocities, &min, &max);

	//init the grid offset
	initGridOffset(true,min,max);
	

	//init gpu struct
	allocate_and_copy_UnifiedParticleSet_vector_cuda(&boundaries_data_cuda, boundaries_data, 1);

	//init the boundaries neighbor searchs
    boundaries_data[0].initNeighborsSearchData(*this, false, false);

	std::cout << "loading boundaries end" << std::endl;


	destructor_activated = true;
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
        oss << fluid_files_folder+"solid_file_" << i << ".txt";

		std::remove(oss.str().c_str());
		vector_dynamic_bodies_data[i].write_to_file(oss.str());
	}


	std::cout << "saving solids end: " << std::endl;
}

void DFSPHCData::read_solids_from_file(bool load_velocities) {

	destructor_activated = false;

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
        oss << fluid_files_folder +"solid_file_" << i << ".txt";
        vector_dynamic_bodies_data[i].load_from_file(oss.str(), load_velocities);
	}

	//init gpu struct
	allocate_and_copy_UnifiedParticleSet_vector_cuda(&vector_dynamic_bodies_data_cuda, vector_dynamic_bodies_data, numDynamicBodies);



	std::cout << "loading solids end" << std::endl;


	destructor_activated = true;
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
	read_last_error_cuda("before:  ");
	if (boundaries_data_cuda != NULL) {
		release_UnifiedParticleSet_vector_cuda(&boundaries_data_cuda, 1);
	}
	read_last_error_cuda("middle:  ");
	if (boundaries_data != NULL) {
		delete[] boundaries_data; boundaries_data = NULL;
	}
	read_last_error_cuda("end:  ");
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
        std::cout << "number of existing rigid bodies do not fit the input data (actual number, input number):" <<
                  numDynamicBodies<<"  ,  "<<vect_new_info.size()<<std::endl;
        exit(1256);
    }

    for (int i = 0; i < numDynamicBodies; ++i) {
        DynamicBody& body = vect_new_info[i];
        vector_dynamic_bodies_data[i].updateDynamicBodiesParticles(body.position, body.velocity, body.q, body.angular_vel);
    }

}

void DFSPHCData::pause_solids() {
    for (int i = 0; i < numDynamicBodies; ++i) {
        vector_dynamic_bodies_data[i].zeroVelocities();
    }

}

void DFSPHCData::zeroFluidVelocities() {
	fluid_data[0].zeroVelocities();
}


void DFSPHCData::handleFLuidLevelControl(RealCuda level) {
	if (level > 0) {
		control_fluid_height_cuda(*this,level);
	}
}


RealCuda DFSPHCData::computeFluidLevel(){
    return find_fluid_height_cuda(*this);
}

void DFSPHCData::getFluidImpactOnDynamicBodies(std::vector<SPH::Vector3d>& sph_forces, std::vector<SPH::Vector3d>& sph_moments,
                                               const std::vector<SPH::Vector3d> &reduction_factors){
    //std::cout<<"conputing impact start"<<std::endl;
    for (int i = 0; i < numDynamicBodies; ++i) {
        Vector3d force, moment;
        compute_fluid_impact_on_dynamic_body_cuda(vector_dynamic_bodies_data[i],force,moment, reduction_factors[i]);
        sph_forces.push_back(force);
        sph_moments.push_back(moment);

        //std::cout<<"force on body "<<i<<":   "<<force.x<<"  "<<force.y<<"  "<<force.z<<"   ("<<force.norm()<<std::endl;
        //std::cout<<"reduction factor "<<i<<":   "<<reduction_factors[i].x<<"  "<<reduction_factors[i].y<<"  "<<reduction_factors[i].z<<std::endl;


    }

}

void DFSPHCData::getFluidBoyancyOnDynamicBodies(std::vector<SPH::Vector3d>& forces, std::vector<SPH::Vector3d>& pts_appli){
    for (int i = 0; i < numDynamicBodies; ++i) {
        Vector3d force, pt_appli;
        compute_fluid_Boyancy_on_dynamic_body_cuda(vector_dynamic_bodies_data[i],force,pt_appli);
        forces.push_back(force);
        pts_appli.push_back(pt_appli);
    }
}


SPH::Vector3d DFSPHCData::getSimulationCenter(){
    return get_simulation_center_cuda(*this);
}


void DFSPHCData::computeRigidBodiesParticlesMass(){
    std::cout<<"updating rigid body particle mass start"<<std::endl;

    //do the comutation for the dynamic bodies
    for (int i=0;i<numDynamicBodies;++i){
        vector_dynamic_bodies_data[i].computeParticlesMass(this);
    }

    //and for the boundaries
    boundaries_data->computeParticlesMass(this);

    std::cout<<"updating rigid body particle mass end"<<std::endl;

}


void DFSPHCData::loadBender2019BoundariesFromCPU(RealCuda* V_rigids_i, Vector3d* X_rigids_i) {
	fluid_data->loadBender2019BoundariesFromCPU(V_rigids_i, X_rigids_i);
}


void DFSPHCData::handleFluidBoundries( Vector3d movement) {
	if (!DynamicWindowInterface::isInitialized()) {
		DynamicWindowInterface::initDynamicWindow(*this);
	}
	DynamicWindowInterface::handleFluidBoundaries(*this, movement);
}


void DFSPHCData::handleBoundariesHeightTest() {
	if (!DynamicWindowInterface::isInitialized()) {
		DynamicWindowInterface::initDynamicWindow(*this);
	}
	DynamicWindowInterface::handleOceanBoundariesTest(*this);
}

void DFSPHCData::initAdvancedRendering(int width, int height) {
	advancedRenderingData = new AdvancedRenderingData();
	advancedRenderingData->init(width, height);
}


void DFSPHCData::runAdvancedRendering(Vector3d eye, Vector3d lookAt) {
	advancedRenderingData->computeDepthBuffer(this, eye, lookAt);
}


void DFSPHCData::checkParticlesPositions(int mode, bool report) {
	check_particles_positions_cuda(*this, mode, report);
}