#ifndef __DFSPHCArraysStructure_h__
#define __DFSPHCArraysStructure_h__

#include "SPlisHSPlasH\BasicTypes.h"
#include <string>
#include <vector>


#include "SPlisHSPlasH\Vector.h"
#include "SPlisHSPlasH\Quaternion.h"

#include "DFSPH_define_c.h"
#include "SPH_kernels.h"

class ParticleSetRenderingData;

namespace SPH
{

class DynamicBody {
public:
    Vector3d position;
    Vector3d velocity;
    Quaternion q;
    Vector3d angular_vel;
};

/**
        This class encapsulate the data needed to RealCudaize the neighbor search for one set of
        particles
    */
class DFSPHCData;
class UnifiedParticleSet;

class NeighborsSearchDataSet {
public:
	int set_id;

    unsigned int numParticles;
    unsigned int numParticlesMax;
    unsigned int* cell_id;
    unsigned int* cell_id_sorted;
    unsigned int* local_id;
    unsigned int* p_id;
    unsigned int* p_id_sorted;
    unsigned int* cell_start_end;
    unsigned int* hist;

    //those 4 variables are used for cub internal computations
    void *d_temp_storage_pair_sort;
    size_t temp_storage_bytes_pair_sort;
    void *d_temp_storage_cumul_hist;
    size_t temp_storage_bytes_cumul_hist;

    bool internal_buffers_allocated;

    Vector3d* intermediate_buffer_v3d ;
    RealCuda* intermediate_buffer_real;

    //pointer for the gpu storage (that should be a copy of this item but allocated on the gpu)
    NeighborsSearchDataSet* gpu_ptr;

    //empty contructor to make static arrays possible
    NeighborsSearchDataSet();

    /**
            allocate the data structure
        */
    NeighborsSearchDataSet(unsigned int numParticles_i, unsigned int numParticlesMax_i);
    ~NeighborsSearchDataSet();


    /**
            this function RealCudaize the computations necessary to initialize:
                p_id_sorted buffer with the particles index sorted by their cell index
                cell_start_end with the number of particles at the start and end of each cell
        */
    void initData(UnifiedParticleSet* particleSet, DFSPHCData &data, bool sort_data);

    /**
            Free computation memory. This cna be called for the boundary as
            keeping the internal computation buffers are only needed at the start
            since the computation is only done once
        */
    void deleteComputationBuffer();


    void updateActiveParticleNumber(unsigned int val);
    void changeMaxParticleNumber(int numParticlesMax_i);


};

class UnifiedParticleSet {
public:



    //this allow the control of the destructor call
    //especially it was create to be able to use temp variable sot copy to cuda
    //but I also need it for the initialisation because doing a=b(params) call the destructur at the end of the line ...
    //so by default the destructor is not activated and I activate it only bafter the data has been fulle initialised
    bool releaseDataOnDestruction;


    int numParticles;
    int numParticlesMax;
    bool has_factor_computation;
    bool velocity_impacted_by_fluid_solver;
    bool is_dynamic_object;

    //for fluid particles only
    RealCuda m_V;//volume
    RealCuda density0;

    //for all particles
    RealCuda* mass;
    Vector3d* pos;
    Vector3d* vel;

    NeighborsSearchDataSet* neighborsDataSet;

    //for particles with factor computation
    RealCuda* density;
    RealCuda* factor;
    RealCuda* densityAdv;
    int* numberOfNeighbourgs;
    int* neighbourgs;

    //for particles whose velocity is controlled by the fluid solver
    Vector3d* acc;
    RealCuda* kappa;
    RealCuda* kappaV;

	//that is for the boundaries handling with bender 2019 method
	//for now I'll put them directly inside the fluid particles structure 
	RealCuda* V_rigids;
	Vector3d* X_rigids;

    //for dynamic object particles
    //the original particle position
    Vector3d* pos0;
    //the force to be transmitted to the physics engine
    Vector3d* F;

    //this is a buffer that is used to make the transition between cuda and the cpu physics engine
    //I need it because to transmit the data to the model I need to convert the values to the
    //type used in the orinal cpu model...
    DynamicBody* rigidBody_cpu;
    Vector3d* F_cpu;

    //data for the rendering
    ParticleSetRenderingData* renderingData;
	Vector3d* color;//not used all of the tme it's mostly a debug fonctionality
	bool has_color_buffer;

    //pointer for the gpu storage (that should be a copy of this item but allocated on the gpu)
    UnifiedParticleSet* gpu_ptr;

    //those two variables are here for cub computationso that they are only allocated once (or when the number
    //of particles change). There are used when computeing the avg at the end of the pressure and div loops
    void     *d_temp_storage;
    size_t   temp_storage_bytes;


    //base contructor (set every array to null and the nb of particles to 0
    UnifiedParticleSet();
    void init();

    //actual constructor
    UnifiedParticleSet(int nbParticles, bool has_factor_computation_i, bool velocity_impacted_by_fluid_solver_i,
                       bool is_dynamic_object_i);
    void init(int nbParticles, bool has_factor_computation_i, bool velocity_impacted_by_fluid_solver_i,
              bool is_dynamic_object_i, bool need_color_buffer);

    //copy contructor
    UnifiedParticleSet(UnifiedParticleSet* other);

    //destructor
    ~UnifiedParticleSet();
    void clear();

    //this function should only be used for the rigid bodies
    void computeParticlesMass(DFSPHCData *data);

    //update particles position from rb position and orientation
    //only do smth for the dynamic bodies
    template<class T>
    void updateDynamicBodiesParticles(T* particleObj);
    void updateDynamicBodiesParticles(Vector3d position, Vector3d velocity, Quaternion q, Vector3d angular_vel);
    void updateDynamicBodiesParticles();

    //tranfer the forces to the cpu buffer
    //only do smth for the dynamic bodies
    void transferForcesToCPU();

    //this does the necessary calls to be able to run the neighbors search later
    void initNeighborsSearchData(DFSPHCData &data, bool sort_data, bool delete_computation_data=false);

    //reset the data
    //also clear the computation buffers
    template<class T>
    void reset(T* particleObj);
    void reset(Vector3d* pos_temp, Vector3d* vel_temp, RealCuda* mass_temp);

    //store a unified dataset to a file
    void write_to_file(std::string file_path);
    //load a unified dataset from a file
    void load_from_file(std::string file_path, bool load_velocities, Vector3d* min_o=NULL, Vector3d* max_o=NULL, bool positions_limitations=true);

    //store the forces to a file in case of a dynamic body
    void write_forces_to_file(std::string file_path);

    FUNCTION inline int* getNeighboursPtr(int particle_id) {
#ifdef INTERLEAVE_NEIGHBORS
		return neighbourgs + particle_id;
#else
        return neighbourgs + particle_id*MAX_NEIGHBOURS;
#endif


    }

    FUNCTION inline unsigned int getNumberOfNeighbourgs(int particle_id, int body_id = 0) {
        //return numberOfNeighbourgs[body_id*numFluidParticles + particle_id];
        return numberOfNeighbourgs[particle_id * 3 + body_id];
    }

    void updateActiveParticleNumber(unsigned int val);
    void changeMaxParticleNumber(int numParticlesMax_i);

    void add_particles(std::vector<Vector3d> pos, std::vector<Vector3d> vel);

    void zeroVelocities();

	void getMinMaxNaive(Vector3d& min, Vector3d& max);

	void loadBender2019BoundariesFromCPU(RealCuda* V_rigids_i, Vector3d* X_rigids_i);

	void resetColor();
};


class FluidModel;
class SimulationDataDFSPH;

class DFSPHCData {
public:

    bool destructor_activated;

    //I need a kernel without all th static class memebers because it seems they do not work on
    //the gpu
    CubicKernelPerso m_kernel;
    PrecomputedCubicKernelPerso m_kernel_precomp;
    AdhesionKernelGPU m_kernel_adhesion;
    CohesionKernelGPU m_kernel_cohesion;

    const Vector3d gravitation = Vector3d(0.0f, -9.81, 0.0f);

    //static size and value all time

#ifdef PRECOMPUTED_KERNELS
	FUNCTION inline RealCuda W(const Vector3d &r) const { return m_kernel_precomp.W(r); }
	FUNCTION inline RealCuda W(const RealCuda r) const { return m_kernel_precomp.W(r); }
	FUNCTION inline Vector3d gradW(const Vector3d &r) const { return m_kernel_precomp.gradW(r); }
	FUNCTION inline RealCuda getKernelRadius() { return m_kernel_precomp.getRadius(); }
	FUNCTION inline RealCuda getKernelRadius() const { return m_kernel_precomp.getRadius(); }
#else
    FUNCTION inline RealCuda W(const Vector3d &r) const { return m_kernel.W(r); }
    FUNCTION inline RealCuda W(const RealCuda r) const { return m_kernel.W(r); }
    FUNCTION inline Vector3d gradW(const Vector3d &r) const { return m_kernel.gradW(r); }
	FUNCTION inline RealCuda getKernelRadius() { return m_kernel.getRadius(); }
	FUNCTION inline RealCuda getKernelRadius() const { return m_kernel.getRadius(); } 
#endif // PRECOMPUTED_KERNELS


    FUNCTION inline RealCuda WAdhesion(const Vector3d &r) const { return m_kernel_adhesion.W(r); }
    FUNCTION inline RealCuda WAdhesion(const RealCuda r) const { return m_kernel_adhesion.W(r); }

    FUNCTION inline RealCuda WCohesion(const Vector3d &r) const { return m_kernel_cohesion.W(r); }
    FUNCTION inline RealCuda WCohesion(const RealCuda r) const { return m_kernel_cohesion.W(r); }


    RealCuda W_zero;
    RealCuda density0;
    RealCuda particleRadius;
    RealCuda viscosity;
    RealCuda m_surfaceTension;
	Vector3i gridOffset;
	Vector3d dynamicWindowTotalDisplacement;

    RealCuda h;
    RealCuda h_future;
    RealCuda h_past;
    RealCuda h_ratio_to_past;
    RealCuda h_ratio_to_past2;
    RealCuda invH;
    RealCuda invH2;
    RealCuda invH_past;
    RealCuda invH2_past;
    RealCuda invH_future;
    RealCuda invH2_future;


    //boundaries particles
    UnifiedParticleSet* fluid_data;
    UnifiedParticleSet* fluid_data_cuda;

    //boundaries particles
    UnifiedParticleSet* boundaries_data;
    UnifiedParticleSet* boundaries_data_cuda;


    //data structure for the dynamic objects
    //both contains the same data but the first one is create with a new on the cpu
    //and the second one is created on the gpu
    //note: in either case the arrays inside are allocated with cuda
    UnifiedParticleSet* vector_dynamic_bodies_data;
    UnifiedParticleSet* vector_dynamic_bodies_data_cuda;
    int numDynamicBodies;
    //this neighbors search structure willwork on the solids as a group
    //this allow a way smaller memory size needed and faster computations
    NeighborsSearchDataSet* neighborsDataSetGroupedDynamicBodies;
    NeighborsSearchDataSet* neighborsDataSetGroupedDynamicBodies_cuda;
    Vector3d* posBufferGroupedDynamicBodies;
    bool is_fluid_aggregated;

    Vector3d* bmin;
    Vector3d* bmax;

    bool damp_borders;
    int damp_borders_steps_count;
    Vector3d* damp_planes;
    int damp_planes_count;

    bool cancel_wave;
    int cancel_wave_steps_count;
    Vector3d* cancel_wave_planes;
    RealCuda cancel_wave_lowest_point;

	std::string fluid_files_folder;


    DFSPHCData();
    DFSPHCData(FluidModel *model);
    ~DFSPHCData();

	int getFluidParticlesCount() { return fluid_data->numParticles; };
	int getFluidParticlesCountMax() { return fluid_data->numParticlesMax; };

    FUNCTION RealCuda getSurfaceTension(){return m_surfaceTension;}

    void readDynamicData(FluidModel *model, SimulationDataDFSPH& data);

    void loadDynamicObjectsData(FluidModel *model=NULL);
    void readDynamicObjectsData(FluidModel *model);

    void reset(FluidModel *model);
	void initGridOffset();

    inline void updateTimeStep(RealCuda h_fut) {
        h_future = h_fut;
        invH_future = 1.0 / h_future;
        invH2_future = 1.0 / (h_future*h_future);
        h_ratio_to_past = h / h_future;
        h_ratio_to_past2 = (h*h) / (h_future*h_future);
    }

    inline void onSimulationStepEnd() {
        h_past = h;
        invH_past = invH;
        invH2_past = invH2;
        h = h_future;
        invH = invH_future;
        invH2 = invH2_future;
    }

    inline RealCuda get_current_timestep(){return h;}

    void write_fluid_to_file();
    void read_fluid_from_file(bool load_velocities);
    void write_boundaries_to_file();
    void read_boundaries_from_file(bool load_velocities);
    void write_solids_to_file();
    void read_solids_from_file(bool load_velocities);

    void clear_fluid_data();
    void clear_boundaries_data();
    void clear_solids_data();


    void update_solids_to_file();
    void update_solids_from_file();

    void update_solids(std::vector<DynamicBody> vect_new_info);
    void pause_solids();

    void zeroFluidVelocities();
    void handleFLuidLevelControl(RealCuda level);
    RealCuda computeFluidLevel();

    void getFluidImpactOnDynamicBodies(std::vector<SPH::Vector3d>& sph_forces, std::vector<SPH::Vector3d>& sph_moments,
                                       const std::vector<SPH::Vector3d>& reduction_factors);
    void getFluidBoyancyOnDynamicBodies(std::vector<SPH::Vector3d>& forces, std::vector<SPH::Vector3d>& pts_appli);

    SPH::Vector3d getSimulationCenter();

    void computeRigidBodiesParticlesMass();


	void loadBender2019BoundariesFromCPU(RealCuda* V_rigids_i, Vector3d* X_rigids_i);

	//don't call that ...
	void handleFluidBoundries(bool loading = false);
};
}

#endif


