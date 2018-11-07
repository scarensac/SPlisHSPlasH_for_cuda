#ifndef __DFSPHCArraysStructure_h__
#define __DFSPHCArraysStructure_h__

#include "SPlisHSPlasH\BasicTypes.h"
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

#include "SPlisHSPlasH\Vector.h"
#include "SPlisHSPlasH\Quaternion.h"

class ParticleSetRenderingData;

namespace SPH
{
	//this is an eigen less cubic kernel
	/** \brief Cubic spline kernel.
	*/
	class CubicKernelPerso
	{
	protected:
		RealCuda m_radius;
		RealCuda m_k;
		RealCuda m_l;
		RealCuda m_W_zero;
	public:
		FUNCTION RealCuda getRadius() { return m_radius; }
		FUNCTION void setRadius(RealCuda val)
		{
			m_radius = val;
			static const RealCuda pi = static_cast<RealCuda>(M_PI);

			const RealCuda h3 = m_radius*m_radius*m_radius;
			m_k = 8.0 / (pi*h3);
			m_l = 48.0 / (pi*h3);
			m_W_zero = W(Vector3d::Zero());
		}

	public:
		FUNCTION RealCuda W(const RealCuda r) const
		{
			RealCuda res = 0.0;
			const RealCuda q = r / m_radius;
			if (q <= 1.0)
			{
				if (q <= 0.5)
				{
					const RealCuda q2 = q*q;
					const RealCuda q3 = q2*q;
					res = m_k * (6.0*q3 - 6.0*q2 + 1.0);
				}
				else
				{
					const RealCuda factor = 1.0 - q;
					//res = m_k * (2.0*pow(1.0 - q, 3));
					res = m_k * (2.0*factor*factor*factor);
				}
			}
			return res;
		}

		FUNCTION inline RealCuda W(const Vector3d &r) const
		{
			return W(r.norm());
		}

		FUNCTION Vector3d gradW(const Vector3d &r) const
		{
			Vector3d res;
			const RealCuda rl = r.norm();
			const RealCuda q = rl / m_radius;
			if (q <= 1.0)
			{
				if (rl > 1.0e-6)
				{
					const Vector3d gradq = r * ((RealCuda) 1.0 / (rl*m_radius));
					if (q <= 0.5)
					{
						res = m_l*q*((RealCuda) 3.0*q - (RealCuda) 2.0)*gradq;
					}
					else
					{
						const RealCuda factor = 1.0 - q;
						res = m_l*(-factor*factor)*gradq;
					}
				}
			}
			else
				res.setZero();

			return res;
		}

		FUNCTION inline RealCuda W_zero() const
		{
			return m_W_zero;
		}
	};

	class PrecomputedCubicKernelPerso
	{
	public:
		RealCuda* m_W;
		RealCuda* m_gradW;
		RealCuda m_radius;
		RealCuda m_radius2;
		RealCuda m_invStepSize;
		RealCuda m_W_zero;
		unsigned int m_resolution;
	public:
		FUNCTION RealCuda getRadius() { return m_radius; }
		void setRadius(RealCuda val);

	public:
		FUNCTION RealCuda W(const Vector3d &r) const
		{
			RealCuda res = 0.0;
			const RealCuda r2 = r.squaredNorm();
			if (r2 <= m_radius2)
			{
				const RealCuda r = sqrt(r2);
				const unsigned int pos = (unsigned int)(r * m_invStepSize);
				res = m_W[pos];
			}
			return res;
		}

		FUNCTION RealCuda W(const RealCuda r) const
		{
			RealCuda res = 0.0;
			if (r <= m_radius)
			{
				const unsigned int pos = (unsigned int)(r * m_invStepSize);
				res = m_W[pos];
			}
			return res;
		}

		FUNCTION Vector3d gradW(const Vector3d &r) const
		{
			Vector3d res;
			const RealCuda r2 = r.squaredNorm();
			if (r2 <= m_radius2)
			{
				const RealCuda rl = sqrt(r2);
				const unsigned int pos = (unsigned int)(rl * m_invStepSize);
				res = m_gradW[pos] * r;
			}
			else
				res.setZero();

			return res;
		}

		FUNCTION inline RealCuda W_zero() const
		{
			return m_W_zero;
		}
	};

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
		void initData(UnifiedParticleSet* particleSet, RealCuda kernel_radius, bool sort_data);

		/**
			Free computation memory. This cna be called for the boundary as
			keeping the internal computation buffers are only needed at the start
			since the computation is only done once
		*/
		void deleteComputationBuffer();


		
		
	};

	class UnifiedParticleSet {
	public:


		//the size is 75 because I checked and the max neighbours I reached was 58
		//so I put some more to be sure. In the end those buffers will stay on the GPU memory
		//so there will be no transfers.
#define MAX_NEIGHBOURS 75
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
			bool is_dynamic_object_i);

		//destructor
		~UnifiedParticleSet();
		void clear();


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
		void initNeighborsSearchData(RealCuda kernel_radius, bool sort_data, bool delete_computation_data=false);

		//reset the data
		//also clear the computation buffers 
		template<class T>
		void reset(T* particleObj);
		void reset(Vector3d* pos_temp, Vector3d* vel_temp, RealCuda* mass_temp);

		//store a unified dataset to a file
		void write_to_file(std::string file_path);
		//load a unified dataset from a file
		void load_from_file(std::string file_path, bool load_velocities);

		//store the forces to a file in case of a dynamic body
		void write_forces_to_file(std::string file_path);

		FUNCTION inline int* getNeighboursPtr(int particle_id) {
			//	return neighbourgs + body_id*numFluidParticles*MAX_NEIGHBOURS + particle_id*MAX_NEIGHBOURS;
			return neighbourgs + particle_id*MAX_NEIGHBOURS;
		}

		FUNCTION inline unsigned int getNumberOfNeighbourgs(int particle_id, int body_id = 0) {
			//return numberOfNeighbourgs[body_id*numFluidParticles + particle_id]; 
			return numberOfNeighbourgs[particle_id * 3 + body_id];
		}

		void update_active_particle_number(unsigned int val);

		void add_particles(std::vector<Vector3d> pos, std::vector<Vector3d> vel);

		void zeroVelocities();

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

		const Vector3d gravitation = Vector3d(0.0f, -9.81, 0.0f);

		//static size and value all time
		FUNCTION inline RealCuda W(const Vector3d &r) const { return m_kernel.W(r); }
		FUNCTION inline RealCuda W(const RealCuda r) const { return m_kernel.W(r); }
		FUNCTION inline Vector3d gradW(const Vector3d &r) const { return m_kernel.gradW(r); }
		FUNCTION inline RealCuda getKernelRadius()  { return m_kernel.getRadius(); }
		RealCuda W_zero;
		RealCuda density0;
		RealCuda particleRadius;
		RealCuda viscosity;
		

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

		bool damp_borders;
		int damp_borders_steps_count;
		Vector3d* bmin;
		Vector3d* bmax;
		Vector3d* damp_planes;
		int damp_planes_count;

		DFSPHCData();
		DFSPHCData(FluidModel *model);
		~DFSPHCData();
		
		void readDynamicData(FluidModel *model, SimulationDataDFSPH& data);

		void loadDynamicObjectsData(FluidModel *model);
		void readDynamicObjectsData(FluidModel *model);

		void reset(FluidModel *model);

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

		void zeroFluidVelocities();
	};
}

#endif


