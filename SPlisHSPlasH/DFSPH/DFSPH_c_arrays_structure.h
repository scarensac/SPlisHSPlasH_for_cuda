#ifndef __DFSPHCArraysStructure_h__
#define __DFSPHCArraysStructure_h__

#include "SPlisHSPlasH\BasicTypes.h"

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

#include "SPlisHSPlasH\Vector.h"

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
					res = m_k * (2.0*pow(1.0 - q, 3));
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

	/**
		This class encapsulate the data needed to RealCudaize the neighbor search for one set of 
		particles
	*/
	class DFSPHCData;

	class NeighborsSearchDataSet {
	public:
		unsigned int numParticles;
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

		//empty contructor to make static arrays possible
		NeighborsSearchDataSet(){}

		/**
			allocate the data structure
		*/
		NeighborsSearchDataSet(unsigned int numParticles_i);
		~NeighborsSearchDataSet();

		/**
			this function RealCudaize the computations necessary to initialize:
				p_id_sorted buffer with the particles index sorted by their cell index
				cell_start_end with the number of particles at the start and end of each cell
		*/
		void initData(DFSPHCData* data, bool is_boundaries);


		/**
			Free computation memory. This cna be called for the boundary as
			keeping the internal computation buffers are only needed at the start
			since the computation is only done once
		*/
		void deleteComputationBuffer();


		
	};

	class RigidBodyContainer {
	public:
		Vector3d* pos;
		Vector3d* vel;
		RealCuda* psi;
		int numParticles;

		//the force to be transmitted to the physics engine
		Vector3d* F;

		//this is a buffer that is used to make the transition between cuda and the cpu physics engine
		//I need it because to transmit the data to the model I need to convert the values to the
		//type used in the orinal cpu model...
		Vector3d* F_cpu;


		// I need to be able to acces to the elements inside both on the cpu and gpu
		NeighborsSearchDataSet* neighborsDataSet;


		ParticleSetRenderingData* renderingData;

		//empty contructor to make static arrays possible
		RigidBodyContainer() {}
		
		//actual constructor
		RigidBodyContainer(int nbParticles);
		

		//this class is used to be able to transmit the info to the neighbor search kernel
		//it only contains the necessary arrays
		class NeighborKernelData {
		public:
			Vector3d* pos;
			unsigned int* p_id_sorted;
			unsigned int* cell_start_end;

			NeighborKernelData( Vector3d* pos_i,unsigned int* p_id_sorted_i,unsigned int* cell_start_end_i):
				pos(pos_i),p_id_sorted(p_id_sorted_i),cell_start_end(cell_start_end_i){}
		};

		NeighborKernelData getNeighborKerneldata() {
			return NeighborKernelData(pos, neighborsDataSet->p_id_sorted,neighborsDataSet->cell_start_end);
		}
	};

	class UnifiedParticleSet {
	public://static size and values when pacticle count constant
		int numParticles;
		bool has_factor_computation;
		bool is_dynamic_object;
		bool velocity_impacted_by_fluid_solver;

		//for all particles
		RealCuda* mass;
		RealCuda* density;
		Vector3d* pos;
		Vector3d* vel;

		NeighborsSearchDataSet* neighborsDataSet;

		//for particles with factor computation
		RealCuda* factor;
		RealCuda* densityAdv;
		int* numberOfNeighbourgs;
		int* neighbourgs;

		//for particles whose velocity is controlled by the fluid solver
		Vector3d* acc;
		RealCuda* kappa;
		RealCuda* kappaV;

		//for dynamic object particles
		//the force to be transmitted to the physics engine
		Vector3d* F;

		//this is a buffer that is used to make the transition between cuda and the cpu physics engine
		//I need it because to transmit the data to the model I need to convert the values to the
		//type used in the orinal cpu model...
		Vector3d* F_cpu;

		//data for the rendering
		ParticleSetRenderingData* renderingDataFluid;

		//base contructor (set every array to null and the nb of particles to 0
		UnifiedParticleSet();

		//actual constructor
		UnifiedParticleSet(int nbParticles, bool has_factor_computation_i, bool is_dynamic_object_i, 
			bool velocity_impacted_by_fluid_solver_i);


		//this class is used to be able to transmit the info to the neighbor search kernel
		//it only contains the necessary arrays
		class NeighborKernelData {
		public:
			Vector3d* pos;
			unsigned int* p_id_sorted;
			unsigned int* cell_start_end;

			NeighborKernelData(Vector3d* pos_i, unsigned int* p_id_sorted_i, unsigned int* cell_start_end_i) :
				pos(pos_i), p_id_sorted(p_id_sorted_i), cell_start_end(cell_start_end_i) {}
		};

		NeighborKernelData getNeighborKerneldata() {
			return NeighborKernelData(pos, neighborsDataSet->p_id_sorted, neighborsDataSet->cell_start_end);
		}
	};

	class FluidModel;
	class SimulationDataDFSPH;

	class DFSPHCData {
	public:
		

		
		//I need a kernel without all th static class memebers because it seems they do not work on
		//the gpu
		CubicKernelPerso m_kernel;
		PrecomputedCubicKernelPerso m_kernel_precomp;

		//the size is 60 because I checked and the max neighbours I reached was 50
		//so I put 10 more to be sure. In the end those buffers will stay on the GPU memory
		//so there will be no transfers.
#define MAX_NEIGHBOURS 75
		const Vector3d gravitation = Vector3d(0.0f, -9.81, 0.0f);

		//static size and value all time
		FUNCTION inline RealCuda W(const Vector3d &r) const { return m_kernel.W(r); }
		FUNCTION inline RealCuda W(const RealCuda r) const { return m_kernel.W(r); }
		FUNCTION inline Vector3d gradW(const Vector3d &r) { return m_kernel.gradW(r); }
		RealCuda W_zero;
		RealCuda density0;
		RealCuda particleRadius;
		RealCuda viscosity;

		//for boundaries everything should be completely static here
		Vector3d* posBoundary;
		Vector3d* velBoundary;
		RealCuda* boundaryPsi;
		int numBoundaryParticles;

		//static size and values when pacticle count constant
		RealCuda* mass;

		//static size but value dynamic
		RealCuda* density;
		Vector3d* posFluid;
		Vector3d* velFluid;
		Vector3d* accFluid;
		int* numberOfNeighbourgs;
		int* neighbourgs;
		RealCuda* factor;

		//the V one is for the divergence
		RealCuda* kappa;
		RealCuda* kappaV;
		//for the terative solvers
		RealCuda* densityAdv;

		int numFluidParticles;
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

		//data sets for the neighbors search
		NeighborsSearchDataSet* neighborsdataSetBoundaries;
		NeighborsSearchDataSet* neighborsdataSetFluid;

	

		//data structure for the dynamic objects
		//both contains the same data but the first one is create with a new on the cpu
		//and the second one is created on the gpu 
		//note: in either case the arrays inside are allocated with cuda
		RigidBodyContainer* vector_dynamic_bodies_data;
		RigidBodyContainer* vector_dynamic_bodies_data_cuda;
		int numDynamicBodies;

		//variables for the rendering (pointer to opengl and cuda interop)
		ParticleSetRenderingData* renderingDataFluid;
		ParticleSetRenderingData* renderingDataBoundaries;


		unsigned int vao_float;
		unsigned int pos_buffer_float;
		unsigned int vel_buffer_float;

		DFSPHCData(FluidModel *model);

		void loadDynamicData(FluidModel *model, const SimulationDataDFSPH& data);
		void readDynamicData(FluidModel *model, SimulationDataDFSPH& data);
		void sortDynamicData(FluidModel *model);

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


		FUNCTION inline int* getNeighboursPtr(int particle_id) {
			//	return neighbourgs + body_id*numFluidParticles*MAX_NEIGHBOURS + particle_id*MAX_NEIGHBOURS;
			return neighbourgs + particle_id*MAX_NEIGHBOURS;
		}

		FUNCTION inline unsigned int getNumberOfNeighbourgs(int particle_id, int body_id = 0) {
			//return numberOfNeighbourgs[body_id*numFluidParticles + particle_id]; 
			return numberOfNeighbourgs[particle_id*3+body_id];
		}

	};
}

#endif


