#ifndef __DFSPHCArraysStructure_h__
#define __DFSPHCArraysStructure_h__

#include "SPlisHSPlasH\BasicTypes.h"

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

#include "SPlisHSPlasH\Vector.h"

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
		void initData(Vector3d* pos, RealCuda kernelRadius);


		/**
			Free computation memory. This cna be called for the boundary as
			keeping the internal computation buffers are only needed at the start
			since the computation is only done once
		*/
		void deleteComputationBuffer();
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
#define MAX_NEIGHBOURS 60
		const Vector3d gravitation = Vector3d(0.0f, -9.81, 0.0f);

		//static size and value all time
		FUNCTION inline RealCuda W(const Vector3d &r) const { return m_kernel_precomp.W(r); }
		FUNCTION inline RealCuda W(const RealCuda r) const { return m_kernel_precomp.W(r); }
		FUNCTION inline Vector3d gradW(const Vector3d &r) { return m_kernel_precomp.gradW(r); }
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

		//the V one is for the warm start
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



		//variables for vao
		//Normaly I should use GLuint but including all gl.h only for that ...
		unsigned int vao;
		unsigned int pos_buffer;
		unsigned int vel_buffer;


		unsigned int vao_float;
		unsigned int pos_buffer_float;
		unsigned int vel_buffer_float;

		DFSPHCData(FluidModel *model);

		void loadDynamicData(FluidModel *model, const SimulationDataDFSPH& data);
		void readDynamicData(FluidModel *model, SimulationDataDFSPH& data);
		void sortDynamicData(FluidModel *model);

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

		FUNCTION inline unsigned int getNeighbour(int particle_id, int neighbour_id, int body_id = 0) {
			return neighbourgs[body_id*numFluidParticles*MAX_NEIGHBOURS + particle_id * MAX_NEIGHBOURS + neighbour_id];
		}

		FUNCTION inline unsigned int getNumberOfNeighbourgs(int particle_id, int body_id = 0) {
			return numberOfNeighbourgs[body_id*numFluidParticles + particle_id];
		}
	};
}

#endif


