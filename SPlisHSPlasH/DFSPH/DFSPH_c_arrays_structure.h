#ifndef __DFSPHCArraysStructure_h__
#define __DFSPHCArraysStructure_h__


typedef double Real;

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
		Real m_radius;
		Real m_k;
		Real m_l;
		Real m_W_zero;
	public:
		FUNCTION Real getRadius() { return m_radius; }
		FUNCTION void setRadius(Real val)
		{
			m_radius = val;
			static const Real pi = static_cast<Real>(M_PI);

			const Real h3 = m_radius*m_radius*m_radius;
			m_k = 8.0 / (pi*h3);
			m_l = 48.0 / (pi*h3);
			m_W_zero = W(Vector3d::Zero());
		}

	public:
		FUNCTION Real W(const Real r) const
		{
			Real res = 0.0;
			const Real q = r / m_radius;
			if (q <= 1.0)
			{
				if (q <= 0.5)
				{
					const Real q2 = q*q;
					const Real q3 = q2*q;
					res = m_k * (6.0*q3 - 6.0*q2 + 1.0);
				}
				else
				{
					res = m_k * (2.0*pow(1.0 - q, 3));
				}
			}
			return res;
		}

		FUNCTION inline Real W(const Vector3d &r) const
		{
			return W(r.norm());
		}

		FUNCTION Vector3d gradW(const Vector3d &r) const
		{
			Vector3d res;
			const Real rl = r.norm();
			const Real q = rl / m_radius;
			if (q <= 1.0)
			{
				if (rl > 1.0e-6)
				{
					const Vector3d gradq = r * ((Real) 1.0 / (rl*m_radius));
					if (q <= 0.5)
					{
						res = m_l*q*((Real) 3.0*q - (Real) 2.0)*gradq;
					}
					else
					{
						const Real factor = 1.0 - q;
						res = m_l*(-factor*factor)*gradq;
					}
				}
			}
			else
				res.setZero();

			return res;
		}

		FUNCTION inline Real W_zero() const
		{
			return m_W_zero;
		}
	};

	class PrecomputedCubicKernelPerso
	{
	public:
		Real* m_W;
		Real* m_gradW;
		Real m_radius;
		Real m_radius2;
		Real m_invStepSize;
		Real m_W_zero;
		unsigned int m_resolution;
	public:
		Real getRadius() { return m_radius; }
		void setRadius(Real val);

	public:
		FUNCTION Real W(const Vector3d &r) const 
		{
			Real res = 0.0;
			const Real r2 = r.squaredNorm();
			if (r2 <= m_radius2)
			{
				const Real r = sqrt(r2);
				const unsigned int pos = (unsigned int)(r * m_invStepSize);
				res = m_W[pos];
			}
			return res;
		}

		FUNCTION Real W(const Real r) const
		{
			Real res = 0.0;
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
			const Real r2 = r.squaredNorm();
			if (r2 <= m_radius2)
			{
				const Real rl = sqrt(r2);
				const unsigned int pos = (unsigned int)(rl * m_invStepSize);
				res = m_gradW[pos] * r;
			}
			else
				res.setZero();

			return res;
		}

		FUNCTION inline Real W_zero() const
		{
			return m_W_zero;
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
#define MAX_NEIGHBOURS 60
		const Vector3d gravitation = Vector3d(0.0, -9.81, 0.0);

		//static size and value all time
		FUNCTION inline Real W(const Vector3d &r) const { return m_kernel_precomp.W(r); }
		FUNCTION inline Real W(const Real r) const { return m_kernel_precomp.W(r); }
		FUNCTION inline Vector3d gradW(const Vector3d &r) { return m_kernel_precomp.gradW(r); }
		Real W_zero;
		double density0;
		double particleRadius;
		double viscosity;

		//for boundaries everything should be completely static here
		Vector3d* posBoundary;
		Vector3d* velBoundary;
		Real* boundaryPsi;
		int numBoundaryParticles;

		//static size and values when pacticle count constant
		Real* mass;

		//static size but value dynamic
		Real* density;
		Vector3d* posFluid;
		Vector3d* velFluid;
		Vector3d* accFluid;
		int* numberOfNeighbourgs;
		int* neighbourgs;
		double* factor;

		//the V one is for the warm start
		Real* kappa;
		Real* kappaV;
		//for the terative solvers
		Real* densityAdv;

		int numFluidParticles;
		double h;
		double h_future;
		double h_past;

		DFSPHCData(FluidModel *model);

		void loadDynamicData(FluidModel *model, const SimulationDataDFSPH& data);
		void readDynamicData(FluidModel *model, SimulationDataDFSPH& data);
		void sortDynamicData(FluidModel *model);

		void reset(FluidModel *model);

		FUNCTION inline unsigned int getNeighbour(int particle_id, int neighbour_id, int body_id = 0) {
			return neighbourgs[body_id*numFluidParticles*MAX_NEIGHBOURS + particle_id * MAX_NEIGHBOURS + neighbour_id];
		}

		FUNCTION inline unsigned int getNumberOfNeighbourgs(int particle_id, int body_id = 0) {
			return numberOfNeighbourgs[body_id*numFluidParticles + particle_id];
		}
	};
}

#endif


