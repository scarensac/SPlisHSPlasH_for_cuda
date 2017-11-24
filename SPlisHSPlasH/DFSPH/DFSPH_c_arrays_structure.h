#ifndef __DFSPHCArraysStructure_h__
#define __DFSPHCArraysStructure_h__

#include "SPlisHSPlasH/SPHKernels.h"


namespace SPH
{
	class FluidModel;
	class SimulationDataDFSPH;

	struct DFSPHCData {
		//the size is 60 because I checked and the max neighbours I reached was 50
		//so I put 10 more to be sure. In the end those buffers will stay on the GPU memory
		//so there will be no transfers.
#define MAX_NEIGHBOURS 60
		const Vector3r gravitation=Vector3r(0.0, -9.81, 0.0);

		//static size and value all time
		Real W(const Vector3r &r) const { return CubicKernel::W(r); }
		Vector3r gradW(const Vector3r &r) { return CubicKernel::gradW(r); }
		Real W_zero;
		double density0;
		double particleRadius;
		double viscosity;

		//for boundaries everything should be completely static here
		Vector3r* posBoundary;
		Vector3r* velBoundary;
		Real* boundaryPsi;
		int numBoundaryParticles;

		//static size and values when pacticle count constant
		Real* mass;

		//static size but value dynamic
		Real* density;
		Vector3r* posFluid;
		Vector3r* velFluid;
		Vector3r* accFluid;
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

		DFSPHCData(FluidModel *model);

		void loadDynamicData(FluidModel *model, const SimulationDataDFSPH& data);
		void readDynamicData(FluidModel *model, SimulationDataDFSPH& data);

		void reset(FluidModel *model);

		FORCE_INLINE unsigned int getNeighbour(int particle_id, int neighbour_id, int body_id = 0) {
			return neighbourgs[body_id*numFluidParticles*MAX_NEIGHBOURS + particle_id * MAX_NEIGHBOURS + neighbour_id];
		}

		FORCE_INLINE unsigned int getNumberOfNeighbourgs(int particle_id, int body_id = 0) {
			return numberOfNeighbourgs[body_id*numFluidParticles + particle_id];
		}
	};
}

#endif
