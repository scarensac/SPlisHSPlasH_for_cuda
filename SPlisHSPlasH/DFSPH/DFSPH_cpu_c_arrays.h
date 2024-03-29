#ifndef __DFSPHCArrays_h__
#define __DFSPHCArrays_h__

#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/TimeStep.h"
#include "SimulationDataDFSPH.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SPlisHSPlasH/DFSPH/DFSPH_c_arrays_structure.h"

namespace SPH
{
	class SimulationDataDFSPH;


	/** \brief This class implements the Divergence-free Smoothed Particle Hydrodynamics approach introduced
	* by Bender and Koschier \cite Bender:2015, \cite Bender2017.
	*/
	class DFSPHCArrays : public TimeStep
	{
	protected:
		DFSPHCData m_data;

		SimulationDataDFSPH m_simulationData;
		unsigned int m_counter;
		const Real m_eps = 1.0e-5;
		bool m_enableDivergenceSolver;

		void computeDFSPHFactor();
		void pressureSolve();
		void divergenceSolve();
		void computeDensityAdv(const unsigned int index, const int numParticles, const Real h, const Real density0);
		void computeDensityChange(const unsigned int index, const Real h, const Real density0);

		/** Perform the neighborhood search for all fluid particles.
		*/
		virtual void performNeighborhoodSearch();
		virtual void emittedParticles(const unsigned int startIndex);


		//c array version of the other files functions
		/** Determine densities of all fluid particles.
		*/
		void computeDensities();
		void clearAccelerations();
		void computeNonPressureForces();
		void viscosity_XSPH();
		void surfaceTension_Akinci2013();

		void updateVelocities(Real h);
		void updatePositions(Real h);

		/** Update time step size by CFL condition.
		*/
		virtual void updateTimeStepSizeCFL(const Real minTimeStepSize);

	public:
		DFSPHCArrays(FluidModel *model);
		virtual ~DFSPHCArrays(void);

		virtual void step();
		virtual void reset();

		bool getEnableDivergenceSolver() const { return m_enableDivergenceSolver; }
		void setEnableDivergenceSolver(bool val) { m_enableDivergenceSolver = val; }

		void checkReal(std::string txt, Real old_v, Real new_v);
		void checkVector3(std::string txt, Vector3d old_v, Vector3d new_v);
		inline void checkVector3(std::string txt, Vector3r old_v, Vector3d new_v) { checkVector3(txt, vector3rTo3d(old_v), new_v); }
		inline void checkVector3(std::string txt, Vector3r old_v, Vector3r new_v) { checkVector3(txt, vector3rTo3d(old_v), vector3rTo3d(new_v)); }
	};
}

#endif
