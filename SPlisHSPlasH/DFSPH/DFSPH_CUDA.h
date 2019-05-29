#ifndef __DFSPH_CUDA_h__
#define __DFSPH_CUDA_h__

#include "SPlisHSPlasH/Common.h"

#ifdef SPLISHSPLASH_FRAMEWORK
#include "SPlisHSPlasH/TimeStep.h"
#include "SimulationDataDFSPH.h"
#endif //SPLISHSPLASH_FRAMEWORK
#include "SPlisHSPlasH/SPHKernels.h"
#include "SPlisHSPlasH/DFSPH/DFSPH_c_arrays_structure.h"

namespace SPH
{
class SimulationDataDFSPH;


/** \brief This class implements the Divergence-free Smoothed Particle Hydrodynamics approach introduced
    * by Bender and Koschier \cite Bender:2015, \cite Bender2017.
    */

#ifdef SPLISHSPLASH_FRAMEWORK
class DFSPHCUDA : public TimeStep
        #else
class DFSPHCUDA
        #endif //SPLISHSPLASH_FRAMEWORK
{
protected:
    DFSPHCData m_data;
    bool show_fluid_timings;

#ifdef SPLISHSPLASH_FRAMEWORK
    SimulationDataDFSPH m_simulationData;
#else
    //those are the members that come from TimeStep
    FluidModel *m_model;
    unsigned int m_iterations;
    unsigned int m_iterationsV;
    RealCuda m_maxError;
    unsigned int m_maxIterations;
    RealCuda m_maxErrorV;
    unsigned int m_maxIterationsV;

    //new members needed for the frameworks outside of splishsplash
    RealCuda desired_time_step;
#endif //SPLISHSPLASH_FRAMEWORK

    unsigned int m_counter;
    const Real m_eps = 1.0e-5;
    bool m_enableDivergenceSolver;

#ifdef SPLISHSPLASH_FRAMEWORK
    void computeDFSPHFactor();
    void pressureSolve();
    template <bool warm_start> void pressureSolveParticle(const unsigned int i);

    void divergenceSolve();
    template <bool warm_start> void divergenceSolveParticle(const unsigned int i);

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
#endif //SPLISHSPLASH_FRAMEWORK



public:
    DFSPHCUDA(FluidModel* model=NULL);
    virtual ~DFSPHCUDA(void);

    virtual void step();
    virtual void reset();

    bool getEnableDivergenceSolver() const { return m_enableDivergenceSolver; }
    void setEnableDivergenceSolver(bool val) { m_enableDivergenceSolver = val; }

    void checkReal(std::string txt, Real old_v, Real new_v);
    void checkVector3(std::string txt, Vector3d old_v, Vector3d new_v);

#ifdef SPLISHSPLASH_FRAMEWORK
    inline void checkVector3(std::string txt, Vector3r old_v, Vector3d new_v) { checkVector3(txt, vector3rTo3d(old_v), new_v); }
    inline void checkVector3(std::string txt, Vector3r old_v, Vector3r new_v) { checkVector3(txt, vector3rTo3d(old_v), vector3rTo3d(new_v)); }
#endif //SPLISHSPLASH_FRAMEWORK

    void renderFluid();
    void renderBoundaries(bool renderWalls);

    bool is_dynamic_bodies_paused;
    void handleDynamicBodiesPause(bool pause);
    void handleSimulationSave(bool save_liquid, bool save_solids, bool save_boundaries);
    void handleSimulationLoad(bool load_liquid, bool load_liquid_velocities, bool load_solids, bool load_solids_velocities,
                              bool load_boundaries, bool load_boundaries_velocities);

    void handleSimulationMovement(Vector3d movement);

    void handleFLuidLevelControl(RealCuda level);
    RealCuda getFluidLevel();

    void updateRigidBodiesStatefromFile();
    void updateRigidBodiesStateToFile();

    void updateRigidBodies(std::vector<DynamicBody> vect_new_info);

    void zeroFluidVelocities();


#ifndef SPLISHSPLASH_FRAMEWORK
    void updateTimeStepDuration(RealCuda duration);
#endif //SPLISHSPLASH_FRAMEWORK
    void forceUpdateRigidBodies();
    void getFluidImpactOnDynamicBodies(std::vector<SPH::Vector3d>& sph_forces, std::vector<SPH::Vector3d>& sph_moments,
                                       const std::vector<SPH::Vector3d>& reduction_factors);
    void getFluidBoyancyOnDynamicBodies(std::vector<SPH::Vector3d>& forces, std::vector<SPH::Vector3d>& pts_appli);
    SPH::Vector3d getSimulationCenter();
};
}

#endif
