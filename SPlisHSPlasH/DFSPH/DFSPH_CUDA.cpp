#include "DFSPH_CUDA.h"

#ifdef SPLISHSPLASH_FRAMEWORK
#include "SPlisHSPlasH/TimeManager.h"
#endif //SPLISHSPLASH_FRAMEWORK
#include "SPlisHSPlasH/SPHKernels.h"
#include <iostream>
#include "SPlisHSPlasH/Utilities/SegmentedTiming.h"
#include "SPlisHSPlasH/Utilities/Timing.h"
#include "DFSPH_cuda_basic.h"
#include <fstream>
#include <sstream>

// BENDER2019_BOUNDARIES includes
#include "SPlisHSPlasH/BoundaryModel_Bender2019.h"
#include "SPlisHSPlasH/StaticRigidBody.h"
#include "SPlisHSPlasH/Utilities/GaussQuadrature.h"
/*
#ifdef SPLISHSPLASH_FRAMEWORK
throw("DFSPHCData::readDynamicData must not be used outside of the SPLISHSPLASH framework");
#else
#endif //SPLISHSPLASH_FRAMEWORK
//*/

using namespace SPH;
using namespace std;

#define USE_WARMSTART
#define USE_WARMSTART_V



DFSPHCUDA::DFSPHCUDA(FluidModel *model) :
    #ifdef SPLISHSPLASH_FRAMEWORK
    TimeStep(model),
    m_simulationData(),
    #endif //SPLISHSPLASH_FRAMEWORK
    m_data(model)
{
#ifdef SPLISHSPLASH_FRAMEWORK
    m_simulationData.init(model);
#else
    m_model=model;
    m_iterations=0;
    m_maxError=0.01;
    m_maxIterations=100;
    m_maxErrorV=0.01;
    m_maxIterationsV=100;
    desired_time_step=m_data.get_current_timestep();
#endif //SPLISHSPLASH_FRAMEWORK
    m_counter = 0;
    m_iterationsV = 0;
    m_enableDivergenceSolver = true;
    is_dynamic_bodies_paused = false;
    show_fluid_timings=false;

#ifdef BENDER2019_BOUNDARIES
	m_boundaryModelBender2019 = new BoundaryModel_Bender2019();
	StaticRigidBody* rbo = dynamic_cast<StaticRigidBody*>(model->getRigidBodyParticleObject(0)->m_rigidBody);
	m_boundaryModelBender2019->initModel(rbo, model);
	SPH::TriangleMesh& mesh = rbo->getGeometry();
	initVolumeMap(m_boundaryModelBender2019);

	
#endif
}

DFSPHCUDA::~DFSPHCUDA(void)
{
}


void DFSPHCUDA::step()
{

    static int count_steps = 0;

	if (TimeManager::getCurrent()->getTime() > 0.5) {
		if ((count_steps % 32) == 0) {
			//handleSimulationMovement(Vector3d(1, 0, 0));
		}
	}

#ifdef SPLISHSPLASH_FRAMEWORK
    m_data.viscosity = m_viscosity->getViscosity();
#else
    m_data.viscosity = 0.02;
#endif //SPLISHSPLASH_FRAMEWORK



    if (true) {
        m_data.destructor_activated = false;

#ifdef OCEAN_BOUNDARIES_PROTOTYPE
    bool moving_borders = false;
    static int count_moving_steps = 0;
        //*


    //test dynamic boundary 
    if ((count_steps > 5) && ((count_steps % 15) == 0))
    {
        handleSimulationMovement(Vector3d(1,0,0));
        moving_borders = true;
        count_moving_steps++;
    }//*/
#endif
        /*
        if (count_steps == 0) {
            std::vector<Vector3d> additional_pos;
            std::vector<Vector3d> additional_vel;

            additional_pos.push_back(Vector3d(0.5, 2, 0.5));
            additional_vel.push_back(Vector3d(0, 0, 0));
            m_data.fluid_data->add_particles(additional_pos, additional_vel);
        }
        //*/

        static float time_avg = 0;
        static unsigned int time_count = 0;
#define NB_TIME_POINTS 9
        std::chrono::steady_clock::time_point tab_timepoint[NB_TIME_POINTS+1];
        std::string tab_name[NB_TIME_POINTS] = { "read dynamic bodies data", "neighbors","divergence", "viscosity","cfl","update vel","density","update pos","dynamic_borders" };
        static float tab_avg[NB_TIME_POINTS] = { 0 };
        int current_timepoint = 0;

        static std::chrono::steady_clock::time_point end;
        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();

        if (!is_dynamic_bodies_paused){
            m_data.loadDynamicObjectsData(m_model);
        }


        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();
        //*
        cuda_neighborsSearch(m_data);

		///TODO change the code so that the boundaries volumes and distances are conputed directly on the GPU
#ifdef BENDER2019_BOUNDARIES
		{

			RealCuda* V_rigids= new RealCuda[m_data.getFluidParticlesCount()];
			Vector3d* X_rigids= new Vector3d[m_data.getFluidParticlesCount()];

			//sadly I need to update the cpu storage positions as long as those calculations are done on cpu
			//TODO replace that by a simple read of the positions no need to update the damn model
			m_data.readDynamicData(m_model,m_simulationData);

			std::cout << m_data.dynamicWindowTotalDisplacement.x << std::endl;
			//do the actual conputation
			int numParticles = m_data.getFluidParticlesCount();
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++)
				{
					computeVolumeAndBoundaryX(i);

					X_rigids[i] = vector3rTo3d(m_boundaryModelBender2019->getBoundaryXj(0, i))+ m_data.dynamicWindowTotalDisplacement;
					V_rigids[i] = m_boundaryModelBender2019->getBoundaryVolume(0, i);;
				}
			}
			

			//now send that info to the GPU
			m_data.loadBender2019BoundariesFromCPU(V_rigids,X_rigids);


			/*
			for (int i = 0; i < (int)numParticles; i++)
			{
				std::cout << "particle " << i << 
					"  v // x // density  " <<
					V_rigids[i]<<"  //  "<< X_rigids[i].x << " " <<X_rigids[i].y << " "<< X_rigids[i].z << "  //  "<<
					std::endl;
			}
			//*/

			delete[](X_rigids);
			delete[](V_rigids);

			
		}
#endif

        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();


        if (m_enableDivergenceSolver)
        {
            m_iterationsV = cuda_divergenceSolve(m_data, m_maxIterationsV, m_maxErrorV);

			
        }
        else
        {
            m_iterationsV = 0;
        }


        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();


        cuda_externalForces(m_data);


        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();


        //cuda_CFL(m_data, 0.0001, m_cflFactor, m_cflMaxTimeStepSize);

#ifdef SPLISHSPLASH_FRAMEWORK
        const Real new_time_step = TimeManager::getCurrent()->getTimeStepSize();
#else
        const Real new_time_step = desired_time_step;
#endif //SPLISHSPLASH_FRAMEWORK
        m_data.updateTimeStep(new_time_step);

        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();


        cuda_update_vel(m_data);


        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();


        m_iterations = cuda_pressureSolve(m_data, m_maxIterations, m_maxError);

		if (false&&count_steps == 0) 	
		{
			/*
			//a simple test showing all the first particle neighbors in order

			{
				int test_particle = 0;
				std::cout << "particle id: " << test_particle << "  neighbors count: " << m_data.fluid_data->getNumberOfNeighbourgs(test_particle, 0) << "  " <<
					m_data.fluid_data->getNumberOfNeighbourgs(test_particle, 1) << "  " <<
					m_data.fluid_data->getNumberOfNeighbourgs(test_particle, 2) << std::endl;
				int numParticles = m_data.fluid_data->numParticles;
				int * end_ptr = m_data.fluid_data->getNeighboursPtr(test_particle);
				int * cur_particle = end_ptr;
				for (int k = 0; k < 3; ++k) {
#ifdef INTERLEAVE_NEIGHBORS
					end_ptr += m_data.fluid_data->getNumberOfNeighbourgs(test_particle, k)*numParticles;
					while (cur_particle != end_ptr) {
						std::cout << *cur_particle << std::endl;
						cur_particle += numParticles;
					}
#else
					end_ptr += m_data.fluid_data->getNumberOfNeighbourgs(test_particle, k);
					while (cur_particle != end_ptr) {
						std::cout << *cur_particle++ << std::endl;
					}
#endif
				}
			}

			//*/

			//this is only for debug purpose
			std::string filename = "boundaries density adv.csv";
				std::remove(filename.c_str());
			ofstream myfile;
			myfile.open(filename, std::ios_base::app);
			if (myfile.is_open()) {
				SPH::UnifiedParticleSet* set = m_data.fluid_data;
				for (int i = 0; i < set->numParticles; ++i) {
					myfile << i << ", " <<set->getNumberOfNeighbourgs(i,0) 
						<< ", " << set->getNumberOfNeighbourgs(i,1) 
						<< ", " << set->getNumberOfNeighbourgs(i,2)
						<< ", " << set->density[i]
						<< ", " << set->densityAdv[i] << std::endl;;

				}
				//myfile << total_time / (count_steps + 1) << ", " << m_iterations << ", " << m_iterationsV << std::endl;;
				myfile.close();
			}
			else {
				std::cout << "failed to open file: " << filename << "   reason: " << std::strerror(errno) << std::endl;
			}
		}
		

        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();


        cuda_update_pos(m_data);

        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();



        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();




        m_data.readDynamicObjectsData(m_model);
        m_data.onSimulationStepEnd();

        // Compute new time

#ifdef SPLISHSPLASH_FRAMEWORK
        TimeManager::getCurrent()->setTimeStepSize(m_data.h);
        TimeManager::getCurrent()->setTime(TimeManager::getCurrent()->getTime() + m_data.h);
#endif //SPLISHSPLASH_FRAMEWORK
        //*/



		//code for timming informations

        if ((current_timepoint-1) != NB_TIME_POINTS) {
            std::cout << "the number of time points does not correspond: "<< current_timepoint<<std::endl;
            exit(325);
        }

        float time_iter = std::chrono::duration_cast<std::chrono::nanoseconds> (tab_timepoint[NB_TIME_POINTS] - tab_timepoint[0]).count() / 1000000.0f;
        float time_between= std::chrono::duration_cast<std::chrono::nanoseconds> (tab_timepoint[0] - end).count() / 1000000.0f;
        static float total_time = 0;
        total_time += time_iter;


        static float iter_pressure_avg = 0;
        static float iter_divergence_avg = 0;
        iter_pressure_avg += m_iterations;
        iter_divergence_avg += m_iterationsV;

        if(show_fluid_timings){
            std::cout << "timestep total: " << total_time / (count_steps + 1) << "   this step: " << time_iter + time_between << "  (" << time_iter << "  " << time_between << ")" << std::endl;
            std::cout << "solver iteration avg (current step) density: " <<  iter_pressure_avg / (count_steps + 1) << " ( " << m_iterations << " )   divergence : " <<
                         iter_divergence_avg / (count_steps + 1) << " ( " << m_iterationsV << " )   divergence : " << std::endl;


            for (int i = 0; i < NB_TIME_POINTS; ++i) {
                float time = std::chrono::duration_cast<std::chrono::nanoseconds> (tab_timepoint[i+1] - tab_timepoint[i]).count() / 1000000.0f;
                tab_avg[i] += time;
                
                if (i == 8) {
                    std::cout << tab_name[i] << "  :" << ((count_moving_steps>0)?(tab_avg[i] / (count_moving_steps + 1)):0) << "  (" << time << ")" << std::endl;
                } else {
                    std::cout << tab_name[i] << "  :" << (tab_avg[i] / (count_steps + 1)) << "  (" << time << ")" << std::endl;
                }
            }
        }

		if (true) {
			if ((count_steps % 50) == 0) {
				std::cout << "time computation for "<< count_steps <<" steps: " << total_time << std::endl;
			}
		}


        if (false){
			static float real_time = 0;
			real_time += new_time_step;
            std::string filename = "timmings_detailled.csv";
            if (count_steps == 0) {
                std::remove(filename.c_str());
            }
            ofstream myfile;
            myfile.open(filename, std::ios_base::app);
            if (myfile.is_open()) {
                //myfile << total_time / (count_steps + 1) << ", " << m_iterations << ", " << m_iterationsV << std::endl;;
				myfile << real_time << ", " << time_iter << ", " << m_iterations << ", " << m_iterationsV << std::endl;;
				myfile.close();
            }
            else {
                std::cout << "failed to open file: " << filename << "   reason: " << std::strerror(errno) << std::endl;
            }
        }


		if (false) {
			if (count_steps > 1500) {
				count_steps = 0;
				total_time = 0;
				iter_pressure_avg = 0;
				iter_divergence_avg = 0;

				for (int i = 0; i < NB_TIME_POINTS; ++i) {
					tab_avg[i] = 0;
				}
			}
		}

        end = std::chrono::steady_clock::now();

        m_data.destructor_activated = true;

    }


#ifdef SPLISHSPLASH_FRAMEWORK
    if (false){
        //original code
        TimeManager *tm = TimeManager::getCurrent();
        const Real h = tm->getTimeStepSize();

        performNeighborhoodSearch();

        const unsigned int numParticles = m_model->numActiveParticles();

        computeDensities();

        START_TIMING("computeDFSPHFactor");
        computeDFSPHFactor();
        STOP_TIMING_AVG;

        if (m_enableDivergenceSolver)
        {
            START_TIMING("divergenceSolve");
            divergenceSolve();
            STOP_TIMING_AVG
        }
        else
            m_iterationsV = 0;

        // Compute accelerations: a(t)
        clearAccelerations();

        computeNonPressureForces();

        updateTimeStepSize();

#pragma omp parallel default(shared)
        {
#pragma omp for schedule(static)  
            for (int i = 0; i < (int)numParticles; i++)
            {
                Vector3r &vel = m_model->getVelocity(0, i);
                vel += h * m_model->getAcceleration(i);
            }
        }

        START_TIMING("pressureSolve");
        pressureSolve();
        STOP_TIMING_AVG;

#pragma omp parallel default(shared)
        {
#pragma omp for schedule(static)  
            for (int i = 0; i < (int)numParticles; i++)
            {
                Vector3r &xi = m_model->getPosition(0, i);
                const Vector3r &vi = m_model->getVelocity(0, i);
                xi += h * vi;
            }
        }

        emitParticles();

        tm->setTime(tm->getTime() + h);
    }
#endif //SPLISHSPLASH_FRAMEWORK




    if(show_fluid_timings){
        static int true_count_steps = 0;
        std::cout << "step finished: " << true_count_steps<<"  "<< count_steps << std::endl;
		true_count_steps++;
    }
	count_steps++;
}

void DFSPHCUDA::reset()
{
#ifdef SPLISHSPLASH_FRAMEWORK
    TimeStep::reset();
    m_simulationData.reset();
#else
    m_iterations = 0;
#endif //SPLISHSPLASH_FRAMEWORK

    m_counter = 0;
    m_iterationsV = 0;
    m_data.reset(m_model);

    /*
    std::vector<Vector3d> additional_pos;
    std::vector<Vector3d> additional_vel;

    additional_pos.push_back(Vector3d(0.5, 2, 0.5));
    additional_vel.push_back(Vector3d(0, 1, 0));
    m_data.fluid_data->add_particles(additional_pos, additional_vel);
    //*/
}


#ifdef SPLISHSPLASH_FRAMEWORK

void DFSPHCUDA::computeDFSPHFactor()
{
    //////////////////////////////////////////////////////////////////////////
    // Init parameters
    //////////////////////////////////////////////////////////////////////////

    const Real h = TimeManager::getCurrent()->getTimeStepSize();
    const int numParticles = (int)m_model->numActiveParticles();

#pragma omp parallel default(shared)
    {
        //////////////////////////////////////////////////////////////////////////
        // Compute pressure stiffness denominator
        //////////////////////////////////////////////////////////////////////////

#pragma omp for schedule(static)  
        for (int i = 0; i < numParticles; i++)
        {
            //////////////////////////////////////////////////////////////////////////
            // Compute gradient dp_i/dx_j * (1/k)  and dp_j/dx_j * (1/k)
            //////////////////////////////////////////////////////////////////////////
            const Vector3r &xi = m_model->getPosition(0, i);
            Real sum_grad_p_k = 0.0;
            Vector3r grad_p_i;
            grad_p_i.setZero();

            //////////////////////////////////////////////////////////////////////////
            // Fluid
            //////////////////////////////////////////////////////////////////////////
            for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
            {
                const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
                const Vector3r &xj = m_model->getPosition(0, neighborIndex);
                const Vector3r grad_p_j = -m_model->getMass(neighborIndex) * m_model->gradW(xi - xj);
                sum_grad_p_k += grad_p_j.squaredNorm();
                grad_p_i -= grad_p_j;
            }

            //////////////////////////////////////////////////////////////////////////
            // Boundary
            //////////////////////////////////////////////////////////////////////////
            for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
            {
                for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, i); j++)
                {
                    const unsigned int neighborIndex = m_model->getNeighbor(pid, i, j);
                    const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
                    const Vector3r grad_p_j = -m_model->getBoundaryPsi(pid, neighborIndex) * m_model->gradW(xi - xj);
                    sum_grad_p_k += grad_p_j.squaredNorm();
                    grad_p_i -= grad_p_j;
                }
            }

            sum_grad_p_k += grad_p_i.squaredNorm();

            //////////////////////////////////////////////////////////////////////////
            // Compute pressure stiffness denominator
            //////////////////////////////////////////////////////////////////////////
            Real &factor = m_simulationData.getFactor(i);

            sum_grad_p_k = max(sum_grad_p_k, m_eps);
            factor = -1.0 / (sum_grad_p_k);
        }
    }
}

void DFSPHCUDA::pressureSolve()
{
    const Real h = TimeManager::getCurrent()->getTimeStepSize();
    const Real h2 = h*h;
    const Real invH = 1.0 / h;
    const Real invH2 = 1.0 / h2;
    const Real density0 = m_model->getDensity0();
    const int numParticles = (int)m_model->numActiveParticles();
    Real avg_density_err = 0.0;

#ifdef USE_WARMSTART			
#pragma omp parallel default(shared)
    {
        //////////////////////////////////////////////////////////////////////////
        // Divide by h^2, the time step size has been removed in
        // the last step to make the stiffness value independent
        // of the time step size
        //////////////////////////////////////////////////////////////////////////
#pragma omp for schedule(static)  
        for (int i = 0; i < (int)numParticles; i++)
        {
            m_simulationData.getKappa(i) = MAX_MACRO(m_simulationData.getKappa(i)*invH2, -0.5);
            //computeDensityAdv(i, numParticles, h, density0);
        }

        //////////////////////////////////////////////////////////////////////////
        // Predict v_adv with external velocities
        //////////////////////////////////////////////////////////////////////////

#pragma omp for schedule(static)  
        for (int i = 0; i < numParticles; i++)
        {
            //if (m_simulationData.getDensityAdv(i) > density0)
            {
                Vector3r &vel = m_model->getVelocity(0, i);
                const Real ki = m_simulationData.getKappa(i);
                const Vector3r &xi = m_model->getPosition(0, i);

                //////////////////////////////////////////////////////////////////////////
                // Fluid
                //////////////////////////////////////////////////////////////////////////
                for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
                {
                    const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
                    const Real kj = m_simulationData.getKappa(neighborIndex);

                    const Real kSum = (ki + kj);
                    if (fabs(kSum) > m_eps)
                    {
                        const Vector3r &xj = m_model->getPosition(0, neighborIndex);
                        const Vector3r grad_p_j = -m_model->getMass(neighborIndex) * m_model->gradW(xi - xj);
                        vel -= h * kSum * grad_p_j;					// ki, kj already contain inverse density
                    }
                }

                //////////////////////////////////////////////////////////////////////////
                // Boundary
                //////////////////////////////////////////////////////////////////////////
                if (fabs(ki) > m_eps)
                {
                    for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
                    {
                        for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, i); j++)
                        {
                            const unsigned int neighborIndex = m_model->getNeighbor(pid, i, j);
                            const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
                            const Vector3r grad_p_j = -m_model->getBoundaryPsi(pid, neighborIndex) * m_model->gradW(xi - xj);
                            const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
                            vel += velChange;

                            m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * velChange * invH;
                        }
                    }
                }
            }
        }
    }
#endif

    //////////////////////////////////////////////////////////////////////////
    // Compute rho_adv
    //////////////////////////////////////////////////////////////////////////
#pragma omp parallel default(shared)
    {
#pragma omp for schedule(static)  
        for (int i = 0; i < numParticles; i++)
        {
            computeDensityAdv(i, numParticles, h, density0);
            m_simulationData.getFactor(i) *= invH2;
#ifdef USE_WARMSTART
            m_simulationData.getKappa(i) = 0.0;
#endif
        }
    }

    m_iterations = 0;

    //////////////////////////////////////////////////////////////////////////
    // Start solver
    //////////////////////////////////////////////////////////////////////////

    // Maximal allowed density fluctuation
    const Real eta = m_maxError * 0.01 * density0;  // maxError is given in percent

    while (((avg_density_err > eta) || (m_iterations < 2)) && (m_iterations < m_maxIterations))
    {
        avg_density_err = 0.0;

#pragma omp parallel default(shared)
        {
            //////////////////////////////////////////////////////////////////////////
            // Compute pressure forces
            //////////////////////////////////////////////////////////////////////////
#pragma omp for schedule(static) 
            for (int i = 0; i < numParticles; i++)
            {
                //////////////////////////////////////////////////////////////////////////
                // Evaluate rhs
                //////////////////////////////////////////////////////////////////////////
                const Real b_i = m_simulationData.getDensityAdv(i) - density0;
                const Real ki = b_i*m_simulationData.getFactor(i);
#ifdef USE_WARMSTART
                m_simulationData.getKappa(i) += ki;
#endif

                Vector3r &v_i = m_model->getVelocity(0, i);
                const Vector3r &xi = m_model->getPosition(0, i);

                //////////////////////////////////////////////////////////////////////////
                // Fluid
                //////////////////////////////////////////////////////////////////////////
                for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
                {
                    const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
                    const Real b_j = m_simulationData.getDensityAdv(neighborIndex) - density0;
                    const Real kj = b_j*m_simulationData.getFactor(neighborIndex);
                    const Real kSum = (ki + kj);
                    if (fabs(kSum) > m_eps)
                    {
                        const Vector3r &xj = m_model->getPosition(0, neighborIndex);
                        const Vector3r grad_p_j = -m_model->getMass(neighborIndex) * m_model->gradW(xi - xj);

                        // Directly update velocities instead of storing pressure accelerations
                        v_i -= h * kSum * grad_p_j;			// ki, kj already contain inverse density
                    }
                }

                //////////////////////////////////////////////////////////////////////////
                // Boundary
                //////////////////////////////////////////////////////////////////////////
                if (fabs(ki) > m_eps)
                {
                    for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
                    {
                        for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, i); j++)
                        {
                            const unsigned int neighborIndex = m_model->getNeighbor(pid, i, j);
                            const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
                            const Vector3r grad_p_j = -m_model->getBoundaryPsi(pid, neighborIndex) * m_model->gradW(xi - xj);

                            // Directly update velocities instead of storing pressure accelerations
                            const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
                            v_i += velChange;

                            m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * velChange * invH;
                        }
                    }
                }
            }


            //////////////////////////////////////////////////////////////////////////
            // Update rho_adv and density error
            //////////////////////////////////////////////////////////////////////////
#pragma omp for reduction(+:avg_density_err) schedule(static) 
            for (int i = 0; i < numParticles; i++)
            {
                computeDensityAdv(i, numParticles, h, density0);

                const Real density_err = m_simulationData.getDensityAdv(i) - density0;
                avg_density_err += density_err;
            }
        }

        avg_density_err /= numParticles;

        m_iterations++;
    }


#ifdef USE_WARMSTART
    //////////////////////////////////////////////////////////////////////////
    // Multiply by h^2, the time step size has to be removed
    // to make the stiffness value independent
    // of the time step size
    //////////////////////////////////////////////////////////////////////////
    for (int i = 0; i < numParticles; i++)
        m_simulationData.getKappa(i) *= h2;
#endif
}

template <bool warm_start>
void  DFSPHCUDA::pressureSolveParticle(const unsigned int i) {

    //////////////////////////////////////////////////////////////////////////
    // Evaluate rhs
    //////////////////////////////////////////////////////////////////////////
    const Real ki = (warm_start) ? m_data.kappa[i] : (m_data.densityAdv[i])*m_data.factor[i];

#ifdef USE_WARMSTART
    if (!warm_start) { m_data.kappa[i] += ki; }
#endif


    Vector3d v_i = Vector3d(0, 0, 0);
    const Vector3d &xi = m_data.posFluid[i];

    //////////////////////////////////////////////////////////////////////////
    // Fluid
    //////////////////////////////////////////////////////////////////////////
    for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i); j++)
    {
        const unsigned int neighborIndex = m_data.getNeighbour(i, j);
        const Real kSum = (ki + ((warm_start) ? m_data.kappa[neighborIndex] : (m_data.densityAdv[neighborIndex])*m_data.factor[neighborIndex]));
        if (fabs(kSum) > m_eps)
        {
            // ki, kj already contain inverse density
            v_i += kSum * m_data.mass[neighborIndex] * m_data.gradW(xi - m_data.posFluid[neighborIndex]);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // Boundary
    //////////////////////////////////////////////////////////////////////////
    if (fabs(ki) > m_eps)
    {
        for (unsigned int pid = 1; pid < 2; pid++)
        {
            for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i, pid); j++)
            {
                const unsigned int neighborIndex = m_data.getNeighbour(i, j, pid);
                const Vector3d delta = ki * m_data.boundaryPsi[neighborIndex] * m_data.gradW(xi - m_data.posBoundary[neighborIndex]);

                v_i += delta;// ki already contains inverse density

                ///TODO reactivate the external forces check the original formula to be sure of the sign
                //m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * ki * grad_p_j;
            }
        }
    }
    // Directly update velocities instead of storing pressure accelerations
    m_data.velFluid[i] += v_i*m_data.h;

}

void DFSPHCUDA::divergenceSolve()
{
    //////////////////////////////////////////////////////////////////////////
    // Init parameters
    //////////////////////////////////////////////////////////////////////////

    const Real h = TimeManager::getCurrent()->getTimeStepSize();
    const Real invH = 1.0 / h;
    const int numParticles = (int)m_model->numActiveParticles();
    const unsigned int maxIter = m_maxIterationsV;
    const Real maxError = m_maxErrorV;
    const Real density0 = m_model->getDensity0();


#ifdef USE_WARMSTART_V
#pragma omp parallel default(shared)
    {
        //////////////////////////////////////////////////////////////////////////
        // Divide by h^2, the time step size has been removed in
        // the last step to make the stiffness value independent
        // of the time step size
        //////////////////////////////////////////////////////////////////////////
#pragma omp for schedule(static)  
        for (int i = 0; i < numParticles; i++)
        {
            m_simulationData.getKappaV(i) = 0.5*MAX_MACRO(m_simulationData.getKappaV(i)*invH, -0.5);
            computeDensityChange(i, h, density0);
        }

#pragma omp for schedule(static)  
        for (int i = 0; i < (int)numParticles; i++)
        {
            if (m_simulationData.getDensityAdv(i) > 0.0)
            {
                Vector3r &vel = m_model->getVelocity(0, i);
                const Real ki = m_simulationData.getKappaV(i);
                const Vector3r &xi = m_model->getPosition(0, i);

                //////////////////////////////////////////////////////////////////////////
                // Fluid
                //////////////////////////////////////////////////////////////////////////
                for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
                {
                    const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
                    const Real kj = m_simulationData.getKappaV(neighborIndex);

                    const Real kSum = (ki + kj);
                    if (fabs(kSum) > m_eps)
                    {
                        const Vector3r &xj = m_model->getPosition(0, neighborIndex);
                        const Vector3r grad_p_j = -m_model->getMass(neighborIndex) * m_model->gradW(xi - xj);
                        vel -= h * kSum * grad_p_j;					// ki, kj already contain inverse density
                    }
                }

                //////////////////////////////////////////////////////////////////////////
                // Boundary
                //////////////////////////////////////////////////////////////////////////
                if (fabs(ki) > m_eps)
                {
                    for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
                    {
                        for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, i); j++)
                        {
                            const unsigned int neighborIndex = m_model->getNeighbor(pid, i, j);
                            const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
                            const Vector3r grad_p_j = -m_model->getBoundaryPsi(pid, neighborIndex) * m_model->gradW(xi - xj);

                            const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
                            vel += velChange;

                            m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * velChange * invH;
                        }
                    }
                }
            }
        }
    }
#endif

    //////////////////////////////////////////////////////////////////////////
    // Compute velocity of density change
    //////////////////////////////////////////////////////////////////////////
#pragma omp parallel default(shared)
    {
#pragma omp for schedule(static)  
        for (int i = 0; i < (int)numParticles; i++)
        {
            computeDensityChange(i, h, density0);
            m_simulationData.getFactor(i) *= invH;

#ifdef USE_WARMSTART_V
            m_simulationData.getKappaV(i) = 0.0;
#endif
        }
    }

    m_iterationsV = 0;

    //////////////////////////////////////////////////////////////////////////
    // Start solver
    //////////////////////////////////////////////////////////////////////////

    // Maximal allowed density fluctuation
    // use maximal density error divided by time step size
    const Real eta = (1.0 / h) * maxError * 0.01 * density0;  // maxError is given in percent

    Real avg_density_err = 0.0;
    while (((avg_density_err > eta) || (m_iterationsV < 1)) && (m_iterationsV < maxIter))
    {
        avg_density_err = 0.0;

        //////////////////////////////////////////////////////////////////////////
        // Perform Jacobi iteration over all blocks
        //////////////////////////////////////////////////////////////////////////
#pragma omp parallel default(shared)
        {
#pragma omp for schedule(static) 
            for (int i = 0; i < (int)numParticles; i++)
            {
                //////////////////////////////////////////////////////////////////////////
                // Evaluate rhs
                //////////////////////////////////////////////////////////////////////////
                const Real b_i = m_simulationData.getDensityAdv(i);
                const Real ki = b_i*m_simulationData.getFactor(i);
#ifdef USE_WARMSTART_V
                m_simulationData.getKappaV(i) += ki;
#endif

                Vector3r &v_i = m_model->getVelocity(0, i);

                const Vector3r &xi = m_model->getPosition(0, i);

                //////////////////////////////////////////////////////////////////////////
                // Fluid
                //////////////////////////////////////////////////////////////////////////
                for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
                {
                    const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
                    const Real b_j = m_simulationData.getDensityAdv(neighborIndex);
                    const Real kj = b_j*m_simulationData.getFactor(neighborIndex);

                    const Real kSum = (ki + kj);
                    if (fabs(kSum) > m_eps)
                    {
                        const Vector3r &xj = m_model->getPosition(0, neighborIndex);
                        const Vector3r grad_p_j = -m_model->getMass(neighborIndex) * m_model->gradW(xi - xj);
                        v_i -= h * kSum * grad_p_j;			// ki, kj already contain inverse density
                    }
                }

                //////////////////////////////////////////////////////////////////////////
                // Boundary
                //////////////////////////////////////////////////////////////////////////
                if (fabs(ki) > m_eps)
                {
                    for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
                    {
                        for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, i); j++)
                        {
                            const unsigned int neighborIndex = m_model->getNeighbor(pid, i, j);
                            const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
                            const Vector3r grad_p_j = -m_model->getBoundaryPsi(pid, neighborIndex) * m_model->gradW(xi - xj);

                            const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
                            v_i += velChange;

                            m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * velChange * invH;
                        }
                    }
                }
            }

            //////////////////////////////////////////////////////////////////////////
            // Update rho_adv and density error
            //////////////////////////////////////////////////////////////////////////
#pragma omp for reduction(+:avg_density_err) schedule(static) 
            for (int i = 0; i < (int)numParticles; i++)
            {
                computeDensityChange(i, h, density0);
                avg_density_err += m_simulationData.getDensityAdv(i);
            }
        }

        avg_density_err /= numParticles;
        m_iterationsV++;
    }

#ifdef USE_WARMSTART_V
    //////////////////////////////////////////////////////////////////////////
    // Multiply by h, the time step size has to be removed
    // to make the stiffness value independent
    // of the time step size
    //////////////////////////////////////////////////////////////////////////
    for (int i = 0; i < numParticles; i++)
        m_simulationData.getKappaV(i) *= h;
#endif

    for (int i = 0; i < numParticles; i++)
    {
        m_simulationData.getFactor(i) *= h;
    }
}

template <bool warm_start>
void DFSPHCUDA::divergenceSolveParticle(const unsigned int i) {
    Vector3d v_i = Vector3d(0, 0, 0);
    //////////////////////////////////////////////////////////////////////////
    // Evaluate rhs
    //////////////////////////////////////////////////////////////////////////
    const Real ki = (warm_start) ? m_data.kappaV[i] : (m_data.densityAdv[i])*m_data.factor[i];

#ifdef USE_WARMSTART_V
    if (!warm_start) { m_data.kappaV[i] += ki; }
#endif

    const Vector3d &xi = m_data.posFluid[i];


    //////////////////////////////////////////////////////////////////////////
    // Fluid
    //////////////////////////////////////////////////////////////////////////
    for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i); j++)
    {
        const unsigned int neighborIndex = m_data.getNeighbour(i, j);
        const Real kSum = (ki + ((warm_start) ? m_data.kappaV[neighborIndex] : (m_data.densityAdv[neighborIndex])*m_data.factor[neighborIndex]));
        if (fabs(kSum) > m_eps)
        {
            // ki, kj already contain inverse density
            v_i += kSum *  m_data.mass[neighborIndex] * m_data.gradW(xi - m_data.posFluid[neighborIndex]);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // Boundary
    //////////////////////////////////////////////////////////////////////////
    if (fabs(ki) > m_eps)
    {
        for (unsigned int pid = 1; pid < 2; pid++)
        {
            for (unsigned int j = 0; j < m_data.getNumberOfNeighbourgs(i, pid); j++)
            {
                const unsigned int neighborIndex = m_data.getNeighbour(i, j, pid);
                ///TODO fuse those lines
                const Vector3d delta = ki * m_data.boundaryPsi[neighborIndex] * m_data.gradW(xi - m_data.posBoundary[neighborIndex]);
                v_i += delta;// ki already contains inverse density

                ///TODO reactivate this for objects see theoriginal sign to see the the actual sign
                //m_model->getForce(pid, neighborIndex) -= m_model->getMass(i) * ki * grad_p_j;
            }
        }
    }

    m_data.velFluid[i] += v_i*m_data.h;
}

void DFSPHCUDA::computeDensityAdv(const unsigned int index, const int numParticles, const Real h, const Real density0)
{
    const Real &density = m_model->getDensity(index);
    Real &densityAdv = m_simulationData.getDensityAdv(index);
    const Vector3r &xi = m_model->getPosition(0, index);
    const Vector3r &vi = m_model->getVelocity(0, index);
    Real delta = 0.0;

    //////////////////////////////////////////////////////////////////////////
    // Fluid
    //////////////////////////////////////////////////////////////////////////
    for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, index); j++)
    {
        const unsigned int neighborIndex = m_model->getNeighbor(0, index, j);
        const Vector3r &xj = m_model->getPosition(0, neighborIndex);
        const Vector3r &vj = m_model->getVelocity(0, neighborIndex);
        delta += m_model->getMass(neighborIndex) * (vi - vj).dot(m_model->gradW(xi - xj));
    }

    //////////////////////////////////////////////////////////////////////////
    // Boundary
    //////////////////////////////////////////////////////////////////////////
    for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
    {
        for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, index); j++)
        {
            const unsigned int neighborIndex = m_model->getNeighbor(pid, index, j);
            const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
            const Vector3r &vj = m_model->getVelocity(pid, neighborIndex);
            delta += m_model->getBoundaryPsi(pid, neighborIndex) * (vi - vj).dot(m_model->gradW(xi - xj));
        }
    }

    densityAdv = density + h*delta;
    densityAdv = max(densityAdv, density0);
}

void DFSPHCUDA::computeDensityChange(const unsigned int index, const Real h, const Real density0)
{
    Real &densityAdv = m_simulationData.getDensityAdv(index);
    const Vector3r &xi = m_model->getPosition(0, index);
    const Vector3r &vi = m_model->getVelocity(0, index);
    densityAdv = 0.0;
    unsigned int numNeighbors = m_model->numberOfNeighbors(0, index);

    //////////////////////////////////////////////////////////////////////////
    // Fluid
    //////////////////////////////////////////////////////////////////////////
    for (unsigned int j = 0; j < numNeighbors; j++)
    {
        const unsigned int neighborIndex = m_model->getNeighbor(0, index, j);
        const Vector3r &xj = m_model->getPosition(0, neighborIndex);
        const Vector3r &vj = m_model->getVelocity(0, neighborIndex);
        densityAdv += m_model->getMass(neighborIndex) * (vi - vj).dot(m_model->gradW(xi - xj));
    }

    //////////////////////////////////////////////////////////////////////////
    // Boundary
    //////////////////////////////////////////////////////////////////////////
    for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
    {
        numNeighbors += m_model->numberOfNeighbors(pid, index);
        for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, index); j++)
        {
            const unsigned int neighborIndex = m_model->getNeighbor(pid, index, j);
            const Vector3r &xj = m_model->getPosition(pid, neighborIndex);
            const Vector3r &vj = m_model->getVelocity(pid, neighborIndex);
            densityAdv += m_model->getBoundaryPsi(pid, neighborIndex) * (vi - vj).dot(m_model->gradW(xi - xj));
        }
    }

    // only correct positive divergence
    densityAdv = MAX_MACRO(densityAdv, 0.0);

    // in case of particle deficiency do not perform a divergence solve
    if (numNeighbors < 20)
        densityAdv = 0.0;
}

void DFSPHCUDA::performNeighborhoodSearch()
{
    if (m_counter % 500 == 0)
    {
        //m_model->performNeighborhoodSearchSort();
        //m_simulationData.performNeighborhoodSearchSort();
        //TimeStep::performNeighborhoodSearchSort();
    }
    m_counter++;

    TimeStep::performNeighborhoodSearch();
}

void DFSPHCUDA::emittedParticles(const unsigned int startIndex)
{
    m_simulationData.emittedParticles(startIndex);
    TimeStep::emittedParticles(startIndex);
}

void DFSPHCUDA::computeDensities()
{
    const unsigned int numParticles = m_model->numActiveParticles();

#pragma omp parallel default(shared)
    {
#pragma omp for schedule(static)  
        for (int i = 0; i < (int)numParticles; i++)
        {
            Real &density = m_model->getDensity(i);

            // Compute current density for particle i
            density = m_model->getMass(i) * m_model->W_zero();
            const Vector3r &xi = m_model->getPosition(0, i);

            //////////////////////////////////////////////////////////////////////////
            // Fluid
            //////////////////////////////////////////////////////////////////////////
            for (unsigned int j = 0; j < m_model->numberOfNeighbors(0, i); j++)
            {
                const unsigned int neighborIndex = m_model->getNeighbor(0, i, j);
                const Vector3r &xj = m_model->getPosition(0, neighborIndex);
                density += m_model->getMass(neighborIndex) * m_model->W(xi - xj);
            }

            //////////////////////////////////////////////////////////////////////////
            // Boundary
            //////////////////////////////////////////////////////////////////////////
            for (unsigned int pid = 1; pid < m_model->numberOfPointSets(); pid++)
            {
                for (unsigned int j = 0; j < m_model->numberOfNeighbors(pid, i); j++)
                {
                    const unsigned int neighborIndex = m_model->getNeighbor(pid, i, j);
                    const Vector3r &xj = m_model->getPosition(pid, neighborIndex);

                    // Boundary: Akinci2012
                    density += m_model->getBoundaryPsi(pid, neighborIndex) * m_model->W(xi - xj);
                }
            }
        }
    }
}

void DFSPHCUDA::clearAccelerations()
{
    const unsigned int count = m_model->numActiveParticles();
    const Vector3r &grav = m_model->getGravitation();
    for (unsigned int i = 0; i < count; i++)
    {
        // Clear accelerations of dynamic particles
        if (m_model->getMass(i) != 0.0)
        {
            m_model->getAcceleration(i) = grav;
        }
    }
}

void DFSPHCUDA::computeNonPressureForces()
{
    START_TIMING("computeNonPressureForces");
    computeSurfaceTension();
    computeViscosity();
    computeVorticity();
    computeDragForce();
    STOP_TIMING_AVG;

}

void DFSPHCUDA::viscosity_XSPH()
{

    const Real h = m_data.h;
    const Real invH = (1.0 / h);

    // Compute viscosity forces (XSPH)
}

void DFSPHCUDA::surfaceTension_Akinci2013()
{
    throw("go look for the code but there are multiples funtion to copy so ...");
}

void DFSPHCUDA::computeVolumeAndBoundaryX(const unsigned int i)
{
	//I leave those just to make the transition easier to code
	const unsigned int nFluids = 1;
	const unsigned int nBoundaries = 1;
	const bool sim2D = false;
	const Real supportRadius = m_model->getSupportRadius();
	const Real particleRadius = m_model->getParticleRadius();
	const Real dt = TimeManager::getCurrent()->getTimeStepSize();
	int fluidModelIndex = 0;
	Vector3r xi = m_model->getPosition(0, i)-vector3dTo3r(m_data.dynamicWindowTotalDisplacement);
	//*

	for (unsigned int pid = 0; pid < nBoundaries; pid++)
	{
		BoundaryModel_Bender2019* bm = m_boundaryModelBender2019;

		Vector3r& boundaryXj = bm->getBoundaryXj(fluidModelIndex, i);
		boundaryXj.setZero();
		Real& boundaryVolume = bm->getBoundaryVolume(fluidModelIndex, i);
		boundaryVolume = 0.0;

		const Vector3r& t = bm->getRigidBodyObject()->getPosition();
		const Matrix3r& R = bm->getRigidBodyObject()->getRotation();

		Eigen::Vector3d normal;
		const Eigen::Vector3d localXi = (R.transpose() * (xi - t)).cast<double>();

		std::array<unsigned int, 32> cell;
		Eigen::Vector3d c0;
		Eigen::Matrix<double, 32, 1> N;
#ifdef USE_FD_NORMAL
		bool chk = bm->getMap()->determineShapeFunctions(0, localXi, cell, c0, N);
#else
		Eigen::Matrix<double, 32, 3> dN;
		bool chk = bm->getMap()->determineShapeFunctions(0, localXi, cell, c0, N, &dN);
#endif
		Real dist = numeric_limits<Real>::max();
		if (chk)
#ifdef USE_FD_NORMAL
			dist = static_cast<Real>(bm->getMap()->interpolate(0, localXi, cell, c0, N));
#else
			dist = static_cast<Real>(bm->getMap()->interpolate(0, localXi, cell, c0, N, &normal, &dN));
#endif
		
		if ((dist > 0.1 * particleRadius) && (dist < supportRadius))
		{
			const Real volume = static_cast<Real>(bm->getMap()->interpolate(1, localXi, cell, c0, N));
			if ((volume > 1e-6) && (volume != numeric_limits<Real>::max()))
			{
				boundaryVolume = volume;

#ifdef USE_FD_NORMAL
				if (sim2D)
					approximateNormal(bm->getMap(), localXi, normal, 2);
				else
					approximateNormal(bm->getMap(), localXi, normal, 3);
#endif
				normal = R.cast<double>() * normal;
				const double nl = normal.norm();
				if (nl > 1.0e-6)
				{
					normal /= nl;
					boundaryXj = (xi - dist * normal.cast<Real>());
				}
				else
				{
					boundaryVolume = 0.0;
				}
			}
			else
			{
				boundaryVolume = 0.0;
			}
		}
		else if (dist <= 0.1 * particleRadius)
		{
			normal = R.cast<double>() * normal;
			const double nl = normal.norm();
			if (nl > 1.0e-6)
			{
				std::string msg = "If this is triggered it means a particle was too close to the border so you'll need to reactivate that code I'm not sure works";
				std::cout<<msg<<std::endl;
				throw(msg);
				/*
				normal /= nl;
				// project to surface
				Real d = -dist;
				d = std::min(d, static_cast<Real>(0.25 / 0.005)* particleRadius* dt);		// get up in small steps
				sim->getFluidModel(fluidModelIndex)->getPosition(i) = (xi + d * normal.cast<Real>());
				// adapt velocity in normal direction
				sim->getFluidModel(fluidModelIndex)->getVelocity(i) += (0.05 - sim->getFluidModel(fluidModelIndex)->getVelocity(i).dot(normal.cast<Real>())) * normal.cast<Real>();
				//*/
			}
			boundaryVolume = 0.0;
		}
		else
		{
			boundaryVolume = 0.0;
		}
	}
     //*/
}


void DFSPHCUDA::initVolumeMap(BoundaryModel_Bender2019* boundaryModel) {
	StaticRigidBody* rbo = dynamic_cast<StaticRigidBody*>(boundaryModel->getRigidBodyObject());
	SPH::TriangleMesh& mesh = rbo->getGeometry();
	std::vector<Vector3r>& x= mesh.getVertices();
	std::vector<unsigned int>& faces= mesh.getFaces();

	const Real supportRadius = m_model->getSupportRadius();
	Discregrid::CubicLagrangeDiscreteGrid* volumeMap;

	
	
	//////////////////////////////////////////////////////////////////////////
	// Generate distance field of object using Discregrid
	//////////////////////////////////////////////////////////////////////////
#ifdef USE_DOUBLE_CUDA
	Discregrid::TriangleMesh sdfMesh(&x[0][0], faces.data(), x.size(), faces.size() / 3);
#else
	// if type is float, copy vector to double vector
	std::vector<double> doubleVec;
	doubleVec.resize(3 * x.size());
	for (unsigned int i = 0; i < x.size(); i++)
		for (unsigned int j = 0; j < 3; j++)
			doubleVec[3 * i + j] = x[i][j];
	Discregrid::TriangleMesh sdfMesh(&doubleVec[0], faces.data(), x.size(), faces.size() / 3);
#endif

	Discregrid::MeshDistance md(sdfMesh);
	Eigen::AlignedBox3d domain;
	for (auto const& x_ : x)
	{
		domain.extend(x_.cast<double>());
	}
	const Real tolerance = 0.0;///TODO set that as a parameter the current valu is just the one I read from the current project github
	domain.max() += (4.0 * supportRadius + tolerance) * Eigen::Vector3d::Ones();
	domain.min() -= (4.0 * supportRadius + tolerance) * Eigen::Vector3d::Ones();

	std::cout << "Domain - min: " << domain.min()[0] << ", " << domain.min()[1] << ", " << domain.min()[2] << std::endl;
	std::cout << "Domain - max: " << domain.max()[0] << ", " << domain.max()[1] << ", " << domain.max()[2] << std::endl;

	Eigen::Matrix<unsigned int, 3, 1> resolutionSDF = Eigen::Matrix<unsigned int, 3, 1>(40, 30, 15);///TODO set that as a parameter the current valu is just the one I read from the current project github
	std::cout << "Set SDF resolution: " << resolutionSDF[0] << ", " << resolutionSDF[1] << ", " << resolutionSDF[2] << std::endl;
	volumeMap = new Discregrid::CubicLagrangeDiscreteGrid(domain, std::array<unsigned int, 3>({ resolutionSDF[0], resolutionSDF[1], resolutionSDF[2] }));
	auto func = Discregrid::DiscreteGrid::ContinuousFunction{};

	//volumeMap->setErrorTolerance(0.001);
	
	Real sign = 1.0;
	bool mapInvert = true; ///TODO set that as a parameter the current valu is just the one I read from the current project github
	if (mapInvert)
		sign = -1.0;
	const Real particleRadius = m_model->getParticleRadius();
	// subtract 0.5 * particle radius to prevent penetration of particles and the boundary
	func = [&md, &sign, &tolerance, &particleRadius](Eigen::Vector3d const& xi) {return sign * (md.signedDistanceCached(xi) - tolerance - 0.5 * particleRadius); };

	std::cout << "Generate SDF" << std::endl;
	volumeMap->addFunction(func, false);

	//////////////////////////////////////////////////////////////////////////
	// Generate volume map of object using Discregrid
	//////////////////////////////////////////////////////////////////////////
	
	const bool sim2D = false;

	auto int_domain = Eigen::AlignedBox3d(Eigen::Vector3d::Constant(-supportRadius), Eigen::Vector3d::Constant(supportRadius));
	Real factor = 1.0;
	if (sim2D)
		factor = 1.75;
	auto volume_func = [&](Eigen::Vector3d const& x)
	{
		auto dist = volumeMap->interpolate(0u, x);
		if (dist > (1.0 + 1.0 )* supportRadius)
		{
			return 0.0;
		}

		auto integrand = [&volumeMap, &x, &supportRadius, &factor](Eigen::Vector3d const& xi) -> double
		{
			if (xi.squaredNorm() > supportRadius* supportRadius)
				return 0.0;

			auto dist = volumeMap->interpolate(0u, x + xi);

			if (dist <= 0.0)
				return 1.0 - 0.1 * dist / supportRadius;
			if (dist < 1.0 / factor * supportRadius)
				return static_cast<double>(CubicKernel::W(factor * static_cast<Real>(dist)) / CubicKernel::W_zero());
			return 0.0;
		};

		double res = 0.0;
		res = 0.8 * GaussQuadrature::integrate(integrand, int_domain, 30);

		return res;
	};
	
	auto cell_diag = volumeMap->cellSize().norm();
	std::cout << "Generate volume map..." << std::endl;
	const bool no_reduction = true;
	volumeMap->addFunction(volume_func, false, [&](Eigen::Vector3d const& x_)
		{
			if (no_reduction)
			{
				return true;
			}
			auto x = x_.cwiseMax(volumeMap->domain().min()).cwiseMin(volumeMap->domain().max());
			auto dist = volumeMap->interpolate(0u, x);
			if (dist == std::numeric_limits<double>::max())
			{
				return false;
			}

			return -6.0 * supportRadius < dist + cell_diag && dist - cell_diag < 2.0 * supportRadius;
		});

	// reduction
	if (!no_reduction)
	{
		std::cout << "Reduce discrete fields...";
		volumeMap->reduceField(0u, [&](Eigen::Vector3d const&, double v)->double
			{
				return 0.0 <= v && v <= 3.0;
			});
		std::cout << "DONE" << std::endl;
	}

	boundaryModel->setMap(volumeMap);

	
	//*/
}


#endif //SPLISHSPLASH_FRAMEWORK

void DFSPHCUDA::checkReal(std::string txt, Real old_v, Real new_v) {
    Real error = std::abs(old_v - new_v);
    //Real trigger = 0;
    Real trigger = std::abs(old_v) * 1E-13;
    if (error > trigger) {
        ostringstream oss;
        oss << "(Real)" << txt << " old/ new: " << old_v << " / " << new_v <<
               " //// " << error << " / " << trigger << std::endl;
        std::cout << oss.str();
        exit(2679);
    }
}

void DFSPHCUDA::checkVector3(std::string txt, Vector3d old_v, Vector3d new_v) {

    Vector3d error = (old_v - new_v).toAbs();
    //Vector3d trigger = Vector3d(0, 0, 0);
    //Vector3d trigger = old_v.toAbs() * 1E-13;
    Vector3d trigger = old_v.toAbs().avg() * 1E-13;
    if (error.x > trigger.x || error.y > trigger.y || error.z > trigger.z) {
        ostringstream oss;
        oss << "(Vector3) " << txt << " error/ trigger: " "(" << error.x << ", " << error.y << ", " << error.z << ")" << " / " <<
               "(" << trigger.x << ", " << trigger.y << ", " << trigger.z << ")" << " //// actual value old/new (x, y, z): " <<
               "(" << old_v.x << ", " << old_v.y << ", " << old_v.z << ")" << " / " <<
               "(" << new_v.x << ", " << new_v.y << ", " << new_v.z << ")" << std::endl;
        std::cout << oss.str();
        exit(2679);
    }

}



void DFSPHCUDA::renderFluid() {
    cuda_renderFluid(&m_data);
}

void DFSPHCUDA::renderBoundaries(bool renderWalls) {
    cuda_renderBoundaries(&m_data, renderWalls);
}


void DFSPHCUDA::handleDynamicBodiesPause(bool pause) {
#ifdef SPLISHSPLASH_FRAMEWORK
    if (pause) {
        //if we are toggleing the pause we need to store the velocities and set the one used for the computations to zero
        for (int id = 2; id < m_model->m_particleObjects.size(); ++id) {
            FluidModel::RigidBodyParticleObject* particleObj = static_cast<FluidModel::RigidBodyParticleObject*>(m_model->m_particleObjects[id]);

            for (int i = 0; i < particleObj->m_v.size(); ++i) {
                particleObj->m_v[i] = Vector3r(0,0,0);
            }
        }
    }
#endif //SPLISHSPLASH_FRAMEWORK

    /*
    FluidModel::RigidBodyParticleObject* particleObjtemp = static_cast<FluidModel::RigidBodyParticleObject*>(m_model->m_particleObjects[2]);
    std::cout << "vel_check: " << particleObjtemp->m_v[0].x() << "  " << particleObjtemp->m_v[0].y() << "  " << particleObjtemp->m_v[0].z() << std::endl;
    //*/
    is_dynamic_bodies_paused = pause;

    if (is_dynamic_bodies_paused){
        m_data.pause_solids();
    }
}



void DFSPHCUDA::handleSimulationSave(bool save_liquid, bool save_solids, bool save_boundaries) {
    if (save_liquid) {
        m_data.write_fluid_to_file();
		
		//save the others fluid data
		std::string file_name = m_data.fluid_files_folder + "general_data.txt";
		std::remove(file_name.c_str());
		std::cout << "saving general data to: " << file_name << std::endl;

		ofstream myfile;
		myfile.open(file_name, std::ios_base::app);
		if (myfile.is_open()) {
			//the timestep
			myfile << m_data.h;
			myfile.close();
		}
		else {
			std::cout << "failed to open file: " << file_name << "   reason: " << std::strerror(errno) << std::endl;
		}
    }

    if (save_boundaries) {
        m_data.write_boundaries_to_file();
    }

    if (save_solids) {
        m_data.write_solids_to_file();
    }

}

void DFSPHCUDA::handleSimulationLoad(bool load_liquid, bool load_liquid_velocities, bool load_solids, bool load_solids_velocities, 
                                     bool load_boundaries, bool load_boundaries_velocities) {

    if (load_boundaries) {
        m_data.read_boundaries_from_file(load_boundaries_velocities);
    }

    if (load_solids) {
        m_data.read_solids_from_file(load_solids_velocities);
    }

    //recompute the particle mass for the rigid particles
    if (load_boundaries||load_solids){
        m_data.computeRigidBodiesParticlesMass();

        //handleSimulationSave(false, true, true);
    }

    if (load_liquid) {
        m_data.read_fluid_from_file(load_liquid_velocities);
		
		//load the others fluid data
		std::string file_name = m_data.fluid_files_folder + "general_data.txt";
		std::cout << "loading general data start: " << file_name << std::endl;

		ifstream myfile;
		myfile.open(file_name);
		if (!myfile.is_open()) {
			std::cout << "trying to read from unexisting file: " << file_name << std::endl;
			exit(256);
		}

		//read the first line
		RealCuda sim_step;
		myfile >> sim_step;
		m_data.updateTimeStep(sim_step);
		m_data.onSimulationStepEnd();
		m_data.updateTimeStep(sim_step);
		m_data.onSimulationStepEnd();

	#ifdef SPLISHSPLASH_FRAMEWORK
		TimeManager::getCurrent()->setTimeStepSize(sim_step);
	#else
		desired_time_step = sim_step;
	#endif //SPLISHSPLASH_FRAMEWORK


		std::cout << "loading general data end: " << std::endl;
    }


}


void DFSPHCUDA::handleSimulationMovement(Vector3d movement) {
    if (movement.norm() > 0.5) {

        m_data.handleFluidBoundries(movement);
    }
}


void DFSPHCUDA::handleFLuidLevelControl(RealCuda level) {
    if (level > 0) {
        m_data.handleFLuidLevelControl(level);
    }
}


RealCuda DFSPHCUDA::getFluidLevel() {
    return m_data.computeFluidLevel();
}



void DFSPHCUDA::updateRigidBodiesStatefromFile() {
    m_data.update_solids_from_file();
}

void DFSPHCUDA::updateRigidBodiesStateToFile() {
    m_data.update_solids_to_file();
}

void DFSPHCUDA::updateRigidBodies(std::vector<DynamicBody> vect_new_info) {
    m_data.update_solids(vect_new_info);
}


void DFSPHCUDA::zeroFluidVelocities() {
    m_data.zeroFluidVelocities();
}


#ifndef SPLISHSPLASH_FRAMEWORK
void DFSPHCUDA::updateTimeStepDuration(RealCuda duration){
    desired_time_step=duration;
}
#endif //SPLISHSPLASH_FRAMEWORK

void DFSPHCUDA::forceUpdateRigidBodies(){
    m_data.loadDynamicObjectsData(m_model);
}

void DFSPHCUDA::getFluidImpactOnDynamicBodies(std::vector<SPH::Vector3d>& sph_forces, std::vector<SPH::Vector3d>& sph_moments,
                                              const std::vector<SPH::Vector3d>& reduction_factors){
    m_data.getFluidImpactOnDynamicBodies(sph_forces,sph_moments, reduction_factors);
}

void DFSPHCUDA::getFluidBoyancyOnDynamicBodies(std::vector<SPH::Vector3d>& forces, std::vector<SPH::Vector3d>& pts_appli){
    m_data.getFluidBoyancyOnDynamicBodies(forces,pts_appli);
}

SPH::Vector3d DFSPHCUDA::getSimulationCenter(){
    return m_data.getSimulationCenter();
}

