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
#include <cmath>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>
#include "SPlisHSPlasH\DFSPH\OpenBoundariesSimple.h" 
#include "SPlisHSPlasH\DFSPH\DynamicWindow.h" 

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
    m_maxErrorV=0.1;
    m_maxIterationsV=100;
    desired_time_step=m_data.get_current_timestep();
#endif //SPLISHSPLASH_FRAMEWORK
	need_stabilization_for_init = false;
    count_steps = 0;
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

    //test_particleshift();
}

DFSPHCUDA::~DFSPHCUDA(void)
{
}


void DFSPHCUDA::step()
{
	//compare_vector3_struct_speed();
	//exit(0);
   
    m_data.fluid_data->resetColor();

	static int count_fluid_particles_initial = m_data.getFluidParticlesCount();

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

	std::chrono::steady_clock::time_point tpStartSimuStep = std::chrono::steady_clock::now();

    if (true) {
        m_data.destructor_activated = false;

		//I'll run my tests here so that I can be sure that the loading did not failed before launching them
		//this is the initialization system
		if (false && count_steps == 0)
		{

			handleFluidInit();
			//handleFluidInitExperiments();

			std::this_thread::sleep_for(std::chrono::nanoseconds(10));
			std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(1));

			count_steps++;
			need_stabilization_for_init = false;
			return;
		}


#ifdef OCEAN_BOUNDARIES_PROTOTYPE
        bool moving_borders = false;
        static int count_moving_steps = 0;
        //*
		//test the simple open boundaries
		
		if (false) {
			bool useOpenBoundaries = true;
			bool useDynamicWindow = false;

			if (useOpenBoundaries)
			{
				applyOpenBoundaries();
			}
			
			
			//qzdqs
			//the dynamic windows diplacment
			if (useDynamicWindow) {
				applyDynamicWindow();
			}

			if (count_steps == 0) {
				std::this_thread::sleep_for(std::chrono::nanoseconds(10));
				std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(2));

				count_steps++;
				return;
			}
		}
		//test boundries height control
	//if (TimeManager::getCurrent()->getTime() < 1.5) 
	{
		//m_data.handleBoundariesHeightTest();
	}


    //test dynamic window
    /*
    if ((count_steps > 5) && ((count_steps % 15) == 0))
    {
        handleSimulationMovement(Vector3d(1,0,0));
        moving_borders = true;
        count_moving_steps++;
        //return;
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
		bool sort_data = true;// (((count_steps) % 5) == 0);
        cuda_neighborsSearch(m_data, sort_data);
		//if (sort_data) {	std::cout << "sorting now" << std::endl << std::endl << std::endl << std::endl << std::endl;}


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

       // std::cout << "self density: " << m_data.W_zero * 0.1 << std::endl;


        /*
        RealCuda min_density = 10000;
        RealCuda max_density = 0;
        for (int j = 0; j < m_data.fluid_data->numParticles; ++j) {
            if((m_data.fluid_data->getNumberOfNeighbourgs(j) + m_data.fluid_data->getNumberOfNeighbourgs(j)) >60)
            min_density = std::fminf(min_density, m_data.fluid_data->density[j]);
            max_density = std::fmaxf(max_density, m_data.fluid_data->density[j]);
           
        }

        std::cout << "min/ max density : " << min_density << "  " << max_density << std::endl;
        
        std::cout << "fluid_level: " << getFluidLevel() << std::endl;
        //*/

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
		
        int nbr_step_apply = -1;
        if (count_steps < nbr_step_apply) {
            //*
            RealCuda clamp_value = m_data.particleRadius * 1 / 0.003;
            clamp_buffer_to_value<Vector3d, 4>(m_data.fluid_data->vel, Vector3d(clamp_value), m_data.fluid_data->numParticles);
            std::cout << "clamping value: " << clamp_value << std::endl;
            //*/
            /*
            RealCuda factor = 0;
            apply_factor_to_buffer(m_data.fluid_data->vel, Vector3d(factor), m_data.fluid_data->numParticles);
            //*/
            
        }
        //std::cout << "count steps: " << count_steps << std::endl;

        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();

        cuda_update_pos(m_data);


        if(false){
            //read data to CPU
            static Vector3d* vel = NULL;
            int size = 0;
            if (m_data.fluid_data->numParticles > size) {
                if (vel != NULL) {
                    delete[] vel;
                }
                vel = new Vector3d[m_data.fluid_data->numParticlesMax];
                size = m_data.fluid_data->numParticlesMax;

            }
            read_UnifiedParticleSet_cuda(*(m_data.fluid_data), NULL, vel, NULL);

            //read the actual evaluation
            RealCuda stabilzationEvaluation = -1;

            for (int i = 0; i < m_data.fluid_data->numParticles; ++i) {
                stabilzationEvaluation = MAX_MACRO_CUDA(stabilzationEvaluation, vel[i].squaredNorm());
            }
            SQRT_MACRO_CUDA(stabilzationEvaluation);

            std::cout << "max velocity: " << stabilzationEvaluation << std::endl;
        }

        nbr_step_apply = -1;
        if (count_steps < nbr_step_apply) {
            //*
            RealCuda clamp_value = 0;// m_data.particleRadius * 0.3 / 0.003;
            clamp_buffer_to_value<Vector3d, 4>(m_data.fluid_data->vel, Vector3d(clamp_value), m_data.fluid_data->numParticles);
            std::cout << "clamping value: " << clamp_value << std::endl;
            //*/
            /*
            RealCuda factor = 0;
            apply_factor_to_buffer(m_data.fluid_data->vel, Vector3d(factor), m_data.fluid_data->numParticles);
            //*/

        }

        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();



        tab_timepoint[current_timepoint++] = std::chrono::steady_clock::now();



		if (!is_dynamic_bodies_paused) {
			m_data.readDynamicObjectsData(m_model);
		}
        m_data.onSimulationStepEnd();
        
        //set_buffer_to_value<Vector3d>(m_data.fluid_data->vel, Vector3d(0), m_data.fluid_data->numParticles);

        // Compute new time

#ifdef SPLISHSPLASH_FRAMEWORK
        TimeManager::getCurrent()->setTimeStepSize(m_data.h);
        TimeManager::getCurrent()->setTime(TimeManager::getCurrent()->getTime() + m_data.h);
#endif //SPLISHSPLASH_FRAMEWORK
        //*/

		m_data.checkParticlesPositions(2);

		//code for timming informations

        if ((current_timepoint-1) != NB_TIME_POINTS) {
            std::cout << "the number of time points does not correspond: "<< current_timepoint<<std::endl;
            exit(325);
        }

        float time_iter = std::chrono::duration_cast<std::chrono::nanoseconds> (tab_timepoint[NB_TIME_POINTS] - tab_timepoint[0]).count() / 1000000.0f;
        float time_between= std::chrono::duration_cast<std::chrono::nanoseconds> (tab_timepoint[0] - end).count() / 1000000.0f;
        static float total_time = 0;
        total_time += time_iter;

		if (false) {
			std::cout << "check density and divergence targetsand iter densityiter/diviter/densityerr/diverr: " <<
				m_maxError << "  " << m_maxIterations << "  " << m_maxErrorV << "  " << m_maxIterationsV << std::endl;
		}

        static float iter_pressure_avg = 0;
        static float iter_divergence_avg = 0;
        iter_pressure_avg += m_iterations;
        iter_divergence_avg += m_iterationsV;

        if(show_fluid_timings){
            std::cout << "timestep total: " << total_time / (count_steps + 1) << "   this step: " << time_iter + time_between << "  (" << time_iter << "  " << time_between << ")" << std::endl;
            std::cout << "solver iteration avg (current step) density: " <<  iter_pressure_avg / (count_steps + 1) << " ( " << m_iterations << " )   divergence : " <<
                         iter_divergence_avg / (count_steps + 1) << " ( " << m_iterationsV << " )  " << std::endl;


            for (int i = 0; i < NB_TIME_POINTS; ++i) {
                float time = std::chrono::duration_cast<std::chrono::nanoseconds> (tab_timepoint[i+1] - tab_timepoint[i]).count() / 1000000.0f;
                tab_avg[i] += time;
				std::cout << i << "  ";
                if (i == 8) {
                    std::cout << tab_name[i] << "  :" << ((count_moving_steps>0)?(tab_avg[i] / (count_moving_steps + 1)):0) << "  (" << time << ")" << std::endl;
                } else {
                    std::cout << tab_name[i] << "  :" << (tab_avg[i] / (count_steps + 1)) << "  (" << time << ")" << std::endl;
                }
            }


			int end_step =  1650;
			bool has_solid = false;
			bool print_avg_velocity = false;
			if (end_step > 0) {

			static std::vector<float> timmings;
			static std::vector<float> iters_divergence;
			static std::vector<float> iters_density;
			static std::vector<Vector3d> avgs_velocity;
			timmings.push_back(time_iter);
			iters_divergence.push_back(m_iterationsV);
			iters_density.push_back(m_iterations);
			if (print_avg_velocity) {
				avgs_velocity.push_back(m_data.getFluidAvgVelocity());
			}

			static std::vector<Vector3d> forces;
			if (has_solid) {
				std::vector<Vector3d> f;
				std::vector<Vector3d> m;
				std::vector<Vector3d> c; c.push_back(1);
				m_data.getFluidImpactOnDynamicBodies(f, m, c);
				forces.push_back(f[0]);
			}

				if((count_steps+1) == end_step ) {
					float desired_recap = 0;
					for (int i = 0; i < NB_TIME_POINTS; ++i) {
						/*
						if (i == 1) { //don't consider the neighbor search
							
							//continue;
						}
						//*/

						float time = tab_avg[i] / (count_steps + 1);
					
						//viscosity
						//the +1 is for the warm start iteration
						if (i == 2) {
							std::cout << time << "  ";
							time *= 3 / (((iter_divergence_avg ) / (count_steps + 1))+1);
							std::cout << time << "  "<< ((iter_divergence_avg + 1) / (count_steps + 1)) << "  " << i << "  " << std::endl;
						}

						//density
						//the +1 is for the warm start iteration
						if (i == 6) {
							std::cout << time << "  ";
							time *= 13 / (((iter_pressure_avg) / (count_steps + 1))+1);
							std::cout << time << "  " << ((iter_pressure_avg + 1) / (count_steps + 1)) << "  " << i << "  " << std::endl;
						}

						desired_recap += time;
					}
					std::cout << "recap result: " << desired_recap << std::endl;
					std::cout << "nbr lost particles: " << count_fluid_particles_initial-m_data.getFluidParticlesCount() << std::endl;


					if (false) {

						

						std::cout << "timestep comp_time iter_density iter_div ";

						if (print_avg_velocity) {
							std::cout << " vx vy vz ";
						}
						if (has_solid) {
							std::cout << " fx fy fz ";
						}
						std::cout << std::endl;
						for (int i = 0; i < timmings.size(); ++i) {
							Vector3d force = forces[i];
							if (i > 0) {
								force -= forces[i-1];
							}

							std::cout << (i + 1)*m_data.get_current_timestep() << " " << timmings[i] << " ";
							std::cout << iters_density[i] << " " << iters_divergence[i] << " ";
							if (print_avg_velocity) {
								std::cout << avgs_velocity[i].toString() << " ";
							}
							if (has_solid) {
								std::cout << force.toString() << " ";
							}
							std::cout << std::endl;
						}
					}

					exit(0);

				}

			}
        }


		//some less spammy timming message
		if (false) {
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

		//output some information ot a file
		if (false) {
			std::string filename = "particle_count_evolution_wedge.csv";
			if (count_steps == 0) {
				std::remove(filename.c_str());
			}
			ofstream myfile;
			myfile.open(filename, std::ios_base::app);
			if (myfile.is_open()) {
				myfile << count_steps<< "  "<<m_data.getFluidParticlesCount() << std::endl;;
				myfile.close();
			}
			else {
				std::cout << "failed to open file: " << filename << "   reason: " << std::strerror(errno) << std::endl;
			}
		}



		//std::cout << "fluid level: " << getFluidLevel() << std::endl;

        end = std::chrono::steady_clock::now();
        //m_data.fluid_data->resetColor();
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


	std::chrono::steady_clock::time_point tpEndSimuStep = std::chrono::steady_clock::now();

	//this gives results in miliseconds
	RealCuda time_simu_step = std::chrono::duration_cast<std::chrono::nanoseconds> (tpEndSimuStep - tpStartSimuStep).count() / 1000000.0f;

	std::cout << "time simu step: " << time_simu_step << std::endl;


    if(show_fluid_timings)
    {
        static int true_count_steps = 0;
        std::cout << "step finished: " << true_count_steps<<"  "<< count_steps << std::endl;
		true_count_steps++;
    }
	count_steps++;
	m_data.fluid_data->resetColor();
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
            v_i += kSum * m_data.getMass(neighborIndex) * m_data.gradW(xi - m_data.posFluid[neighborIndex]);
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
            v_i += kSum *  m_data.getMass(neighborIndex) * m_data.gradW(xi - m_data.posFluid[neighborIndex]);
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

    if (load_liquid||load_boundaries||load_solids) {
        count_steps = 0;
    }

}


void DFSPHCUDA::handleFluidInit() {

	RestFLuidLoaderInterface::StabilizationParameters params;
	params.method = 0;
	params.max_iterEval = 30;

	if (params.method == 0) {
		params.stabilizationItersCount = 10;
		params.useDivergenceSolver = true;
		params.useExternalForces = true;

		params.timeStep = 0.003;

	}
	
	{
		//this can be used to load any boundary shape that has a config and no existing fluid

		bool keep_existing_fluid = false;
		int simulation_config = 15;

		Vector3d normal_gravitation = m_data.gravitation;
		//m_data.gravitation.y *= 5;

		static std::default_random_engine e;
		static std::uniform_real_distribution<> dis(-1, 1); // rage -1 ; 1


		RestFLuidLoaderInterface::InitParameters paramsInit;
		RestFLuidLoaderInterface::TaggingParameters paramsTagging;
		RestFLuidLoaderInterface::LoadingParameters paramsLoading;

		paramsInit.show_debug = true;
		paramsTagging.show_debug = true;
		params.show_debug = true;
		paramsLoading.show_debug = true;

		paramsLoading.load_raw_untaged_data = false;

		int step_size = 60;
		{
			std::vector<RealCuda> vect_t1_internal;
			std::vector<RealCuda> vect_t2_internal;
			std::vector<RealCuda> vect_t3_internal;
			std::vector<int> vect_count_stabilization_iter_internal;
			std::vector<int> vect_count_selection_iter_internal;



			for (int k = 0; k <1; ++k) {
				//if i keep the existing fluid i need to reload it from memory each loop
				if (keep_existing_fluid) {
					m_data.read_fluid_from_file(false);
				}


				std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

				//handleSimulationLoad(true,false,false,false,false,false);

				//if (k == 0) 
				{
					paramsInit.clear_data = false;
					paramsInit.air_particles_restriction = 1;
					paramsInit.center_loaded_fluid = true;
					paramsInit.keep_existing_fluid = false;
					paramsInit.simulation_config = simulation_config;
					paramsInit.apply_additional_offset = true;
					paramsInit.additional_offset = Vector3d(dis(e), dis(e), dis(e))*m_data.particleRadius * 2;

					RestFLuidLoaderInterface::init(m_data, paramsInit);
				}

				std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();

				//*
				paramsTagging.useRule2 = false;
				paramsTagging.useRule3 = true;
				paramsTagging.useStepSizeRegulator = true;
				paramsTagging.min_step_density = 5;
				paramsTagging.step_density = step_size;
				paramsTagging.density_end = 999;
				paramsTagging.keep_existing_fluid = paramsInit.keep_existing_fluid;
				paramsTagging.output_density_information = true;

				paramsLoading.load_fluid = true;
				paramsLoading.keep_existing_fluid = keep_existing_fluid;

				RestFLuidLoaderInterface::initializeFluidToSurface(m_data, true, paramsTagging, paramsLoading);
				//*/
				std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
				/*
				params.method = 0;
				params.timeStep = 0.003;
				{
				params.stabilize_tagged_only = true;
				params.postUpdateVelocityDamping = false;
				params.postUpdateVelocityClamping = false;
				params.preUpdateVelocityClamping = false;


				params.maxErrorD = 0.05;

				params.useDivergenceSolver = true;
				params.useExternalForces = true;

				params.preUpdateVelocityDamping = true;
				params.preUpdateVelocityDamping_val = 0.8;

				params.stabilizationItersCount = 10;

				params.reduceDampingAndClamping = true;
				params.reduceDampingAndClamping_val = std::powf(0.2f / params.preUpdateVelocityDamping_val, 1.0f / (params.stabilizationItersCount - 1));

				params.clearWarmstartAfterStabilization = false;
				}
				params.runCheckParticlesPostion = true;
				params.interuptOnLostParticle = false;
				params.reloadFluid = false;
				params.evaluateStabilization = false;
				params.min_stabilization_iter = 2;
				params.stable_velocity_max_target = m_data.particleRadius*0.25 / m_data.get_current_timestep();
				params.stable_velocity_avg_target = m_data.particleRadius*0.025 / m_data.get_current_timestep();

				std::chrono::steady_clock::time_point tp5 = std::chrono::steady_clock::now();

				RestFLuidLoaderInterface::stabilizeFluid(m_data, params);

				//if there was a fail do not count the run in the values
				if (!params.stabilization_sucess) {
				continue;
				}
				//*/

				std::chrono::steady_clock::time_point tp4 = std::chrono::steady_clock::now();

				RealCuda time_p1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000000.0f;
				RealCuda time_p2 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp3 - tp2).count() / 1000000000.0f;
				RealCuda time_p3 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp4 - tp3).count() / 1000000000.0f;

				vect_t1_internal.push_back(time_p1);
				vect_t2_internal.push_back(time_p2);
				vect_t3_internal.push_back(time_p3);
				vect_count_selection_iter_internal.push_back(paramsTagging.count_iter);
				vect_count_stabilization_iter_internal.push_back(params.count_iter_o);

				//std::cout << "direct timmings output: " << time_p1 << "  " << time_p2 << "  " << time_p3 << "  " << std::endl;
			}

			if (!vect_t1_internal.empty()) {


				std::sort(vect_t1_internal.begin(), vect_t1_internal.end());
				std::sort(vect_t2_internal.begin(), vect_t2_internal.end());
				std::sort(vect_t3_internal.begin(), vect_t3_internal.end());
				std::sort(vect_count_selection_iter_internal.begin(), vect_count_selection_iter_internal.end());
				std::sort(vect_count_stabilization_iter_internal.begin(), vect_count_stabilization_iter_internal.end());

				//*
				for (int k = 0; k < vect_t1_internal.size(); ++k) {
					std::cout << k << "   " << vect_t1_internal[k] << " + " << vect_t2_internal[k] <<
						" + " << vect_t3_internal[k] << " = " <<
						vect_t1_internal[k] + vect_t2_internal[k] + vect_t3_internal[k] <<
						" // " << vect_count_selection_iter_internal[k] << " // " << vect_count_stabilization_iter_internal[k] << std::endl;
				}
				//*/
				//some density informations
				std::cout << "density info: " << paramsTagging.avg_density_o << "  " << paramsTagging.min_density_o << "  " <<
					paramsTagging.max_density_o << "  " << paramsTagging.stdev_density_o / paramsTagging.avg_density_o * 100 << "  " << std::endl;


				int idMedian = std::floor((vect_t2_internal.size() - 1) / 2.0f);


				std::cout << "median values" << std::endl;
				std::cout << "count valid runs: " << vect_t1_internal.size() << std::endl;
				std::cout << "time init" << " ; " << "time selection" <<
					" ; " << "time stabilization" << " ; " <<
					"total time" <<
					" ; " << "coutn iter selection" << " ; " << "count iter stabilization" << std::endl;
				std::cout << vect_t1_internal[idMedian] << " ; " << vect_t2_internal[idMedian] <<
					" ; " << vect_t3_internal[idMedian] << " ; " <<
					vect_t1_internal[idMedian] + vect_t2_internal[idMedian] + vect_t3_internal[idMedian] <<
					" ; " << vect_count_selection_iter_internal[idMedian] << " ; " << vect_count_stabilization_iter_internal[idMedian] << std::endl;

			}

		}

		m_data.gravitation = normal_gravitation;


	}

	count_steps = 0;
}


void DFSPHCUDA::handleFluidInitExperiments() {

		//*
		RestFLuidLoaderInterface::StabilizationParameters params;
		params.method = 0;
		params.max_iterEval = 30;

		if (params.method == 0) {
			params.stabilizationItersCount = 10;
			params.useDivergenceSolver = true;
			params.useExternalForces = true;

			params.timeStep = 0.003;

		}
		else if (params.method == 1) {
			params.stabilizationItersCount = 30;
			params.timeStep = 0.0001;
			RealCuda delta_s = m_data.particleRadius * 2;
			//when based on gamma gradiant fast evolution on a single perturbated particle
			//params.p_b = 10000;//25000 * delta_s;
			//based on gamma slow evolution on a single particle
			//params.p_b = 10;//25000 * delta_s;
			//based on gamma slow evolution on a single particle
			//params.p_b = 100;//25000 * delta_s;
			//values used when doing density based gradiant
			//params.p_b = 1/1.0;//25000 * delta_s;
			//values used when doing the density estimation gradiant
			//params.p_b = 1.0 / 10000;//25000 * delta_s;
			//values used when doing the density estimation gradiant v2
			//params.p_b = -1.0 / 1000;//25000 * delta_s;
			//values used when pushing the particles from the border
			//params.p_b = 100;//25000 * delta_s;
			//values used when using the attraction model
			params.p_b = 100;//25000 * delta_s;
							 //deactivate
							 //params.p_b = 0;//25000 * delta_s;

							 //deactivate
			params.k_r = 0;
			//params.k_r = 0.01;//150 * delta_s * delta_s * 0.03 / 3200.0;
			//params.k_r = 1;//150 * delta_s * delta_s * 0.03 / 3200.0;

			//params.k_r *= 100;
			//params.p_b *= 20;

			//zeta as a pure damping coefficient directly on the velocity
			params.zeta = 1;
			params.zetaChangeFrequency = 100000;
			params.zetaChangeCoefficient = 0.999;
			//params.zeta = 2 * (SQRT_MACRO_CUDA(delta_s) + 1) / delta_s;
		}

		int run_type = 11;
		if (run_type == 0) {

			if (params.method == 0) {

				/*
				params.preUpdateVelocityDamping = true;
				params.postUpdateVelocityDamping = true;
				params.preUpdateVelocityClamping = true;
				params.preUpdateVelocityDamping_val = 0.80;
				params.postUpdateVelocityDamping_val = 0.2;
				params.preUpdateVelocityClamping_val = 4;
				params.postUpdateVelocityClamping = false;
				//*/

				//*
				params.preUpdateVelocityDamping = false;
				params.postUpdateVelocityDamping = false;
				params.postUpdateVelocityClamping = false;
				params.preUpdateVelocityClamping = false;
				//*/

				//params.preUpdateVelocityClamping = false;
				//params.preUpdateVelocityDamping = false;
				//params.postUpdateVelocityDamping_val = 0;
				//params.maxErrorD = 0.1;
				params.maxErrorD = 0.05;

				//params.useDivergenceSolver = false;
				//params.useExternalForces = false;

				params.preUpdateVelocityDamping = true;
				params.preUpdateVelocityDamping_val = 0.8;

				params.postUpdateVelocityDamping = false;
				params.postUpdateVelocityDamping_val = 1;

				params.reduceDampingAndClamping = true;
				params.reduceDampingAndClamping_val = 0.80;

			}
			else if (params.method == 1) {
			}

			std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

			RealCuda avg_eval = 0;
			RealCuda max_eval = 0;
			params.evaluateStabilization = false;
			int count_eval = 1;
			for (int i = 0; i < count_eval; ++i) {
				//RestFLuidLoaderInterface::init(m_data);
				//RestFLuidLoaderInterface::initializeFluidToSurface(m_data);

				RestFLuidLoaderInterface::stabilizeFluid(m_data, params);

				max_eval = MAX_MACRO_CUDA(max_eval, params.stabilzationEvaluation1);
				avg_eval += params.stabilzationEvaluation1;

				std::cout << params.preUpdateVelocityDamping_val << "  " << params.postUpdateVelocityDamping_val <<
					"  " << params.preUpdateVelocityClamping_val << "  " << params.postUpdateVelocityClamping_val << "  " << params.stabilzationEvaluation1 << std::endl;


				// params.clearWarmstartAfterStabilization = false;
			}
			avg_eval /= count_eval;

			std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();
			RealCuda time_opti = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000000.0f;
			std::cout << "fluid initialization finished and took (in s): " << time_opti << std::endl;


			if (count_eval > 1) {
				std::cout << "stabilisation evaluation (avg/max): " << avg_eval << "    " << max_eval << std::endl;
			}
			//exit(0); 
		}
		else if (run_type == 1) {

			std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

			//*
			std::ofstream myfile("stabilization_method_comparison.csv", std::ofstream::trunc);
			if (myfile.is_open())
			{
				myfile << "pre_damp post_damp pre_clamp post_clamp eval" << std::endl;
				myfile.close();
			}
			else {
				std::cout << "failed opening the file at init" << std::endl;
			}
			//run the eval
			if (params.method == 0) {
				RealCuda damping_min = 0;
				RealCuda damping_max = 1;
				RealCuda clamping_min = 0;
				RealCuda clamping_max = m_data.particleRadius / params.timeStep;
				int nbr_sampling_by_dim = 11;

				RealCuda damping_step = damping_max - damping_min;
				damping_step /= (nbr_sampling_by_dim - 1);
				damping_step -= damping_step / 1000000;
				RealCuda clamping_step = clamping_max - clamping_min;
				clamping_step /= (nbr_sampling_by_dim - 1);
				clamping_step -= clamping_step / 1000000;

				//the sustraction is to make absolutly sure that even with the float precision limit I don't go over the min or max
				RealCuda damping_min_real = damping_max;
				RealCuda clamping_min_real = clamping_max;
				for (int i = 0; i < (nbr_sampling_by_dim - 1); ++i) {
					damping_min_real -= damping_step;
					clamping_min_real -= clamping_step;
				}
				std::cout << "check damping step by calculating min theorical/real/step:  " << damping_min << "   " << damping_min_real << "   " << damping_step << std::endl;
				std::cout << "check clamping step by calculating min theorical/real/step: " << clamping_min << "   " << clamping_min_real << "   " << clamping_step << std::endl;

				params.preUpdateVelocityDamping = true;
				params.postUpdateVelocityDamping = true;
				params.preUpdateVelocityClamping = true;
				params.postUpdateVelocityClamping = true;
				params.preUpdateVelocityDamping_val = damping_max;
				params.postUpdateVelocityDamping_val = damping_max;
				params.preUpdateVelocityClamping_val = clamping_max;
				params.postUpdateVelocityClamping_val = clamping_max;

				for (int j = 0; j< nbr_sampling_by_dim; j++) {
					for (int k = 0; k< nbr_sampling_by_dim; k++) {
						for (int l = 0; l< nbr_sampling_by_dim; l++) {
							for (int m = 0; m< nbr_sampling_by_dim; m++) {
								RestFLuidLoaderInterface::stabilizeFluid(m_data, params);

								std::ofstream myfile("stabilization_method_comparison.csv", std::ofstream::app);
								if (myfile.is_open())
								{
									myfile << params.preUpdateVelocityDamping_val << "  " << params.postUpdateVelocityDamping_val <<
										"  " << params.preUpdateVelocityClamping_val << "  " << params.postUpdateVelocityClamping_val << "  " << params.stabilzationEvaluation1 << std::endl;
									myfile.close();
								}
								else {
									std::cout << "failed opening the file during eval" << std::endl;
								}
								params.postUpdateVelocityClamping_val -= clamping_step;
							}
							params.postUpdateVelocityClamping_val = clamping_max;
							params.preUpdateVelocityClamping_val -= clamping_step;
							if (params.preUpdateVelocityClamping_val < 2) {
								break;
							}
						}
						params.preUpdateVelocityClamping_val = clamping_max;
						params.postUpdateVelocityDamping_val -= damping_step;
					}
					params.postUpdateVelocityDamping_val = damping_max;
					params.preUpdateVelocityDamping_val -= damping_step;
					if (params.preUpdateVelocityDamping_val< 0.55) {
						break;
					}
				}
			}
			else if (params.method == 1) {
			}



			std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();
			RealCuda time_opti = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000000.0f;

			std::cout << "stabilisation evaluation finished and took (in s): " << time_opti << std::endl;
			exit(0);
			//*/

		}
		else if (run_type == 2) {
			//here is an optimisation that challenge the step size versus the Nbr of simulation steps required for stability

			std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

			//*

			RestFLuidLoaderInterface::TaggingParameters tagging_params;
			tagging_params.useRule2 = false;
			tagging_params.useRule3 = true;
			tagging_params.step_density = 59;
			tagging_params.keep_existing_fluid = false;
			RestFLuidLoaderInterface::LoadingParameters paramsLoading;
			paramsLoading.load_fluid = true;
			paramsLoading.keep_existing_fluid = false;
			RestFLuidLoaderInterface::initializeFluidToSurface(m_data, true, tagging_params, paramsLoading);

			//*/

			{
				params.stabilize_tagged_only = true;

				params.preUpdateVelocityDamping = false;
				params.postUpdateVelocityDamping = false;
				params.postUpdateVelocityClamping = false;
				params.preUpdateVelocityClamping = false;
				//*/

				params.maxErrorD = 0.05;

				params.useDivergenceSolver = true;
				params.useExternalForces = true;

				params.preUpdateVelocityDamping = true;
				params.preUpdateVelocityDamping_val = 0.8;

				params.stabilizationItersCount = 20;

				params.reduceDampingAndClamping = false;
				params.reduceDampingAndClamping_val = 0.0;
				//params.reduceDampingAndClamping_val = std::powf(0.1f / params.preUpdateVelocityDamping_val, 1.0f / (params.stabilizationItersCount - 1));
			}

			params.stabilizationItersCount = 5;

			RealCuda avg_eval = 0;
			RealCuda max_eval = 0;
			params.evaluateStabilization = false;
			int count_eval = 1;
			std::vector<RealCuda> vect_eval_1;
			std::vector<RealCuda> vect_eval_2;
			std::vector<RealCuda> vect_eval_3;



			std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();
			for (int i = 0; i < count_eval; ++i) {
				//RestFLuidLoaderInterface::init(m_data);
				//RestFLuidLoaderInterface::initializeFluidToSurface(m_data);
				//params.stabilizationItersCount = count_eval-i;

				RestFLuidLoaderInterface::stabilizeFluid(m_data, params);

				max_eval = MAX_MACRO_CUDA(max_eval, params.stabilzationEvaluation1);
				avg_eval += params.stabilzationEvaluation1;

				std::cout << tagging_params.step_density << "  " << params.stabilizationItersCount <<
					"  " << (params.stabilzationEvaluation1*params.timeStepEval) / m_data.particleRadius <<
					"  " << (params.stabilzationEvaluation2* params.timeStepEval) / m_data.particleRadius << "  " << params.stabilzationEvaluation3 << std::endl;


				vect_eval_1.push_back((params.stabilzationEvaluation1* params.timeStepEval) / m_data.particleRadius);
				vect_eval_2.push_back((params.stabilzationEvaluation2* params.timeStepEval) / m_data.particleRadius);
				vect_eval_3.push_back(params.stabilzationEvaluation3);

				// params.clearWarmstartAfterStabilization = false;
			}
			avg_eval /= count_eval;

			std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
			RealCuda time_ini = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000000.0f;
			RealCuda time_opti = std::chrono::duration_cast<std::chrono::nanoseconds> (tp3 - tp2).count() / 1000000000.0f;
			std::cout << "fluid initialization finished and took (in s) ini/opti: " << time_ini << "  " << time_opti << std::endl;

			for (int i = 0; i < vect_eval_1.size(); ++i) {
				std::cout << tagging_params.step_density << "  " << count_eval - i << "  " << vect_eval_1[i] <<
					"  " << vect_eval_2[i] << "  " << vect_eval_3[i] << std::endl;

			}

			if (count_eval > 1) {
				std::cout << "stabilisation evaluation (avg/max): " << avg_eval << "    " << max_eval << std::endl;

			}

			std::cout << "count particles after init: " << m_data.fluid_data->numParticles << std::endl;
		}
		else if (run_type == 3) {
			//this one will try to slightly improve the particle distribution befor going for the simulation
			//on a single run
			//*

			std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

			params.method = 3;
			params.p_b = 100;
			params.stabilizationItersCount = 10;

			params.timeStep = 0.0001;
			RealCuda delta_s = m_data.particleRadius * 2;

			params.evaluateStabilization = false;
			params.reloadFluid = true;
			//RestFLuidLoaderInterface::stabilizeFluid(m_data, params);
			//*/


			std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();
			params.method = 0;
			params.timeStep = 0.003;
			{
				params.stabilize_tagged_only = true;
				params.postUpdateVelocityDamping = false;
				params.postUpdateVelocityClamping = false;
				params.preUpdateVelocityClamping = false;
				//*/

				params.maxErrorD = 0.05;

				params.useDivergenceSolver = true;
				params.useExternalForces = true;

				params.preUpdateVelocityDamping = true;
				params.preUpdateVelocityDamping_val = 0.8;

				params.stabilizationItersCount = 6;

				params.reduceDampingAndClamping = true;
				params.reduceDampingAndClamping_val = std::powf(0.2f / params.preUpdateVelocityDamping_val, 1.0f / (params.stabilizationItersCount - 1));

				params.clearWarmstartAfterStabilization = false;
				params.maxIterD = 10;
			}
			params.runCheckParticlesPostion = true;
			params.interuptOnLostParticle = false;
			params.reloadFluid = true;
			RestFLuidLoaderInterface::stabilizeFluid(m_data, params);


			std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
			RealCuda time_p1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000000.0f;
			RealCuda time_p2 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp3 - tp2).count() / 1000000000.0f;


			std::cout << "fluid initialization finished and took (in s) time_displacement//time_simu_damped: " << time_p1 << "  " << time_p2 << std::endl;

		}
		else if (run_type == 4) {
			//same as 3 but it run the optimisation version similar to 2

			RealCuda avg_eval = 0;
			RealCuda max_eval = 0;
			int count_eval = 10;
			std::vector<RealCuda> vect_eval_1;
			std::vector<RealCuda> vect_eval_2;
			std::vector<RealCuda> vect_eval_3;

			for (int i = 0; i < (count_eval - 2); ++i) {

				{
					params.method = 1;
					params.p_b = 100;
					params.stabilizationItersCount = 10;


					params.zeta = 1;
					params.zetaChangeFrequency = 100000;
					params.zetaChangeCoefficient = 0.999;
					params.timeStep = 0.0001;
					RealCuda delta_s = m_data.particleRadius * 2;

					params.evaluateStabilization = false;
					params.reloadFluid = true;
					//*
					RestFLuidLoaderInterface::stabilizeFluid(m_data, params);
					params.reloadFluid = false;
					//*/
				}

				params.method = 0;
				params.timeStep = 0.003;
				{
					params.postUpdateVelocityDamping = false;
					params.postUpdateVelocityClamping = false;
					params.preUpdateVelocityClamping = false;
					//*/

					params.maxErrorD = 0.05;

					params.useDivergenceSolver = true;
					params.useExternalForces = true;

					params.preUpdateVelocityDamping = true;
					params.preUpdateVelocityDamping_val = 0.8;

					params.stabilizationItersCount = count_eval - i;

					params.reduceDampingAndClamping = true;
					params.reduceDampingAndClamping_val = std::powf(0.1f / params.preUpdateVelocityDamping_val, 1.0f / (params.stabilizationItersCount - 1));
				}

				params.evaluateStabilization = true;
				params.max_iterEval = 30;
				RestFLuidLoaderInterface::stabilizeFluid(m_data, params);


				/*
				std::cout << params.stabilizationItersCount << "  " << (params.stabilzationEvaluation1 * params.timeStepEval) / m_data.particleRadius <<
				"  " << (params.stabilzationEvaluation2 * params.timeStepEval) / m_data.particleRadius << "  " << params.stabilzationEvaluation3 << std::endl;
				//*/

				vect_eval_1.push_back((params.stabilzationEvaluation1 * params.timeStepEval) / m_data.particleRadius);
				vect_eval_2.push_back((params.stabilzationEvaluation2 * params.timeStepEval) / m_data.particleRadius);
				vect_eval_3.push_back(params.stabilzationEvaluation3);

				// params.clearWarmstartAfterStabilization = false;
			}

			std::cout << "eval_recap" << std::endl;
			for (int i = 0; i < vect_eval_1.size(); ++i) {
				std::cout << count_eval - i << "  " << vect_eval_1[i] <<
					"  " << vect_eval_2[i] << "  " << vect_eval_3[i] << std::endl;

			}

		}
		else if (run_type == 5) {
			//the run type with preexisting ffluid
			params.keep_existing_fluid = true;
			std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

			/*

			RestFLuidLoaderInterface::TaggingParameters tagging_params;
			tagging_params.step_density = -1;
			if (!(need_stabilization_for_init&&params.keep_existing_fluid)) {
			tagging_params.useRule2 = false;
			tagging_params.useRule3 = true;
			tagging_params.step_density = 50;
			tagging_params.keep_existing_fluid = params.keep_existing_fluid;
			RestFLuidLoaderInterface::initializeFluidToSurface(m_data, true, tagging_params, true);
			}
			//*/

			//this one will try to slightly improve the particle distribution befor going for the simulation
			//on a single run
			//*

			std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();

			params.method = 3;
			params.p_b = 100;
			params.stabilizationItersCount = 10;

			params.timeStep = 0.0001;
			RealCuda delta_s = m_data.particleRadius * 2;

			params.evaluateStabilization = false;
			params.reloadFluid = false;
			//RestFLuidLoaderInterface::stabilizeFluid(m_data, params);
			//*/


			std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
			params.method = 0;
			params.timeStep = 0.003;
			{
				params.stabilize_tagged_only = true;
				params.postUpdateVelocityDamping = false;
				params.postUpdateVelocityClamping = false;
				params.preUpdateVelocityClamping = false;
				//*/

				params.maxErrorD = 0.05;

				params.useDivergenceSolver = true;
				params.useExternalForces = true;

				params.preUpdateVelocityDamping = true;
				params.preUpdateVelocityDamping_val = 0.8;

				params.stabilizationItersCount = 10;

				params.reduceDampingAndClamping = true;
				params.reduceDampingAndClamping_val = std::powf(0.2f / params.preUpdateVelocityDamping_val, 1.0f / (params.stabilizationItersCount - 1));

				params.clearWarmstartAfterStabilization = false;
				params.maxIterD = 100;
			}
			params.runCheckParticlesPostion = true;
			params.interuptOnLostParticle = false;
			params.reloadFluid = true;
			RestFLuidLoaderInterface::stabilizeFluid(m_data, params);


			std::chrono::steady_clock::time_point tp4 = std::chrono::steady_clock::now();
			RealCuda time_p1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000000.0f;
			RealCuda time_p2 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp3 - tp2).count() / 1000000000.0f;
			RealCuda time_p3 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp3 - tp2).count() / 1000000000.0f;


			std::cout << "fluid initialization finished and took (in s) time_select//time_displacement//time_simu_damped: " <<
				time_p1 << "  " << time_p2 << "  " << time_p3 << std::endl;


			std::cout << "count particles after init: " << m_data.fluid_data->numParticles << std::endl;
		}
		else if (run_type == 6) {
			//here a test for an avg with vairuous random offsets

			std::chrono::steady_clock::time_point tp0 = std::chrono::steady_clock::now();
			RestFLuidLoaderInterface::InitParameters paramsInit;

			paramsInit.air_particles_restriction = 1;
			paramsInit.center_loaded_fluid = true;
			paramsInit.keep_existing_fluid = false;
			paramsInit.simulation_config = 0;
			paramsInit.apply_additional_offset = true;
			static std::default_random_engine e;
			static std::uniform_real_distribution<> dis(-1, 1); // rage -1 ; 1
			paramsInit.additional_offset = Vector3d(dis(e), dis(e), dis(e))*m_data.particleRadius * 2;

			RestFLuidLoaderInterface::init(m_data, paramsInit);

			std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

			RestFLuidLoaderInterface::TaggingParameters paramsTagging;
			paramsTagging.useRule2 = false;
			paramsTagging.useRule3 = true;
			paramsTagging.step_density = 25;
			paramsTagging.keep_existing_fluid = paramsInit.keep_existing_fluid;


			RestFLuidLoaderInterface::LoadingParameters paramsLoading;
			paramsLoading.load_fluid = true;
			paramsLoading.keep_existing_fluid = paramsInit.keep_existing_fluid;

			RestFLuidLoaderInterface::initializeFluidToSurface(m_data, true, paramsTagging, paramsLoading);

			std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();

			params.method = 0;
			params.timeStep = 0.003;
			{
				params.keep_existing_fluid = paramsInit.keep_existing_fluid;
				params.stabilize_tagged_only = true;
				params.postUpdateVelocityDamping = false;
				params.postUpdateVelocityClamping = false;
				params.preUpdateVelocityClamping = false;
				//*/

				params.maxErrorD = 0.05;

				params.useDivergenceSolver = true;
				params.useExternalForces = true;

				params.preUpdateVelocityDamping = true;
				params.preUpdateVelocityDamping_val = 0.8;

				params.stabilizationItersCount = 10;

				params.reduceDampingAndClamping = true;
				params.reduceDampingAndClamping_val = std::powf(0.2f / params.preUpdateVelocityDamping_val, 1.0f / (params.stabilizationItersCount - 1));

				params.clearWarmstartAfterStabilization = false;
				//params.maxIterD = 10;
			}
			params.runCheckParticlesPostion = true;
			params.interuptOnLostParticle = false;
			params.reloadFluid = true;
			RestFLuidLoaderInterface::stabilizeFluid(m_data, params);

			std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
			RealCuda time_p1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000000.0f;
			RealCuda time_p2 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp3 - tp2).count() / 1000000000.0f;


			std::cout << "fluid initialization finished and took (in s) time_displacement//time_simu_damped: " << time_p1 << "  " << time_p2 << std::endl;

		}
		else if (run_type == 7) {
			//this is the experiment to study the impact of the selection stepsize

			std::vector<unsigned int> vect_step_size;
			std::vector<RealCuda> vect_t1_medians;
			std::vector<RealCuda> vect_t2_medians;
			std::vector<RealCuda> vect_t3_medians;
			std::vector<RealCuda> vect_tselection_medians;
			std::vector<RealCuda> vect_density_min_medians;
			std::vector<RealCuda> vect_density_max_medians;
			std::vector<RealCuda> vect_density_avg_medians;
			std::vector<RealCuda> vect_density_stdev_medians;
			std::vector<int> vect_count_iter_medians;
			static std::default_random_engine e;
			static std::uniform_real_distribution<> dis(-1, 1); // rage -1 ; 1


			RestFLuidLoaderInterface::InitParameters paramsInit;
			RestFLuidLoaderInterface::TaggingParameters paramsTagging;
			RestFLuidLoaderInterface::LoadingParameters paramsLoading;

			params.show_debug = false;
			paramsInit.show_debug = false;
			paramsTagging.show_debug = false;
			paramsLoading.show_debug = false;

			paramsInit.keep_existing_fluid = false;

			bool use_protection_rule = true;
			bool use_step_size_regulator = true;
			bool run_stabilization = true;

			bool loopstepsize = true;
			for (int step_size = 60; step_size < 66; step_size += 1) {
				//*
				if (loopstepsize&&step_size == 65) {
					loopstepsize = false;
					use_protection_rule = false;
					step_size = 60;
				}
				//*/

				std::vector<RealCuda> vect_t1_internal;
				std::vector<RealCuda> vect_t2_internal;
				std::vector<RealCuda> vect_t3_internal;
				std::vector<RealCuda> vect_tselection_internal;
				std::vector<RealCuda> vect_density_min_internal;
				std::vector<RealCuda> vect_density_max_internal;
				std::vector<RealCuda> vect_density_avg_internal;
				std::vector<RealCuda> vect_density_stdev_internal;
				std::vector<int> vect_count_iter_internal;


				for (int k = 0; k < 51; ++k) {
					std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

					//handleSimulationLoad(true,false,false,false,false,false);

					//if (k == 0) 
					{
						paramsInit.clear_data = false;
						paramsInit.air_particles_restriction = 1;
						paramsInit.center_loaded_fluid = true;
						paramsInit.keep_existing_fluid = false;
						paramsInit.simulation_config = 6;
						paramsInit.apply_additional_offset = true;
						paramsInit.additional_offset = Vector3d(dis(e), dis(e), dis(e))*m_data.particleRadius * 2;

						RestFLuidLoaderInterface::init(m_data, paramsInit);
					}

					std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();

					//*
					paramsTagging.useRule2 = false;
					paramsTagging.useRule3 = use_protection_rule;
					paramsTagging.useStepSizeRegulator = use_step_size_regulator;
					paramsTagging.step_density = step_size;
					paramsTagging.min_step_density = 5;
					paramsTagging.keep_existing_fluid = paramsInit.keep_existing_fluid;
					paramsTagging.output_density_information = true;

					paramsLoading.load_fluid = true;
					paramsLoading.keep_existing_fluid = paramsInit.keep_existing_fluid;

					RestFLuidLoaderInterface::initializeFluidToSurface(m_data, true, paramsTagging, paramsLoading);
					//*/
					std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();

					if (run_stabilization) {
						params.method = 0;
						params.timeStep = 0.003;
						{
							params.stabilize_tagged_only = true;
							params.postUpdateVelocityDamping = false;
							params.postUpdateVelocityClamping = false;
							params.preUpdateVelocityClamping = false;


							params.maxErrorD = 0.05;
							params.useMaxErrorDPreciseAtMinIter = false;
							params.maxErrorDPrecise = 0.01;

							params.useDivergenceSolver = true;
							params.useExternalForces = true;

							params.preUpdateVelocityDamping = true;
							params.preUpdateVelocityDamping_val = 0.8;

							params.stabilizationItersCount = 10;

							params.reduceDampingAndClamping = true;
							params.reduceDampingAndClamping_val = std::powf(0.2f / params.preUpdateVelocityDamping_val, 1.0f / (params.stabilizationItersCount - 1));

							params.clearWarmstartAfterStabilization = false;
						}
						params.runCheckParticlesPostion = true;
						params.interuptOnLostParticle = false;
						params.reloadFluid = false;
						params.evaluateStabilization = false;
						params.min_stabilization_iter = 2;
						params.stable_velocity_max_target = m_data.particleRadius*0.25 / m_data.get_current_timestep();
						params.stable_velocity_avg_target = m_data.particleRadius*0.05 / m_data.get_current_timestep();


						RestFLuidLoaderInterface::stabilizeFluid(m_data, params);
						//*/

					}
					std::chrono::steady_clock::time_point tp4 = std::chrono::steady_clock::now();

					RealCuda time_p1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000000.0f;
					RealCuda time_p2 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp3 - tp2).count() / 1000000000.0f;
					RealCuda time_p3 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp4 - tp3).count() / 1000000000.0f;

					vect_count_iter_internal.push_back(paramsTagging.count_iter);
					vect_t1_internal.push_back(time_p1);
					vect_t2_internal.push_back(time_p2);
					vect_t3_internal.push_back(time_p3);
					vect_tselection_internal.push_back(paramsTagging.time_total);

					//I would also like to know the min and max density
					//the eaiest way if to init the neighborhood and to call the divergence init
					//actually it's useless to do it this way since there are no air particles it fucks up the min
					/*
					cuda_neighborsSearch(m_data, false);
					cuda_divergence_warmstart_init(m_data);

					RealCuda min_density = 10000;
					RealCuda max_density = 0;
					for (int j = 0; j < m_data.fluid_data->numParticles; ++j) {
					min_density = std::fminf(min_density, m_data.fluid_data->density[j]);
					max_density = std::fmaxf(max_density, m_data.fluid_data->density[j]);
					}


					vect_density_min_internal.push_back(min_density);
					vect_density_max_internal.push_back(max_density);
					//*/
					//ok so this version output the density from the init to surface function
					//and also only repport the densities of active particles
					vect_density_min_internal.push_back(paramsTagging.min_density_o);
					vect_density_max_internal.push_back(paramsTagging.max_density_o);
					vect_density_avg_internal.push_back(paramsTagging.avg_density_o);
					vect_density_stdev_internal.push_back(paramsTagging.stdev_density_o);

				}

				/*
				for (int k = 0; k < vect_t1_internal.size(); ++k) {
				std::cout << k << "  " << vect_t1_internal[k] << "  " << vect_tselection_internal[k] << "  " <<
				vect_count_iter_internal[k] << "  " << vect_density_min_internal[k] << "  " << vect_density_max_internal[k] << std::endl;
				}
				//*/

				std::sort(vect_t1_internal.begin(), vect_t1_internal.end());
				std::sort(vect_t2_internal.begin(), vect_t2_internal.end());
				std::sort(vect_t3_internal.begin(), vect_t3_internal.end());
				std::sort(vect_tselection_internal.begin(), vect_tselection_internal.end());
				std::sort(vect_count_iter_internal.begin(), vect_count_iter_internal.end());
				std::sort(vect_density_min_internal.begin(), vect_density_min_internal.end());
				std::sort(vect_density_max_internal.begin(), vect_density_max_internal.end());
				std::sort(vect_density_avg_internal.begin(), vect_density_avg_internal.end());
				std::sort(vect_density_stdev_internal.begin(), vect_density_stdev_internal.end());

				/*
				for (int k = 0; k < vect_t1_internal.size(); ++k) {
				std::cout << k << "  " << vect_t1_internal[k] << "  " << vect_tselection_internal[k] << "  " <<
				vect_count_iter_internal[k] << "  " << vect_density_min_internal[k] << "  " << vect_density_max_internal[k] << std::endl;
				}
				//*/


				int idMedian = std::floor((vect_t2_internal.size() - 1) / 2.0f);

				//std::cout << "id median: " << idMedian << std::endl;


				vect_count_iter_medians.push_back(vect_count_iter_internal[idMedian]);
				vect_t1_medians.push_back(vect_t1_internal[idMedian]);
				vect_t2_medians.push_back(vect_t2_internal[idMedian]);
				vect_t3_medians.push_back(vect_t3_internal[idMedian]);
				vect_tselection_medians.push_back(vect_tselection_internal[idMedian]);
				vect_step_size.push_back(paramsTagging.step_density);
				vect_density_min_medians.push_back(vect_density_min_internal[idMedian]);
				vect_density_max_medians.push_back(vect_density_max_internal[idMedian]);
				vect_density_avg_medians.push_back(vect_density_avg_internal[idMedian]);
				vect_density_stdev_medians.push_back(vect_density_stdev_internal[idMedian]);
			}

			std::cout << std::endl;

			std::cout << "iter" << " ; " << "step size" << " ; " << "time init" << " ; " <<
				" time selection" << " ; " << " time stabilization" << " ; " <<
				"time stabilization internal" << " ; " << " count iter stabilization"
				<< " ; " << "avg density"
				<< " ; " << "min density" << " ; " << "max density" << " ; " << "stdev density" << std::endl;

			for (int k = 0; k < vect_step_size.size(); ++k) {
				std::cout << k << " ; " << vect_step_size[k] << " ; " << vect_t1_medians[k] << " ; " <<
					vect_t2_medians[k] << " ; " << vect_t3_medians[k] << " ; " <<
					vect_tselection_medians[k] << " ; " << vect_count_iter_medians[k]
					<< " ; " << vect_density_avg_medians[k]
					<< " ; " << vect_density_min_medians[k] << " ; " << vect_density_max_medians[k] <<
					" ; " << vect_density_stdev_medians[k] << std::endl;
			}

		}
		else if (run_type == 8) {
			//this is the experiment to compare the fluif-fluif initialization with the fluid-boundary initialization
			//normaly config 7 is the boundary-fluid and config 8 is the fluid-fluid one

			bool keep_existing_fluid = false;
			int simulation_config = (keep_existing_fluid ? 8 : 7);

			Vector3d normal_gravitation = m_data.gravitation;
			//m_data.gravitation.y *= 5;

			static std::default_random_engine e;
			static std::uniform_real_distribution<> dis(-1, 1); // rage -1 ; 1


			RestFLuidLoaderInterface::InitParameters paramsInit;
			RestFLuidLoaderInterface::TaggingParameters paramsTagging;
			RestFLuidLoaderInterface::LoadingParameters paramsLoading;


			paramsInit.show_debug = true;
			paramsTagging.show_debug = true;
			params.show_debug = true;
			paramsLoading.show_debug = true;

			if (!keep_existing_fluid) {
				paramsLoading.neighbors_tagging_distance_coef = 5;
			}

			int step_size = 60;
			{
				std::vector<RealCuda> vect_t1_internal;
				std::vector<RealCuda> vect_t2_internal;
				std::vector<RealCuda> vect_t3_internal;
				std::vector<int> vect_count_stabilization_iter_internal;
				std::vector<int> vect_count_selection_iter_internal;



				for (int k = 0; k < 2; ++k) {
					//if i keep the existing fluid i need to reload it from memory each loop
					if (keep_existing_fluid) {
						m_data.read_fluid_from_file(false);
					}


					std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

					//handleSimulationLoad(true,false,false,false,false,false);

					//if (k == 0) 
					{
						paramsInit.clear_data = false;
						paramsInit.air_particles_restriction = 1;
						paramsInit.center_loaded_fluid = true;
						paramsInit.keep_existing_fluid = keep_existing_fluid;
						paramsInit.simulation_config = simulation_config;
						paramsInit.apply_additional_offset = true;
						paramsInit.additional_offset = Vector3d(dis(e), dis(e), dis(e))*m_data.particleRadius * 2;

						RestFLuidLoaderInterface::init(m_data, paramsInit);
					}

					std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();

					//*
					paramsTagging.useRule2 = false;
					paramsTagging.useRule3 = true;
					paramsTagging.useStepSizeRegulator = true;
					paramsTagging.min_step_density = 5;
					paramsTagging.step_density = step_size;
					paramsTagging.density_end = 999;
					paramsTagging.keep_existing_fluid = paramsInit.keep_existing_fluid;
					paramsTagging.output_density_information = true;

					paramsLoading.load_fluid = true;
					paramsLoading.keep_existing_fluid = keep_existing_fluid;

					RestFLuidLoaderInterface::initializeFluidToSurface(m_data, true, paramsTagging, paramsLoading);
					//*/
					std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
					//*
					params.method = 0;
					params.timeStep = 0.003;
					{
						params.stabilize_tagged_only = true;
						params.postUpdateVelocityDamping = false;
						params.postUpdateVelocityClamping = false;
						params.preUpdateVelocityClamping = false;


						params.maxErrorD = 0.05;

						params.useDivergenceSolver = true;
						params.useExternalForces = true;

						params.preUpdateVelocityDamping = true;
						params.preUpdateVelocityDamping_val = 0.8;

						params.stabilizationItersCount = 10;

						params.reduceDampingAndClamping = true;
						params.reduceDampingAndClamping_val = std::powf(0.2f / params.preUpdateVelocityDamping_val, 1.0f / (params.stabilizationItersCount - 1));

						params.clearWarmstartAfterStabilization = false;
					}
					params.runCheckParticlesPostion = true;
					params.interuptOnLostParticle = false;
					params.reloadFluid = false;
					params.evaluateStabilization = false;
					params.min_stabilization_iter = 2;
					params.stable_velocity_max_target = m_data.particleRadius*0.25 / m_data.get_current_timestep();
					params.stable_velocity_avg_target = m_data.particleRadius*0.025 / m_data.get_current_timestep();

					std::chrono::steady_clock::time_point tp5 = std::chrono::steady_clock::now();

					RestFLuidLoaderInterface::stabilizeFluid(m_data, params);
					//*/

					std::chrono::steady_clock::time_point tp4 = std::chrono::steady_clock::now();

					RealCuda time_p1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000000.0f;
					RealCuda time_p2 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp3 - tp2).count() / 1000000000.0f;
					RealCuda time_p3 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp4 - tp3).count() / 1000000000.0f;

					vect_t1_internal.push_back(time_p1);
					vect_t2_internal.push_back(time_p2);
					vect_t3_internal.push_back(time_p3);
					vect_count_selection_iter_internal.push_back(paramsTagging.count_iter);
					vect_count_stabilization_iter_internal.push_back(params.count_iter_o);

					//std::cout << "direct timmings output: " << time_p1 << "  " << time_p2 << "  " << time_p3 << "  " << std::endl;
				}


				std::sort(vect_t1_internal.begin(), vect_t1_internal.end());
				std::sort(vect_t2_internal.begin(), vect_t2_internal.end());
				std::sort(vect_t3_internal.begin(), vect_t3_internal.end());
				std::sort(vect_count_selection_iter_internal.begin(), vect_count_selection_iter_internal.end());
				std::sort(vect_count_stabilization_iter_internal.begin(), vect_count_stabilization_iter_internal.end());

				//*
				for (int k = 0; k < vect_t1_internal.size(); ++k) {
					std::cout << k << "   " << vect_t1_internal[k] << " + " << vect_t2_internal[k] <<
						" + " << vect_t3_internal[k] << " = " <<
						vect_t1_internal[k] + vect_t2_internal[k] + vect_t3_internal[k] <<
						" // " << vect_count_selection_iter_internal[k] << " // " << vect_count_stabilization_iter_internal[k] << std::endl;
				}
				//*/
				//some density informations
				std::cout << "density info: " << paramsTagging.avg_density_o << "  " << paramsTagging.min_density_o << "  " <<
					paramsTagging.max_density_o << "  " << paramsTagging.stdev_density_o / paramsTagging.avg_density_o * 100 << "  " << std::endl;


				int idMedian = std::floor((vect_t2_internal.size() - 1) / 2.0f);


				std::cout << "median values" << std::endl;
				std::cout << "time init" << " ; " << "time selection" <<
					" ; " << "time stabilization" << " ; " <<
					"total time" <<
					" ; " << "coutn iter selection" << " ; " << "count iter stabilization" << std::endl;
				std::cout << vect_t1_internal[idMedian] << " ; " << vect_t2_internal[idMedian] <<
					" ; " << vect_t3_internal[idMedian] << " ; " <<
					vect_t1_internal[idMedian] + vect_t2_internal[idMedian] + vect_t3_internal[idMedian] <<
					" ; " << vect_count_selection_iter_internal[idMedian] << " ; " << vect_count_stabilization_iter_internal[idMedian] << std::endl;



			}

			m_data.gravitation = normal_gravitation;


		}
		else if (run_type == 9) {
			//this is the experiment with the pyramid shaped boundaries
			//for that experiment with have 3 meters of fluid inside the pyramid

			bool keep_existing_fluid = false;
			int simulation_config = 1;

			Vector3d normal_gravitation = m_data.gravitation;
			//m_data.gravitation.y *= 5;

			static std::default_random_engine e;
			static std::uniform_real_distribution<> dis(-1, 1); // rage -1 ; 1


			RestFLuidLoaderInterface::InitParameters paramsInit;
			RestFLuidLoaderInterface::TaggingParameters paramsTagging;
			RestFLuidLoaderInterface::LoadingParameters paramsLoading;

			paramsInit.show_debug = false;
			paramsTagging.show_debug = false;
			params.show_debug = false;
			paramsLoading.show_debug = false;

			int step_size = 60;
			{
				std::vector<RealCuda> vect_t1_internal;
				std::vector<RealCuda> vect_t2_internal;
				std::vector<RealCuda> vect_t3_internal;
				std::vector<int> vect_count_stabilization_iter_internal;
				std::vector<int> vect_count_selection_iter_internal;



				for (int k = 0; k <51; ++k) {
					//if i keep the existing fluid i need to reload it from memory each loop
					if (keep_existing_fluid) {
						m_data.read_fluid_from_file(false);
					}


					std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

					//handleSimulationLoad(true,false,false,false,false,false);

					//if (k == 0) 
					{
						paramsInit.clear_data = false;
						paramsInit.air_particles_restriction = 1;
						paramsInit.center_loaded_fluid = true;
						paramsInit.keep_existing_fluid = false;
						paramsInit.simulation_config = simulation_config;
						paramsInit.apply_additional_offset = true;
						paramsInit.additional_offset = Vector3d(dis(e), dis(e), dis(e))*m_data.particleRadius * 2;

						RestFLuidLoaderInterface::init(m_data, paramsInit);
					}

					std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();

					//*
					paramsTagging.useRule2 = false;
					paramsTagging.useRule3 = true;
					paramsTagging.useStepSizeRegulator = true;
					paramsTagging.min_step_density = 5;
					paramsTagging.step_density = step_size;
					paramsTagging.density_end = 999;
					paramsTagging.keep_existing_fluid = paramsInit.keep_existing_fluid;
					paramsTagging.output_density_information = true;

					paramsLoading.load_fluid = true;
					paramsLoading.keep_existing_fluid = keep_existing_fluid;

					RestFLuidLoaderInterface::initializeFluidToSurface(m_data, true, paramsTagging, paramsLoading);
					//*/
					std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
					//*
					params.method = 0;
					params.timeStep = 0.003;
					{
						params.stabilize_tagged_only = true;
						params.postUpdateVelocityDamping = false;
						params.postUpdateVelocityClamping = false;
						params.preUpdateVelocityClamping = false;


						params.maxErrorD = 0.05;

						params.useDivergenceSolver = true;
						params.useExternalForces = true;

						params.preUpdateVelocityDamping = true;
						params.preUpdateVelocityDamping_val = 0.8;

						params.stabilizationItersCount = 10;

						params.reduceDampingAndClamping = true;
						params.reduceDampingAndClamping_val = std::powf(0.2f / params.preUpdateVelocityDamping_val, 1.0f / (params.stabilizationItersCount - 1));

						params.clearWarmstartAfterStabilization = false;
					}
					params.runCheckParticlesPostion = true;
					params.interuptOnLostParticle = false;
					params.reloadFluid = false;
					params.evaluateStabilization = false;
					params.min_stabilization_iter = 2;
					params.stable_velocity_max_target = m_data.particleRadius*0.25 / m_data.get_current_timestep();
					params.stable_velocity_avg_target = m_data.particleRadius*0.025 / m_data.get_current_timestep();

					std::chrono::steady_clock::time_point tp5 = std::chrono::steady_clock::now();

					RestFLuidLoaderInterface::stabilizeFluid(m_data, params);

					//if there was a fail do not count the run in the values
					if (!params.stabilization_sucess) {
						continue;
					}
					//*/

					std::chrono::steady_clock::time_point tp4 = std::chrono::steady_clock::now();

					RealCuda time_p1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000000.0f;
					RealCuda time_p2 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp3 - tp2).count() / 1000000000.0f;
					RealCuda time_p3 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp4 - tp3).count() / 1000000000.0f;

					vect_t1_internal.push_back(time_p1);
					vect_t2_internal.push_back(time_p2);
					vect_t3_internal.push_back(time_p3);
					vect_count_selection_iter_internal.push_back(paramsTagging.count_iter);
					vect_count_stabilization_iter_internal.push_back(params.count_iter_o);

					//std::cout << "direct timmings output: " << time_p1 << "  " << time_p2 << "  " << time_p3 << "  " << std::endl;
				}

				if (!vect_t1_internal.empty()) {


					std::sort(vect_t1_internal.begin(), vect_t1_internal.end());
					std::sort(vect_t2_internal.begin(), vect_t2_internal.end());
					std::sort(vect_t3_internal.begin(), vect_t3_internal.end());
					std::sort(vect_count_selection_iter_internal.begin(), vect_count_selection_iter_internal.end());
					std::sort(vect_count_stabilization_iter_internal.begin(), vect_count_stabilization_iter_internal.end());

					//*
					for (int k = 0; k < vect_t1_internal.size(); ++k) {
						std::cout << k << "   " << vect_t1_internal[k] << " + " << vect_t2_internal[k] <<
							" + " << vect_t3_internal[k] << " = " <<
							vect_t1_internal[k] + vect_t2_internal[k] + vect_t3_internal[k] <<
							" // " << vect_count_selection_iter_internal[k] << " // " << vect_count_stabilization_iter_internal[k] << std::endl;
					}
					//*/
					//some density informations
					std::cout << "density info: " << paramsTagging.avg_density_o << "  " << paramsTagging.min_density_o << "  " <<
						paramsTagging.max_density_o << "  " << paramsTagging.stdev_density_o / paramsTagging.avg_density_o * 100 << "  " << std::endl;


					int idMedian = std::floor((vect_t2_internal.size() - 1) / 2.0f);


					std::cout << "median values" << std::endl;
					std::cout << "count valid runs: " << vect_t1_internal.size() << std::endl;
					std::cout << "time init" << " ; " << "time selection" <<
						" ; " << "time stabilization" << " ; " <<
						"total time" <<
						" ; " << "coutn iter selection" << " ; " << "count iter stabilization" << std::endl;
					std::cout << vect_t1_internal[idMedian] << " ; " << vect_t2_internal[idMedian] <<
						" ; " << vect_t3_internal[idMedian] << " ; " <<
						vect_t1_internal[idMedian] + vect_t2_internal[idMedian] + vect_t3_internal[idMedian] <<
						" ; " << vect_count_selection_iter_internal[idMedian] << " ; " << vect_count_stabilization_iter_internal[idMedian] << std::endl;

				}

			}

			m_data.gravitation = normal_gravitation;


		}
		else if (run_type == 10) {
			//this is the experiment with the floating cube
			//for this experiment we have a floating cube that was rotated 45 degree on top of the fluid

			bool keep_existing_fluid = false;
			int simulation_config = 4;

			Vector3d normal_gravitation = m_data.gravitation;
			//m_data.gravitation.y *= 5;

			static std::default_random_engine e;
			static std::uniform_real_distribution<> dis(-1, 1); // rage -1 ; 1


			RestFLuidLoaderInterface::InitParameters paramsInit;
			RestFLuidLoaderInterface::TaggingParameters paramsTagging;
			RestFLuidLoaderInterface::LoadingParameters paramsLoading;

			paramsInit.show_debug = true;
			paramsTagging.show_debug = true;
			params.show_debug = true;
			paramsLoading.show_debug = true;


			int step_size = 60;
			{
				std::vector<RealCuda> vect_t1_internal;
				std::vector<RealCuda> vect_t2_internal;
				std::vector<RealCuda> vect_t3_internal;
				std::vector<int> vect_count_stabilization_iter_internal;
				std::vector<int> vect_count_selection_iter_internal;



				for (int k = 0; k <1; ++k) {
					//if i keep the existing fluid i need to reload it from memory each loop
					if (keep_existing_fluid) {
						m_data.read_fluid_from_file(false);
					}


					std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

					//handleSimulationLoad(true,false,false,false,false,false);

					//if (k == 0) 
					{
						paramsInit.clear_data = false;
						paramsInit.air_particles_restriction = 1;
						paramsInit.center_loaded_fluid = true;
						paramsInit.keep_existing_fluid = false;
						paramsInit.simulation_config = simulation_config;
						paramsInit.apply_additional_offset = true;
						paramsInit.additional_offset = Vector3d(dis(e), dis(e), dis(e))*m_data.particleRadius * 2;

						RestFLuidLoaderInterface::init(m_data, paramsInit);
					}

					std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();

					//*
					paramsTagging.useRule2 = false;
					paramsTagging.useRule3 = true;
					paramsTagging.useStepSizeRegulator = true;
					paramsTagging.min_step_density = 5;
					paramsTagging.step_density = step_size;
					paramsTagging.density_end = 999;
					paramsTagging.keep_existing_fluid = paramsInit.keep_existing_fluid;
					paramsTagging.output_density_information = true;

					paramsLoading.load_fluid = true;
					paramsLoading.keep_existing_fluid = keep_existing_fluid;

					RestFLuidLoaderInterface::initializeFluidToSurface(m_data, true, paramsTagging, paramsLoading);
					//*/
					std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
					//*
					params.method = 0;
					params.timeStep = 0.003;
					{
						params.stabilize_tagged_only = true;
						params.postUpdateVelocityDamping = false;
						params.postUpdateVelocityClamping = false;
						params.preUpdateVelocityClamping = false;


						params.maxErrorD = 0.05;

						params.useDivergenceSolver = true;
						params.useExternalForces = true;

						params.preUpdateVelocityDamping = true;
						params.preUpdateVelocityDamping_val = 0.8;

						params.stabilizationItersCount = 10;

						params.reduceDampingAndClamping = true;
						params.reduceDampingAndClamping_val = std::powf(0.2f / params.preUpdateVelocityDamping_val, 1.0f / (params.stabilizationItersCount - 1));

						params.clearWarmstartAfterStabilization = false;
					}
					params.runCheckParticlesPostion = true;
					params.interuptOnLostParticle = false;
					params.reloadFluid = false;
					params.evaluateStabilization = false;
					params.min_stabilization_iter = 2;
					params.stable_velocity_max_target = m_data.particleRadius*0.25 / m_data.get_current_timestep();
					params.stable_velocity_avg_target = m_data.particleRadius*0.025 / m_data.get_current_timestep();

					std::chrono::steady_clock::time_point tp5 = std::chrono::steady_clock::now();

					RestFLuidLoaderInterface::stabilizeFluid(m_data, params);

					//if there was a fail do not count the run in the values
					if (!params.stabilization_sucess) {
						continue;
					}
					//*/

					std::chrono::steady_clock::time_point tp4 = std::chrono::steady_clock::now();

					RealCuda time_p1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000000.0f;
					RealCuda time_p2 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp3 - tp2).count() / 1000000000.0f;
					RealCuda time_p3 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp4 - tp3).count() / 1000000000.0f;

					vect_t1_internal.push_back(time_p1);
					vect_t2_internal.push_back(time_p2);
					vect_t3_internal.push_back(time_p3);
					vect_count_selection_iter_internal.push_back(paramsTagging.count_iter);
					vect_count_stabilization_iter_internal.push_back(params.count_iter_o);

					//std::cout << "direct timmings output: " << time_p1 << "  " << time_p2 << "  " << time_p3 << "  " << std::endl;
				}

				if (!vect_t1_internal.empty()) {


					std::sort(vect_t1_internal.begin(), vect_t1_internal.end());
					std::sort(vect_t2_internal.begin(), vect_t2_internal.end());
					std::sort(vect_t3_internal.begin(), vect_t3_internal.end());
					std::sort(vect_count_selection_iter_internal.begin(), vect_count_selection_iter_internal.end());
					std::sort(vect_count_stabilization_iter_internal.begin(), vect_count_stabilization_iter_internal.end());

					//*
					for (int k = 0; k < vect_t1_internal.size(); ++k) {
						std::cout << k << "   " << vect_t1_internal[k] << " + " << vect_t2_internal[k] <<
							" + " << vect_t3_internal[k] << " = " <<
							vect_t1_internal[k] + vect_t2_internal[k] + vect_t3_internal[k] <<
							" // " << vect_count_selection_iter_internal[k] << " // " << vect_count_stabilization_iter_internal[k] << std::endl;
					}
					//*/
					//some density informations
					std::cout << "density info: " << paramsTagging.avg_density_o << "  " << paramsTagging.min_density_o << "  " <<
						paramsTagging.max_density_o << "  " << paramsTagging.stdev_density_o / paramsTagging.avg_density_o * 100 << "  " << std::endl;


					int idMedian = std::floor((vect_t2_internal.size() - 1) / 2.0f);


					std::cout << "median values" << std::endl;
					std::cout << "count valid runs: " << vect_t1_internal.size() << std::endl;
					std::cout << "time init" << " ; " << "time selection" <<
						" ; " << "time stabilization" << " ; " <<
						"total time" <<
						" ; " << "coutn iter selection" << " ; " << "count iter stabilization" << std::endl;
					std::cout << vect_t1_internal[idMedian] << " ; " << vect_t2_internal[idMedian] <<
						" ; " << vect_t3_internal[idMedian] << " ; " <<
						vect_t1_internal[idMedian] + vect_t2_internal[idMedian] + vect_t3_internal[idMedian] <<
						" ; " << vect_count_selection_iter_internal[idMedian] << " ; " << vect_count_stabilization_iter_internal[idMedian] << std::endl;

				}

			}

			m_data.gravitation = normal_gravitation;


		}
		else if (run_type == 11) {
			//this can be used to load any boundary shape that has a config and no existing fluid

			bool keep_existing_fluid = false;
			int simulation_config = 15;

			Vector3d normal_gravitation = m_data.gravitation;
			//m_data.gravitation.y *= 5;

			static std::default_random_engine e;
			static std::uniform_real_distribution<> dis(-1, 1); // rage -1 ; 1


			RestFLuidLoaderInterface::InitParameters paramsInit;
			RestFLuidLoaderInterface::TaggingParameters paramsTagging;
			RestFLuidLoaderInterface::LoadingParameters paramsLoading;

			paramsInit.show_debug = true;
			paramsTagging.show_debug = true;
			params.show_debug = true;
			paramsLoading.show_debug = true;

			paramsLoading.load_raw_untaged_data = false;

			int step_size = 60;
			{
				std::vector<RealCuda> vect_t1_internal;
				std::vector<RealCuda> vect_t2_internal;
				std::vector<RealCuda> vect_t3_internal;
				std::vector<int> vect_count_stabilization_iter_internal;
				std::vector<int> vect_count_selection_iter_internal;



				for (int k = 0; k <1; ++k) {
					//if i keep the existing fluid i need to reload it from memory each loop
					if (keep_existing_fluid) {
						m_data.read_fluid_from_file(false);
					}


					std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

					//handleSimulationLoad(true,false,false,false,false,false);

					//if (k == 0) 
					{
						paramsInit.clear_data = false;
						paramsInit.air_particles_restriction = 1;
						paramsInit.center_loaded_fluid = true;
						paramsInit.keep_existing_fluid = false;
						paramsInit.simulation_config = simulation_config;
						paramsInit.apply_additional_offset = true;
						paramsInit.additional_offset = Vector3d(dis(e), dis(e), dis(e))*m_data.particleRadius * 2;

						RestFLuidLoaderInterface::init(m_data, paramsInit);
					}

					std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();

					//*
					paramsTagging.useRule2 = false;
					paramsTagging.useRule3 = true;
					paramsTagging.useStepSizeRegulator = true;
					paramsTagging.min_step_density = 5;
					paramsTagging.step_density = step_size;
					paramsTagging.density_end = 999;
					paramsTagging.keep_existing_fluid = paramsInit.keep_existing_fluid;
					paramsTagging.output_density_information = true;

					paramsLoading.load_fluid = true;
					paramsLoading.keep_existing_fluid = keep_existing_fluid;

					RestFLuidLoaderInterface::initializeFluidToSurface(m_data, true, paramsTagging, paramsLoading);
					//*/
					std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
					/*
					params.method = 0;
					params.timeStep = 0.003;
					{
					params.stabilize_tagged_only = true;
					params.postUpdateVelocityDamping = false;
					params.postUpdateVelocityClamping = false;
					params.preUpdateVelocityClamping = false;


					params.maxErrorD = 0.05;

					params.useDivergenceSolver = true;
					params.useExternalForces = true;

					params.preUpdateVelocityDamping = true;
					params.preUpdateVelocityDamping_val = 0.8;

					params.stabilizationItersCount = 10;

					params.reduceDampingAndClamping = true;
					params.reduceDampingAndClamping_val = std::powf(0.2f / params.preUpdateVelocityDamping_val, 1.0f / (params.stabilizationItersCount - 1));

					params.clearWarmstartAfterStabilization = false;
					}
					params.runCheckParticlesPostion = true;
					params.interuptOnLostParticle = false;
					params.reloadFluid = false;
					params.evaluateStabilization = false;
					params.min_stabilization_iter = 2;
					params.stable_velocity_max_target = m_data.particleRadius*0.25 / m_data.get_current_timestep();
					params.stable_velocity_avg_target = m_data.particleRadius*0.025 / m_data.get_current_timestep();

					std::chrono::steady_clock::time_point tp5 = std::chrono::steady_clock::now();

					RestFLuidLoaderInterface::stabilizeFluid(m_data, params);

					//if there was a fail do not count the run in the values
					if (!params.stabilization_sucess) {
					continue;
					}
					//*/

					std::chrono::steady_clock::time_point tp4 = std::chrono::steady_clock::now();

					RealCuda time_p1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000000.0f;
					RealCuda time_p2 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp3 - tp2).count() / 1000000000.0f;
					RealCuda time_p3 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp4 - tp3).count() / 1000000000.0f;

					vect_t1_internal.push_back(time_p1);
					vect_t2_internal.push_back(time_p2);
					vect_t3_internal.push_back(time_p3);
					vect_count_selection_iter_internal.push_back(paramsTagging.count_iter);
					vect_count_stabilization_iter_internal.push_back(params.count_iter_o);

					//std::cout << "direct timmings output: " << time_p1 << "  " << time_p2 << "  " << time_p3 << "  " << std::endl;
				}

				if (!vect_t1_internal.empty()) {


					std::sort(vect_t1_internal.begin(), vect_t1_internal.end());
					std::sort(vect_t2_internal.begin(), vect_t2_internal.end());
					std::sort(vect_t3_internal.begin(), vect_t3_internal.end());
					std::sort(vect_count_selection_iter_internal.begin(), vect_count_selection_iter_internal.end());
					std::sort(vect_count_stabilization_iter_internal.begin(), vect_count_stabilization_iter_internal.end());

					//*
					for (int k = 0; k < vect_t1_internal.size(); ++k) {
						std::cout << k << "   " << vect_t1_internal[k] << " + " << vect_t2_internal[k] <<
							" + " << vect_t3_internal[k] << " = " <<
							vect_t1_internal[k] + vect_t2_internal[k] + vect_t3_internal[k] <<
							" // " << vect_count_selection_iter_internal[k] << " // " << vect_count_stabilization_iter_internal[k] << std::endl;
					}
					//*/
					//some density informations
					std::cout << "density info: " << paramsTagging.avg_density_o << "  " << paramsTagging.min_density_o << "  " <<
						paramsTagging.max_density_o << "  " << paramsTagging.stdev_density_o / paramsTagging.avg_density_o * 100 << "  " << std::endl;


					int idMedian = std::floor((vect_t2_internal.size() - 1) / 2.0f);


					std::cout << "median values" << std::endl;
					std::cout << "count valid runs: " << vect_t1_internal.size() << std::endl;
					std::cout << "time init" << " ; " << "time selection" <<
						" ; " << "time stabilization" << " ; " <<
						"total time" <<
						" ; " << "coutn iter selection" << " ; " << "count iter stabilization" << std::endl;
					std::cout << vect_t1_internal[idMedian] << " ; " << vect_t2_internal[idMedian] <<
						" ; " << vect_t3_internal[idMedian] << " ; " <<
						vect_t1_internal[idMedian] + vect_t2_internal[idMedian] + vect_t3_internal[idMedian] <<
						" ; " << vect_count_selection_iter_internal[idMedian] << " ; " << vect_count_stabilization_iter_internal[idMedian] << std::endl;

				}

			}

			m_data.gravitation = normal_gravitation;


		}




}


void DFSPHCUDA::applyOpenBoundaries() {
	if (count_steps == 0) {
		OpenBoundariesSimpleInterface::InitParameters initParams;
		initParams.show_debug = true;
		initParams.simulation_config = 1001;
		OpenBoundariesSimpleInterface::init(m_data, initParams);


	}
	else {
		std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

		OpenBoundariesSimpleInterface::ApplyParameters applyParams;
		applyParams.show_debug = false;
		applyParams.allowedNewDistance = m_data.particleRadius*1.75;
		applyParams.allowedNewDensity = 700;
		applyParams.useInflow = true;
		applyParams.useOutflow = true;
		OpenBoundariesSimpleInterface::applyOpenBoundary(m_data, applyParams);


		std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();
		RealCuda time_p1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000.0f;

		//std::cout << "time handling open boundaries: " << time_p1 << std::endl;
	}


	//this is the output for the experiment to study how well the openboundary absorb the perturbations
	if (false) {
		//essencially I only what to measure the energy left in the fluid
		//the energy is the sum of the square of the velocities of all particles
		//read data to CPU
		static Vector3d* vel = NULL;
		static Vector3d* pos = NULL;
		int size = 0;
		if (m_data.fluid_data->numParticles > size) {
			if (vel != NULL) {
				delete[] vel;
			}
			if (pos != NULL) {
				delete[] pos;
			}
			vel = new Vector3d[m_data.fluid_data->numParticlesMax];
			pos = new Vector3d[m_data.fluid_data->numParticlesMax];
			size = m_data.fluid_data->numParticlesMax;

		}
		read_UnifiedParticleSet_cuda(*(m_data.fluid_data), pos, vel, NULL);

		//read the actual evaluation
		RealCuda stabilzationEvaluation = -1;

		RealCuda sumSqVelLeft = 0;
		RealCuda sumSqVelRight = 0;
		int countParticlesLeft = 0;
		int countParticlesRight = 0;
		for (int i = 0; i < m_data.fluid_data->numParticles; ++i) {
			if (pos[i].z < 0) {
				sumSqVelLeft += vel[i].squaredNorm();
				countParticlesLeft++;
			}
			else {
				sumSqVelRight += vel[i].squaredNorm();
				countParticlesRight++;
			}
		}

		std::cout << count_steps << " ; " << count_steps * 0.003
			<< " ; " << countParticlesLeft
			<< " ; " << countParticlesRight
			<< " ; " << sumSqVelLeft
			<< " ; " << sumSqVelRight
			<< std::endl;

		std::string filename = "temp56.csv";
		if (count_steps == 0) {
			std::remove(filename.c_str());
		}
		ofstream myfile;
		myfile.open(filename, std::ios_base::app);
		if (myfile.is_open()) {
			myfile << count_steps << " ; " << count_steps * 0.003
				<< " ; " << countParticlesLeft
				<< " ; " << countParticlesRight
				<< " ; " << sumSqVelLeft
				<< " ; " << sumSqVelRight
				<< std::endl;


			//myfile << total_time / (count_steps + 1) << ", " << m_iterations << ", " << m_iterationsV << std::endl;;
			myfile.close();
		}
		else {
			std::cout << "failed to open file: " << filename << "   reason: " << std::strerror(errno) << std::endl;
		}
	}
}

void DFSPHCUDA::applyDynamicWindow() {
	if (count_steps == 0) {
		DynamicWindowInterface::InitParameters initParams;
		initParams.show_debug = true;
		initParams.simulation_config = 5;
		initParams.air_particles_restriction = 1;
		initParams.keep_existing_fluid = false;
		initParams.clear_data = false;
		initParams.max_allowed_displacement = m_data.getKernelRadius() * 4;
		DynamicWindowInterface::init(m_data, initParams);

	}
	else {
		Vector3d potentialDisplacement(0, 0, 0);
		if (m_data.numDynamicBodies > 0) {
			Vector3d current_center = m_data.dynamicWindowTotalDisplacement;
			Vector3d cur_interest_position = m_data.vector_dynamic_bodies_data[0].rigidBody_cpu->position;
			potentialDisplacement = cur_interest_position - current_center;
			potentialDisplacement.y = 0;
		}
		else {
			if (count_steps % 10 == 0) {
				potentialDisplacement = Vector3d(m_data.getKernelRadius() * 3, 0, 0);
			}
		}


		//m_data.vector_dynamic_bodies_data->rigidBody_cpu->position;
		//if (count_steps == 1)
		//if ((count_steps % 10) == 0)
		//if (false)
		if (potentialDisplacement.norm() > (m_data.getKernelRadius()*2.5))
		{
			//set a limit to the displacement
			RealCuda max_displacement_norm = (m_data.getKernelRadius()*3.5);
			if (potentialDisplacement.norm() > max_displacement_norm) {
				potentialDisplacement.toUnit();
				potentialDisplacement *= max_displacement_norm;
			}



			std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();


			DynamicWindowInterface::TaggingParameters paramsTagging;
			DynamicWindowInterface::LoadingParameters paramsLoading;
			DynamicWindowInterface::StabilizationParameters paramsStabilization;

			paramsTagging.show_debug = false;
			paramsLoading.show_debug = false;
			paramsStabilization.show_debug = false;

			bool run_stabilization = true;

			//qsdqs
			{
				//paramsTagging.displacement = Vector3d(-m_data.getKernelRadius() * 3, 0, 0);
				paramsTagging.displacement = potentialDisplacement;
				paramsTagging.useRule2 = false;
				paramsTagging.useRule3 = true;
				paramsTagging.useStepSizeRegulator = true;
				paramsTagging.min_step_density = 5;
				paramsTagging.step_density = 60;
				paramsTagging.density_end = 999;
				paramsTagging.keep_existing_fluid = true;
				paramsTagging.output_density_information = false;
				paramsTagging.output_timming_information = false;




				paramsLoading.load_fluid = true;
				paramsLoading.set_up_tagging = run_stabilization;
				paramsLoading.keep_existing_fluid = true;
				paramsLoading.tag_active_neigbors = false;
				paramsLoading.neighbors_tagging_distance_coef = 2;
				paramsLoading.tag_active_neigbors_use_repetition_approach = false;

				DynamicWindowInterface::initializeFluidToSurface(m_data, paramsTagging, paramsLoading);
			}



			std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();

			if (run_stabilization) {
				paramsStabilization.max_iterEval = 30;


				paramsStabilization.method = 0;
				paramsStabilization.timeStep = 0.003;
				{
					paramsStabilization.stabilize_tagged_only = true;
					paramsStabilization.postUpdateVelocityDamping = false;
					paramsStabilization.postUpdateVelocityClamping = false;
					paramsStabilization.preUpdateVelocityClamping = false;


					paramsStabilization.maxErrorD = 0.05;

					paramsStabilization.useDivergenceSolver = false;
					paramsStabilization.useExternalForces = true;

					paramsStabilization.preUpdateVelocityDamping = true;
					paramsStabilization.preUpdateVelocityDamping_val = 0.8;

					paramsStabilization.stabilizationItersCount = 10;


					paramsStabilization.reduceDampingAndClamping = true;
					paramsStabilization.reduceDampingAndClamping_val = std::powf(0.2f /
						paramsStabilization.preUpdateVelocityDamping_val, 1.0f / (paramsStabilization.stabilizationItersCount - 1));

					paramsStabilization.clearWarmstartAfterStabilization = false;


					paramsStabilization.stabilizationItersCount = 10;
				}
				paramsStabilization.runCheckParticlesPostion = false;
				paramsStabilization.interuptOnLostParticle = false;
				paramsStabilization.reloadFluid = false;
				paramsStabilization.evaluateStabilization = false;
				//making that number laarge essencially desactivete the system
				paramsStabilization.min_stabilization_iter = 1000;
				paramsStabilization.stable_velocity_max_target = m_data.particleRadius*0.25 / m_data.get_current_timestep();
				paramsStabilization.stable_velocity_avg_target = m_data.particleRadius*0.025 / m_data.get_current_timestep();


				DynamicWindowInterface::stabilizeFluid(m_data, paramsStabilization);
			}



			std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();

			//this gives results in miliseconds
			RealCuda time_p1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000.0f;
			RealCuda time_p2 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp3 - tp2).count() / 1000000.0f;

			std::cout << " time dynamic window tag+load/stab/total: " << time_p1 << "  " << time_p2 << "  " <<
				time_p1 + time_p2 << std::endl;
			//std::this_thread::sleep_for(std::chrono::nanoseconds(10));
			//std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(2));

			//count_steps++;
			//return;
		}

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


void DFSPHCUDA::initAdvancedRendering(int width, int height) {
    m_data.initAdvancedRendering(width, height);
}


