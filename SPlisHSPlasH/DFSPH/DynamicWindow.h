#ifndef DYNAMIC_WINDOW
#define DYNAMIC_WINDOW


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"

/**
Dynamic Window

This system allow the user to move the simulated area around as if there was an infinite ocean.
	- this system is a specialization of the RestFluidLoader system.
The calls to the actual class function should be done through the class DynamicWindowInterface

Requirements:
- Specify the simulation boundary shape in a configuration in the init() function
	this can be done either by using the BufferFluidSurface, by using either geometric primitives or actual triangular meshes

- create a file containing a distribution of particles that represent the sampling point that can be used to add new fluid particles
	- This should be done by simulating a fluid for the same boundary shape until it is at rest
	- the name of this file should be specified in the init function and should be located in
		a folder named folderDynamicWindow in the folder used to save fluid distribution

System use:

- first call the init function after having create the boundary particles
- then call the applyOpenBoundary function at the start of the simulation step

*/
namespace SPH {
	class DynamicWindowInterface {

	public:
		static void clear();
		
		struct InitParameters {
			bool clear_data;

			bool show_debug;

			bool center_loaded_fluid;
			bool apply_additional_offset;
			Vector3d additional_offset;

			bool keep_existing_fluid;

			int air_particles_restriction;

			int simulation_config;

			RealCuda max_allowed_displacement;

			InitParameters() {
				clear_data = false;
				show_debug = false;
				center_loaded_fluid = false;
				apply_additional_offset = false;
				additional_offset;

				keep_existing_fluid = false;

				air_particles_restriction=1;

				simulation_config = 0;

				max_allowed_displacement = 0.5;
			}

		};

		static void init(DFSPHCData& data, InitParameters& params);


		static bool isInitialized();

		struct TaggingParameters {
			bool show_debug;
			bool keep_existing_fluid;

			RealCuda density_start ;
			RealCuda density_end;
			RealCuda step_density;

			bool useRule2;
			RealCuda min_density;

			bool useRule3;
			RealCuda density_delta_threshold;

			bool useStepSizeRegulator;
            RealCuda min_step_density;
			RealCuda step_to_target_delta_change_trigger_ratio{ 1 };


			//here the parameters for the displacement
			Vector3d displacement;


			//here are some output values
			unsigned int count_iter;
			RealCuda time_total;

			bool output_density_information;
			RealCuda min_density_o;
			RealCuda max_density_o;
			RealCuda avg_density_o;
			RealCuda stdev_density_o;

			bool output_timming_information;

			TaggingParameters(){
				show_debug = false;

				density_start = 1900;
				density_end = 1000;
				step_density = 50;

				useRule2 = false;
				min_density = 905;
			
				useRule3 = false;
				density_delta_threshold = 5;

				useStepSizeRegulator = true;
				min_step_density = 5;

				keep_existing_fluid = false;

				count_iter = 0; 
				time_total=0;
				
				displacement = Vector3d(0, 0, 0);

				output_density_information = false;
				min_density_o = 10000;
				max_density_o = 0;
				avg_density_o = 0;
				stdev_density_o = 0;

				output_timming_information = false;
			}
		};

		struct LoadingParameters {
			bool load_fluid;
			bool keep_air_particles;
			bool set_up_tagging;
			bool keep_existing_fluid;

			bool show_debug;

			RealCuda neighbors_tagging_distance_coef;
			bool tag_active_neigbors;
			bool tag_active_neigbors_use_repetition_approach;

			LoadingParameters() {
				load_fluid=true;
				keep_air_particles = false;
				set_up_tagging = true;
				keep_existing_fluid = false;
				show_debug = false;
				tag_active_neigbors=true;
				neighbors_tagging_distance_coef = 2;
				tag_active_neigbors_use_repetition_approach=false;
			}
		};

		//ok here I'll test a system to initialize a volume of fluid from
		//a large wolume of fluid (IE a technique to iinit the fluid at rest)
		static void initializeFluidToSurface(SPH::DFSPHCData& data, TaggingParameters& params,
			LoadingParameters& params_loading);


		//this struct is only to be more flexible in the addition of stabilization methods in the stabilizeFluid function 
		struct StabilizationParameters {
			int method;
			int stabilizationItersCount;
			RealCuda timeStep;
			bool reloadFluid;
			bool keep_existing_fluid;
			bool stabilize_tagged_only;
			bool stabilization_sucess;

			bool show_debug;

			//those are parameters for when using the SPH simulation step to stabilize the fluid
			bool useDivergenceSolver;
			bool useDensitySolver;
			bool useExternalForces;
			RealCuda maxErrorV;
			RealCuda maxIterV;
			RealCuda maxErrorD;
			RealCuda maxIterD;
			bool useMaxErrorDPreciseAtMinIter;
			RealCuda maxErrorDPrecise;

			bool clearWarmstartAfterStabilization;

			bool preUpdateVelocityClamping;
			RealCuda preUpdateVelocityClamping_val ;
			bool preUpdateVelocityDamping ;
			RealCuda preUpdateVelocityDamping_val ;
			bool postUpdateVelocityClamping ;
			RealCuda postUpdateVelocityClamping_val ;
			bool postUpdateVelocityDamping;
			RealCuda postUpdateVelocityDamping_val;
			bool reduceDampingAndClamping;
			RealCuda reduceDampingAndClamping_val;
			RealCuda countLostParticlesLimit;


			int min_stabilization_iter;
			RealCuda stable_velocity_max_target;
			RealCuda stable_velocity_avg_target;


			bool runCheckParticlesPostion;
			bool interuptOnLostParticle;

			//params for the particle packing method
			RealCuda p_b;//2500 * delta_s;
			RealCuda k_r;// 150 * delta_s * delta_s;
			RealCuda zeta;// 2 * (SQRT_MACRO_CUDA(delta_s) + 1) / delta_s;
			int zetaChangeFrequency;
			RealCuda zetaChangeCoefficient;




			//params for the evaluation
			RealCuda evaluateStabilization;
			RealCuda stabilzationEvaluation1;
			RealCuda stabilzationEvaluation2;
			RealCuda stabilzationEvaluation3;
			RealCuda maxErrorVEval;
			RealCuda maxIterVEval;
			RealCuda maxErrorDEval;
			RealCuda maxIterDEval;
			RealCuda timeStepEval;
			int max_iterEval;

			//some output values for details
			int count_iter_o;

			StabilizationParameters() {
				method = -1;
				stabilizationItersCount = 5;
				timeStep = 0.003;
				reloadFluid = true;
				keep_existing_fluid = false;
				stabilize_tagged_only = false;
				stabilization_sucess = true;

				show_debug = false;

				useDivergenceSolver = true;
				useDensitySolver = true;
				useExternalForces = true;
				maxErrorV = 0.1;
				maxIterV = 100;
				maxErrorD = 0.01;
				maxIterD = 100;
				useMaxErrorDPreciseAtMinIter=false;
				maxErrorDPrecise=0.01;

				clearWarmstartAfterStabilization = true;

				preUpdateVelocityClamping = false;
				preUpdateVelocityClamping_val = 0;
				preUpdateVelocityDamping = false;
				preUpdateVelocityDamping_val = 0;
				postUpdateVelocityClamping = false;
				postUpdateVelocityClamping_val = 0;
				postUpdateVelocityDamping = false;
				postUpdateVelocityDamping_val = 0;
				postUpdateVelocityDamping = false;
				postUpdateVelocityDamping_val = 0;
				countLostParticlesLimit = 10;


				min_stabilization_iter = 2;
				stable_velocity_max_target = 0;
				stable_velocity_avg_target = 0;

				runCheckParticlesPostion = true;
				interuptOnLostParticle = true;


				p_b = -1;
				k_r = -1;
				zeta = 1;
				zetaChangeFrequency=1;
				zetaChangeCoefficient=0.997;

				evaluateStabilization = true;
				stabilzationEvaluation1 = -1;
				stabilzationEvaluation2 = -1;
				stabilzationEvaluation3 = -1;
				maxErrorVEval = 0.1;
				maxIterVEval = 100;
				maxErrorDEval = 0.01;
				maxIterDEval = 100;
				timeStepEval = 0.003;
				max_iterEval = 5;

				count_iter_o = 0;
			}
		};

		//so this is a function that will be used to move around the particles in the fluid to 
		//improove the stability of the fluid when the first time step is ran
		static void stabilizeFluid(SPH::DFSPHCData& data, StabilizationParameters& params);

	};

}

#endif //DYNAMIC_WINDOW