#ifndef REST_FLUID_LOADER
#define REST_FLUID_LOADER


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"



namespace SPH {
	class RestFLuidLoaderInterface {

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

			InitParameters() {
				clear_data = false;
				show_debug = false;
				center_loaded_fluid = false;
				apply_additional_offset = false;
				additional_offset;

				keep_existing_fluid = false;

				air_particles_restriction=1;

				simulation_config = 0;
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
			RealCuda min_step_density;

			bool useRule2;
			RealCuda min_density;

			bool useRule3;
			RealCuda density_delta_threshold;


			//here are some output values
			unsigned int count_iter;
			RealCuda time_total;

			RealCuda min_density_o;
			RealCuda max_density_o;
			RealCuda avg_density_o;

			TaggingParameters(){
				show_debug = false;

				density_start = 1900;
				density_end = 1000;
				step_density = 50;
				min_step_density = 5;

				useRule2 = false;
				min_density = 905;
			
				useRule3 = false;
				density_delta_threshold = 5;

				keep_existing_fluid = false;

				count_iter = 0; 
				time_total=0;
				min_density_o = 10000;
				max_density_o = 0;
				avg_density_o = 0;
			}
		};

		struct LoadingParameters {
			bool load_fluid;
			bool keep_air_particles;
			bool set_up_tagging;
			bool keep_existing_fluid;

			bool show_debug;

			RealCuda neighbors_tagging_distance_coef;

			LoadingParameters() {
				load_fluid=true;
				keep_air_particles = false;
				set_up_tagging = true;
				keep_existing_fluid = false;
				show_debug = false;
				neighbors_tagging_distance_coef = 2;
			}
		};

		//ok here I'll test a system to initialize a volume of fluid from
		//a large wolume of fluid (IE a technique to iinit the fluid at rest)
		static void initializeFluidToSurface(SPH::DFSPHCData& data, bool center_loaded_fluid, TaggingParameters& params,
			LoadingParameters& params_loading, bool output_min_max_density = false);


		//this struct is only to be more flexible in the addition of stabilization methods in the stabilizeFluid function 
		struct StabilizationParameters {
			int method;
			int stabilizationItersCount;
			RealCuda timeStep;
			bool reloadFluid;
			bool keep_existing_fluid;
			bool stabilize_tagged_only;

			bool show_debug;

			//those are parameters for when using the SPH simulation step to stabilize the fluid
			bool useDivergenceSolver;
			bool useDensitySolver;
			bool useExternalForces;
			RealCuda maxErrorV;
			RealCuda maxIterV;
			RealCuda maxErrorD;
			RealCuda maxIterD;

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
			RealCuda stable_velocity_target;


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

			StabilizationParameters() {
				method = -1;
				stabilizationItersCount = 5;
				timeStep = 0.003;
				reloadFluid = true;
				keep_existing_fluid = false;
				stabilize_tagged_only = false;

				show_debug = false;

				useDivergenceSolver = true;
				useDensitySolver = true;
				useExternalForces = true;
				maxErrorV = 0.1;
				maxIterV = 100;
				maxErrorD = 0.01;
				maxIterD = 100;
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
				stable_velocity_target = 0;

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
			}
		};

		//so this is a function that will be used to move around the particles in the fluid to 
		//improove the stability of the fluid when the first time step is ran
		static void stabilizeFluid(SPH::DFSPHCData& data, StabilizationParameters& params);

	};

}

#endif //REST_FLUID_LOADER