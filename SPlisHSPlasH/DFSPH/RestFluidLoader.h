#ifndef REST_FLUID_LOADER
#define REST_FLUID_LOADER


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"



namespace SPH {
	class RestFLuidLoaderInterface {

	public:
		
		static void init(DFSPHCData& data, bool center_loaded_fluid, bool keep_existing_fluid);


		static bool isInitialized();

		struct TaggingParameters {
			RealCuda density_start ;
			RealCuda density_end;
			RealCuda step_density;

			bool useRule2;
			RealCuda min_density;

			bool useRule3;
			RealCuda density_delta_threshold;

			bool keep_existing_fluid;

			TaggingParameters(){
				density_start = 1900;
				density_end = 1001;
				step_density = 50;

				useRule2 = false;
				min_density = 905;
			
				useRule3 = false;
				density_delta_threshold = 5;

				keep_existing_fluid = false;
			}
		};

		struct LoadingParameters {
			bool load_fluid;
			bool keep_air_particles;
			bool set_up_tagging;
			bool keep_existing_fluid;

			LoadingParameters() {
				load_fluid=true;
				keep_air_particles = false;
				set_up_tagging = true;
				keep_existing_fluid = false;
			}
		};

		//ok here I'll test a system to initialize a volume of fluid from
		//a large wolume of fluid (IE a technique to iinit the fluid at rest)
		static void initializeFluidToSurface(SPH::DFSPHCData& data, bool center_loaded_fluid, TaggingParameters& params, 
			bool load_fluid=true, bool keep_existing_fluid=false);


		//this struct is only to be more flexible in the addition of stabilization methods in the stabilizeFluid function 
		struct StabilizationParameters {
			int method;
			int stabilizationItersCount;
			RealCuda timeStep;
			bool reloadFluid;

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