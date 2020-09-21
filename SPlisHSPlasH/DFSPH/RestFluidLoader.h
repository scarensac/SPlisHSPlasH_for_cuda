#ifndef REST_FLUID_LOADER
#define REST_FLUID_LOADER


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"



namespace SPH {
	class RestFLuidLoaderInterface {

	public:
		
		static void init(DFSPHCData& data);

		static bool isInitialized();
		
		//ok here I'll test a system to initialize a volume of fluid from
		//a large wolume of fluid (IE a technique to iinit the fluid at rest)
		static void initializeFluidToSurface(SPH::DFSPHCData& data);


		//this struct is only to be more flexible in the addition of stabilization methods in the stabilizeFluid function 
		struct StabilizationParameters {
			int method;
			int stabilizationItersCount;
			RealCuda timeStep;

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
			bool postUpdateVelocityDamping ;
			RealCuda postUpdateVelocityDamping_val ;



			bool runCheckParticlesPostion;
			bool interuptOnLostParticle;

			//params for the particle packing method
			RealCuda p_b;//2500 * delta_s;
			RealCuda k_r;// 150 * delta_s * delta_s;
			RealCuda zeta;// 2 * (SQRT_MACRO_CUDA(delta_s) + 1) / delta_s;



			//params for the evaluation
			RealCuda evaluateStabilization;
			RealCuda stabilzationEvaluation;
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

				runCheckParticlesPostion = true;
				interuptOnLostParticle = true;


				p_b = -1;
				k_r = -1;
				zeta = -1;

				evaluateStabilization = true;
				stabilzationEvaluation = -1;
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