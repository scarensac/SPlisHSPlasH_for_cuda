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
			int max_iter;
			RealCuda threshold;

			bool useDivergenceSolver;
			bool useDensitySolver;
			bool useExternalForces;
			RealCuda maxErrorV;
			RealCuda maxIterV;
			RealCuda maxErrorD;
			RealCuda maxIterD;
			RealCuda timeStep;

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

			RealCuda evaluateStabilization;
			RealCuda stabilzationEvaluation;
			RealCuda maxErrorVEval;
			RealCuda maxIterVEval;
			RealCuda maxErrorDEval;
			RealCuda maxIterDEval;
			RealCuda timeStepEval;

			StabilizationParameters() {
				method = -1;

				useDivergenceSolver = true;
				useDensitySolver = true;
				useExternalForces = true;
				maxErrorV = 0.1;
				maxIterV = 100;
				maxErrorD = 0.01;
				maxIterD = 100;
				timeStep = 0.003;

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

				evaluateStabilization = true;
				stabilzationEvaluation = -1;
				maxErrorVEval = 0.1;
				maxIterVEval = 100;
				maxErrorDEval = 0.01;
				maxIterDEval = 100;
				timeStepEval = 0.003;
			}
		};

		//so this is a function that will be used to move around the particles in the fluid to 
		//improove the stability of the fluid when the first time step is ran
		static void stabilizeFluid(SPH::DFSPHCData& data, StabilizationParameters& params);

	};

}

#endif //REST_FLUID_LOADER