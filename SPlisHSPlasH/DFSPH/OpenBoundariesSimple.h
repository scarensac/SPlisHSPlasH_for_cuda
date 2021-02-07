#ifndef OPEN_BOUNDARIES_SIMPLE
#define OPEN_BOUNDARIES_SIMPLE


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"



namespace SPH {
	class OpenBoundariesSimpleInterface {

	public:
		struct InitParameters {
			bool show_debug;
			int simulation_config;
			
			InitParameters(){
				show_debug = false;
				simulation_config = 0;
			}
		};

		static void init(DFSPHCData& data, InitParameters& params);


		struct ApplyParameters {
			bool show_debug;

			//inflow parameters
			bool useInflow;
			RealCuda allowedNewDistance;

			//outflow parameters
			bool useOutflow;

			ApplyParameters() {
				show_debug = false;
				allowedNewDistance = -1;
				useInflow = true;
				useOutflow = true;
			}
		};

		static void applyOpenBoundary(DFSPHCData& data, ApplyParameters& params);
	};

}

#endif //OPEN_BOUNDARIES_SIMPLE