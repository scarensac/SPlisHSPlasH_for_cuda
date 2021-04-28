#ifndef OPEN_BOUNDARIES_SIMPLE
#define OPEN_BOUNDARIES_SIMPLE


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"

/**
Open Boundary Simple

This system allow the user to have a simple version of an open boundary that prevent the reflection of
	waves generated inside the simulation space on the solid boundary.
The calls to the actual class function should be done through the class OpenBoundarySimpleInterface

Requirements:
- Specify the simulation boundary shape in a configuration in the init() function
	this can be done either by using the BufferFluidSurface, by using either geometric primitives or actual triangular meshes

- create a file containing a distribution of particles that represent the sampling point that can be used to add new fluid particles
	- the easy way is to take a fluid at rest for the used boundary shape 
		and the init function will only keep the particle positions close to the boundary
	- the name of this file should be specified in the init function and should be located in 
		a folder named inflowFolder in the folder used to save fluid distribution

System use:

- first call the init function after having create the boundary particles
- then call the applyOpenBoundary function at the start of the simulation step

*/

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
			RealCuda allowedNewDensity;

			//outflow parameters
			bool useOutflow;

			ApplyParameters() {
				show_debug = false;
				allowedNewDistance = -1;
				allowedNewDensity = -1;
				useInflow = true;
				useOutflow = true;
			}
		};

		static void applyOpenBoundary(DFSPHCData& data, ApplyParameters& params);
	};

}

#endif //OPEN_BOUNDARIES_SIMPLE