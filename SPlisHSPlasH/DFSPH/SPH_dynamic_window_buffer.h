#ifndef SPH_DYNAMIC_WINDOW_BUFFER
#define SPH_DYNAMIC_WINDOW_BUFFER


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"



namespace SPH {
	class DynamicWindowInterface {

	public:

		static void initDynamicWindow(DFSPHCData& data);

		static bool isInitialized();


		static void handleFluidBoundaries(SPH::DFSPHCData& data, SPH::Vector3d movement = SPH::Vector3d(0, 0, 0));

		static void clearDynamicWindow();

		static void handleOceanBoundariesTest(SPH::DFSPHCData& data);

		//ok here I'll test a system to initialize a volume of fluid from
		//a large wolume of fluid (IE a technique to iinit the fluid at rest)
		static void initializeFluidToSurface(SPH::DFSPHCData& data);

	};

}

#endif //DFSPH_STATIC_VAR_STRUCT