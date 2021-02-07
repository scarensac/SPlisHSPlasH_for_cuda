#ifndef SPH_DYNAMIC_WINDOW_BUFFER
#define SPH_DYNAMIC_WINDOW_BUFFER


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"



namespace SPH {
	class DynamicWindowV1Interface {

	public:

		static void initDynamicWindowV1(DFSPHCData& data);

		static bool isInitialized();


		static void handleFluidBoundaries(SPH::DFSPHCData& data, SPH::Vector3d movement = SPH::Vector3d(0, 0, 0));

		static void clearDynamicWindowV1();

		static void handleOceanBoundariesTest(SPH::DFSPHCData& data);



	};

}

#endif //DFSPH_STATIC_VAR_STRUCT