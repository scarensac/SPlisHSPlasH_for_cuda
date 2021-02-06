#ifndef OPEN_BOUNDARIES_SIMPLE
#define OPEN_BOUNDARIES_SIMPLE


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"



namespace SPH {
	class OpenBoundariesSimpleInterface {

	public:
		struct InitParameters {
			InitParameters{

			}
		};

		static void init(DFSPHCData& data, InitParameters& params);
	};

}

#endif //OPEN_BOUNDARIES_SIMPLE