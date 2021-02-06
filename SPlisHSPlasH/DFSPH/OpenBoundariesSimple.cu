#include "OpenBoundariesSimple.h"
#include "DFSPH_core_cuda.h"

#include <stdio.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <sstream>
#include <fstream>

#include "DFSPH_define_cuda.h"
#include "DFSPH_macro_cuda.h"
#include "DFSPH_static_variables_structure_cuda.h"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DFSPH_c_arrays_structure.h"
#include "cub.cuh"

#include "SPlisHSPlasH/Utilities/SegmentedTiming.h"
#include "SPH_other_systems_cuda.h"


#include <curand.h>
#include <curand_kernel.h>

#include "basic_kernels_cuda.cuh"


namespace SPH {
	class OpenBoundariesSimple {
	public:
		bool _isinitialized;

		OpenBoundariesSimple() {
			_isinitialized = false;
		};

		~OpenBoundariesSimple() {

		};

		static OpenBoundariesSimple& getStructure() {
			static OpenBoundariesSimple obs;
			return obs;
		}

		static void init(DFSPHCData& data, OpenBoundariesSimpleInterface::InitParameters& params);
	};
}


void OpenBoundariesSimpleInterface::init(DFSPHCData& data, OpenBoundariesSimpleInterface::InitParameters& params) {
	OpenBoundariesSimple::getStructure().init(data, params);
}

void OpenBoundariesSimple::init(DFSPHCData& data, OpenBoundariesSimpleInterface::InitParameters& params) {
	
}