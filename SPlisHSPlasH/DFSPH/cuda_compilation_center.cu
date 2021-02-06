//the two removal tag must be high enought to ensure thay'll be at the end
#define TAG_REMOVAL  310000000
#define TAG_REMOVAL_CANDIDATE  320000000

//those are the structural tags, they MUST be in that particular order of value
#define TAG_ACTIVE 10000000
#define TAG_ACTIVE_NEIGHBORS 20000000
#define TAG_ACTIVE_NEIGHBORS_ORDER_2 30000000
#define TAG_UNTAGGED 40000000
#define TAG_AIR 50000000
#define TAG_AIR_ACTIVE_NEIGHBORS 60000000

//other tag for internal functioning theeir actualvalue is unimportant
#define TAG_SAVE 80000000

//some additiona tags for debug
#define TAG_1 90000000
#define TAG_2 100000000
#define TAG_3 110000000

//another removal tag for backward compatibility
#define REMOVAL_TAG  TAG_REMOVAL



#include "basic_kernels_cuda.cu"
#include "DFSPH_core_cuda.cu"
#include "DFSPH_memory_management_cuda.cu"
#include "DFSPH_rendering_cuda.cu"
#include "DFSPH_static_variables_structure_cuda.cu"
#include "BufferFluidSurface.cu"
#include "BorderHeightMap.cu"
#include "SPH_other_systems_cuda.cu"
#include "SPH_dynamic_window_buffer.cu"
#include "RestFluidLoader.cu"
#include "OpenBoundariesSimple.cu"

//test2