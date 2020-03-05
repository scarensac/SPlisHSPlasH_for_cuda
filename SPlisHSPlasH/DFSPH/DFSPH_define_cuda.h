#ifndef DFSPH_DEFINE_CUDA
#define DFSPH_DEFINE_CUDA

#include "DFSPH_define_c.h"

#define BLOCKSIZE 128
#define m_eps 1.0e-5
#define CELL_ROW_LENGTH 128
#define CELL_COUNT CELL_ROW_LENGTH*CELL_ROW_LENGTH*CELL_ROW_LENGTH

//use warm start
//#define USE_WARMSTART //for density
#define USE_WARMSTART_V //for divergence

//apply physics values for static boundaries particles
//#define COMPUTE_BOUNDARIES_DYNAMIC_PROPERTiES
#ifdef COMPUTE_BOUNDARIES_DYNAMIC_PROPERTiES
//#define USE_BOUNDARIES_DYNAMIC_PROPERTiES
#endif

//use bit shift for dynamic bodies particles index
#define BITSHIFT_INDEX_DYNAMIC_BODIES

//using norton bitshift for the cells is slower than using a normal index, not that much though
//#define BITSHIFT_INDEX_NEIGHBORS_CELL
//#define USE_COMPLETE


//print debug messages in cuda functions (may not activate /deactivate all messages)
//#define SHOW_MESSAGES_IN_CUDA_FUNCTIONS

#endif 