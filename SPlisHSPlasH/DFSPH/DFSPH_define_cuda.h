#ifndef DFSPH_DEFINE_CUDA
#define DFSPH_DEFINE_CUDA

#include "DFSPH_define_c.h"

#define BLOCKSIZE 128
#define m_eps 1.0e-5
#define CELL_ROW_LENGTH 256
#define CELL_COUNT CELL_ROW_LENGTH*CELL_ROW_LENGTH*CELL_ROW_LENGTH

//use warm start
#define USE_WARMSTART //for density
#define USE_WARMSTART_V //for divergence

//apply physics values for static boundaries particles
//#define COMPUTE_BOUNDARIES_DYNAMIC_PROPERTiES
#ifdef COMPUTE_BOUNDARIES_DYNAMIC_PROPERTiES
//#define USE_BOUNDARIES_DYNAMIC_PROPERTiES
#endif

//use bit shift for dynamic bodies particles index
#define BITSHIFT_INDEX_DYNAMIC_BODIES

//using norton bitshift for the cells is slower than using a normal index, not that much though
#define LINEAR_INDEX_NEIGHBORS_CELL
//#define MORTON_INDEX_NEIGHBORS_CELL
//#define HILBERT_INDEX_NEIGHBORS_CELL

//this has to be activated if you don't want the linear advanced index though it is automaticaly activated if 
//an index other than direct calculation linear is used
#define USE_COMPLETE

//activating this will read the index from memory, the actual index is chosen from the earlier define
#define INDEX_NEIGHBORS_CELL_FROM_STORAGE

//activating this will read the index from memory, the actual index is chosen from the earlier define
///!!!WARNING!!! currently this REQUIRES having the interleaved neighbors
#define SORT_NEIGHBORS

//if activated this will create ranges when expliring the neighbor structure so that
//when neighbors are registered they are already sorted
#define NEIGHBORS_RANGE_EXPLORATION


//print debug messages in cuda functions (may not activate /deactivate all messages)
//#define SHOW_MESSAGES_IN_CUDA_FUNCTIONS

//use the position based formalism for the density contraint
//NOTE: does not seems to work if the desired density error is realy low
//#define USE_POSITION_BASED_DENSITY_CONTRAINT 

#endif 