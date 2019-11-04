#ifndef DFSPH_MACRO_CUDA
#define DFSPH_MACRO_CUDA

#include "DFSPH_define_cuda.h"
#include "cuda_runtime.h"

#include <cstdlib>
#include <cstdio>


////////////////////////////////////////////////////
/////////        CUDA ERROR CHECK      /////////////
////////////////////////////////////////////////////


//easy function to check errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: error %d: %s %s %d\n", (int)code, cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

////////////////////////////////////////////////////
/////////          MEMORY CLEAR        /////////////
////////////////////////////////////////////////////


#define FREE_PTR(ptr) if(ptr!=NULL){delete ptr; ptr=NULL;};
#define CUDA_FREE_PTR(ptr) if(ptr!=NULL){cudaFree(ptr); ptr=NULL;};

////////////////////////////////////////////////////
/////////Kernel function constant/global////////////
////////////////////////////////////////////////////


#ifdef PRECOMPUTED_KERNELS_USE_CONSTANT_MEMORY
#define KERNEL_W(data,val) get_constant_W_cuda(val)
#define KERNEL_GRAD_W(data,val) get_constant_grad_W_cuda(val)
#else
#define KERNEL_W(data,val) data.W(val)
#define KERNEL_GRAD_W(data, val) data.gradW(val)
#endif


////////////////////////////////////////////////////
/////////DYNAMIC BODIES PARTICLES INDEX/////////////
////////////////////////////////////////////////////

#ifdef BITSHIFT_INDEX_DYNAMIC_BODIES
#define WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX(body_index,particle_index) WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX_BITSHIFT(body_index,particle_index)
#define READ_DYNAMIC_BODIES_PARTICLES_INDEX(neighbors_ptr,body_index,particle_index) READ_DYNAMIC_BODIES_PARTICLES_INDEX_BITSHIFT(neighbors_ptr,body_index,particle_index)
#else
#define WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX(body_index,particle_index) WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX_ADDITION(body_index,particle_index)
#define READ_DYNAMIC_BODIES_PARTICLES_INDEX(neighbors_ptr,body_index,particle_index) READ_DYNAMIC_BODIES_PARTICLES_INDEX_ADDITION(neighbors_ptr,body_index,particle_index)
#endif

//those defines are to create and read the dynamic bodies indexes
#define WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX_BITSHIFT(body_index,particle_index)  body_index + (particle_index << 0x8)
#define WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX_ADDITION(body_index,particle_index)  particle_index + (body_index * 1000000)


//WARNING his one declare the body/particle index by itself
//you just have to give it the variable name you want
#define READ_DYNAMIC_BODIES_PARTICLES_INDEX_BITSHIFT(neighbors_idx, body_index,particle_index)  \
    const unsigned int particle_index = neighbors_idx >> 0x8;\
    const unsigned int body_index = neighbors_idx & 0xFF;

#define READ_DYNAMIC_BODIES_PARTICLES_INDEX_ADDITION(neighbors_idx, body_index,particle_index)   \
    const unsigned int particle_index = neighbors_idx % (1000000);\
    const unsigned int body_index=neighbors_idx / 1000000;





////////////////////////////////////////////////////
/////////   NEIGHBORS ITERATIONS       /////////////
//////////////////////////////////////////////////// 

/////////   FROM STRUCTURE       /////////////

#define ITER_NEIGHBORS_INIT_FROM_STRUCTURE_BASE(data,particleSet,index)\
	RealCuda radius_sq = data.getKernelRadius();\
	Vector3d pos = particleSet->pos[index];\
	Vector3d pos_cell = (pos / radius_sq) + data.gridOffset;\
	int x = pos_cell.x;\
	int y = pos_cell.y;\
	int z = pos_cell.z;\
	radius_sq *= radius_sq;


#define ITER_NEIGHBORS_FROM_STRUCTURE_BASE(neighborsDataSet,positions,code){\
    for (int k = -1; k < 2; ++k) {\
    for (int m = -1; m < 2; ++m) {\
    for (int n = -1; n < 2; ++n) {\
    unsigned int cur_cell_id = COMPUTE_CELL_INDEX(x + n, y + k, z + m);\
    unsigned int end = neighborsDataSet->cell_start_end[cur_cell_id + 1];\
    for (unsigned int cur_particle = neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {\
    unsigned int j = neighborsDataSet->p_id_sorted[cur_particle];\
    if ((pos - positions[j]).squaredNorm() < radius_sq) {\
    code\
}\
}\
}\
}\
}\
}

//since this version use the std index to be able to iterate on 3 successive cells
//I can do the -1 at the start on x.
//one thing: it x=0 then we can only iterate 2 cells at a time
#define ITER_NEIGHBORS_INIT_FROM_STRUCTURE_LINEAR_ADVANCED(data,particleSet,index)\
	ITER_NEIGHBORS_INIT_FROM_STRUCTURE_BASE(data,particleSet,index)\
	unsigned int successive_cells_count = (x > 0) ? 3 : 2;\
	x = (x > 0) ? x - 1 : x;


#define ITER_NEIGHBORS_FROM_STRUCTURE_LINEAR_ADVANCED(neighborsDataSet,positions,code){\
    for (int k = -1; k < 2; ++k) {\
    for (int m = -1; m < 2; ++m) {\
    unsigned int cur_cell_id = COMPUTE_CELL_INDEX(x, y + k, z + m);\
    unsigned int end = neighborsDataSet->cell_start_end[cur_cell_id + successive_cells_count];\
    for (unsigned int cur_particle = neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {\
    unsigned int j = neighborsDataSet->p_id_sorted[cur_particle];\
    if ((pos - positions[j]).squaredNorm() < radius_sq) {\
    code\
}\
}\
}\
}\
}

#ifdef USE_COMPLETE
#define ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, particleSet, index) ITER_NEIGHBORS_INIT_FROM_STRUCTURE_BASE(data, particleSet, index)
#define ITER_NEIGHBORS_FROM_STRUCTURE(neighborsDataSet,positions,code) ITER_NEIGHBORS_FROM_STRUCTURE_BASE(neighborsDataSet,positions,code)
#else
#define ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, particleSet, index) ITER_NEIGHBORS_INIT_FROM_STRUCTURE_LINEAR_ADVANCED(data, particleSet, index)
#define ITER_NEIGHBORS_FROM_STRUCTURE(neighborsDataSet,positions,code) ITER_NEIGHBORS_FROM_STRUCTURE_LINEAR_ADVANCED(neighborsDataSet,positions,code)
#endif

///TODO that if(i!=j) is only a correct solution if there is only one fluid set and if we are computing the properties for that fluid set
#define ITER_NEIGHBORS_FLUID_FROM_STRUCTURE(data,particleSet,index,code){\
	const SPH::UnifiedParticleSet& body = data.fluid_data_cuda[0];\
	ITER_NEIGHBORS_FROM_STRUCTURE(body.neighborsDataSet, body.pos,{if(index!=j){const unsigned int neighborIndex = j;code}})};

#define ITER_NEIGHBORS_BOUNDARIES_FROM_STRUCTURE(data,particleSet,index,code){\
	const SPH::UnifiedParticleSet& body = data.boundaries_data_cuda[0];\
	ITER_NEIGHBORS_FROM_STRUCTURE(body.neighborsDataSet, body.pos,{const unsigned int neighborIndex = j;code})};



#ifdef GROUP_DYNAMIC_BODIES_NEIGHBORS_SEARCH
#define ITER_NEIGHBORS_SOLIDS_FROM_STRUCTURE(data,particleSet,index,code){\
if (data.numDynamicBodies > 0) {\
ITER_NEIGHBORS_FROM_STRUCTURE(data.neighborsDataSetGroupedDynamicBodies_cuda, data.posBufferGroupedDynamicBodies,\
{ int body_id = 0; int count_particles_previous_bodies = 0;\
while ((count_particles_previous_bodies + data.vector_dynamic_bodies_data_cuda[body_id].numParticles)<j) {\
	count_particles_previous_bodies += data.vector_dynamic_bodies_data_cuda[body_id].numParticles;\
	body_id++;\
}\
const SPH::UnifiedParticleSet& body = m_data.vector_dynamic_bodies_data_cuda[body_id];\
const unsigned int neighborIndex = j - count_particles_previous_bodies;\
code;\
}\
)\
}\
}
#else
#define ITER_NEIGHBORS_SOLIDS_FROM_STRUCTURE(data,index,code){\
if (data.numDynamicBodies > 0) {\
for (int id_body = 0; id_body < data.numDynamicBodies; ++id_body) {\
const SPH::UnifiedParticleSet& body = m_data.vector_dynamic_bodies_data_cuda[id_body];\
ITER_NEIGHBORS_FROM_STRUCTURE(body.neighborsDataSet, body.pos,\
const unsigned int neighborIndex = j;\
code;\
)\
}\
}\
}
#endif


/////////   FROM STORAGE       /////////////
#define ITER_NEIGHBORS_INIT_FROM_STORAGE_BASE(data,particleSet,index) int* neighbors_ptr = particleSet->getNeighboursPtr(index); int* end_ptr = neighbors_ptr;

#ifdef INTERLEAVE_NEIGHBORS
#define WRITTE_AND_ADVANCE_NEIGHBORS(neighbors_storage_ptr_cur_pos,index) *neighbors_storage_ptr_cur_pos = index; neighbors_storage_ptr_cur_pos+=numParticles;
#define READ_AND_ADVANCE_NEIGHBOR(var_name,ptr_cur_pos)  const unsigned int var_name = *ptr_cur_pos; ptr_cur_pos+=numParticles;
#define ADVANCE_END_PTR(ptr,nb_neighbors) ptr+= nb_neighbors*numParticles;
#define ITER_NEIGHBORS_INIT_FROM_STORAGE(data,particleSet,index) ITER_NEIGHBORS_INIT_FROM_STORAGE_BASE(data,particleSet,index); int numParticles=particleSet->numParticles;
#else
#define WRITTE_AND_ADVANCE_NEIGHBORS(neighbors_storage_ptr_cur_pos,index) *neighbors_storage_ptr_cur_pos++ = index;
#define READ_AND_ADVANCE_NEIGHBOR(var_name,ptr_cur_pos)  const unsigned int var_name = *ptr_cur_pos++;
#define ADVANCE_END_PTR(ptr,nb_neighbors) ptr+= nb_neighbors;
#define ITER_NEIGHBORS_INIT_FROM_STORAGE(data,particleSet,index) ITER_NEIGHBORS_INIT_FROM_STORAGE_BASE(data,particleSet,index)
#endif


#define ITER_NEIGHBORS_FLUID_FROM_STORAGE(data,particleSet,index,code){\
	ADVANCE_END_PTR(end_ptr,particleSet->getNumberOfNeighbourgs(index));\
    const SPH::UnifiedParticleSet& body = *(data.fluid_data_cuda);\
    while (neighbors_ptr != end_ptr)\
{\
    READ_AND_ADVANCE_NEIGHBOR(neighborIndex,neighbors_ptr)\
    code;\
    }\
    }


#define ITER_NEIGHBORS_BOUNDARIES_FROM_STORAGE(data,particleSet,index,code){\
    const SPH::UnifiedParticleSet& body = *(data.boundaries_data_cuda);\
	ADVANCE_END_PTR(end_ptr,particleSet->getNumberOfNeighbourgs(index,1));\
    while (neighbors_ptr != end_ptr)\
{\
    READ_AND_ADVANCE_NEIGHBOR(neighborIndex,neighbors_ptr)\
    code; \
    }\
    }


#define ITER_NEIGHBORS_SOLIDS_FROM_STORAGE(data,particleSet,index,code){\
	ADVANCE_END_PTR(end_ptr, particleSet->getNumberOfNeighbourgs(index,2)); \
    while (neighbors_ptr != end_ptr)\
{\
	READ_AND_ADVANCE_NEIGHBOR(dummy,neighbors_ptr);\
    READ_DYNAMIC_BODIES_PARTICLES_INDEX(dummy, bodyIndex, neighborIndex);\
    const SPH::UnifiedParticleSet& body = data.vector_dynamic_bodies_data_cuda[bodyIndex];\
    code; \
    }\
    }


#ifdef STORE_PARTICLE_NEIGHBORS
#define ITER_NEIGHBORS_INIT(data,particleSet,index) ITER_NEIGHBORS_INIT_FROM_STORAGE(data,particleSet,index)  
#define ITER_NEIGHBORS_FLUID(data,particleSet,index,code) ITER_NEIGHBORS_FLUID_FROM_STORAGE(data,particleSet,index,code)
#define ITER_NEIGHBORS_BOUNDARIES(data,particleSet,index,code) ITER_NEIGHBORS_BOUNDARIES_FROM_STORAGE(data,particleSet,index,code)
#define ITER_NEIGHBORS_SOLIDS(data,particleSet,index,code) ITER_NEIGHBORS_SOLIDS_FROM_STORAGE(data,particleSet,index,code)
#else
#define ITER_NEIGHBORS_INIT(data,particleSet,index) ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data,particleSet,index)  
#define ITER_NEIGHBORS_FLUID(data,particleSet,index,code) ITER_NEIGHBORS_FLUID_FROM_STRUCTURE(data,particleSet,index,code)
#define ITER_NEIGHBORS_BOUNDARIES(data,particleSet,index,code) ITER_NEIGHBORS_BOUNDARIES_FROM_STRUCTURE(data,particleSet,index,code)
#define ITER_NEIGHBORS_SOLIDS(data,particleSet,index,code) ITER_NEIGHBORS_SOLIDS_FROM_STRUCTURE(data,particleSet,index,code)
#endif



////////////////////////////////////////////////////
/////////NEIGHBORS STRUCT CONSTRUCTION /////////////
////////////////////////////////////////////////////

#ifdef BITSHIFT_INDEX_NEIGHBORS_CELL

#ifndef USE_COMPLETE
#define USE_COMPLETE
#endif

__device__ void interleave_2_bits_magic_numbers(unsigned int& x) {
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
}
__device__ unsigned int compute_morton_magic_numbers(unsigned int x, unsigned int y, unsigned int z) {
	interleave_2_bits_magic_numbers(x);
	interleave_2_bits_magic_numbers(y);
	interleave_2_bits_magic_numbers(z);

	return x | (y << 1) | (z << 2);
}

#define COMPUTE_CELL_INDEX(x,y,z) compute_morton_magic_numbers(x,y,z)

#else
#define COMPUTE_CELL_INDEX(x,y,z) (x)+(z)*CELL_ROW_LENGTH+(y)*CELL_ROW_LENGTH*CELL_ROW_LENGTH
#endif

#endif //DFSPH_MACRO_CUDA


