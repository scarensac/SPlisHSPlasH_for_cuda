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
#define READ_DYNAMIC_BODIES_PARTICLES_INDEX_BITSHIFT(neighbors_ptr, body_index,particle_index)  \
    const unsigned int identifier = *neighbors_ptr++;\
    const unsigned int particle_index = identifier >> 0x8;\
    const unsigned int body_index = identifier & 0xFF;

#define READ_DYNAMIC_BODIES_PARTICLES_INDEX_ADDITION(neighbors_ptr, body_index,particle_index)   \
    const unsigned int identifier = *neighbors_ptr++;\
    const unsigned int particle_index = identifier % (1000000);\
    const unsigned int body_index=identifier / 1000000;





////////////////////////////////////////////////////
/////////   NEIGHBORS ITERATIONS       /////////////
////////////////////////////////////////////////////

#define ITER_NEIGHBORS_INIT(index) int* neighbors_ptr = particleSet->getNeighboursPtr(index); int* end_ptr = neighbors_ptr;

#define ITER_NEIGHBORS_FLUID(index,code){\
    end_ptr += particleSet->getNumberOfNeighbourgs(index);\
    const SPH::UnifiedParticleSet& body = *(m_data.fluid_data_cuda);\
    while (neighbors_ptr != end_ptr)\
{\
    const unsigned int neighborIndex = *neighbors_ptr++;\
    code;\
    }\
    }


#define ITER_NEIGHBORS_BOUNDARIES(index,code){\
    const SPH::UnifiedParticleSet& body = *(m_data.boundaries_data_cuda);\
    end_ptr += particleSet->getNumberOfNeighbourgs(index, 1);\
    while (neighbors_ptr != end_ptr)\
{\
    const unsigned int neighborIndex = *neighbors_ptr++;\
    code; \
    }\
    }


#define ITER_NEIGHBORS_SOLIDS(index,code){\
    end_ptr += particleSet->getNumberOfNeighbourgs(index, 2);\
    while (neighbors_ptr != end_ptr)\
{\
    READ_DYNAMIC_BODIES_PARTICLES_INDEX(neighbors_ptr, bodyIndex, neighborIndex);\
    const SPH::UnifiedParticleSet& body = m_data.vector_dynamic_bodies_data_cuda[bodyIndex];\
    code; \
    }\
    }



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

