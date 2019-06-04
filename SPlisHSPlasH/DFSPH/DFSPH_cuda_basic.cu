
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DFSPH_cuda_basic.h"
#include <stdio.h>
#include "DFSPH_c_arrays_structure.h"
#include "cub.cuh"
#include <chrono>
#include <iostream>
#include <thread>
#include <curand.h>
#include <curand_kernel.h>

//#define SHOW_MESSAGES_IN_CUDA_FUNCTIONS

#define BLOCKSIZE 256
#define m_eps 1.0e-5
#define CELL_ROW_LENGTH 64
#define CELL_COUNT CELL_ROW_LENGTH*CELL_ROW_LENGTH*CELL_ROW_LENGTH

#define USE_WARMSTART
#define USE_WARMSTART_V
#define COMPUTATION_BOUNDARIES_FULL

#define BITSHIFT_INDEX_DYNAMIC_BODIES

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

//using norton bitshift for the cells is slower than using a normal index, not that much though
//#define BITSHIFT_INDEX_NEIGHBORS_CELL
//#define USE_COMPLETE


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



//those two variables are the identifiers that  link the ongle buffers to cuda
//cudaGraphicsResource_t vboRes_pos;
//cudaGraphicsResource_t vboRes_vel;

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

/*
//this is the bases for all kernels based function
__global__ void DFSPH__kernel(SPH::DFSPHCData m_data) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= m_data.numFluidParticles) { return; }

}
void cuda_(SPH::DFSPHCData& data) {
int numBlocks = (data.numFluidParticles + BLOCKSIZE - 1) / BLOCKSIZE;
DFSPH__kernel << <numBlocks, BLOCKSIZE >> > (data);

cudaError_t cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cuda_compute_density failed: %d\n", (int)cudaStatus);
exit(1598);
}
}
//*/

FUNCTION inline int* getNeighboursPtr(int * neighbourgs, int particle_id) {
    //	return neighbourgs + body_id*numFluidParticles*MAX_NEIGHBOURS + particle_id*MAX_NEIGHBOURS;
    return neighbourgs + particle_id*MAX_NEIGHBOURS;
}

FUNCTION inline unsigned int getNumberOfNeighbourgs(int* numberOfNeighbourgs, int particle_id, int body_id = 0) {
    //return numberOfNeighbourgs[body_id*numFluidParticles + particle_id];
    return numberOfNeighbourgs[particle_id * 3 + body_id];
}

#define FREE_PTR(ptr) if(ptr!=NULL){delete ptr; ptr=NULL;};
#define CUDA_FREE_PTR(ptr) if(ptr!=NULL){cudaFree(ptr); ptr=NULL;};

//Static Variable Structure kernels
__global__ void initCurand_kernel(curandState *state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1) { return; }

    curand_init(1234, 0, 0, state);
}

//Static Variable Structure
class SVS_CU{
public:
    static SVS_CU* get(bool free=false){
        static SVS_CU* v=NULL;

        if(free){
            if(v!=NULL){
                delete v; v=NULL;
            }
        }else{
            if (v==NULL){
                v=new SVS_CU();
            }
        }

        return v;
    }



    SVS_CU(){
        cudaMalloc(&(avg_density_err), sizeof(RealCuda));
        shuffle_index = NULL;


        cudaMalloc(&(curand_state), sizeof(curandState));
        initCurand_kernel << <1, 1 >> > (curand_state);
        gpuErrchk(cudaDeviceSynchronize());

        cudaMallocManaged(&(count_rmv_particles), sizeof(int));
        cudaMallocManaged(&(count_possible_particles), sizeof(int));
        cudaMallocManaged(&(count_moved_particles), sizeof(int));
        cudaMallocManaged(&(count_invalid_position), sizeof(int));

        cudaMallocManaged(&(column_max_height), CELL_ROW_LENGTH*CELL_ROW_LENGTH * sizeof(RealCuda));
        cudaMallocManaged(&(tagged_particles_count),sizeof(int));
        cudaMallocManaged(&(count_created_particles), sizeof(int));

        cudaMallocManaged(&(force_cuda),sizeof(Vector3d));
        cudaMallocManaged(&(moment_cuda),sizeof(Vector3d));
        cudaMallocManaged(&(pt_cuda),sizeof(Vector3d));
    }

    virtual ~SVS_CU(){
        CUDA_FREE_PTR( avg_density_err);
        CUDA_FREE_PTR( shuffle_index);
        CUDA_FREE_PTR(curand_state);
        CUDA_FREE_PTR(count_rmv_particles);
        CUDA_FREE_PTR(count_possible_particles);
        CUDA_FREE_PTR(count_moved_particles);
        CUDA_FREE_PTR( count_invalid_position);
        CUDA_FREE_PTR(column_max_height);
        CUDA_FREE_PTR( tagged_particles_count);
        CUDA_FREE_PTR( count_created_particles);
        CUDA_FREE_PTR( force_cuda);
        CUDA_FREE_PTR( moment_cuda);
        CUDA_FREE_PTR( pt_cuda);
    }

    static void particleNumberChanged(){
        CUDA_FREE_PTR(get()->shuffle_index);
    }

    //cuda variables
    RealCuda* avg_density_err;
    unsigned int* shuffle_index;
    curandState *curand_state;
    int* count_rmv_particles;
    int* count_possible_particles;
    int* count_moved_particles;
    int* count_invalid_position;
    RealCuda* column_max_height;
    int* tagged_particles_count;
    int* count_created_particles;
    Vector3d* force_cuda;
    Vector3d* moment_cuda;
    Vector3d* pt_cuda;


    //cpu variables

};


__global__ void get_min_max_pos_kernel(SPH::UnifiedParticleSet* particleSet, Vector3d* min_o, Vector3d *max_o, RealCuda particle_radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1) { return; }

    //the problem I have is that there wont be a particle in the exact corner
    //I I'll iter on some particles to be sure to reach smth near the corner
    Vector3d min = particleSet->pos[0];
    Vector3d max = particleSet->pos[particleSet->numParticles - 1];

    for (int k = 0; k < 10; ++k) {
        Vector3d p_min = particleSet->pos[k];
        Vector3d p_max = particleSet->pos[particleSet->numParticles - (1+k)];

        if (min.x > p_min.x) { min.x = p_min.x; }
        if (min.y > p_min.y) { min.y = p_min.y; }
        if (min.z > p_min.z) { min.z = p_min.z; }

        if (max.x < p_max.x) { max.x = p_max.x; }
        if (max.y < p_max.y) { max.y = p_max.y; }
        if (max.z < p_max.z) { max.z = p_max.z; }
    }

    min += 2*particle_radius;
    max -= 2*particle_radius;

    *min_o = min;
    *max_o = max;
}

__device__ void computeDensityChange(const SPH::DFSPHCData& m_data, SPH::UnifiedParticleSet* particleSet, const unsigned int index) {
    unsigned int numNeighbors = particleSet->getNumberOfNeighbourgs(index);
    // in case of particle deficiency do not perform a divergence solve
    if (numNeighbors < 20) {
        for (unsigned int pid = 1; pid < 3; pid++)
        {
            numNeighbors += particleSet->getNumberOfNeighbourgs(index, pid);
        }
    }
    if (numNeighbors < 20) {
        particleSet->densityAdv[index] = 0;
    }
    else {
        RealCuda densityAdv = 0;
        const Vector3d &xi = particleSet->pos[index];
        const Vector3d &vi = particleSet->vel[index];
        //////////////////////////////////////////////////////////////////////////
        // Fluid
        //////////////////////////////////////////////////////////////////////////
        ITER_NEIGHBORS_INIT(index);

        ITER_NEIGHBORS_FLUID(
                    index,
                    densityAdv += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_data.gradW(xi - body.pos[neighborIndex]));
                );
        //////////////////////////////////////////////////////////////////////////
        // Boundary
        //////////////////////////////////////////////////////////////////////////
        ITER_NEIGHBORS_BOUNDARIES(
                    index,
                    densityAdv += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_data.gradW(xi - body.pos[neighborIndex]));
                );

        //////////////////////////////////////////////////////////////////////////
        // Dynamic Bodies
        //////////////////////////////////////////////////////////////////////////
        ITER_NEIGHBORS_SOLIDS(
                    index,
                    densityAdv += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_data.gradW(xi - body.pos[neighborIndex]));
                );

        // only correct positive divergence
        particleSet->densityAdv[index] = MAX_MACRO_CUDA(densityAdv, 0.0);
    }
}



template <bool warm_start> __device__ void divergenceSolveParticle(SPH::DFSPHCData& m_data, SPH::UnifiedParticleSet* particleSet, const unsigned int i) {
    Vector3d v_i = Vector3d(0, 0, 0);
    //////////////////////////////////////////////////////////////////////////
    // Evaluate rhs
    //////////////////////////////////////////////////////////////////////////
    const RealCuda ki = (warm_start) ? particleSet->kappaV[i] : (particleSet->densityAdv[i])*particleSet->factor[i];

#ifdef USE_WARMSTART_V
    if (!warm_start) { particleSet->kappaV[i] += ki; }
#endif

    const Vector3d &xi = particleSet->pos[i];


    //////////////////////////////////////////////////////////////////////////
    // Fluid
    //////////////////////////////////////////////////////////////////////////
    ITER_NEIGHBORS_INIT(i);

    ITER_NEIGHBORS_FLUID(
                i,
                const RealCuda kSum = (ki + ((warm_start) ? body.kappaV[neighborIndex] : (body.densityAdv[neighborIndex])*body.factor[neighborIndex]));
            if (fabs(kSum) > m_eps)
    {
        // ki, kj already contain inverse density
        v_i += kSum *  body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);
    }
    );

#ifdef COMPUTATION_BOUNDARIES_FULL
	ITER_NEIGHBORS_BOUNDARIES(
		i,
		const RealCuda kSum = (ki + ((warm_start) ? body.kappaV[neighborIndex] : (body.densityAdv[neighborIndex])*body.factor[neighborIndex]));
	if (fabs(kSum) > m_eps)
	{
		// ki, kj already contain inverse density
		v_i += kSum *  body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);
	}
	);
#endif


    if (fabs(ki) > m_eps)
    {
        //////////////////////////////////////////////////////////////////////////
        // Boundary
        //////////////////////////////////////////////////////////////////////////
#ifndef COMPUTATION_BOUNDARIES_FULL
        ITER_NEIGHBORS_BOUNDARIES(
                    i,
                    const Vector3d delta = ki * body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);
                v_i += delta;// ki already contains inverse density
        );
#endif

        //////////////////////////////////////////////////////////////////////////
        // Dynamic bodies
        //////////////////////////////////////////////////////////////////////////

        ITER_NEIGHBORS_SOLIDS(
                    i,
                    Vector3d delta = ki * body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);
                v_i += delta;// ki already contains inverse density

        //we apply the force to the body particle (no invH since it has been fatorized at the end)
        delta *= -particleSet->mass[i];
        atomicAdd(&(body.F[neighborIndex].x), delta.x);
        atomicAdd(&(body.F[neighborIndex].y), delta.y);
        atomicAdd(&(body.F[neighborIndex].z), delta.z);
        );
    }

    particleSet->vel[i] += v_i*m_data.h;
}

__device__ void computeDensityAdv(SPH::DFSPHCData& m_data, SPH::UnifiedParticleSet* particleSet, const unsigned int index) {
    const Vector3d xi = particleSet->pos[index];
    const Vector3d vi = particleSet->vel[index];
    RealCuda delta = 0;


    //////////////////////////////////////////////////////////////////////////
    // Fluid
    //////////////////////////////////////////////////////////////////////////
    ITER_NEIGHBORS_INIT(index);

    ITER_NEIGHBORS_FLUID(
                index,
                delta += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_data.gradW(xi - body.pos[neighborIndex]));
            );

    //////////////////////////////////////////////////////////////////////////
    // Boundary
    //////////////////////////////////////////////////////////////////////////
    ITER_NEIGHBORS_BOUNDARIES(
                index,
                delta += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_data.gradW(xi - body.pos[neighborIndex]));
            );

    //////////////////////////////////////////////////////////////////////////
    // Dynamic bodies
    //////////////////////////////////////////////////////////////////////////
    ITER_NEIGHBORS_SOLIDS(
                index,
                delta += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_data.gradW(xi - body.pos[neighborIndex]));
            )

            particleSet->densityAdv[index] = MAX_MACRO_CUDA(particleSet->density[index] + m_data.h_future*delta - m_data.density0, 0.0);


#ifdef USE_WARMSTART
    particleSet->kappa[index] += (particleSet->densityAdv[index])*particleSet->factor[index];

#endif
}

__device__ void computeDensityAdv(const unsigned int index, Vector3d* posFluid, Vector3d* velFluid, int* neighbourgs, int * numberOfNeighbourgs,
                                  RealCuda* mass, SPH::PrecomputedCubicKernelPerso m_kernel_precomp, RealCuda* boundaryPsi, Vector3d* posBoundary, Vector3d* velBoundary,
                                  SPH::UnifiedParticleSet* vector_dynamic_bodies_data_cuda, RealCuda* densityAdv, RealCuda* density, RealCuda h_future, RealCuda density0) {
    const Vector3d xi = posFluid[index];
    const Vector3d vi = velFluid[index];
    RealCuda delta = 0;

    //////////////////////////////////////////////////////////////////////////
    // Fluid
    //////////////////////////////////////////////////////////////////////////
    int* neighbors_ptr = getNeighboursPtr(neighbourgs, index);
    int* end_ptr = neighbors_ptr + getNumberOfNeighbourgs(numberOfNeighbourgs, index);
    while (neighbors_ptr != end_ptr)
    {
        const unsigned int neighborIndex = *neighbors_ptr++;
        delta += mass[neighborIndex] * (vi - velFluid[neighborIndex]).dot(m_kernel_precomp.gradW(xi - posFluid[neighborIndex]));
    }

    //////////////////////////////////////////////////////////////////////////
    // Boundary
    //////////////////////////////////////////////////////////////////////////
    end_ptr += getNumberOfNeighbourgs(numberOfNeighbourgs, index, 1);
    while (neighbors_ptr != end_ptr)
    {
        const unsigned int neighborIndex = *neighbors_ptr++;
        delta += boundaryPsi[neighborIndex] * (vi - velBoundary[neighborIndex]).dot(m_kernel_precomp.gradW(xi - posBoundary[neighborIndex]));
    }

    //////////////////////////////////////////////////////////////////////////
    // Dynamic bodies
    //////////////////////////////////////////////////////////////////////////
    end_ptr += getNumberOfNeighbourgs(numberOfNeighbourgs, index, 2);
    while (neighbors_ptr != end_ptr)
    {
        READ_DYNAMIC_BODIES_PARTICLES_INDEX(neighbors_ptr, bodyIndex, neighborIndex);
        SPH::UnifiedParticleSet& body = vector_dynamic_bodies_data_cuda[bodyIndex];
        delta += body.mass[neighborIndex] * (vi - body.vel[neighborIndex]).dot(m_kernel_precomp.gradW(xi - body.pos[neighborIndex]));
    }




    densityAdv[index] = MAX_MACRO_CUDA(density[index] + h_future*delta - density0, 0.0);
}

template <bool warm_start> __device__ void pressureSolveParticle(SPH::DFSPHCData& m_data, SPH::UnifiedParticleSet* particleSet, const unsigned int i) {
    //////////////////////////////////////////////////////////////////////////
    // Evaluate rhs
    //////////////////////////////////////////////////////////////////////////
    const RealCuda ki = (warm_start) ? particleSet->kappa[i] : (particleSet->densityAdv[i])*particleSet->factor[i];

#ifdef USE_WARMSTART
    //if (!warm_start) { particleSet->kappa[i] += ki; } //moved to the evaluation
#endif


    Vector3d v_i = Vector3d(0, 0, 0);
    const Vector3d &xi = particleSet->pos[i];

    //////////////////////////////////////////////////////////////////////////
    // Fluid
    //////////////////////////////////////////////////////////////////////////
    ITER_NEIGHBORS_INIT(i);

    ITER_NEIGHBORS_FLUID(
                i,
                const RealCuda kSum = (ki + ((warm_start) ? body.kappa[neighborIndex] : (body.densityAdv[neighborIndex])*body.factor[neighborIndex]));
            if (fabs(kSum) > m_eps)
    {
        // ki, kj already contain inverse density
        v_i += kSum * body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);
    }
    );

#ifdef COMPUTATION_BOUNDARIES_FULL
	ITER_NEIGHBORS_BOUNDARIES(
		i,
		const RealCuda kSum = (ki + ((warm_start) ? body.kappa[neighborIndex] : (body.densityAdv[neighborIndex])*body.factor[neighborIndex]));
	if (fabs(kSum) > m_eps)
	{
		// ki, kj already contain inverse density
		v_i += kSum * body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);
	}
	);
#endif

    if (fabs(ki) > m_eps)
    {
        //////////////////////////////////////////////////////////////////////////
        // Boundary
        //////////////////////////////////////////////////////////////////////////

#ifndef COMPUTATION_BOUNDARIES_FULL
        ITER_NEIGHBORS_BOUNDARIES(
                    i,
                    v_i += ki * body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);
                );
#endif


        //////////////////////////////////////////////////////////////////////////
        // Dynamic bodies
        //////////////////////////////////////////////////////////////////////////
        ITER_NEIGHBORS_SOLIDS(
                    i,
                    Vector3d delta = ki * body.mass[neighborIndex] * m_data.gradW(xi - body.pos[neighborIndex]);
                v_i += delta;// ki already contains inverse density

        //we apply the force to the body particle (no invH since it has been fatorized at the end)
        delta *= -particleSet->mass[i];
        atomicAdd(&(body.F[neighborIndex].x), delta.x);
        atomicAdd(&(body.F[neighborIndex].y), delta.y);
        atomicAdd(&(body.F[neighborIndex].z), delta.z);
        );
    }

    // Directly update velocities instead of storing pressure accelerations
    particleSet->vel[i] += v_i*m_data.h_future;
}

template <bool ignore_when_no_fluid_near>
__global__ void DFSPH_divergence_warmstart_init_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    if (ignore_when_no_fluid_near) {
        if (particleSet->getNumberOfNeighbourgs(i) == 0) {
            return;
        }
    }

    particleSet->kappaV[i] = MAX_MACRO_CUDA(particleSet->kappaV[i] * m_data.h_ratio_to_past / 2, -0.5);
    //computeDensityChange(m_data, i);


    //I can actually make the factor and density computation here
    {
        //////////////////////////////////////////////////////////////////////////
        // Compute gradient dp_i/dx_j * (1/k)  and dp_j/dx_j * (1/k)
        //////////////////////////////////////////////////////////////////////////
        const Vector3d &xi = particleSet->pos[i];
        const Vector3d &vi = particleSet->vel[i];
        RealCuda sum_grad_p_k = 0;
        Vector3d grad_p_i;
        grad_p_i.setZero();

        RealCuda density = particleSet->mass[i] * m_data.W_zero;
        RealCuda densityAdv = 0;

        //////////////////////////////////////////////////////////////////////////
        // Fluid
        //////////////////////////////////////////////////////////////////////////
        ITER_NEIGHBORS_INIT(i);

        ITER_NEIGHBORS_FLUID(
                    i,
                    const Vector3d &xj = body.pos[neighborIndex];
                density += body.mass[neighborIndex] * m_data.W(xi - xj);
        const Vector3d grad_p_j = body.mass[neighborIndex] * m_data.gradW(xi - xj);
        sum_grad_p_k += grad_p_j.squaredNorm();
        grad_p_i += grad_p_j;
        densityAdv += (vi - body.vel[neighborIndex]).dot(grad_p_j);
        );


        //////////////////////////////////////////////////////////////////////////
        // Boundary
        //////////////////////////////////////////////////////////////////////////
        ITER_NEIGHBORS_BOUNDARIES(
                    i,
                    const Vector3d &xj = body.pos[neighborIndex];
                density += body.mass[neighborIndex] * m_data.W(xi - xj);
        const Vector3d grad_p_j = body.mass[neighborIndex] * m_data.gradW(xi - xj);
        sum_grad_p_k += grad_p_j.squaredNorm();
        grad_p_i += grad_p_j;
        densityAdv += (vi - body.vel[neighborIndex]).dot(grad_p_j);
        );

        //////////////////////////////////////////////////////////////////////////
        // Dynamic bodies
        //////////////////////////////////////////////////////////////////////////
        //*
        ITER_NEIGHBORS_SOLIDS(
                    i,
                    const Vector3d &xj = body.pos[neighborIndex];
                density += body.mass[neighborIndex] * m_data.W(xi - xj);
        const Vector3d grad_p_j = body.mass[neighborIndex] * m_data.gradW(xi - xj);
        sum_grad_p_k += grad_p_j.squaredNorm();
        grad_p_i += grad_p_j;
        densityAdv += (vi - body.vel[neighborIndex]).dot(grad_p_j);
        );
        //*/


        sum_grad_p_k += grad_p_i.squaredNorm();

        //////////////////////////////////////////////////////////////////////////
        // Compute pressure stiffness denominator
        //////////////////////////////////////////////////////////////////////////
        particleSet->factor[i] = (-m_data.invH / (MAX_MACRO_CUDA(sum_grad_p_k, m_eps)));
        particleSet->density[i] = density;

        //end the density adv computation
        unsigned int numNeighbors = particleSet->getNumberOfNeighbourgs(i);
        // in case of particle deficiency do not perform a divergence solve
        if (numNeighbors < 20) {
            for (unsigned int pid = 1; pid < 3; pid++)
            {
                numNeighbors += particleSet->getNumberOfNeighbourgs(i, pid);
            }
        }
        if (numNeighbors < 20) {
            particleSet->densityAdv[i] = 0;
        }
        else {
            particleSet->densityAdv[i] = MAX_MACRO_CUDA(densityAdv, 0.0);

        }

    }

}

void cuda_divergence_warmstart_init(SPH::DFSPHCData& data) {
    {//fluid
        int numBlocks = (data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
        DFSPH_divergence_warmstart_init_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
    }

	//*
    if (data.boundaries_data[0].has_factor_computation){//boundaries
        int numBlocks = (data.boundaries_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
        DFSPH_divergence_warmstart_init_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data[0].gpu_ptr);
    }
	//*/


    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda_divergence_warmstart_init failed: %d\n", (int)cudaStatus);
        exit(1598);
    }
}

//WARNING !!! this is not suposed to be called for the fluid this function is used for boundaries and object for witch 
//doing the velocity variation computation makes no sense but still need the accumulation of kappa for the warm start
__global__ void DFSPH_divergence_accumulate_kappaV_kernel(SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	const RealCuda ki =  (particleSet->densityAdv[i])*particleSet->factor[i];
    particleSet->kappaV[i] += ki; 
}


template<bool warmstart> __global__ void DFSPH_divergence_compute_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    if (warmstart) {
        if (particleSet->densityAdv[i] > 0.0) {
            divergenceSolveParticle<warmstart>(m_data, particleSet, i);
        }
    }
    else {
        divergenceSolveParticle<warmstart>(m_data, particleSet, i);
    }

}

template<bool warmstart> void cuda_divergence_compute(SPH::DFSPHCData& data) {
	{//fluid
		int numBlocks = (data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_divergence_compute_kernel<warmstart> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
	}

	//*
	
	if (data.boundaries_data[0].has_factor_computation) {//boundaries 
		if (!warmstart) {
			int numBlocks = (data.boundaries_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
			DFSPH_divergence_accumulate_kappaV_kernel << <numBlocks, BLOCKSIZE >> > (data.boundaries_data[0].gpu_ptr);
		}
	}
	//*/

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda_divergence_compute failed: %d\n", (int)cudaStatus);
        exit(1598);
    }
}
template void cuda_divergence_compute<true>(SPH::DFSPHCData& data);
template void cuda_divergence_compute<false>(SPH::DFSPHCData& data);

__global__ void DFSPH_divergence_init_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    {
#ifdef USE_WARMSTART_V
        particleSet->kappaV[i] = 0;
#endif

        ///TODO when doing this kernel I can actually fuse the code for all those computation to limit the number
        ///of time I read the particles positions
        computeDensityChange(m_data, particleSet, i);

#ifndef USE_WARMSTART_V
        //I can actually make the factor and desity computation here
        {
            //////////////////////////////////////////////////////////////////////////
            // Compute gradient dp_i/dx_j * (1/k)  and dp_j/dx_j * (1/k)
            //////////////////////////////////////////////////////////////////////////
            const Vector3d &xi = particleSet->pos[i];
            RealCuda sum_grad_p_k = 0;
            Vector3d grad_p_i;
            grad_p_i.setZero();

            RealCuda density = particleSet->mass[i] * m_data.W_zero;

            //////////////////////////////////////////////////////////////////////////
            // Fluid
            //////////////////////////////////////////////////////////////////////////
            ITER_NEIGHBORS_INIT(i);

            ITER_NEIGHBORS_FLUID(
                        i,
                        const Vector3d &xj = body.pos[neighborIndex];
                    density += body.mass[neighborIndex] * m_data.W(xi - xj);
            const Vector3d grad_p_j = body.mass[neighborIndex] * m_data.gradW(xi - xj);
            sum_grad_p_k += grad_p_j.squaredNorm();
            grad_p_i += grad_p_j;
            );

            //////////////////////////////////////////////////////////////////////////
            // Boundary
            //////////////////////////////////////////////////////////////////////////
            ITER_NEIGHBORS_BOUNDARIES(
                        i,
                        const Vector3d &xj = body.pos[neighborIndex];
                    density += body.mass[neighborIndex] * m_data.W(xi - xj);
            const Vector3d grad_p_j = body.mass[neighborIndex] * m_data.gradW(xi - xj);
            sum_grad_p_k += grad_p_j.squaredNorm();
            grad_p_i += grad_p_j;
            );

            //////////////////////////////////////////////////////////////////////////
            // Dynamic bodies
            //////////////////////////////////////////////////////////////////////////
            //*
            ITER_NEIGHBORS_SOLIDS(
                        i,
                        const Vector3d &xj = body.pos[neighborIndex];
                    density += body.mass[neighborIndex] * m_data.W(xi - xj);
            const Vector3d grad_p_j = body.mass[neighborIndex] * m_data.gradW(xi - xj);
            sum_grad_p_k += grad_p_j.squaredNorm();
            grad_p_i += grad_p_j;
            );
            //*/


            sum_grad_p_k += grad_p_i.squaredNorm();

            //////////////////////////////////////////////////////////////////////////
            // Compute pressure stiffness denominator
            //////////////////////////////////////////////////////////////////////////
            particleSet->factor[i] = (-m_data.invH / (MAX_MACRO_CUDA(sum_grad_p_k, m_eps)));
            particleSet->density[i] = density;

        }
#endif


    }

}

void cuda_divergence_init(SPH::DFSPHCData& data) {
	{//fluid
		int numBlocks = (data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_divergence_init_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
	}

	//*
	if (data.boundaries_data[0].has_factor_computation) {//boundaries 
		int numBlocks = (data.boundaries_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_divergence_init_kernel << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data[0].gpu_ptr);
	}
	//*/

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda_divergence_init failed: %d\n", (int)cudaStatus);
        exit(1598);
    }
}

__global__ void DFSPH_divergence_loop_end_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet, RealCuda* avg_density_err) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    computeDensityChange(m_data, particleSet, i);
    //atomicAdd(avg_density_err, m_data.densityAdv[i]);
}

RealCuda cuda_divergence_loop_end(SPH::DFSPHCData& data) {
    RealCuda* avg_density_err = SVS_CU::get()->avg_density_err;

	{//fluid
		int numBlocks = (data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_divergence_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr, avg_density_err);
	}

	//*
	if (data.boundaries_data[0].has_factor_computation) {//boundaries 
		int numBlocks = (data.boundaries_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_divergence_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data[0].gpu_ptr, avg_density_err);
	}
	//*/


    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda_divergence_loop_end failed: %d\n", (int)cudaStatus);
        exit(1598);
    }


    // Run sum-reduction
    cub::DeviceReduce::Sum(data.fluid_data->d_temp_storage, data.fluid_data->temp_storage_bytes, data.fluid_data->densityAdv, avg_density_err, data.fluid_data[0].numParticles);
    gpuErrchk(cudaDeviceSynchronize());


    RealCuda result = 0;
    gpuErrchk(cudaMemcpy(&result, avg_density_err, sizeof(RealCuda), cudaMemcpyDeviceToHost));

    return result;
}

// also prepare the normals for the adhesion force
__global__ void DFSPH_viscosityXSPH_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    //I set the gravitation directly here to lover the number of kernels
    Vector3d ai = Vector3d(0, 0, 0);
    Vector3d ni = Vector3d(0, 0, 0);
    const Vector3d &xi = particleSet->pos[i];
    const Vector3d &vi = particleSet->vel[i];

    //////////////////////////////////////////////////////////////////////////
    // Fluid
    //////////////////////////////////////////////////////////////////////////
    ITER_NEIGHBORS_INIT(i);

    //*
    ITER_NEIGHBORS_FLUID(
                i,
                Vector3d xixj=xi - body.pos[neighborIndex];
            RealCuda mass_div_density=body.mass[neighborIndex] / body.density[neighborIndex];
    ai -= m_data.invH * m_data.viscosity * (mass_div_density) * (vi - body.vel[neighborIndex]) * m_data.W(xixj);
    ni += mass_div_density * m_data.gradW(xixj);
    )
    //*/
    /*
            //viscosity only
            ITER_NEIGHBORS_FLUID(
                        i,
                        ai -= m_data.invH * m_data.viscosity * (body.mass[neighborIndex] / body.density[neighborIndex]) *
                    (vi - body.vel[neighborIndex]) * m_data.W(xi - body.pos[neighborIndex]);

                    )//*/

    particleSet->acc[i] = m_data.gravitation + ai;

    //*
    //I'm gona use the vector3D used for the agglomerated neigbor search to store the normals
    ni *= m_data.getKernelRadius();
    m_data.posBufferGroupedDynamicBodies[i]=ni;
    //*/
}


__global__ void DFSPH_applySurfaceAkinci2013SurfaceTension_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    //for more lisability of the code
    Vector3d* normals=m_data.posBufferGroupedDynamicBodies;
    RealCuda supportRadius=m_data.getKernelRadius();
    RealCuda k = m_data.getSurfaceTension();
    RealCuda density0=m_data.density0;

    //I set the gravitation directly here to lover the number of kernels
    Vector3d ai = Vector3d(0, 0, 0);
    Vector3d ni = normals[i];
    RealCuda rhoi = particleSet->density[i];
    const Vector3d &xi = particleSet->pos[i];

    ITER_NEIGHBORS_INIT(i);

    //////////////////////////////////////////////////////////////////////////
    // Fluid
    //////////////////////////////////////////////////////////////////////////

    ITER_NEIGHBORS_FLUID(
                i,
                RealCuda K_ij = 2.0*density0 / (rhoi + body.density[neighborIndex]);

            Vector3d accel=Vector3d(0,0,0);


    // Cohesion force
    Vector3d xixj = xi - body.pos[neighborIndex];
    const Real length2 = xixj.squaredNorm();
    if (length2 > 1.0e-9)
    {
        xixj = ((Real) 1.0 / sqrt(length2)) * xixj;
        accel -= k * body.mass[neighborIndex] * xixj * m_data.WCohesion(xixj);
    }

    // Curvature
    accel -= k * supportRadius* (ni - normals[neighborIndex]);

    ai += K_ij * accel;
    //*/
    );
    //////////////////////////////////////////////////////////////////////////
    // Boundary
    //////////////////////////////////////////////////////////////////////////
    ITER_NEIGHBORS_BOUNDARIES(
                i,
                // adhesion force
                Vector3d xixj = (xi - body.pos[neighborIndex]);
            const Real length2 = xixj.squaredNorm();
    if (length2 > 1.0e-9)
    {
        xixj = ((Real) 1.0 / sqrt(length2)) * xixj;
        ai -= k * body.mass[neighborIndex] * xixj * m_data.WAdhesion(xixj);
    }
    );

    //////////////////////////////////////////////////////////////////////////
    // Dynamic Bodies
    //////////////////////////////////////////////////////////////////////////
    ITER_NEIGHBORS_SOLIDS(
                i,
                // adhesion force
                Vector3d xixj = (xi - body.pos[neighborIndex]);
            const Real length2 = xixj.squaredNorm();
    if (length2 > 1.0e-9)
    {
        xixj = ((Real) 1.0 / sqrt(length2)) * xixj;
        ai -= k * body.mass[neighborIndex] * xixj * m_data.WAdhesion(xixj);
    }
    );

    particleSet->acc[i]+=ai;
}


void cuda_externalForces(SPH::DFSPHCData& data) {
    int numBlocks = (data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
    DFSPH_viscosityXSPH_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda_viscosityXSPH failed: %d\n", (int)cudaStatus);
        exit(1598);
    }

    //end the computations for the surface tension

    DFSPH_applySurfaceAkinci2013SurfaceTension_kernel<< <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void DFSPH_CFL_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet particleSet, RealCuda* maxVel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m_data.fluid_data[0].numParticles) { return; }

    for (unsigned int i = 0; i < m_data.fluid_data[0].numParticles; i++)
    {
        const RealCuda velMag = (particleSet.vel[i] + particleSet.acc[i] * m_data.h).squaredNorm();
        if (velMag > *maxVel)
            *maxVel = velMag;
    }
}

__global__ void DFSPH_CFLVelSquaredNorm_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet, RealCuda* sqaredNorm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    sqaredNorm[i] = (particleSet->vel[i] + particleSet->acc[i] * m_data.h).squaredNorm();
}

__global__ void DFSPH_CFLAdvanced_kernel(SPH::DFSPHCData m_data, RealCuda *max, int *mutex, unsigned int n)
{
    unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int stride = gridDim.x*blockDim.x;
    unsigned int offset = 0;

    __shared__ RealCuda cache[256];


    RealCuda temp = 0;
    while (index + offset < n) {
        int i = index + offset;
        const RealCuda velMag = (m_data.fluid_data_cuda->vel[i] + m_data.fluid_data_cuda->acc[i] * m_data.h).squaredNorm();
        temp = fmaxf(temp, velMag);

        offset += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();


    // reduction
    unsigned int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] = MAX_MACRO_CUDA(cache[threadIdx.x], cache[threadIdx.x + i]);
        }

        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0) {
        while (atomicCAS(mutex, 0, 1) != 0);  //lock
        *max = MAX_MACRO_CUDA(*max, cache[0]);
        atomicExch(mutex, 0);  //unlock
    }
}

void cuda_CFL(SPH::DFSPHCData& m_data, const RealCuda minTimeStepSize, RealCuda m_cflFactor, RealCuda m_cflMaxTimeStepSize) {

    //we compute the square norm

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    RealCuda* out_buff;
    cudaMalloc(&(out_buff), sizeof(RealCuda));

    if (true) {

        //cub version
        static RealCuda* temp_buff = NULL;
        if (temp_buff == NULL) {
            cudaMallocManaged(&(temp_buff), m_data.fluid_data[0].numParticles * sizeof(RealCuda));
        }
        int numBlocks = (m_data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
        DFSPH_CFLVelSquaredNorm_kernel << <numBlocks, BLOCKSIZE >> > (m_data, m_data.fluid_data[0].gpu_ptr, temp_buff);

        cudaError_t cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cuda_cfl squared norm failed: %d\n", (int)cudaStatus);
            exit(1598);
        }

        // Determine temporary device storage requirements
        static void     *d_temp_storage = NULL;
        static size_t   temp_storage_bytes = 0;
        if (d_temp_storage == NULL) {
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, temp_buff, out_buff, m_data.fluid_data[0].numParticles);
            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
        }
        // Run max-reduction
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, temp_buff, out_buff, m_data.fluid_data[0].numParticles);

    }
    else {
        //manual
        int *d_mutex;
        cudaMalloc((void**)&d_mutex, sizeof(int));
        cudaMemset(d_mutex, 0, sizeof(float));

        int numBlocks = (m_data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
        DFSPH_CFLAdvanced_kernel << < numBlocks, BLOCKSIZE >> > (m_data, out_buff, d_mutex, m_data.fluid_data[0].numParticles);

        cudaError_t cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cuda_cfl failed: %d\n", (int)cudaStatus);
            exit(1598);
        }
        cudaFree(d_mutex);
    }
    RealCuda maxVel;
    cudaMemcpy(&maxVel, out_buff, sizeof(RealCuda), cudaMemcpyDeviceToHost);
    cudaFree(out_buff);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    RealCuda h = m_data.h;

    // Approximate max. time step size
    h = m_cflFactor * .4 * (2.0*m_data.particleRadius / (sqrt(maxVel)));

    h = min(h, m_cflMaxTimeStepSize);
    h = max(h, minTimeStepSize);

    m_data.updateTimeStep(h);//*/


    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();



    float time_search = std::chrono::duration_cast<std::chrono::nanoseconds> (t1 - t0).count() / 1000000.0f;
    float time_comp = std::chrono::duration_cast<std::chrono::nanoseconds> (t2 - t1).count() / 1000000.0f;

    printf("Time to do cfl (search,comp): %f    %f\n", time_search, time_comp);
}

__global__ void DFSPH_update_vel_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    particleSet->vel[i] += m_data.h * particleSet->acc[i];

#ifdef USE_WARMSTART	
    //done here to have one less kernel
    particleSet->kappa[i] = MAX_MACRO_CUDA(particleSet->kappa[i] * m_data.h_ratio_to_past2, -0.5);
#endif
}




void cuda_update_vel(SPH::DFSPHCData& data) {
    int numBlocks = (data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
    DFSPH_update_vel_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda_update_vel failed: %d\n", (int)cudaStatus);
        exit(1598);
    }


}

//WARNING !!! this is not suposed to be called for the fluid this function is used for boundaries and object for witch 
//doing the velocity variation computation makes no sense but still need the accumulation of kappa for the warm start
__global__ void DFSPH_density_accumulate_kappa_kernel(SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	const RealCuda ki = (particleSet->densityAdv[i])*particleSet->factor[i];
	particleSet->kappa[i] += ki;
}

template<bool warmstart> __global__ void DFSPH_pressure_compute_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    pressureSolveParticle<warmstart>(m_data, particleSet, i);

}

template<bool warmstart> void cuda_pressure_compute(SPH::DFSPHCData& data) {
	{//fluid
		int numBlocks = (data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_pressure_compute_kernel<warmstart> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
	}

	if (data.boundaries_data[0].has_factor_computation) {//boundaries 
		if (!warmstart) {
			int numBlocks = (data.boundaries_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
			DFSPH_density_accumulate_kappa_kernel << <numBlocks, BLOCKSIZE >> > (data.boundaries_data[0].gpu_ptr);
		}
	}

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda_pressure_compute failed: %d\n", (int)cudaStatus);
        exit(1598);
    }
}
template void cuda_pressure_compute<true>(SPH::DFSPHCData& data);
template void cuda_pressure_compute<false>(SPH::DFSPHCData& data);



__global__ void DFSPH_pressure_init_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

#ifdef USE_WARMSTART
    particleSet->kappa[i] = 0;
#endif

    particleSet->factor[i] *= m_data.invH_future;

    computeDensityAdv(m_data, particleSet, i);


}

void cuda_pressure_init(SPH::DFSPHCData& data) {
    {//fluid
        int numBlocks = (data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
        DFSPH_pressure_init_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);
    }
	
	if (data.boundaries_data[0].has_factor_computation) {//boundaries 
		int numBlocks = (data.boundaries_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_pressure_init_kernel << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data[0].gpu_ptr);
	}


    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda_pressure_init failed: %d\n", (int)cudaStatus);
        exit(1598);
    }
}


__global__ void DFSPH_pressure_loop_end_kernel(SPH::DFSPHCData m_data, SPH::UnifiedParticleSet* particleSet, RealCuda* avg_density_err) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }


    computeDensityAdv(m_data, particleSet, i);
    //atomicAdd(avg_density_err, m_data.densityAdv[i]);
}
/*
__global__ void DFSPH_pressure_loop_end_kernel(int numFluidParticles, Vector3d* posFluid, Vector3d* velFluid, int* neighbourgs, int * numberOfNeighbourgs,
    RealCuda* mass, SPH::PrecomputedCubicKernelPerso m_kernel_precomp, RealCuda* boundaryPsi, Vector3d* posBoundary, Vector3d* velBoundary,
    SPH::UnifiedParticleSet* vector_dynamic_bodies_data_cuda, RealCuda* densityAdv, RealCuda* density, RealCuda h_future, RealCuda density0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numFluidParticles) { return; }

    computeDensityAdv(i, posFluid, velFluid, neighbourgs, numberOfNeighbourgs,
        mass, m_kernel_precomp, boundaryPsi, posBoundary, velBoundary,
        vector_dynamic_bodies_data_cuda, densityAdv, density, h_future, density0);
}//*/
//*/
RealCuda cuda_pressure_loop_end(SPH::DFSPHCData& data) {

    std::chrono::steady_clock::time_point p0 = std::chrono::steady_clock::now();

    RealCuda* avg_density_err = SVS_CU::get()->avg_density_err;

	{//fluid
		int numBlocks = (data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_pressure_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr, avg_density_err);
	}

	if (data.boundaries_data[0].has_factor_computation) {//boundaries 
		int numBlocks = (data.boundaries_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
		DFSPH_pressure_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data[0].gpu_ptr, avg_density_err);
	}



    /*
    ///LOL the detailed implementation is slower so no need to even think about developping data
    DFSPH_pressure_loop_end_kernel << <numBlocks, BLOCKSIZE >> > (data.numFluidParticles, data.posFluid, data.velFluid,
    data.neighbourgs, data.numberOfNeighbourgs,
    data.mass, data.m_kernel_precomp, data.boundaryPsi, data.posBoundary, data.velBoundary,
    data.vector_dynamic_bodies_data_cuda, data.densityAdv, data.density, data.h_future, data.density0);
    //*/

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda_pressure_loop_end failed: %d\n", (int)cudaStatus);
        exit(1598);
    }

    std::chrono::steady_clock::time_point p1 = std::chrono::steady_clock::now();

    // Run sum-reduction
    cub::DeviceReduce::Sum(data.fluid_data->d_temp_storage, data.fluid_data->temp_storage_bytes, data.fluid_data->densityAdv, avg_density_err, data.fluid_data[0].numParticles);


    RealCuda result = 0;
    gpuErrchk(cudaMemcpy(&result, avg_density_err, sizeof(RealCuda), cudaMemcpyDeviceToHost));


    std::chrono::steady_clock::time_point p2 = std::chrono::steady_clock::now();
    float time1 = std::chrono::duration_cast<std::chrono::nanoseconds> (p1 - p0).count() / 1000000.0f;
    float time2 = std::chrono::duration_cast<std::chrono::nanoseconds> (p2 - p1).count() / 1000000.0f;

    //std::cout << "pressure loop end details: " << time1 << "  " << time2 << std::endl;

    return result;
}

__global__ void DFSPH_update_pos_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    if (data.damp_borders) {
        /*
        RealCuda max_vel_sq = (data.particleRadius / 2.0f) / data.h;
        max_vel_sq *= max_vel_sq;
        RealCuda cur_vel_sq = particleSet->vel[i].squaredNorm();
        if (cur_vel_sq> max_vel_sq)
        {
            particleSet->vel[i] *= max_vel_sq / cur_vel_sq;
        }//*/

        RealCuda affected_distance_sq= data.particleRadius*6;
        affected_distance_sq *= affected_distance_sq;

        for (int k = 0; k < data.damp_planes_count; ++k) {
            Vector3d plane = data.damp_planes[k];
            if ((particleSet->pos[i] * plane.abs() / plane.norm() - plane).squaredNorm() < affected_distance_sq) {
                if (data.damp_borders_steps_count>1){
                    RealCuda max_vel_sq = (data.particleRadius / 25.0f) / data.h;
                    max_vel_sq *= max_vel_sq;
                    RealCuda cur_vel_sq = particleSet->vel[i].squaredNorm();
                    if (cur_vel_sq> max_vel_sq)
                    {
                        particleSet->vel[i] *= max_vel_sq / cur_vel_sq;
                    }
                    //if we triggered once no need to check for the other planes
                    break;
                }else{
                    particleSet->vel[i] *= 0.1;
                }
            }
        }
    }

    if (data.cancel_wave){
        RealCuda affected_distance_sq= data.getKernelRadius();
        affected_distance_sq *= affected_distance_sq;
        for (int k = 0; k < 2; ++k) {
            Vector3i plane = data.cancel_wave_planes[k];
            if ((particleSet->pos[i] * plane.abs() / plane.norm() - plane).squaredNorm() < affected_distance_sq) {
                //particleSet->vel[i]=Vector3d(0,1,0);
            }
        }
        Vector3d axis=data.cancel_wave_planes[0].abs()/data.cancel_wave_planes[0].norm();
        if(particleSet->pos[i].y>data.cancel_wave_lowest_point){
            if((particleSet->pos[i].dot(axis))<(data.cancel_wave_planes[0].dot(axis))){
                if((particleSet->vel[i].dot(axis))<0){
                    particleSet->vel[i]-=particleSet->vel[i]*axis;
                }
            }

            if((particleSet->pos[i].dot(axis))>(data.cancel_wave_planes[1].dot(axis))){
                if((particleSet->vel[i].dot(axis))>0){
                    particleSet->vel[i]-=particleSet->vel[i]*axis;
                }
            }
        }

    }


    particleSet->pos[i] += data.h * particleSet->vel[i];
}



void cuda_update_pos(SPH::DFSPHCData& data) {


    int numBlocks = (data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
    DFSPH_update_pos_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data[0].gpu_ptr);


    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda_update_pos failed: %d\n", (int)cudaStatus);
        exit(1598);
    }

    if (data.damp_borders){
        for (int k = 0; k < data.damp_planes_count; ++k) {
            Vector3d plane = data.damp_planes[k];
            //std::cout << "damping plane: " << plane.x << "  " << plane.y << "  " << plane.z << std::endl;
        }
        data.damp_borders_steps_count--;
        if (data.damp_borders_steps_count == 0) {
            data.damp_borders = false;
        }
    }
    if (data.cancel_wave){
        //*

        for (int k = 0; k < 2; ++k) {
            Vector3d plane = data.cancel_wave_planes[k];
            std::cout << "cancel wave plane: " << plane.x << "  " << plane.y << "  " << plane.z << std::endl;
        }
        data.cancel_wave_steps_count--;
        if (data.cancel_wave_steps_count == 0) {
            data.cancel_wave = false;
        }
        //*/
    }

}


int cuda_divergenceSolve(SPH::DFSPHCData& m_data, const unsigned int maxIter, const RealCuda maxError) {
    //////////////////////////////////////////////////////////////////////////
    // Init parameters
    //////////////////////////////////////////////////////////////////////////

    const RealCuda h = m_data.h;
    const int numParticles = m_data.fluid_data[0].numParticles;
    const RealCuda density0 = m_data.density0;

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

#ifdef USE_WARMSTART_V
    cuda_divergence_warmstart_init(m_data);

    std::chrono::steady_clock::time_point m0 = std::chrono::steady_clock::now();
    cuda_divergence_compute<true>(m_data);
#endif

    std::chrono::steady_clock::time_point m1 = std::chrono::steady_clock::now();
    //////////////////////////////////////////////////////////////////////////
    // Compute velocity of density change
    //////////////////////////////////////////////////////////////////////////
    cuda_divergence_init(m_data);

    std::chrono::steady_clock::time_point m2 = std::chrono::steady_clock::now();

    unsigned int m_iterationsV = 0;

    //////////////////////////////////////////////////////////////////////////
    // Start solver
    //////////////////////////////////////////////////////////////////////////

    // Maximal allowed density fluctuation
    // use maximal density error divided by time step size
    const RealCuda eta = maxError * 0.01 * density0 / h;  // maxError is given in percent

    float time_3_1 = 0;
    float time_3_2 = 0;
    RealCuda avg_density_err = 0.0;
    while (((avg_density_err > eta) || (m_iterationsV < 3)) && (m_iterationsV < maxIter))
    {

        //////////////////////////////////////////////////////////////////////////
        // Perform Jacobi iteration over all blocks
        //////////////////////////////////////////////////////////////////////////
        std::chrono::steady_clock::time_point p0 = std::chrono::steady_clock::now();
        cuda_divergence_compute<false>(m_data);
        std::chrono::steady_clock::time_point p1 = std::chrono::steady_clock::now();

        avg_density_err = cuda_divergence_loop_end(m_data);
        std::chrono::steady_clock::time_point p2 = std::chrono::steady_clock::now();

        avg_density_err /= numParticles;
        m_iterationsV++;

        time_3_1 += std::chrono::duration_cast<std::chrono::nanoseconds> (p1 - p0).count() / 1000000.0f;
        time_3_2 += std::chrono::duration_cast<std::chrono::nanoseconds> (p2 - p1).count() / 1000000.0f;
    }

    /*
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    float time_0 = std::chrono::duration_cast<std::chrono::nanoseconds> (m0 - start).count() / 1000000.0f;
    float time_1 = std::chrono::duration_cast<std::chrono::nanoseconds> (m1 - m0).count() / 1000000.0f;
    float time_2 = std::chrono::duration_cast<std::chrono::nanoseconds> (m2 - m1).count() / 1000000.0f;
    float time_3 = std::chrono::duration_cast<std::chrono::nanoseconds> (end - m2).count() / 1000000.0f;

    std::cout << "detail pressure solve (iter total (varible_comp warm_comp init actual_comp (t1 t2))): " << m_iterationsV << "  " << time_0+ time_1 + time_2 + time_3 <<
        "  (" << time_0 << "  " << time_1 << "  " << time_2 << "  " << time_3 << "(" << time_3_1 << " " << time_3_2 << ") )" << std::endl;

    //*/
    return m_iterationsV;
}



int cuda_pressureSolve(SPH::DFSPHCData& m_data, const unsigned int m_maxIterations, const RealCuda m_maxError) {
    const RealCuda density0 = m_data.density0;
    const int numParticles = (int)m_data.fluid_data[0].numParticles;
    RealCuda avg_density_err = 0.0;


    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();


#ifdef USE_WARMSTART		
    cuda_pressure_compute<true>(m_data);
#endif

    std::chrono::steady_clock::time_point m1 = std::chrono::steady_clock::now();

    //////////////////////////////////////////////////////////////////////////
    // Compute rho_adv
    //////////////////////////////////////////////////////////////////////////
    cuda_pressure_init(m_data);


    std::chrono::steady_clock::time_point m2 = std::chrono::steady_clock::now();


    unsigned int m_iterations = 0;

    //////////////////////////////////////////////////////////////////////////
    // Start solver
    //////////////////////////////////////////////////////////////////////////

    // Maximal allowed density fluctuation
    const RealCuda eta = m_maxError * 0.01 * density0;  // maxError is given in percent

    float time_3_1 = 0;
    float time_3_2 = 0;
    while (((avg_density_err > eta) || (m_iterations < 2)) && (m_iterations < m_maxIterations))
    {
        std::chrono::steady_clock::time_point p0 = std::chrono::steady_clock::now();
        cuda_pressure_compute<false>(m_data);
        std::chrono::steady_clock::time_point p1 = std::chrono::steady_clock::now();
        avg_density_err = cuda_pressure_loop_end(m_data);
        std::chrono::steady_clock::time_point p2 = std::chrono::steady_clock::now();
        avg_density_err /= numParticles;

        m_iterations++;

        time_3_1 += std::chrono::duration_cast<std::chrono::nanoseconds> (p1 - p0).count() / 1000000.0f;
        time_3_2 += std::chrono::duration_cast<std::chrono::nanoseconds> (p2 - p1).count() / 1000000.0f;
    }
    /*
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    float time_1 = std::chrono::duration_cast<std::chrono::nanoseconds> (m1 - start).count() / 1000000.0f;
    float time_2 = std::chrono::duration_cast<std::chrono::nanoseconds> (m2 - m1).count() / 1000000.0f;
    float time_3 = std::chrono::duration_cast<std::chrono::nanoseconds> (end - m2).count() / 1000000.0f;

    std::cout << "detail pressure solve (iter total (warm init actual_comp (t1 t2))): " <<m_iterations <<"  "<< time_1 + time_2 +time_3 <<
    "  (" << time_1 << "  " << time_2<< "  "<< time_3 <<"("<< time_3_1<<" "<< time_3_2<<") )" << std::endl;

    //*/

    return m_iterations;

}


template<unsigned int grid_size, bool z_curve>
__global__ void DFSPH_computeGridIdx_kernel(Vector3d* in, unsigned int* out, RealCuda kernel_radius, unsigned int num_particles,
                                            Vector3i gridOffset) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) { return; }

    if (z_curve) {

    }
    else {
        //the offset is used to be able to use a small grid bu placing the simulation correctly inside it
        Vector3d pos = (in[i] / kernel_radius) + gridOffset;
        pos.toFloor();
        out[i] = COMPUTE_CELL_INDEX(pos.x, pos.y, pos.z);
    }
}

__global__ void DFSPH_Histogram_kernel(unsigned int* in, unsigned int* out, unsigned int num_particles) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) { return; }

    atomicAdd(&(out[in[i]]), 1);

}

__global__ void DFSPH_setBufferValueToItself_kernel(unsigned int* buff, unsigned int buff_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buff_size) { return; }

    buff[i] = i;
}

__global__ void DFSPH_setVector3dBufferToZero_kernel(Vector3d* buff, unsigned int buff_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buff_size) { return; }

    buff[i] = Vector3d(0, 0, 0);
}

template <bool is_fluid_container>
__global__ void DFSPH_neighborsSearch_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }


    RealCuda radius_sq = data.m_kernel_precomp.getRadius();
    Vector3d pos = particleSet->pos[i];
    Vector3d pos_cell = (pos / radius_sq) + data.gridOffset; //on that line the radius is not yet squared
    pos_cell.toFloor();
    int x = pos_cell.x;
    int y = pos_cell.y;
    int z = pos_cell.z;
    radius_sq *= radius_sq;

    unsigned int nb_neighbors_fluid = 0;
    unsigned int nb_neighbors_boundary = 0;
    unsigned int nb_neighbors_dynamic_objects = 0;
    int* cur_neighbor_ptr = particleSet->neighbourgs + i*MAX_NEIGHBOURS;
    //int neighbors_fluid[MAX_NEIGHBOURS];//doing it with local buffer was not faster
    //int neighbors_boundary[MAX_NEIGHBOURS];

#ifdef USE_COMPLETE
    ///this version uses the morton indexes
#define ITER_CELLS_FOR_BODY(input_body,code){\
    const SPH::UnifiedParticleSet& body = input_body;\
    for (int k = -1; k < 2; ++k) {\
    for (int m = -1; m < 2; ++m) {\
    for (int n = -1; n < 2; ++n) {\
    unsigned int cur_cell_id = COMPUTE_CELL_INDEX(x + n, y + k, z + m);\
    unsigned int end = body.neighborsDataSet->cell_start_end[cur_cell_id + 1];\
    for (unsigned int cur_particle = body.neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {\
    unsigned int j = body.neighborsDataSet->p_id_sorted[cur_particle];\
    if ((pos - body.pos[j]).squaredNorm() < radius_sq) {\
    code\
}\
}\
}\
}\
}\
}
#else
    ///this version uses  standart indexes

    //since this version use the std index to be able to iterate on 3 successive cells
    //I can do the -1 at the start on x.
    //one thing: it x=0 then we can only iterate 2 cells at a time
    unsigned int successive_cells_count = (x > 0) ? 3 : 2;
    x = (x > 0) ? x - 1 : x;

#define ITER_CELLS_FOR_BODY(neighborsDataSet_i,pos_body_particles_i,code){\
    SPH::NeighborsSearchDataSet* neighborsDataSet= neighborsDataSet_i;\
    Vector3d* pos_body_particles=pos_body_particles_i;\
    for (int k = -1; k < 2; ++k) {\
    for (int m = -1; m < 2; ++m) {\
    unsigned int cur_cell_id = COMPUTE_CELL_INDEX(x, y + k, z + m);\
    unsigned int end = neighborsDataSet->cell_start_end[cur_cell_id + successive_cells_count];\
    for (unsigned int cur_particle = neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {\
    unsigned int j = neighborsDataSet->p_id_sorted[cur_particle];\
    if ((pos - pos_body_particles[j]).squaredNorm() < radius_sq) {\
    code\
}\
}\
}\
}\
}
#endif


    if (data.is_fluid_aggregated){
        int neighbors_solids[MAX_NEIGHBOURS];

        //dynamic bodies
        if (data.numDynamicBodies >0) {

#ifdef GROUP_DYNAMIC_BODIES_NEIGHBORS_SEARCH
            ITER_CELLS_FOR_BODY(data.neighborsDataSetGroupedDynamicBodies_cuda, data.posBufferGroupedDynamicBodies,
                                if(j<data.fluid_data_cuda->numParticles){
                                    if (i != j) { *cur_neighbor_ptr++ = j;	nb_neighbors_fluid++; }
                                }else{int body_id=0; int count_particles_previous_bodies=data.fluid_data_cuda->numParticles;
                                      while((count_particles_previous_bodies+data.vector_dynamic_bodies_data_cuda[body_id].numParticles)<j ){
                                          count_particles_previous_bodies+=data.vector_dynamic_bodies_data_cuda[body_id].numParticles;
                                          body_id++;
                                      }
                                      //*cur_neighbor_ptr++ = WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX(body_id, j-count_particles_previous_bodies);
                                      neighbors_solids[nb_neighbors_dynamic_objects]=WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX(body_id, j-count_particles_previous_bodies);
                                      nb_neighbors_dynamic_objects++;} )
        #else
            for (int id_body = 0; id_body < data.numDynamicBodies; ++id_body) {
                ITER_CELLS_FOR_BODY(data.vector_dynamic_bodies_data_cuda[id_body].neighborsDataSet, data.vector_dynamic_bodies_data_cuda[id_body].pos,
                                    *cur_neighbor_ptr++ = WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX(id_body, j); nb_neighbors_dynamic_objects++; )
            }
#endif

        }else{
            //fluid
            ITER_CELLS_FOR_BODY(data.fluid_data_cuda[0].neighborsDataSet, data.fluid_data_cuda[0].pos,
                    if (i != j) { *cur_neighbor_ptr++ = j;	nb_neighbors_fluid++; });
        }

        //boundaries
        ITER_CELLS_FOR_BODY(data.boundaries_data_cuda[0].neighborsDataSet, data.boundaries_data_cuda[0].pos,
                *cur_neighbor_ptr++ = j; nb_neighbors_boundary++; );


        //copy the dynamic bodies at the end
        for (int j=0;j<nb_neighbors_dynamic_objects;++j){
            *cur_neighbor_ptr++=neighbors_solids[j];
        }


    }else{
        //uses the standart version
        //fluid
        ITER_CELLS_FOR_BODY(data.fluid_data_cuda[0].neighborsDataSet, data.fluid_data_cuda[0].pos,
                if (!is_fluid_container || i != j) { *cur_neighbor_ptr++ = j;	nb_neighbors_fluid++; });

        //boundaries
        ITER_CELLS_FOR_BODY(data.boundaries_data_cuda[0].neighborsDataSet, data.boundaries_data_cuda[0].pos,
			if (is_fluid_container || i != j) { *cur_neighbor_ptr++ = j; nb_neighbors_boundary++; });


        if (data.numDynamicBodies > 0) {

#ifdef GROUP_DYNAMIC_BODIES_NEIGHBORS_SEARCH
            ITER_CELLS_FOR_BODY(data.neighborsDataSetGroupedDynamicBodies_cuda, data.posBufferGroupedDynamicBodies,
            {int body_id=0; int count_particles_previous_bodies=0;
             while((count_particles_previous_bodies+data.vector_dynamic_bodies_data_cuda[body_id].numParticles)<j ){
                 count_particles_previous_bodies+=data.vector_dynamic_bodies_data_cuda[body_id].numParticles;
                 body_id++;
             }
             *cur_neighbor_ptr++ = WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX(body_id, j-count_particles_previous_bodies);
             nb_neighbors_dynamic_objects++;} )
        #else
            for (int id_body = 0; id_body < data.numDynamicBodies; ++id_body) {
                ITER_CELLS_FOR_BODY(data.vector_dynamic_bodies_data_cuda[id_body].neighborsDataSet, data.vector_dynamic_bodies_data_cuda[id_body].pos,
                                    *cur_neighbor_ptr++ = WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX(id_body, j); nb_neighbors_dynamic_objects++; )
            }
#endif

        }
    }



    particleSet->numberOfNeighbourgs[3 * i] = nb_neighbors_fluid;
    particleSet->numberOfNeighbourgs[3 * i + 1] = nb_neighbors_boundary;
    particleSet->numberOfNeighbourgs[3 * i + 2] = nb_neighbors_dynamic_objects;

    //memcpy((neighbors_buff + i*MAX_NEIGHBOURS*2), neighbors_fluid, sizeof(int)*nb_neighbors_fluid);
    //memcpy((neighbors_buff + i*MAX_NEIGHBOURS * 2 + MAX_NEIGHBOURS), neighbors_boundary, sizeof(int)*nb_neighbors_boundary);


}

__global__ void DFSPH_neighborsSearchBasic_kernel(unsigned int numFluidParticles, RealCuda radius,
                                                  SPH::UnifiedParticleSet* fluid_data,
                                                  SPH::UnifiedParticleSet* boundaries_data,
                                                  SPH::UnifiedParticleSet* vect_dynamic_bodies, int nb_dynamic_bodies) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numFluidParticles) { return; }


    RealCuda radius_sq = radius;
    Vector3d pos = fluid_data->pos[i];
    radius_sq *= radius_sq;

    unsigned int nb_neighbors_fluid = 0;
    unsigned int nb_neighbors_boundary = 0;
    unsigned int nb_neighbors_dynamic_objects = 0;
    int* cur_neighbor_ptr = fluid_data->neighbourgs + i*MAX_NEIGHBOURS;

    for (int k = 0; k < fluid_data->numParticles; ++k) {
        if (i != k) {
            if ((fluid_data->pos[k] - pos).squaredNorm() < radius_sq) {
                *cur_neighbor_ptr++ = k;	nb_neighbors_fluid++;
            }
        }
    }

    /*
    for (int k = 0; k < boundaries_data->numParticles; ++k) {
        if ((boundaries_data->pos[k] - pos).squaredNorm() < radius_sq) {
            *cur_neighbor_ptr++ = k; nb_neighbors_boundary++;
        }
    }
    //*/

    /*
    for (int id_body = 0; id_body < nb_dynamic_bodies; ++id_body) {
        for (int k = 0; k < vect_dynamic_bodies[id_body].numParticles; ++k) {
            if ((vect_dynamic_bodies[id_body].pos[k] - pos).squaredNorm() < radius_sq) {
                *cur_neighbor_ptr++ = WRITTE_DYNAMIC_BODIES_PARTICLES_INDEX(id_body, k); nb_neighbors_dynamic_objects++;
            }
        }
    }
    //*/


    fluid_data->numberOfNeighbourgs[3 * i] = nb_neighbors_fluid;
    fluid_data->numberOfNeighbourgs[3 * i + 1] = nb_neighbors_boundary;
    fluid_data->numberOfNeighbourgs[3 * i + 2] = nb_neighbors_dynamic_objects;

}

void cuda_neighborsSearchInternal_sortParticlesId(Vector3d* pos, RealCuda kernel_radius, Vector3i gridOffset, int numParticles,
                                                  void **d_temp_storage_pair_sort, size_t   &temp_storage_bytes_pair_sort,
                                                  unsigned int* cell_id, unsigned int* cell_id_sorted,
                                                  unsigned int* p_id, unsigned int* p_id_sorted) {
    cudaError_t cudaStatus;


    /*
    //some test for the definition domain (it is just for debugging purposes)
    //check for negatives values
    for (int i = 0; i < numParticles; ++i) {
    Vector3d temp = (pos[i] / kernel_radius) + 2;
    if (temp.x <= 0 || temp.y <= 0 || temp.z <= 0 ) {
    fprintf(stderr, "negative coordinates: %d\n", (int)i);
    exit(1598);
    }
    }


    //find the bounding box of the particles
    Vector3d min = pos[0];
    Vector3d max = pos[0];
    for (int i = 0; i < numParticles; ++i) {

    if (pos[i].x < min.x) { min.x = pos[i].x; }
    if (pos[i].y < min.y) { min.y = pos[i].y; }
    if (pos[i].z < min.z) { min.z = pos[i].z; }

    if (pos[i].x > max.x) { max.x = pos[i].x; }
    if (pos[i].y > max.y) { max.y = pos[i].y; }
    if (pos[i].z > max.z) { max.z = pos[i].z; }

    }
    fprintf(stderr, "min: %f // %f // %f\n", min.x, min.y, min.z);
    fprintf(stderr, "max: %f // %f // %f\n", max.x, max.y, max.z);
    fprintf(stderr, "description: %f\n", CELL_ROW_LENGTH*kernel_radius);
    exit(1598);
    //*/
    int numBlocks = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;


    //compute the idx of the cell for each particles
    DFSPH_computeGridIdx_kernel<CELL_ROW_LENGTH, false> << <numBlocks, BLOCKSIZE >> > (pos, cell_id,
                                                                                       kernel_radius, numParticles, gridOffset);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "idxs failed: %d\n", (int)cudaStatus);
        exit(1598);
    }


    //do the actual sort
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(*d_temp_storage_pair_sort, temp_storage_bytes_pair_sort,
                                    cell_id, cell_id_sorted, p_id, p_id_sorted, numParticles);
    //*/


    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "sort failed: %d\n", (int)cudaStatus);
        exit(1598);
    }

}

void cuda_neighborsSearchInternal_computeCellStartEnd(int numParticles, unsigned int* cell_id_sorted,
                                                      unsigned int* hist, void **d_temp_storage_cumul_hist, size_t   &temp_storage_bytes_cumul_hist, unsigned int* cell_start_end) {
    cudaError_t cudaStatus;
    int numBlocks = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;


    //Now we need to determine the start and end of each cell
    //init the histogram values. Maybe doing it wiith thrust fill is faster.
    //the doc is not realy clear
    cudaMemset(hist, 0, (CELL_COUNT + 1) * sizeof(unsigned int));

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "histogram value reset failed: %d\n", (int)cudaStatus);
        exit(1598);
    }

    //compute the actual histogram (done here with atomic adds)
    DFSPH_Histogram_kernel << <numBlocks, BLOCKSIZE >> > (cell_id_sorted, hist, numParticles);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "histogram failed: " << (int)cudaStatus << std::endl;
        exit(1598);
    }

    //transformour histogram to a cumulative histogram to have  the start and end of each cell
    //note: the exlusive sum make so that each cell will contains it's start value
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(*d_temp_storage_cumul_hist, temp_storage_bytes_cumul_hist, hist, cell_start_end, (CELL_COUNT + 1));

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cumulative histogram failed: %d\n", (int)cudaStatus);
        exit(1598);
    }
}



//this is the bases for all kernels based function
template<typename T>
__global__ void DFSPH_sortFromIndex_kernel(T* in, T* out, unsigned int* index, unsigned int nbElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbElements) { return; }

    out[i] = in[index[i]];
}




#include <sstream>
void cuda_sortData(SPH::UnifiedParticleSet& particleSet, unsigned int * sort_id) {
    //*
    unsigned int numParticles = particleSet.neighborsDataSet->numParticles;
    int numBlocks = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
    unsigned int *p_id_sorted = sort_id;

    Vector3d* intermediate_buffer_v3d = particleSet.neighborsDataSet->intermediate_buffer_v3d;
    RealCuda* intermediate_buffer_real = particleSet.neighborsDataSet->intermediate_buffer_real;

    //pos
    DFSPH_sortFromIndex_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (particleSet.pos, intermediate_buffer_v3d, p_id_sorted, numParticles);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(particleSet.pos, intermediate_buffer_v3d, numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

    //vel
    DFSPH_sortFromIndex_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (particleSet.vel, intermediate_buffer_v3d, p_id_sorted, numParticles);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(particleSet.vel, intermediate_buffer_v3d, numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

    //mass
    DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.mass, intermediate_buffer_real, p_id_sorted, numParticles);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(particleSet.mass, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

    if (particleSet.has_factor_computation) {
        //kappa
        DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.kappa, intermediate_buffer_real, p_id_sorted, numParticles);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(particleSet.kappa, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

        //kappav
        DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.kappaV, intermediate_buffer_real, p_id_sorted, numParticles);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(particleSet.kappaV, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
    }



    //now that everything is sorted we can set each particle index to itself
    gpuErrchk(cudaMemcpy(p_id_sorted, particleSet.neighborsDataSet->p_id, numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

}







__global__ void generateShuffleIndex_kernel(unsigned int *shuffle_index, unsigned int nbElements, curandState *state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1) { return; }

    for (int j = 0; j < nbElements; ++j) {
        shuffle_index[j] = j;
    }



    curandState localState = *state;
    for (int j = 0; j < nbElements; ++j) {
        float x = curand_uniform(&localState);
        x *= nbElements;
        unsigned int idx = x;
        if (x < nbElements) {
            unsigned int temp = shuffle_index[idx];
            shuffle_index[idx] = shuffle_index[i];
            shuffle_index[i] = temp;
        }
    }
    *state = localState;
}

template<class T>
__global__ void fillRandom_kernel(unsigned int *buff, unsigned int nbElements, T min, T max, curandState *state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1) { return; }

    curandState localState = *state;
    for (int j = 0; j < nbElements; ++j) {
        T x= curand(&localState);
        x *= (max-min);
        x += min;
        buff[i] = x;
    }
    *state = localState;
}


void cuda_shuffleData(SPH::UnifiedParticleSet& particleSet) {
    unsigned int numParticles = particleSet.numParticles;
    int numBlocks = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;

    //create a random sorting index
    unsigned int* shuffle_index = SVS_CU::get()->shuffle_index;
    curandState *state= SVS_CU::get()->curand_state;
    if (shuffle_index == NULL) {
        cudaMallocManaged(&(SVS_CU::get()->shuffle_index), particleSet.numParticlesMax * sizeof(unsigned int));
        shuffle_index = SVS_CU::get()->shuffle_index;
        gpuErrchk(cudaDeviceSynchronize());
    }


    generateShuffleIndex_kernel << <1, 1 >> > (shuffle_index, numParticles, state);
    gpuErrchk(cudaDeviceSynchronize());


    unsigned int *p_id_sorted = shuffle_index;

    Vector3d* intermediate_buffer_v3d = particleSet.neighborsDataSet->intermediate_buffer_v3d;
    RealCuda* intermediate_buffer_real = particleSet.neighborsDataSet->intermediate_buffer_real;

    //pos
    DFSPH_sortFromIndex_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (particleSet.pos, intermediate_buffer_v3d, p_id_sorted, numParticles);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(particleSet.pos, intermediate_buffer_v3d, numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

    //vel
    DFSPH_sortFromIndex_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (particleSet.vel, intermediate_buffer_v3d, p_id_sorted, numParticles);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(particleSet.vel, intermediate_buffer_v3d, numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

    //mass
    DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.mass, intermediate_buffer_real, p_id_sorted, numParticles);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(particleSet.mass, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

    if (particleSet.has_factor_computation) {
        //kappa
        DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.kappa, intermediate_buffer_real, p_id_sorted, numParticles);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(particleSet.kappa, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

        //kappav
        DFSPH_sortFromIndex_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (particleSet.kappaV, intermediate_buffer_real, p_id_sorted, numParticles);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(particleSet.kappaV, intermediate_buffer_real, numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
    }



}


//this is the bases for all kernels based function
//I also use that kernel to reset the force

__global__ void DFSPH_updateDynamicObjectParticles_kernel(int numParticles, Vector3d* pos, Vector3d* vel, Vector3d* pos0,
                                                          Vector3d position, Vector3d velocity, Quaternion q, Vector3d angular_vel, Vector3d* F) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) { return; }

    //reset the force
    F[i] = Vector3d(0, 0, 0);

    //update location and velocity
    pos[i] = q.rotate(pos0[i]) + position;
    vel[i] = angular_vel.cross(pos[i] - position) + velocity;

}

void update_dynamicObject_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& particle_set) {
    if (particle_set.is_dynamic_object) {
        int numBlocks = (particle_set.numParticles + BLOCKSIZE - 1) / BLOCKSIZE;


        //update the particle location and velocity
        DFSPH_updateDynamicObjectParticles_kernel << <numBlocks, BLOCKSIZE >> > (particle_set.numParticles,
                                                                                 particle_set.pos, particle_set.vel, particle_set.pos0,
                                                                                 particle_set.rigidBody_cpu->position, particle_set.rigidBody_cpu->velocity,
                                                                                 particle_set.rigidBody_cpu->q, particle_set.rigidBody_cpu->angular_vel,
                                                                                 particle_set.F);

        //also we can use that time to reset the force buffer
        //directly done in the other kernel
        //DFSPH_setVector3dBufferToZero_kernel << <numBlocks, BLOCKSIZE >> > (container.F, container.numParticles);

        cudaError_t cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "update_dynamicObject_UnifiedParticleSet_cuda failed: %d\n", (int)cudaStatus);
            exit(1369);
        }
    }
}





__global__ void apply_delta_to_buffer_kernel(Vector3d* buffer, Vector3d delta, const unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) { return; }

    buffer[i] += delta;
}





__global__ void remove_particle_layer_kernel(SPH::UnifiedParticleSet* particleSet, Vector3d movement, Vector3d* min, Vector3d *max,
                                             RealCuda kernel_radius, Vector3i gridOffset,
                                             int* count_moved_particles, int* count_possible_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    Vector3d source_id = *min;
    Vector3d target_id = *max;

    if (movement.abs() != movement) {
        source_id = *max;
        target_id = *min;
    }


    Vector3d motion_axis = (movement / movement.norm()).abs();

    //compute the source and target cell row, we only keep the component in the direction of the motion
    source_id = (source_id / kernel_radius) + gridOffset ;
    source_id.toFloor();
    source_id *= motion_axis;

    target_id = (target_id / kernel_radius) + gridOffset;
    target_id.toFloor();
    target_id *= motion_axis;

    //compute the elll row for the particle and only keep the  component in the direction of the motion
    Vector3d pos = (particleSet->pos[i] / kernel_radius) + gridOffset;
    pos.toFloor();
    pos *= motion_axis;

    //I'll tag the particles that need to be moved with 25000000
    particleSet->neighborsDataSet->cell_id[i] = 0;

    if (pos == (source_id+movement)) {
        //I'll also move the paticles away
        particleSet->pos[i].y += 2.0f;
        particleSet->neighborsDataSet->cell_id[i] = 25000000;
        atomicAdd(count_moved_particles, 1);

    }else if (pos == (target_id - movement)) {
        int id = atomicAdd(count_possible_particles, 1);
        particleSet->neighborsDataSet->p_id_sorted[id] = i;
    }else if (pos == target_id || pos == source_id) {
        //move the particles that are on the border
        particleSet->pos[i] += movement*kernel_radius;
    }

}

__global__ void adapt_inserted_particles_position_kernel(SPH::UnifiedParticleSet* particleSet, int* count_moved_particles, int* count_possible_particles, 
                                                         Vector3d mov_pos, Vector3d plane_for_remaining) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    if (particleSet->neighborsDataSet->cell_id[i] == 25000000) {
        int id = atomicAdd(count_moved_particles, 1);

        if (id < (*count_possible_particles)) {
            int ref_particle_id = particleSet->neighborsDataSet->p_id_sorted[id];
            particleSet->pos[i] = particleSet->pos[ref_particle_id] + mov_pos;
            particleSet->vel[i] = particleSet->vel[ref_particle_id];
            particleSet->kappa[i] = particleSet->kappa[ref_particle_id];
            particleSet->kappaV[i] = particleSet->kappaV[ref_particle_id];

            particleSet->neighborsDataSet->cell_id[i] = 0;
        }
        else {
            //particleSet->pos[i]= plane_for_remaining;

        }
    }

}

__global__ void find_column_max_height_kernel(SPH::UnifiedParticleSet* particleSet, RealCuda* column_max_height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= CELL_ROW_LENGTH*CELL_ROW_LENGTH) { return; }

    int z = i / CELL_ROW_LENGTH;
    int x = i - z*CELL_ROW_LENGTH;

    RealCuda max_height = -2;

    for (int y = CELL_ROW_LENGTH - 1; y >= 0; --y) {
        int cell_id=COMPUTE_CELL_INDEX(x, y, z);
        if (particleSet->neighborsDataSet->cell_start_end[cell_id + 1] != particleSet->neighborsDataSet->cell_start_end[cell_id]) {
            unsigned int end = particleSet->neighborsDataSet->cell_start_end[cell_id + 1];
            for (unsigned int cur_particle = particleSet->neighborsDataSet->cell_start_end[cell_id]; cur_particle < end; ++cur_particle) {
                unsigned int j = particleSet->neighborsDataSet->p_id_sorted[cur_particle];
                if (particleSet->pos[j].y > max_height) {
                    max_height = particleSet->pos[j].y;
                }
            }
            break;
        }
    }

    column_max_height[i] = max_height;

}

__global__ void translate_borderline_particles_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, RealCuda* column_max_height,
                                                      int* count_moved_particles,
                                                      int* moved_particles_min, int* moved_particles_max, int* count_invalid_position,
                                                      Vector3d movement, int count_possible_pos, int count_remaining_pos,
                                                      RealCuda start_height_min,  RealCuda start_height_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    RealCuda affected_distance_sq = data.particleRadius*1.5;
    affected_distance_sq *= affected_distance_sq;

    RealCuda precise_affected_distance_sq = data.particleRadius*1.5;
    precise_affected_distance_sq *= precise_affected_distance_sq;



    //compute tsome constants
    Vector3d min=*data.bmin;
    Vector3d max=*data.bmax;
    RealCuda p_distance = data.particleRadius * 2;
    Vector3d plane_unit = movement.abs() / movement.norm();
    bool positive_motion = plane_unit.dot(movement)>0;
    Vector3d plane_unit_perp = (Vector3d(1, 0, 1) - plane_unit);
    //I need to know the width I have
    Vector3d width = (max) - (min);
    //and only kee the component oriented perpendicular with the plane
    int max_count_width =  width.dot(plane_unit_perp) / p_distance;
    //idk why but with that computation it's missing one particle so I'll add it
    max_count_width ++;

    int max_row=(width.dot(plane_unit)/5)/p_distance;


    //just basic one that move the particle above for testing putposes
    /*
    for (int k = 0; k < 2; ++k) {
        Vector3d plane = data.damp_planes[k];
        if ((particleSet->pos[i] * plane_unit - plane).squaredNorm() < affected_distance_sq) {
            particleSet->pos[i].y += 2.0f;
            break;
        }
    }
    return;
    //*/
    bool remaining_particle=particleSet->neighborsDataSet->cell_id[i] == 25000000;

    //so I know I onlyhave 2 damp planes the first one being the one near the min
    for (int k = 0; k < 2; ++k) {

        Vector3d plane = data.damp_planes[k];
        if (((particleSet->pos[i] * plane_unit - plane).squaredNorm() < affected_distance_sq)||remaining_particle) {
            //let's try to estimate the density to see if there are actual surpression
            bool distance_too_short=false;
            if (remaining_particle){
                distance_too_short=true;
            }else{

                if (k==0){
                    //we can do a simple distance check in essence

                    Vector3d cur_particle_pos=particleSet->pos[i];


                    Vector3i cell_pos=(particleSet->pos[i]/data.getKernelRadius()).toFloor()+data.gridOffset;
                    cell_pos+=Vector3i(0,-1,0);
                    //ok since I want to explore the bottom cell firts I need to move in the plane
                    cell_pos-=plane_unit_perp;

                    //potential offset
                    Vector3d particle_offset=Vector3d(0,0,0);
                    //*



                    //we skipp some cases to only move the particles that are on one side
                    if (positive_motion){
                        //for positive motion the lower plane is on the source
                        if (plane_unit.dot(cur_particle_pos) <= plane_unit.dot(data.damp_planes[0])){
                            continue;
                            cell_pos += plane_unit*1;//since the particle lower than that have already been moved in the direction once
                        }else{
                            cell_pos -= plane_unit*2;
                        }
                    }else{
                        //if the motion is negative then the lower plane is the target
                        if (plane_unit.dot(cur_particle_pos) <= plane_unit.dot(data.damp_planes[0])){
                            //the cell that need to be explored are on row away from us
                            cell_pos+=plane_unit;
                        }else{
                            continue;
                            //we need to move the particle we are checking toward on rows in the direction of the movement
                            particle_offset=plane_unit*data.getKernelRadius()*-1;
                        }
                    }
                    //*/

                    //I only need to check if the other side of the jonction border is too close, no need to check the same side since
                    //it was part of a fluid at rest
                    for (int k=0;k<3;++k){//that's y
                        for (int l=0;l<3;++l){//that's the coordinate in the plane

                            Vector3i cur_cell_pos=cell_pos+plane_unit_perp*l;
                            int cur_cell_id=COMPUTE_CELL_INDEX(cur_cell_pos.x,cur_cell_pos.y+k,cur_cell_pos.z);
                            UnifiedParticleSet* body=data.fluid_data_cuda;
                            NeighborsSearchDataSet* neighborsDataSet=body->neighborsDataSet;
                            unsigned int end = neighborsDataSet->cell_start_end[cur_cell_id+1];
                            for (unsigned int cur_particle = neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {
                                unsigned int j = neighborsDataSet->p_id_sorted[cur_particle];
                                if ((cur_particle_pos - (body->pos[j]+particle_offset)).squaredNorm() < precise_affected_distance_sq) {
                                    distance_too_short=true;
                                    break;
                                }
                            }
                        }
                        if (distance_too_short){break;}
                    }

                }else{
                    Vector3d cur_particle_pos=particleSet->pos[i];


                    Vector3i cell_pos=(particleSet->pos[i]/data.getKernelRadius()).toFloor()+data.gridOffset;
                    cell_pos+=Vector3i(0,-1,0);
                    //ok since I want to explore the bottom cell firts I need to move in the plane
                    cell_pos-=plane_unit_perp;

                    //on the target side the cell of the right side are a copy of the left side !
                    // so we have to check the row agaisnt itself
                    //but we will have to translate the particles depending on the side we are on
                    Vector3d particle_offset=Vector3d(0,0,0);


                    if (positive_motion){
                        if (plane_unit.dot(cur_particle_pos) > plane_unit.dot(data.damp_planes[1])) {
                            //the cell that need to be explored are on row away from us
                            cell_pos-=plane_unit;
                        }else{
                            continue;
                            //we need to move the particle we are checking toward on rows in the direction of the movement
                            particle_offset=plane_unit*data.getKernelRadius();
                        }
                    }else{
                        if (plane_unit.dot(cur_particle_pos) > plane_unit.dot(data.damp_planes[1])) {
                            continue;
                            cell_pos -= plane_unit*1;//since the particle lower than that have already been moved in the direction once
                        }else{
                            cell_pos += plane_unit*2;
                        }

                    }


                    //I only need to check if the other side of the jonction border is too close, no need to check the same side since
                    //it was part of a fluid at rest
                    for (int k=0;k<3;++k){//that's y
                        for (int l=0;l<3;++l){//that's the coordinate in the plane

                            Vector3i cur_cell_pos=cell_pos+plane_unit_perp*l;
                            int cur_cell_id=COMPUTE_CELL_INDEX(cur_cell_pos.x,cur_cell_pos.y+k,cur_cell_pos.z);
                            UnifiedParticleSet* body=data.fluid_data_cuda;
                            NeighborsSearchDataSet* neighborsDataSet=body->neighborsDataSet;
                            unsigned int end = neighborsDataSet->cell_start_end[cur_cell_id + 1];
                            for (unsigned int cur_particle = neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {
                                unsigned int j = neighborsDataSet->p_id_sorted[cur_particle];
                                if ((cur_particle_pos - (body->pos[j]+particle_offset)).squaredNorm() < precise_affected_distance_sq) {
                                    distance_too_short=true;
                                    break;
                                }
                            }
                            if (distance_too_short){break;}
                        }
                    }

                }
            }


            if (!distance_too_short){
                //that mean this particle is not too close for another and there is no need to handle it
                continue;
            }else{
                //for testing purposes
                //particleSet->pos[i].y+=2.0f;
                //return;
            }


            //get a unique id to compute the position
            //int id = atomicAdd((k==0)? moved_particles_back : moved_particles_front, 1);
            int id = atomicAdd(count_moved_particles, 1);


            //before trying to position it above the surface, if we sill have palce in front of the fluid we will put it there
            if (count_remaining_pos>0){
                id-= count_remaining_pos;
                if (id<0){
                    id+=count_possible_pos;
                    //so so it means we have to put that paricle in the new layer
                    int ref_particle_id = particleSet->neighborsDataSet->p_id_sorted[id];
                    particleSet->pos[i] = particleSet->pos[ref_particle_id] + movement*data.getKernelRadius();
                    particleSet->vel[i] = particleSet->vel[ref_particle_id];
                    particleSet->kappa[i] = particleSet->kappa[ref_particle_id];
                    particleSet->kappaV[i] = particleSet->kappaV[ref_particle_id];

                    particleSet->neighborsDataSet->cell_id[i] = 0;

                    //andwe have to reexecute the loop since that new pos may be next to a border
                    k=-1;
                    continue;
                }
            }




            //*
            //we place one third of the particles in the front the rest is placed in the back
            bool near_min=(positive_motion)? true : false;
            if ((id%10)==0){
                near_min=!near_min;
            }

            //we repeat until the particle is placed
            for(;;){

                if (near_min){
                    id = atomicAdd(moved_particles_min, 1);
                }else{
                    id = atomicAdd(moved_particles_max, 1);
                }
                //*/



                //and compute the particle position
                int row_count = id / max_count_width;
                int level_count = row_count / max_row;

                Vector3d pos_local = Vector3d(0, 0, 0);
                pos_local.y += level_count*(p_distance*0.80);
                //the 1 or -1 at the end is because the second iter start at the max and it need to go reverse
                pos_local += (plane_unit*p_distance*(row_count - level_count*max_row) + plane_unit_perp*p_distance*(id - row_count*max_count_width))*((near_min) ? 1 : -1);
                //just a simple interleave on y
                if (level_count & 1 != 0) {
                    pos_local += (Vector3d(1,0,1)*(p_distance / 2.0f))*((near_min) ? 1 : -1);
                }

                //now I need to find the first possible position
                //it depends if we are close to the min of to the max
                Vector3d pos_f = (near_min) ? min : max;

                //and for the height we need to find the column
                Vector3d pos_temp = (pos_f + pos_local);

                //now the problem is that the column id wontains the height befoore any particle movement;
                //so from the id I have here I need to know the corresponding id before any particle movement
                //the easiest way is to notivce that anything before the first plane and after the secodn plane have been moved
                //anything else is still the same
                if (near_min){
                    //0 is the min plane
                    if (plane_unit.dot(pos_temp) < plane_unit.dot(data.damp_planes[0])){
                        pos_temp -= (movement*data.getKernelRadius());
                    }
                }else{
                    //1 is the max plane
                    if (plane_unit.dot(pos_temp) > plane_unit.dot(data.damp_planes[1])) {
                        pos_temp -= (movement*data.getKernelRadius());
                    }
                }

                pos_temp= pos_temp / data.getKernelRadius() + data.gridOffset;
                pos_temp.toFloor();

                //read the actual height
                int column_id = pos_temp.x + pos_temp.z*CELL_ROW_LENGTH;

                RealCuda start_height=((near_min) ? start_height_min : start_height_max) + p_distance + p_distance/4.0;
                RealCuda min_height= column_max_height[column_id] + p_distance;

                if((start_height+pos_local.y)<min_height){
                    //this means we can't place a particle here so I need to get another
                    atomicAdd(count_invalid_position, 1);
                    continue;
                }

                pos_f.y = start_height;
                pos_f += pos_local;


                particleSet->pos[i] = pos_f;
                particleSet->vel[i] = Vector3d(0,0,0);
                particleSet->kappa[i] = 0;
                particleSet->kappaV[i] = 0;

                //if he particle was moved we are done
                return;

            }
        }
    }
}

//#define SHOW_MESSAGES_IN_CUDA_FUNCTIONS
void move_simulation_cuda(SPH::DFSPHCData& data, Vector3d movement) {
    data.damp_planes_count = 0;
    //compute the movement on the position and the axis
    Vector3d mov_pos = movement*data.getKernelRadius();
    Vector3d mov_axis = (movement.abs()) / movement.norm();

    //we store the min and max before the movement of the solid particles
    get_min_max_pos_kernel << <1, 1 >> > (data.boundaries_data->gpu_ptr, data.bmin, data.bmax, data.particleRadius);
    gpuErrchk(cudaDeviceSynchronize());

#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
    std::cout << "test min_max: " << data.bmin->x << " " << data.bmin->y << " " << data.bmin->z << " " << data.bmax->x << " " << data.bmax->y << " " << data.bmax->z << std::endl;
#endif
    //move the boundaries
    //we need to move the positions
    SPH::UnifiedParticleSet* particleSet = data.boundaries_data;
    {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();


        unsigned int numParticles = particleSet->numParticles;
        int numBlocks = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;

        //move the particles
        apply_delta_to_buffer_kernel<< <numBlocks, BLOCKSIZE >> > (particleSet->pos, mov_pos, numParticles);
        gpuErrchk(cudaDeviceSynchronize());



#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        float time = std::chrono::duration_cast<std::chrono::nanoseconds> (end - start).count() / 1000000.0f;
        std::cout << "time to move solid particles simu: " << time << " ms" << std::endl;
#endif
    }

    //and now the fluid
    particleSet = data.fluid_data;
    {
        //I'll need the information of whih cell contains which particles
        particleSet->initNeighborsSearchData(data, false);


        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        //first I need the highest particle for each cell
        static RealCuda* column_max_height = NULL;
        if (column_max_height == NULL) {
            cudaMallocManaged(&(column_max_height), CELL_ROW_LENGTH*CELL_ROW_LENGTH * sizeof(RealCuda));
        }
        {
            int numBlocks = (CELL_ROW_LENGTH*CELL_ROW_LENGTH + BLOCKSIZE - 1) / BLOCKSIZE;
            find_column_max_height_kernel << <numBlocks, BLOCKSIZE >> > (particleSet->gpu_ptr, column_max_height);
            gpuErrchk(cudaDeviceSynchronize());
        }




        //for the fluid I don't want to "move"the fluid, I have to rmv some particles and
        //add others to change the simulation area of the fluid
        //the particles that I'll remove are the ones in the second layer when a linear index is used
        //to find the second layer just take the first particle and you add 1to the cell id on the desired direction
        unsigned int numParticles = particleSet->numParticles;
        int numBlocks = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;

        //to remove the particles the easiest way is to attribute a huge id to the particles I want to rmv and them to
        //sort the particles but that id followed by lowering the particle number
        int* count_rmv_particles = SVS_CU::get()->count_rmv_particles;
        int* count_possible_particles = SVS_CU::get()->count_possible_particles;
        int* count_moved_particles = SVS_CU::get()->count_moved_particles; //this is for later
        int* count_invalid_position = SVS_CU::get()->count_invalid_position; //this is for later
        gpuErrchk(cudaMemset(count_rmv_particles, 0, sizeof(int)));
        gpuErrchk(cudaMemset(count_possible_particles, 0, sizeof(int)));

        //this flag tjhe particles that need tobe moved and store the index of the particles that are in the target row
        //also apply the movement to the border rows
        remove_particle_layer_kernel << <numBlocks, BLOCKSIZE >> > (particleSet->gpu_ptr, movement, data.bmin, data.bmax, data.getKernelRadius(),
                                                                    data.gridOffset, count_rmv_particles, count_possible_particles);
        gpuErrchk(cudaDeviceSynchronize());

#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
        std::cout << "count particle delta: (moved particles, possible particles)" << *count_rmv_particles <<"  "<< *count_possible_particles<< std::endl;
#endif
        std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

        //compute the positions of the 2 planes where there is a junction
        //the first of the two planes need to be the source one
        //calc the postion of the jonction planes
        //we updata the min max so that it now considers the new borders
        get_min_max_pos_kernel << <1, 1 >> > (data.boundaries_data->gpu_ptr, data.bmin, data.bmax, data.particleRadius);
        gpuErrchk(cudaDeviceSynchronize());
#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
        std::cout << "test min_max_2: " << data.bmin->x << " " << data.bmin->y << " " << data.bmin->z << " " << data.bmax->x << " " << data.bmax->y << " " << data.bmax->z << std::endl;
#endif

        //min plane
        RealCuda min_plane_precision = data.particleRadius / 1000;
        Vector3d plane = (*data.bmin)*mov_axis;
        plane /= data.getKernelRadius();
        plane.toFloor();
        plane += (movement.abs() == movement)?movement:(movement.abs()*2);
        plane *= data.getKernelRadius();
        //we need to prevent going to close to 0,0,0
        if (plane.norm() < min_plane_precision) {
            plane = mov_axis*min_plane_precision;
        }
        data.damp_planes[data.damp_planes_count++] = plane;

        //max plane
        plane = (*data.bmax)*mov_axis;
        plane /= data.getKernelRadius();
        plane.toFloor();
        plane -= (movement.abs() == movement)?movement:0;
        plane *= data.getKernelRadius();
        //we need to prevent going to close to 0,0,0
        if (plane.norm() < min_plane_precision) {
            plane = mov_axis*min_plane_precision;
        }
        data.damp_planes[data.damp_planes_count++] = plane;

        //always save the source
        if (movement.abs() == movement) {
            plane= data.damp_planes[data.damp_planes_count - 2];
        }

        //now modify the position of the particles that need to be moved in the new layers
        //if there are more particle that neeed to be moved than available positions
        //I'll put the additional particles in the junction plance on the side where particles have been removed
        gpuErrchk(cudaMemset(count_rmv_particles, 0, sizeof(int)));
        adapt_inserted_particles_position_kernel << <numBlocks, BLOCKSIZE >> > (particleSet->gpu_ptr, count_rmv_particles, count_possible_particles,
                                                                                mov_pos, plane);
        gpuErrchk(cudaDeviceSynchronize());


        std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();





        //trigger the damping mechanism
        data.damp_borders = true;
        if (data.damp_borders){
            data.damp_borders_steps_count = 10;
            add_border_to_damp_planes_cuda(data);
        }


        //what I what here is the minimum height in the area where the article will be placed near the surface
        //for both the min side and the max side
        int min_mov_dir=((*data.bmin/data.getKernelRadius()+data.gridOffset)*mov_axis).toFloor().norm();
        int max_mov_dir=((*data.bmax/data.getKernelRadius()+data.gridOffset)*mov_axis).toFloor().norm();

        Vector3d width = (*data.bmax) - (*data.bmin);
        int placement_row=(width.dot(mov_axis)/5)/data.getKernelRadius();

        RealCuda height_near_min=100000;
        RealCuda height_near_max=100000;

        for (int j=0;j<CELL_ROW_LENGTH;++j){
            for (int i=0;i<CELL_ROW_LENGTH;++i){
                int column_id = i + j*CELL_ROW_LENGTH;
                if (column_max_height[column_id]<=0){
                    continue;
                }


                int id_mov_dir=Vector3d(i,0,j).dot(mov_axis);


                if (abs(id_mov_dir-min_mov_dir)<placement_row){
                    height_near_min=MIN_MACRO(height_near_min,column_max_height[column_id]);
                }

                if (abs(id_mov_dir-max_mov_dir)<placement_row){
                    height_near_max=MIN_MACRO(height_near_max,column_max_height[column_id]);
                }
            }
        }
#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
        std::cout<<"check start positon (front back): "<<height_near_max<<"   "<<height_near_min<<std::endl;
#endif





        //transate the particles that are too close to the jonction planes
        int count_possible_pos=*count_possible_particles;
        int count_remaining_pos=MAX_MACRO(count_possible_pos-(*count_rmv_particles),0);

        gpuErrchk(cudaMemset(count_rmv_particles, 0, sizeof(int)));
        gpuErrchk(cudaMemset(count_possible_particles, 0, sizeof(int)));
        gpuErrchk(cudaMemset(count_moved_particles, 0, sizeof(int)));
        gpuErrchk(cudaMemset(count_invalid_position, 0, sizeof(int)));
        data.destructor_activated = false;
        translate_borderline_particles_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, column_max_height,
                                                                             count_moved_particles,
                                                                             count_rmv_particles, count_possible_particles, count_invalid_position,
                                                                             movement, count_possible_pos, count_remaining_pos,
                                                                             height_near_min, height_near_max);
        gpuErrchk(cudaDeviceSynchronize());
        data.destructor_activated = true;


#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
        std::cout<<"number of particles displaced: "<<*count_moved_particles-*count_invalid_position<<"  with "<<
                   *count_rmv_particles+*count_possible_particles <<" at the surface and "<<
                   *count_invalid_position<<" rerolled positions"<<std::endl;

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        float time = std::chrono::duration_cast<std::chrono::nanoseconds> (tp1 - start).count() / 1000000.0f;
        float time_1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000.0f;
        float time_2 = std::chrono::duration_cast<std::chrono::nanoseconds> (end - tp2).count() / 1000000.0f;
        std::cout << "time to move fluid simu: " << time + time_1 + time_2 << " ms  (" << time << "  " << time_1 << "  " << time_2 << ")" << std::endl;
#endif


        //add the wave canceler
        //do nnot use it it do not works properly
        data.cancel_wave = false;
        if (data.cancel_wave){
            data.damp_borders_steps_count = 10;
            //fix the height at chich I have to start stoping the wave
            RealCuda global_height = 0;
            int count_existing_columns = 0;
            for (int i = 0; i < CELL_ROW_LENGTH*CELL_ROW_LENGTH; ++i) {
                if (column_max_height[i] > 0) {
                    global_height += column_max_height[i];
                    count_existing_columns++;
                }
            }
            global_height /= count_existing_columns;
            data.cancel_wave_lowest_point=global_height/2.0;

            //and now fix the 2plane where the wave needs to be stoped
            data.cancel_wave_planes[0]=(*data.bmin)*mov_axis + mov_axis*placement_row*data.getKernelRadius();
            data.cancel_wave_planes[1]=(*data.bmax)*mov_axis - mov_axis*placement_row*data.getKernelRadius();
        }

    }

    //we can now update the offset on the grid
    data.gridOffset-=movement;

    //and we need ot updatethe neighbor structure for the static particles
    //I'll take the easy way and just rerun the neighbor computation
    //there shoudl eb a faster way but it will be enougth for now
    data.boundaries_data->initNeighborsSearchData(data, false);
}

void add_border_to_damp_planes_cuda(SPH::DFSPHCData& data) {

    get_min_max_pos_kernel << <1, 1 >> > (data.boundaries_data->gpu_ptr, data.bmin, data.bmax, data.particleRadius);
    gpuErrchk(cudaDeviceSynchronize());


    RealCuda min_plane_precision = data.particleRadius / 1000;
    data.damp_planes[data.damp_planes_count ++] = Vector3d((abs(data.bmin->x) > min_plane_precision) ? data.bmin->x : min_plane_precision, 0, 0);
    data.damp_planes[data.damp_planes_count ++] = Vector3d((abs(data.bmax->x) > min_plane_precision) ? data.bmax->x : min_plane_precision, 0, 0);
    data.damp_planes[data.damp_planes_count ++] = Vector3d(0, 0, (abs(data.bmin->z) > min_plane_precision) ? data.bmin->z : min_plane_precision);
    data.damp_planes[data.damp_planes_count ++] = Vector3d(0, 0, (abs(data.bmax->z) > min_plane_precision) ? data.bmax->z : min_plane_precision);
    data.damp_planes[data.damp_planes_count ++] = Vector3d(0, (abs(data.bmin->y) > min_plane_precision) ? data.bmin->y : min_plane_precision, 0);

}

//the logic will be I'll get the 5 highest particles and then keep the median
//this mean the actual height willbbe slightly higher but it's a good tradeoff
//the problem with this method is that it can't handle realy low valumes of fluid...
///TODO find a better way ... maybe just keeping the highest is fine since I'll take the median of every columns anyway ...
__global__ void find_splashless_column_max_height_kernel(SPH::UnifiedParticleSet* particleSet, RealCuda* column_max_height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= CELL_ROW_LENGTH*CELL_ROW_LENGTH) { return; }

    int z = i / CELL_ROW_LENGTH;
    int x = i - z*CELL_ROW_LENGTH;

    //this array store the highest heights for the column
    //later values are higher
    int count_values_for_median=3;
    //RealCuda max_height[5] = { -2, -2, -2, -2, -2 };
    RealCuda max_height[3] = { -2, -2, -2};
    int count_actual_values = 0;

    for (int y = CELL_ROW_LENGTH - 1; y >= 0; --y) {
        int cell_id = COMPUTE_CELL_INDEX(x, y, z);
        if (particleSet->neighborsDataSet->cell_start_end[cell_id + 1] != particleSet->neighborsDataSet->cell_start_end[cell_id]) {
            unsigned int end = particleSet->neighborsDataSet->cell_start_end[cell_id + 1];
            for (unsigned int cur_particle = particleSet->neighborsDataSet->cell_start_end[cell_id]; cur_particle < end; ++cur_particle) {
                unsigned int j = particleSet->neighborsDataSet->p_id_sorted[cur_particle];
                count_actual_values++;
                RealCuda cur_height = particleSet->pos[j].y;
                int is_superior = -1;
                //so I need to find the right cell of the max array
                //the boolean will indicate the id of the last cell for which the new height was superior
                for (int k = 0; k < count_values_for_median; ++k) {
                    if (cur_height> max_height[k]) {
                        is_superior = k;
                    }
                }
                if (is_superior > -1) {
                    //Now I need to propagate the values in the array to make place for the new one
                    for (int k = 0; k < is_superior; ++k) {
                        max_height[k] = max_height[k+1];
                    }
                    max_height[is_superior] = cur_height;
                }
            }
            if (count_actual_values>(count_values_for_median-1)){
                break;
            }
        }
    }

    //and we keep the median value only if there are enougth particles in the column (so that the result is relatively correct)
    column_max_height[i] = (count_actual_values>(count_values_for_median-1))?max_height[(count_values_for_median-1)/2]:-2;

}

__global__ void tag_particles_above_limit_hight_kernel(SPH::UnifiedParticleSet* particleSet, RealCuda target_height, int* count_flagged_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    //put the particles that will be removed at the end
    if (particleSet->pos[i].y > target_height) {
        particleSet->neighborsDataSet->cell_id[i] = 30000000;
        atomicAdd(count_flagged_particles, 1);
    }
}

__global__ void place_additional_particles_right_above_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, RealCuda* column_max_height,
                                                              int count_new_particles, Vector3d border_range, int* count_created_particles) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count_new_particles) { return; }

    Vector3d min = *data.bmin;
    Vector3d max = *data.bmax;
    RealCuda p_distance = data.particleRadius * 2;
    //I need to know the width I have
    Vector3d width = (max)-(min);
    width.toAbs();
    Vector3d max_count_width = width / p_distance;
    max_count_width.toFloor();
    //idk why but with that computation it's missing one particle so I'll add it
    max_count_width+=1;



    //and compute the particle position
    int row_count = id / max_count_width.x;
    int level_count = row_count / max_count_width.z;

    Vector3d pos_local = Vector3d(0, 0, 0);
    pos_local.y += level_count*(p_distance*0.80);
    pos_local.x += (id - row_count*max_count_width.x)*p_distance;
    pos_local.z += (row_count - level_count*max_count_width.z)*p_distance;
    //just a simple interleave on y
    if (level_count & 1 != 0) {
        pos_local += Vector3d(1, 0, 1)*(p_distance / 2.0f);
    }

    //now I need to find the first possible position
    //it depends if we are close to the min of to the max
    Vector3d pos_f = min;

    //and for the height we need to find the column
    Vector3d pos_temp = (pos_f + pos_local);

    //now if required check if the particle is near enougth from the border
    int effective_id = 0;
    if (border_range.squaredNorm() > 0) {
        min += border_range;
        max -= border_range;
        //
        if (!(pos_temp.x<min.x || pos_temp.z<min.z || pos_temp.x>max.x || pos_temp.z>max.z)) {
            return;
        }
        effective_id=atomicAdd(count_created_particles, 1);
    }
    else {
        effective_id = id;
    }


    //read the actual height
    pos_temp = pos_temp / data.getKernelRadius() + data.gridOffset;
    pos_temp.toFloor();
    int column_id = pos_temp.x + pos_temp.z*CELL_ROW_LENGTH;
    pos_f.y = column_max_height[column_id] + p_distance+ p_distance/4.0;

    pos_f += pos_local;

    int global_id = effective_id + particleSet->numParticles;
    particleSet->mass[global_id] = particleSet->mass[0];
    particleSet->pos[global_id] = pos_f;
    particleSet->vel[global_id] = Vector3d(0, 0, 0);
    particleSet->kappa[global_id] = 0;
    particleSet->kappaV[global_id] = 0;
}


RealCuda find_fluid_height_cuda(SPH::DFSPHCData& data){
    SPH::UnifiedParticleSet* particleSet = data.fluid_data;


    //we will need the neighbors data to know where the particles are
    particleSet->initNeighborsSearchData(data, false);



    //so first i need to kow the fluid height
    //the main problem is that I don't want to consider splash particles
    //so I need a special kernel for that
    //first I need the highest particle for each cell
    RealCuda* column_max_height = SVS_CU::get()->column_max_height;
    {
        int numBlocks = (CELL_ROW_LENGTH*CELL_ROW_LENGTH + BLOCKSIZE - 1) / BLOCKSIZE;
        //find_column_max_height_kernel << <numBlocks, BLOCKSIZE >> > (particleSet->gpu_ptr, column_max_height);
        find_splashless_column_max_height_kernel << <numBlocks, BLOCKSIZE >> > (particleSet->gpu_ptr, column_max_height);
        gpuErrchk(cudaDeviceSynchronize());
    }

    //now I keep the avg of all the cells containing enought particles
    //technicaly i'd prefer the median but it would require way more computations
    //also doing it on the gpu would be better but F it for now
    RealCuda global_height = 0;
    int count_existing_columns = 0;
    for (int i = 0; i < CELL_ROW_LENGTH*CELL_ROW_LENGTH; ++i) {
        if (column_max_height[i] > 0) {
            global_height += column_max_height[i];
            count_existing_columns++;
        }
    }
    global_height /= count_existing_columns;

#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
    std::cout << "global height detected: " << global_height << "  over column count " << count_existing_columns << std::endl;
#endif

    return global_height;
}



void control_fluid_height_cuda(SPH::DFSPHCData& data, RealCuda target_height) {
#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
    std::cout << "start fluid level control" << std::endl;
#endif 

    SPH::UnifiedParticleSet* particleSet = data.fluid_data;

    RealCuda global_height = find_fluid_height_cuda(data);


    //I'll take an error margin of 5 cm for now
    if (abs(global_height - target_height) < 0.05) {
        return;
    }

    //now we have 2 possible cases
    //either not enougth particles, or too many

    if (global_height > target_height) {
        //so we have to many particles
        //to rmv them, I'll flag the particles above the limit
        int* tagged_particles_count = SVS_CU::get()->tagged_particles_count;
        *tagged_particles_count = 0;

        unsigned int numParticles = particleSet->numParticles;
        int numBlocks = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;

        //tag the particles and count them
        tag_particles_above_limit_hight_kernel << <numBlocks, BLOCKSIZE >> > (particleSet->gpu_ptr, target_height, tagged_particles_count);
        gpuErrchk(cudaDeviceSynchronize());

        //now use the same process as when creating the neighbors structure to put the particles to be removed at the end
        cub::DeviceRadixSort::SortPairs(particleSet->neighborsDataSet->d_temp_storage_pair_sort, particleSet->neighborsDataSet->temp_storage_bytes_pair_sort,
                                        particleSet->neighborsDataSet->cell_id, particleSet->neighborsDataSet->cell_id_sorted,
                                        particleSet->neighborsDataSet->p_id, particleSet->neighborsDataSet->p_id_sorted, particleSet->numParticles);
        gpuErrchk(cudaDeviceSynchronize());
        cuda_sortData(*particleSet, particleSet->neighborsDataSet->p_id_sorted);
        gpuErrchk(cudaDeviceSynchronize());

        //and now you can update the number of particles
        int new_num_particles = particleSet->numParticles - *tagged_particles_count;
        particleSet->updateActiveParticleNumber(new_num_particles);
#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
        std::cout << "new number of particles: " << particleSet->numParticles << std::endl;
#endif

    }
    else {
        //here we are missing fluid particles
        //Ahahahah... ok there is no way in hell I have a correct solution for that ...
        //but let's build smth
        //so let's supose that there are no objects near the borders of the fluid
        //and I'll add the particles there sright above the existing particles

        //so first I need to have the min max and the max height for each column (the actual one even taking the plash into consideration
        get_min_max_pos_kernel << <1, 1 >> > (data.boundaries_data->gpu_ptr, data.bmin, data.bmax, data.particleRadius);
        gpuErrchk(cudaDeviceSynchronize());

        RealCuda* column_max_height = SVS_CU::get()->column_max_height;

        {
            int numBlocks = (CELL_ROW_LENGTH*CELL_ROW_LENGTH + BLOCKSIZE - 1) / BLOCKSIZE;
            find_column_max_height_kernel << <numBlocks, BLOCKSIZE >> > (particleSet->gpu_ptr, column_max_height);
            gpuErrchk(cudaDeviceSynchronize());
        }



        //so now add particles near the border (let's say in the 2 column near the fluid border
        //untill you reach the desired liquid level there
        //note, if there are no rigid bodies in the simulation I can add the fluid particles everywhere

        //count the number of new particles
        Vector3d min = *data.bmin;
        Vector3d max = *data.bmax;
        RealCuda p_distance = data.particleRadius * 2;
        //I need to know the width I have
        Vector3d width = (max)-(min);
        Vector3d max_count_width = width / p_distance;

        //the 0.8 is because the particles will be interleaved and slightly compresses to be closer to a fluid at rest
        max_count_width.y=(target_height - global_height) / (p_distance);
        max_count_width.toFloor();
        //idk why but with that computation it's missing one particle so I'll add it
        max_count_width += 1;


        int count_new_particles = max_count_width.x*max_count_width.y*max_count_width.z;

#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
        std::cout << "num particles to be added: " << count_new_particles << std::endl;
#endif

        if(count_new_particles==0){
            throw("asked creating 0 particles, I can just skip it and return but for now it'll stop the program because it should not happen");
        }



        //if we need more than the max number of particles then we have to reallocate everything
        if ((particleSet->numParticles + count_new_particles) > particleSet->numParticlesMax) {
            change_fluid_max_particle_number(data,(particleSet->numParticles + count_new_particles)*1.5);
        }



        int numBlocks= (count_new_particles + BLOCKSIZE - 1) / BLOCKSIZE;
        data.destructor_activated = false;
        Vector3d border_range =width/3;
        border_range.y=0;

        std::cout<<"border_range: "<<border_range.x<<" "<<border_range.y<<" "<<border_range.z<<std::endl;

        int* count_created_particles = SVS_CU::get()->count_created_particles;
        *count_created_particles = 0;
        //and place the particles in the simulation
        place_additional_particles_right_above_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, column_max_height,
                                                                                     count_new_particles, border_range,
                                                                                     (border_range.squaredNorm()>0) ? count_created_particles : NULL);


        gpuErrchk(cudaDeviceSynchronize());
        data.destructor_activated = true;


        //and now you can update the number of particles
        int added_particles=((border_range.squaredNorm()>0)? (*count_created_particles) :count_new_particles);
        int new_num_particles = particleSet->numParticles + added_particles;
        particleSet->updateActiveParticleNumber(new_num_particles);
#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
        std::cout << "new number of particles: " << particleSet->numParticles << "with num added particles: "<<added_particles<< std::endl;
#endif
    }

#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
    std::cout << "end fluid level control" << std::endl;
#endif

}




Vector3d get_simulation_center_cuda(SPH::DFSPHCData& data){
    //get the min and max
    get_min_max_pos_kernel << <1, 1 >> > (data.boundaries_data->gpu_ptr, data.bmin, data.bmax, data.particleRadius);
    gpuErrchk(cudaDeviceSynchronize());

    //std::cout<<"get_simulation_center_cuda min max: "<<
    //           data.bmin->x<<"  "<<data.bmin->z<<"  "<<data.bmax->x<<"  "<<data.bmax->z<<std::endl;

    //and computethe center
    return ((*data.bmax)+(*data.bmin))/2;
}

__global__ void get_min_max_pos_naive_kernel(SPH::UnifiedParticleSet* particleSet, int* mutex, Vector3d* min_o, Vector3d *max_o) {

	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= particleSet->numParticles) { return; }

	if (index == 0) {
		*min_o = particleSet->pos[0];
		*max_o = particleSet->pos[0];
	}

	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ Vector3d cache_min[BLOCKSIZE];
	__shared__ Vector3d cache_max[BLOCKSIZE];


	Vector3d temp_min = Vector3d(0, 0, 0);
	Vector3d temp_max = Vector3d(0, 0, 0);
	while (index + offset < particleSet->numParticles) {
		Vector3d pos = particleSet->pos[index + offset];

		temp_min.toMin(pos);
		temp_max.toMax(pos);

		offset += stride;
	}

	cache_min[threadIdx.x] = temp_min;
	cache_max[threadIdx.x] = temp_max;

	__syncthreads();



	// reduction
	// TODO you cna optimize that with this link https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	unsigned int i = BLOCKSIZE/2;
	while (i != 0) {
		if (threadIdx.x < i) {
			cache_min[threadIdx.x].toMin(cache_min[threadIdx.x + i]);
			cache_max[threadIdx.x].toMax(cache_max[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		while (atomicCAS(mutex, 0, 1) != 0);  //lock

		min_o->toMin(cache_min[0]);
		max_o->toMax(cache_max[0]);
		atomicExch(mutex, 0);  //unlock
	}
}

void get_UnifiedParticleSet_min_max_naive_cuda(SPH::UnifiedParticleSet& particleSet, Vector3d& min, Vector3d& max){
	Vector3d* min_cuda;
	Vector3d* max_cuda;

	
	cudaMallocManaged(&(min_cuda), sizeof(Vector3d));
	cudaMallocManaged(&(max_cuda), sizeof(Vector3d));
	
	//manual

	
	{
		int *d_mutex;
		cudaMalloc((void**)&d_mutex, sizeof(int));
		cudaMemset(d_mutex, 0, sizeof(float));
		
		get_min_max_pos_naive_kernel << <BLOCKSIZE, BLOCKSIZE >> > (particleSet.gpu_ptr, d_mutex, min_cuda, max_cuda);
		gpuErrchk(cudaDeviceSynchronize());
		
		
		cudaFree(d_mutex);
	}

	min = *min_cuda;
	max = *max_cuda;

	cudaFree(min_cuda);
	cudaFree(max_cuda);
}




__global__ void compute_dynamic_body_particle_mass_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleSet->numParticles) { return; }

    Real delta = 0;//data.W_zero;

    RealCuda radius_sq = data.m_kernel_precomp.getRadius();
    Vector3d pos = particleSet->pos[i];
    Vector3d pos_cell = (pos / radius_sq) + data.gridOffset; //on that line the radius is not yet squared
    pos_cell.toFloor();
    int x = pos_cell.x;
    int y = pos_cell.y;
    int z = pos_cell.z;
    radius_sq *= radius_sq;


    //since this version use the std index to be able to iterate on 3 successive cells
    //I can do the -1 at the start on x.
    //one thing: it x=0 then we can only iterate 2 cells at a time
    unsigned int successive_cells_count = (x > 0) ? 3 : 2;
    x = (x > 0) ? x - 1 : x;


    const SPH::UnifiedParticleSet& body = *particleSet;
    for (int k = -1; k < 2; ++k) {
        for (int m = -1; m < 2; ++m) {
            unsigned int cur_cell_id = COMPUTE_CELL_INDEX(x, y + k, z + m);
            unsigned int end = body.neighborsDataSet->cell_start_end[cur_cell_id + successive_cells_count];
            for (unsigned int cur_particle = body.neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {
                unsigned int j = body.neighborsDataSet->p_id_sorted[cur_particle];
                if ((pos - body.pos[j]).squaredNorm() < radius_sq) {
                    if (i != j) { delta += data.W(pos - body.pos[j]); }
                }
            }
        }
    }


    const Real volume = 1.0 / delta;
    particleSet->mass[i] = particleSet->density0 * volume;
}

void compute_UnifiedParticleSet_particles_mass_cuda(SPH::DFSPHCData& data, SPH::UnifiedParticleSet& container){
    int numBlocks = (container.numParticles + BLOCKSIZE - 1) / BLOCKSIZE;

    container.initNeighborsSearchData(data, false);


    data.destructor_activated = false;
    compute_dynamic_body_particle_mass_kernel << <numBlocks, BLOCKSIZE >> > (data, container.gpu_ptr);
    gpuErrchk(cudaDeviceSynchronize());
    data.destructor_activated = true;
}

void cuda_neighborsSearch(SPH::DFSPHCData& data) {

    //std::chrono::steady_clock::time_point begin_global = std::chrono::steady_clock::now();
    static unsigned int time_count = 0;
    float time_global;
    static float time_avg_global = 0;
    time_count++;

    /*
    if (time_count<5) {
        cuda_shuffleData(data.fluid_data[0]);
        std::cout << "randomizing particle order" << std::endl;
    }
    //*/

    bool need_sort = ((time_count%15)==0);

    if (need_sort){
        //std::cout<<"doing full neighbor search"<<std::endl;
    }

    bool old_fluid_aggregated=data.is_fluid_aggregated;
    cudaError_t cudaStatus;
    if (true){
        if (need_sort&&data.is_fluid_aggregated){
            data.is_fluid_aggregated=false;
        }


        //*
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        //*/

        //first let's generate the cell start end for the dynamic bodies
#ifdef GROUP_DYNAMIC_BODIES_NEIGHBORS_SEARCH
        cuda_initNeighborsSearchDataSetGroupedDynamicBodies(data);
#else
        for (int i = 0; i < data.numDynamicBodies; ++i) {
            SPH::UnifiedParticleSet& body = data.vector_dynamic_bodies_data[i];
            body.initNeighborsSearchData(data.m_kernel_precomp.getRadius(), false);
        }
#endif
        std::chrono::steady_clock::time_point middle = std::chrono::steady_clock::now();

        //no need to ever do it forthe boundaries since they don't ever move

        //now update the cell start end of the fluid particles
        if ((!data.is_fluid_aggregated)||data.numDynamicBodies<1){

            //since it the init iter I'll sort both even if it's the boundaries
            static int step_count = 0;
            step_count++;

            data.fluid_data->initNeighborsSearchData(data, need_sort);


            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "before neighbors search: %d\n", (int)cudaStatus);
                exit(1598);
            }


        }

        /*

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        float time0;
        float time1;
        static float time_avg = 0;
        time0 = std::chrono::duration_cast<std::chrono::nanoseconds> (middle - begin).count() / 1000000.0f;
        time1 = std::chrono::duration_cast<std::chrono::nanoseconds> (end - middle).count() / 1000000.0f;

        time_avg += time0+time1;
        printf("Time to generate cell start end: %f ms (%f,%f)   avg: %f ms \n", time0+time1,time0,time1, time_avg / time_count);

        if (time_count > 150) {
            time_avg = 0;
        }
        //*/


    }
    //and we can now do the actual search of the neaighbor for eahc fluid particle
    if (true)
    {
        //*
        float time;
        static float time_avg = 0;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        //*/

        //cuda way
        int numBlocks = (data.fluid_data[0].numParticles + BLOCKSIZE - 1) / BLOCKSIZE;

        //*
        DFSPH_neighborsSearch_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data_cuda);

		//*
        if (data.boundaries_data->has_factor_computation) {
            //DFSPH_neighborsSearch_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data_cuda);
        }
		//*/

        //*/
        /*
        //this test show that even just computing the neighbors for the fluid particle
        //with a basic method take more time than building the whole structure
        DFSPH_neighborsSearchBasic_kernel << <numBlocks, BLOCKSIZE >> > (data.numFluidParticles,
            data.m_kernel_precomp.getRadius(),
            data.fluid_data_cuda,
            data.boundaries_data_cuda,
            data.vector_dynamic_bodies_data_cuda, data.numDynamicBodies);
        //*/

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cuda neighbors search failed: " << (int)cudaStatus << std::endl;
            exit(1598);
        }

        /*
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() / 1000000.0f;

        time_avg += time;
        printf("Time to generate neighbors buffers: %f ms   avg: %f ms \n", time, time_avg / time_count);

        if (time_count > 150) {
            time_avg = 0;
            time_count = 0;
        }
        //*/



        /*
        {
            //a simple check to know the max nbr of neighbors
            static int absolute_max = 0;
            int max = 0;

            static int absolute_max_d[3] = { 0 };
            int max_d[3] = { 0 };



            for (int j = 0; j < data.fluid_data->getNumberOfNeighbourgs(j); j++)
            {
                //check the global value
                int count_neighbors = 0;
                for (int k = 0; k < 3; ++k) {
                    count_neighbors += data.fluid_data->getNumberOfNeighbourgs(j, k);
                }
                if (count_neighbors > max)max = count_neighbors;

                //chekc the max for each category
                for (unsigned int k = 0; k < 3; ++k) {
                    if ((int)data.fluid_data->getNumberOfNeighbourgs(j,k) > max_d[k])max_d[k] = data.fluid_data->getNumberOfNeighbourgs(j,k);
                }

            }
            if (max>absolute_max)absolute_max = max;
            for (unsigned int k = 0; k < 3; ++k) {
                if (max_d[k]>absolute_max_d[k])absolute_max_d[k] = max_d[k];
            }
            printf("max nbr of neighbors %d  (%d) \n", absolute_max, max);
            printf("max nbr of neighbors %d  (%d)      absolute max  fluid // boundaries // bodies   %d // %d // %d\n",
            absolute_max, max, absolute_max_d[0], absolute_max_d[1], absolute_max_d[2]);
        }


        //*/
		/*
		{
			//another test ot be sure the contruction of the boundries neighbors works orrectly
			if (data.boundaries_data->has_factor_computation) {
				//a simple check to know the  nbr of neighbors of the first boundries particle

				int nb_neighbors[3] = { 0 };
				
				for (int k = 0; k < 3; ++k) {
					nb_neighbors[k] = data.boundaries_data->getNumberOfNeighbourgs(0, k);
				}

				printf(" nbr of neighbors %d     fluid // boundaries // bodies   %d // %d // %d\n",
					nb_neighbors[0] + nb_neighbors[1] + nb_neighbors[2], nb_neighbors[0], nb_neighbors[1], nb_neighbors[2]);
			}
		}

		//*/
    }

    //reactive the aggragation if we desactivated it because a sort was required
    if (need_sort&&old_fluid_aggregated){
        data.is_fluid_aggregated=true;
    }


	
    /*
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    time_global = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin_global).count() / 1000000.0f;

    time_avg_global += time_global;
    printf("time taken by the neighbor function: %f ms   avg: %f ms \n", time_global, time_avg_global / time_count);
    //*/
}



void cuda_initNeighborsSearchDataSet(SPH::UnifiedParticleSet& particleSet, SPH::NeighborsSearchDataSet& dataSet,
                                     SPH::DFSPHCData& data, bool sortBuffers){



    //com the id
    cuda_neighborsSearchInternal_sortParticlesId(particleSet.pos, data.getKernelRadius(), data.gridOffset, dataSet.numParticles,
                                                 &dataSet.d_temp_storage_pair_sort, dataSet.temp_storage_bytes_pair_sort, dataSet.cell_id, dataSet.cell_id_sorted,
                                                 dataSet.p_id, dataSet.p_id_sorted);



    //since it the init iter I'll sort both even if it's the boundaries
    if (sortBuffers) {
        cuda_sortData(particleSet, dataSet.p_id_sorted);
    }



    //and now I cna compute the start and end of each cell :)
    cuda_neighborsSearchInternal_computeCellStartEnd(dataSet.numParticles, dataSet.cell_id_sorted, dataSet.hist,
                                                     &dataSet.d_temp_storage_cumul_hist, dataSet.temp_storage_bytes_cumul_hist, dataSet.cell_start_end);




}


__global__ void DFSPH_fill_aggregated_pos_buffer_kernel(SPH::DFSPHCData data, unsigned int num_particles) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) { return; }

    if (data.is_fluid_aggregated){
        if (i<data.fluid_data_cuda->numParticles){

            //writte de pos
            data.posBufferGroupedDynamicBodies[i]=data.fluid_data_cuda->pos[i];

            return;
        }
    }

    //find the current dynamic body
    int count_particles_previous_bodies=(data.is_fluid_aggregated)?data.fluid_data_cuda->numParticles:0;
    int body_id=0;
    while((count_particles_previous_bodies+data.vector_dynamic_bodies_data_cuda[body_id].numParticles)<i ){
        count_particles_previous_bodies+=data.vector_dynamic_bodies_data_cuda[body_id].numParticles;
        body_id++;
    }

    //writte de pos
    data.posBufferGroupedDynamicBodies[i]=data.vector_dynamic_bodies_data_cuda[body_id].pos[i-count_particles_previous_bodies];
}

void cuda_initNeighborsSearchDataSetGroupedDynamicBodies(SPH::DFSPHCData& data){
    if (data.numDynamicBodies<1){
        return;
    }

    SPH::NeighborsSearchDataSet& dataSet=*(data.neighborsDataSetGroupedDynamicBodies);


    //before anything I need to update the number of active particles
    int numParticles=(data.is_fluid_aggregated)?data.fluid_data[0].numParticles:0;
    for(int i=0;i<data.numDynamicBodies;++i){
        numParticles+=data.vector_dynamic_bodies_data[i].numParticles;
    }

    if (dataSet.numParticles!=numParticles){
        if(numParticles<=(int)dataSet.numParticlesMax){
            dataSet.updateActiveParticleNumber(numParticles);
        }else{
            std::ostringstream oss;
            oss<<"TODO::I need to add particles to the grouped data struct when the number of particle goes above the max"<<
                 " current max: "<<dataSet.numParticlesMax<<"  number of particles: "<<numParticles<<std::endl;
            throw(oss.str());
        }
    }

    // now fill itr
    int numBlocks = (dataSet.numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
    DFSPH_fill_aggregated_pos_buffer_kernel<< <numBlocks, BLOCKSIZE >> > (data, dataSet.numParticles);
    gpuErrchk(cudaDeviceSynchronize());

    //and now we can do the neighbor search
    //com the id
    cuda_neighborsSearchInternal_sortParticlesId(data.posBufferGroupedDynamicBodies, data.getKernelRadius(), data.gridOffset, dataSet.numParticles,
                                                 &dataSet.d_temp_storage_pair_sort, dataSet.temp_storage_bytes_pair_sort, dataSet.cell_id, dataSet.cell_id_sorted,
                                                 dataSet.p_id, dataSet.p_id_sorted);



    //and now I cna compute the start and end of each cell :)
    cuda_neighborsSearchInternal_computeCellStartEnd(dataSet.numParticles, dataSet.cell_id_sorted, dataSet.hist,
                                                     &dataSet.d_temp_storage_cumul_hist, dataSet.temp_storage_bytes_cumul_hist, dataSet.cell_start_end);



}


void cuda_renderFluid(SPH::DFSPHCData& data) {
    cuda_opengl_renderParticleSet(*data.fluid_data->renderingData, data.fluid_data[0].numParticles);
}



void cuda_renderBoundaries(SPH::DFSPHCData& data, bool renderWalls) {
    if (renderWalls) {
        cuda_opengl_renderParticleSet(*(data.boundaries_data->renderingData), data.boundaries_data->numParticles);
    }

    for (int i = 0; i < data.numDynamicBodies; ++i) {
        SPH::UnifiedParticleSet& body= data.vector_dynamic_bodies_data[i];
        cuda_opengl_renderParticleSet(*body.renderingData, body.numParticles);
    }
}

/*
THE NEXT FUNCTIONS ARE FOR THE RENDERING
*/


void cuda_opengl_initParticleRendering(ParticleSetRenderingData& renderingData, unsigned int numParticles,
                                       Vector3d** pos, Vector3d** vel) {
    glGenVertexArrays(1, &renderingData.vao); // Créer le VAO
    glBindVertexArray(renderingData.vao); // Lier le VAO pour l'utiliser


    glGenBuffers(1, &renderingData.pos_buffer);
    // selectionne le buffer pour l'initialiser
    glBindBuffer(GL_ARRAY_BUFFER, renderingData.pos_buffer);
    // dimensionne le buffer actif sur array_buffer, l'alloue et l'initialise avec les positions des sommets de l'objet
    glBufferData(GL_ARRAY_BUFFER,
                 /* length */	numParticles * sizeof(Vector3d),
                 /* data */      NULL,
                 /* usage */     GL_DYNAMIC_DRAW);
    //set it to the attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FORMAT, GL_FALSE, 0, 0);

    glGenBuffers(1, &renderingData.vel_buffer);
    // selectionne le buffer pour l'initialiser
    glBindBuffer(GL_ARRAY_BUFFER, renderingData.vel_buffer);
    // dimensionne le buffer actif sur array_buffer, l'alloue et l'initialise avec les positions des sommets de l'objet
    glBufferData(GL_ARRAY_BUFFER,
                 /* length */	numParticles * sizeof(Vector3d),
                 /* data */      NULL,
                 /* usage */     GL_DYNAMIC_DRAW);
    //set it to the attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FORMAT, GL_FALSE, 0, 0);

    // nettoyage
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Registration with CUDA.
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&renderingData.pos, renderingData.pos_buffer, cudaGraphicsRegisterFlagsNone));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&renderingData.vel, renderingData.vel_buffer, cudaGraphicsRegisterFlagsNone));

    //link the pos and vel buffer to cuda
    gpuErrchk(cudaGraphicsMapResources(1, &renderingData.pos, 0));
    gpuErrchk(cudaGraphicsMapResources(1, &renderingData.vel, 0));

    //set the openglbuffer for direct use in cuda
    Vector3d* vboPtr = NULL;
    size_t size = 0;

    // pos
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &size, renderingData.pos));//get cuda ptr
    *pos = vboPtr;

    // vel
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &size, renderingData.vel));//get cuda ptr
    *vel = vboPtr;

}

void cuda_opengl_releaseParticleRendering(ParticleSetRenderingData& renderingData) {
    //unlink the pos and vel buffer from cuda
    gpuErrchk(cudaGraphicsUnmapResources(1, &(renderingData.pos), 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &(renderingData.vel), 0));

    //delete the opengl buffers
    glDeleteBuffers(1, &renderingData.vel_buffer);
    glDeleteBuffers(1, &renderingData.pos_buffer);
    glDeleteVertexArrays(1, &renderingData.vao);
}

void cuda_opengl_renderParticleSet(ParticleSetRenderingData& renderingData, unsigned int numParticles) {

    //unlink the pos and vel buffer from cuda
    gpuErrchk(cudaGraphicsUnmapResources(1, &(renderingData.pos), 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &(renderingData.vel), 0));

    //Actual opengl rendering
    // link the vao
    glBindVertexArray(renderingData.vao);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //show it
    glDrawArrays(GL_POINTS, 0, numParticles);

    // unlink the vao
    glBindVertexArray(0);

    //link the pos and vel buffer to cuda
    gpuErrchk(cudaGraphicsMapResources(1, &renderingData.pos, 0));
    gpuErrchk(cudaGraphicsMapResources(1, &renderingData.vel, 0));

}






/*
THE NEXT FUNCTIONS ARE FOR THE MEMORY ALLOCATION
*/

void allocate_DFSPHCData_base_cuda(SPH::DFSPHCData& data) {
    if (data.damp_planes == NULL) {
        cudaMallocManaged(&(data.damp_planes), sizeof(Vector3d) * 10);
    }
    if (data.cancel_wave_planes == NULL) {
        cudaMallocManaged(&(data.cancel_wave_planes), sizeof(Vector3i) * 2);
    }

    if (data.bmin == NULL) {

        cudaMallocManaged(&(data.bmin), sizeof(Vector3d));
        cudaMallocManaged(&(data.bmax), sizeof(Vector3d));
    }

    //alloc static variables
    SVS_CU::get();
}

void free_DFSPHCData_base_cuda(SPH::DFSPHCData& data) {
    CUDA_FREE_PTR(data.damp_planes);
    CUDA_FREE_PTR(data.cancel_wave_planes);


    CUDA_FREE_PTR(data.bmin);
    CUDA_FREE_PTR(data.bmax);

    //free static variables
    SVS_CU::get(true);
}



void allocate_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& container) {

    //cudaMalloc(&(container.pos), container.numParticles * sizeof(Vector3d)); //use opengl buffer with cuda interop
    //cudaMalloc(&(container.vel), container.numParticles * sizeof(Vector3d)); //use opengl buffer with cuda interop
    cudaMalloc(&(container.mass), container.numParticlesMax * sizeof(RealCuda));


    if (container.has_factor_computation) {
        //*
        cudaMallocManaged(&(container.numberOfNeighbourgs), container.numParticlesMax * 3 * sizeof(int));
        cudaMallocManaged(&(container.neighbourgs), container.numParticlesMax * MAX_NEIGHBOURS * sizeof(int));

        cudaMalloc(&(container.density), container.numParticlesMax * sizeof(RealCuda));
        cudaMalloc(&(container.factor), container.numParticlesMax * sizeof(RealCuda));
        cudaMalloc(&(container.densityAdv), container.numParticlesMax * sizeof(RealCuda));

        cudaMalloc(&(container.kappa), container.numParticlesMax * sizeof(RealCuda));
        cudaMalloc(&(container.kappaV), container.numParticlesMax * sizeof(RealCuda));
        
		if (container.velocity_impacted_by_fluid_solver) {
            cudaMalloc(&(container.acc), container.numParticlesMax * sizeof(Vector3d));

            //I need the allocate the memory cub need to compute the reduction
            //I need the avg pointer because cub require it (but i'll clear after the cub call)
            RealCuda* avg_density_err = SVS_CU::get()->avg_density_err;

            container.d_temp_storage=NULL;
            container.temp_storage_bytes=0;
            cub::DeviceReduce::Sum(container.d_temp_storage, container.temp_storage_bytes,
                                   container.densityAdv, avg_density_err, container.numParticlesMax);
            // Allocate temporary storage
            cudaMalloc(&(container.d_temp_storage), container.temp_storage_bytes);

        }
        //*/

    }

    if (container.is_dynamic_object) {
        cudaMalloc(&(container.pos0), container.numParticlesMax * sizeof(Vector3d));
        cudaMalloc(&(container.F), container.numParticlesMax * sizeof(Vector3d));
    }

    gpuErrchk(cudaDeviceSynchronize());
}

void release_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& container) {

    CUDA_FREE_PTR(container.mass);

    if (container.has_factor_computation) {
        //*
        CUDA_FREE_PTR(container.numberOfNeighbourgs);
        CUDA_FREE_PTR(container.neighbourgs);

        CUDA_FREE_PTR(container.density);
        CUDA_FREE_PTR(container.factor);
        CUDA_FREE_PTR(container.densityAdv);

        CUDA_FREE_PTR(container.kappa);
        CUDA_FREE_PTR(container.kappaV);
        if (container.velocity_impacted_by_fluid_solver) {
            CUDA_FREE_PTR(container.acc);

            CUDA_FREE_PTR(container.d_temp_storage);
            container.temp_storage_bytes = 0;
        }
        //*/

    }

    if (container.is_dynamic_object) {
        CUDA_FREE_PTR(container.F);
    }

    //delete the cpu buffers if there are some
    if (container.is_dynamic_object) {
        FREE_PTR(container.F_cpu);
    }

}


void load_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& container, Vector3d* pos, Vector3d* vel, RealCuda* mass) {
    gpuErrchk(cudaMemcpy(container.pos, pos, container.numParticles * sizeof(Vector3d), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(container.vel, vel, container.numParticles * sizeof(Vector3d), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(container.mass, mass, container.numParticles * sizeof(RealCuda), cudaMemcpyHostToDevice));

    if (container.is_dynamic_object) {
        int numBlocks = (container.numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
        gpuErrchk(cudaMemcpy(container.pos0, pos, container.numParticles * sizeof(Vector3d), cudaMemcpyHostToDevice));
        DFSPH_setVector3dBufferToZero_kernel << <numBlocks, BLOCKSIZE >> > (container.F, container.numParticles);
    }

    if (container.has_factor_computation) {

        gpuErrchk(cudaMemset(container.kappa, 0, container.numParticles * sizeof(RealCuda)));
        gpuErrchk(cudaMemset(container.kappaV, 0, container.numParticles * sizeof(RealCuda)));
        if (container.velocity_impacted_by_fluid_solver) {
        }
    }

}

void read_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& container, Vector3d* pos, Vector3d* vel, RealCuda* mass, Vector3d* pos0) {
    if (pos != NULL) {
        gpuErrchk(cudaMemcpy(pos, container.pos, container.numParticles * sizeof(Vector3d), cudaMemcpyDeviceToHost));
    }

    if (vel != NULL) {
        gpuErrchk(cudaMemcpy(vel, container.vel, container.numParticles * sizeof(Vector3d), cudaMemcpyDeviceToHost));
    }

    if (mass != NULL) {
        gpuErrchk(cudaMemcpy(mass, container.mass,  container.numParticles * sizeof(RealCuda), cudaMemcpyDeviceToHost));
    }

    if (container.is_dynamic_object&&pos0 != NULL) {
        gpuErrchk(cudaMemcpy(pos0, container.pos0, container.numParticles * sizeof(Vector3d), cudaMemcpyDeviceToHost));
    }
}

void read_rigid_body_force_cuda(SPH::UnifiedParticleSet& container) {
    if (container.is_dynamic_object) {
        if (container.F_cpu == NULL) {
            container.F_cpu = new Vector3d[container.numParticles];
        }

        gpuErrchk(cudaMemcpy(container.F_cpu, container.F, container.numParticles * sizeof(Vector3d), cudaMemcpyDeviceToHost));
    }
}


__global__ void compute_fluid_impact_on_dynamic_body_kernel(SPH::UnifiedParticleSet* container, Vector3d rb_position,
                                                            Vector3d* force, Vector3d* moment, Vector3d reduction_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= container->numParticles) { return; }

    Vector3d F,M;

    F=container->F[i];
    F*=reduction_factor;
    M=(container->pos[i]-rb_position).cross(F);

    atomicAdd(&(force->x),F.x);
    atomicAdd(&(force->y),F.y);
    atomicAdd(&(force->z),F.z);
    atomicAdd(&(moment->x),M.x);
    atomicAdd(&(moment->y),M.y);
    atomicAdd(&(moment->z),M.z);
}

void compute_fluid_impact_on_dynamic_body_cuda(SPH::UnifiedParticleSet& container, Vector3d& force, Vector3d& moment,
                                               const Vector3d& reduction_factor){
    Vector3d* force_cuda = SVS_CU::get()->force_cuda;
    Vector3d* moment_cuda = SVS_CU::get()->moment_cuda;
    *force_cuda=Vector3d(0,0,0);
    *moment_cuda=Vector3d(0,0,0);


    int numBlocks = (container.numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
    compute_fluid_impact_on_dynamic_body_kernel << <numBlocks, BLOCKSIZE >> > (container.gpu_ptr,
                                                                               container.rigidBody_cpu->position, force_cuda,
                                                                               moment_cuda, reduction_factor);
    gpuErrchk(cudaDeviceSynchronize());

    force=*force_cuda;
    moment=*moment_cuda;
}

__global__ void compute_fluid_boyancy_on_dynamic_body_kernel(SPH::UnifiedParticleSet* container, Vector3d* force, Vector3d* pt_appli) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= container->numParticles) { return; }



    //I use the abs just in case for some reason the vertical force is negative ...
    //By this I mena that the y component also contains the y component of the drag. but there
    //is no way to extract the actual boyancy, soand approximation will have to do
    RealCuda boyancy=container->F[i].y;
    RealCuda boyancy_abs=abs(boyancy);
    if(boyancy_abs>0){
        Vector3d pt=container->pos[i]*boyancy_abs;

        //in the x componant I'll store the total abs
        atomicAdd(&(force->x),boyancy_abs);
        atomicAdd(&(force->y),boyancy);
        atomicAdd(&(pt_appli->x),pt.x);
        atomicAdd(&(pt_appli->y),pt.y);
        atomicAdd(&(pt_appli->z),pt.z);
    }
}

void compute_fluid_Boyancy_on_dynamic_body_cuda(SPH::UnifiedParticleSet& container, Vector3d& force, Vector3d& pt_appli){
    Vector3d* force_cuda = SVS_CU::get()->force_cuda;
    Vector3d* pt_cuda = SVS_CU::get()->pt_cuda;
    *force_cuda=Vector3d(0,0,0);
    *pt_cuda=Vector3d(0,0,0);

    int numBlocks = (container.numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
    compute_fluid_boyancy_on_dynamic_body_kernel << <numBlocks, BLOCKSIZE >> > (container.gpu_ptr, force_cuda, pt_cuda);
    gpuErrchk(cudaDeviceSynchronize());

    force=*force_cuda;
    //if the sum of the force is non zero
    if(abs(force.y)>0){
        pt_appli=*pt_cuda;

        //now compute the avg to get the actual point
        pt_appli= pt_appli/force.x;
        //and clear the x component
        force.x=0;
    }else{
        force=Vector3d(0,0,0);
        pt_appli=Vector3d(0,0,0);
    }
}

void allocate_and_copy_UnifiedParticleSet_vector_cuda(SPH::UnifiedParticleSet** out_vector, SPH::UnifiedParticleSet* in_vector, int numSets) {

    gpuErrchk(cudaMalloc(out_vector, numSets * sizeof(SPH::UnifiedParticleSet)));

    //now set the gpu_ptr in eahc object so that it points to the right place
    for (int i = 0; i < numSets; ++i) {
        in_vector[i].gpu_ptr = *out_vector + i;
    }

    //before being able to fill the gpu array we need to make a copy of the data structure since
    //we will have to change the neighborsdataset from the cpu to the gpu
    //*
    SPH::UnifiedParticleSet* temp;
    temp = new SPH::UnifiedParticleSet[numSets];
    std::copy(in_vector, in_vector + numSets, temp);

    for (int i = 0; i < numSets; ++i) {
        SPH::UnifiedParticleSet& body = temp[i];

        //we need to toggle the flag that prevent the destructor from beeing called on release
        //since it's the cpu version that clear the memory buffers that are common to the two structures
        body.releaseDataOnDestruction = false;

        //the gpu unified particle set has a irect pointer to the gpu neughbor dataset
        body.neighborsDataSet=body.neighborsDataSet->gpu_ptr;
    }
    //*/

    gpuErrchk(cudaMemcpy(*out_vector, temp, numSets * sizeof(SPH::UnifiedParticleSet), cudaMemcpyHostToDevice));

    //Now I have to update the pointer of the cpu set so that it point to the gpu structure
    delete[] temp;

}



void allocate_grouped_neighbors_struct_cuda(SPH::DFSPHCData& data){
    std::cout<<"initialising aggregated structure"<<std::endl;
	
	if (data.numDynamicBodies < 1) {
		std::cout << "no dynamic bodies detected" << std::endl;
		
		//ok so I need the grouped buffer because I reuse it for external forces computation
		//note if I want to rmv that restriction I only have to use the one 
		//TODO apply that modification
		cudaMalloc(&(data.posBufferGroupedDynamicBodies), data.fluid_data->numParticlesMax * sizeof(Vector3d));
		
		return;
	}

    if (data.neighborsDataSetGroupedDynamicBodies!=NULL||data.posBufferGroupedDynamicBodies!=NULL){
        throw("allocate_grouped_neighbors_struct_cuda already allocated");
    }

    int numParticles=0;
    int numParticlesMax=0;
    if (data.is_fluid_aggregated){
        numParticles+=data.fluid_data->numParticles;
        numParticlesMax+=data.fluid_data->numParticlesMax;
    }

    for(int i=0;i<data.numDynamicBodies;++i){
        numParticles+= data.vector_dynamic_bodies_data[i].numParticles;
        numParticlesMax+=data.vector_dynamic_bodies_data[i].numParticlesMax;
    }

    //allocate the dataset
    data.neighborsDataSetGroupedDynamicBodies=new SPH::NeighborsSearchDataSet(numParticles,numParticlesMax);

    //read gpu ptr
    data.neighborsDataSetGroupedDynamicBodies_cuda=data.neighborsDataSetGroupedDynamicBodies->gpu_ptr;


    //now it's like the normal neighbor search excapt that we have to iterate on all the solid particles
    //instead of just one buffer
    //the easiest way is to build a new pos array that contains all the dynamic particles
    cudaMalloc(&(data.posBufferGroupedDynamicBodies), numParticlesMax * sizeof(Vector3d));

}


void free_grouped_neighbors_struct_cuda(SPH::DFSPHCData& data){
    FREE_PTR(data.neighborsDataSetGroupedDynamicBodies);

    CUDA_FREE_PTR(data.posBufferGroupedDynamicBodies);
}

void update_neighborsSearchBuffers_UnifiedParticleSet_vector_cuda(SPH::UnifiedParticleSet** out_vector, SPH::UnifiedParticleSet* in_vector, int numSets) {
    SPH::UnifiedParticleSet* temp;
    temp = new SPH::UnifiedParticleSet[numSets];

    gpuErrchk(cudaMemcpy(temp, *out_vector, numSets * sizeof(SPH::UnifiedParticleSet), cudaMemcpyDeviceToHost));

    for (int i = 0; i < numSets; ++i) {
        SPH::UnifiedParticleSet& body = temp[i];

        //we need to toggle the flag that prevent the destructor from beeing called on release
        //since it's the cpu version that clear the memory buffers that are common to the two structures
        body.releaseDataOnDestruction = false;

        //update the neighbor dataset to the cpu
        gpuErrchk(cudaMemcpy(body.neighborsDataSet, in_vector[i].neighborsDataSet,
                             sizeof(SPH::NeighborsSearchDataSet), cudaMemcpyHostToDevice));

    }

    gpuErrchk(cudaMemcpy(*out_vector, temp, numSets * sizeof(SPH::UnifiedParticleSet), cudaMemcpyHostToDevice));


    delete[] temp;
}



void release_UnifiedParticleSet_vector_cuda(SPH::UnifiedParticleSet** vector, int numSets) {
    //to be able to release the internal buffer I need firt to copy everything back to the cpu
    //then release the internal buffers
    //then release the UnifiedParticleSet
    //*
    SPH::UnifiedParticleSet* temp;
    temp = new SPH::UnifiedParticleSet[numSets];


    gpuErrchk(cudaMemcpy(temp, *vector, numSets * sizeof(SPH::UnifiedParticleSet), cudaMemcpyDeviceToHost));

    for (int i = 0; i < numSets; ++i) {
        cudaFree(temp[i].neighborsDataSet); temp[i].neighborsDataSet = NULL;
    }

    cudaFree(*vector); *vector = NULL;
}



template<class T> __global__ void cuda_setBufferToValue_kernel(T* buff, T value, unsigned int buff_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buff_size) { return; }

    buff[i] = value;
}

template<class T> __global__ void cuda_updateParticleCount_kernel(T* container, unsigned int numParticles) {
    //that kernel wil only ever use one thread so I sould noteven need that
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1) { return; }

    container->numParticles = numParticles;
}



template<class T> void update_active_particle_number_cuda(T& container) {
    //And now I need to update the particle count in the gpu structures
    //the easiest way is to use a kernel with just one thread used
    //the other way would be to copy the data back to the cpu then update the value before sending it back to the cpu
    cuda_updateParticleCount_kernel<T> << <1, 1 >> > (container.gpu_ptr, container.numParticles);

    gpuErrchk(cudaDeviceSynchronize());
}

void change_fluid_max_particle_number(SPH::DFSPHCData& data, int numParticlesMax){
    //update the fluid dataset
    data.fluid_data[0].changeMaxParticleNumber(numParticlesMax);

    //and update the aggregated neighbor search if it is used
#ifdef GROUP_DYNAMIC_BODIES_NEIGHBORS_SEARCH
        free_grouped_neighbors_struct_cuda(data);

        allocate_grouped_neighbors_struct_cuda(data);
#endif


}


void change_max_particle_number(SPH::UnifiedParticleSet& container, int numParticlesMax){
    //we need to copy the existing data so let's start in order.
    //first disactivate the destructor
    bool old_release_on_destruct=container.releaseDataOnDestruction;
    container.releaseDataOnDestruction=false;

    //the easy way is to create a dummy, backup the existing buffers inside
    //alloc the new buffers
    //do the copy
    //delete the temps storage

    SPH::UnifiedParticleSet dummy;
    dummy=container;
    //remove the pointer on the neighbor search data since it is handled separately
    dummy.neighborsDataSet=NULL;
    dummy.gpu_ptr=NULL;


    //*
    //now change the number of particle
    container.numParticlesMax=numParticlesMax;

    //the rendering data

    //allocate new
    container.renderingData=new ParticleSetRenderingData();
    cuda_opengl_initParticleRendering(*container.renderingData,numParticlesMax,&container.pos,&container.vel);

    //now we need to copy the data
    gpuErrchk(cudaMemcpy(container.pos, dummy.pos, dummy.numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(container.vel, dummy.vel, dummy.numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

    //and the rest of the data
    allocate_UnifiedParticleSet_cuda(container);

    //and fill it with the old data for buffers that need to be kept
    gpuErrchk(cudaMemcpy(container.mass, dummy.mass, dummy.numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(container.kappa, dummy.kappa, dummy.numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(container.kappaV, dummy.kappaV, dummy.numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

    //*/
    //finaly we need to update the data that is stored on the gpu
    //hopefully ne need for more allocation I can just copy the new ptr addresses
    SPH::NeighborsSearchDataSet* back_up=container.neighborsDataSet;
    container.neighborsDataSet=container.neighborsDataSet->gpu_ptr;
    gpuErrchk(cudaMemcpy(container.gpu_ptr, &container, sizeof(SPH::UnifiedParticleSet), cudaMemcpyHostToDevice));
    container.neighborsDataSet=back_up;
    //and free the data from the dummy
    dummy.clear();


    container.releaseDataOnDestruction=old_release_on_destruct;
}


void change_max_particle_number(SPH::NeighborsSearchDataSet& dataSet, int numParticlesMax){
    if (!dataSet.internal_buffers_allocated){
        throw("only consider fully allocated dataset for now");
    }


    release_neighbors_search_data_set(dataSet,false,true,true);

    //and allocate it back
    dataSet.numParticlesMax=numParticlesMax;

    allocate_neighbors_search_data_set(dataSet,false,true,false);
}

void add_particles_cuda(SPH::UnifiedParticleSet& container, int num_additional_particles, const Vector3d* pos, const Vector3d* vel) {
    //can't use memeset for the mass so I have to make a kernel for the set
    int numBlocks = (num_additional_particles + BLOCKSIZE - 1) / BLOCKSIZE;
    cuda_setBufferToValue_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (container.mass,
                                                                          container.m_V*container.density0, container.numParticles+num_additional_particles);



    gpuErrchk(cudaMemcpy(container.pos + container.numParticles, pos, num_additional_particles * sizeof(Vector3d), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(container.vel + container.numParticles, vel, num_additional_particles * sizeof(Vector3d), cudaMemcpyHostToDevice));


    gpuErrchk(cudaMemset(container.kappa + container.numParticles, 0, num_additional_particles * sizeof(RealCuda)));
    gpuErrchk(cudaMemset(container.kappaV + container.numParticles, 0, num_additional_particles * sizeof(RealCuda)));

    //update the particle count
    container.updateActiveParticleNumber(container.numParticles + num_additional_particles);


    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "add_particles_cuda failed: " << (int)cudaStatus << std::endl;
        exit(1598);
    }


}

template<class T> void set_buffer_to_value(T* buff, T val, int size) {
    //can't use memeset for the mass so I have to make a kernel for the  set
    int numBlocks = (size + BLOCKSIZE - 1) / BLOCKSIZE;
    cuda_setBufferToValue_kernel<T> << <numBlocks, BLOCKSIZE >> > (buff, val, size);

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "set_buffer_to_value failed: " << (int)cudaStatus << std::endl;
        exit(1598);
    }
}


void allocate_precomputed_kernel_managed(SPH::PrecomputedCubicKernelPerso& kernel, bool minimize_managed) {

    if (minimize_managed) {
        cudaMalloc(&(kernel.m_W), kernel.m_resolution * sizeof(RealCuda));
        cudaMalloc(&(kernel.m_gradW), (kernel.m_resolution + 1) * sizeof(RealCuda));
    }
    else {
        fprintf(stderr, "trying to use managed buffers for the kernels\n");
        exit(1256);
        //cudaMallocManaged(&(kernel.m_W), kernel.m_resolution * sizeof(RealCuda));
        //cudaMallocManaged(&(kernel.m_gradW), (kernel.m_resolution + 1) * sizeof(RealCuda));
    }
}

void free_precomputed_kernel_managed(SPH::PrecomputedCubicKernelPerso& kernel) {
    CUDA_FREE_PTR(kernel.m_W);
    CUDA_FREE_PTR(kernel.m_gradW);
}


void init_precomputed_kernel_from_values(SPH::PrecomputedCubicKernelPerso& kernel, RealCuda* w, RealCuda* grad_W) {
    cudaError_t cudaStatus;
    //W
    cudaStatus = cudaMemcpy(kernel.m_W,
                            w,
                            kernel.m_resolution * sizeof(RealCuda),
                            cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "precomputed initialization of W from data failed: %d\n", (int)cudaStatus);
        exit(1598);
    }

    //grad W
    cudaStatus = cudaMemcpy(kernel.m_gradW,
                            grad_W,
                            (kernel.m_resolution + 1) * sizeof(RealCuda),
                            cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "precomputed initialization of grad W from data failed: %d\n", (int)cudaStatus);
        exit(1598);
    }

}


void allocate_neighbors_search_data_set(SPH::NeighborsSearchDataSet& dataSet, bool result_buffers_only, bool particle_related_only,
                                        bool allocate_gpu) {

    //result buffers
    cudaMallocManaged(&(dataSet.p_id_sorted), dataSet.numParticlesMax * sizeof(unsigned int));
    if (!particle_related_only){
        cudaMallocManaged(&(dataSet.cell_start_end), (CELL_COUNT + 1) * sizeof(unsigned int));
    }

    //allocate the mem for fluid particles
    if(!result_buffers_only){
        cudaMallocManaged(&(dataSet.cell_id), dataSet.numParticlesMax * sizeof(unsigned int));
        cudaMallocManaged(&(dataSet.cell_id_sorted), dataSet.numParticlesMax * sizeof(unsigned int));
        cudaMallocManaged(&(dataSet.local_id), dataSet.numParticlesMax * sizeof(unsigned int));
        cudaMallocManaged(&(dataSet.p_id), dataSet.numParticlesMax * sizeof(unsigned int));

        cudaMalloc(&(dataSet.intermediate_buffer_v3d), dataSet.numParticlesMax * sizeof(Vector3d));
        cudaMalloc(&(dataSet.intermediate_buffer_real), dataSet.numParticlesMax * sizeof(RealCuda));

        //reset the particle id
        int numBlocks = (dataSet.numParticlesMax + BLOCKSIZE - 1) / BLOCKSIZE;
        DFSPH_setBufferValueToItself_kernel << <numBlocks, BLOCKSIZE >> > (dataSet.p_id, dataSet.numParticlesMax);
        DFSPH_setBufferValueToItself_kernel << <numBlocks, BLOCKSIZE >> > (dataSet.p_id_sorted, dataSet.numParticlesMax);
        gpuErrchk(cudaDeviceSynchronize());


        //cub pair sort
        dataSet.temp_storage_bytes_pair_sort = 0;
        dataSet.d_temp_storage_pair_sort = NULL;
        cub::DeviceRadixSort::SortPairs(dataSet.d_temp_storage_pair_sort, dataSet.temp_storage_bytes_pair_sort,
                                        dataSet.cell_id, dataSet.cell_id_sorted, dataSet.p_id, dataSet.p_id_sorted, dataSet.numParticlesMax);
        gpuErrchk(cudaDeviceSynchronize());
        cudaMalloc(&(dataSet.d_temp_storage_pair_sort), dataSet.temp_storage_bytes_pair_sort);


        if (!particle_related_only){
            cudaMallocManaged(&(dataSet.hist), (CELL_COUNT + 1) * sizeof(unsigned int));

            //cub histogram
            dataSet.temp_storage_bytes_cumul_hist = 0;
            dataSet.d_temp_storage_cumul_hist = NULL;
            cub::DeviceScan::ExclusiveSum(dataSet.d_temp_storage_cumul_hist, dataSet.temp_storage_bytes_cumul_hist,
                                          dataSet.hist, dataSet.cell_start_end, (CELL_COUNT + 1));
            gpuErrchk(cudaDeviceSynchronize());
            cudaMalloc(&(dataSet.d_temp_storage_cumul_hist), dataSet.temp_storage_bytes_cumul_hist);
        }
    }


    /*
    std::cout << "neighbors struct num byte allocated cub (numParticlesMax pair_sort cumul_hist)" << dataSet.numParticlesMax << "  " <<
                 dataSet.temp_storage_bytes_pair_sort << "  " << dataSet.temp_storage_bytes_cumul_hist << std::endl;
    //*/


    dataSet.internal_buffers_allocated = true;

    if (allocate_gpu){
        //duplicate the neighbor dataset to the gpu
        gpuErrchk(cudaMalloc(&(dataSet.gpu_ptr), sizeof(SPH::NeighborsSearchDataSet)));
    }
    //and copy the gpu data if the buffer exists
    if (dataSet.gpu_ptr!=NULL){
        gpuErrchk(cudaMemcpy(dataSet.gpu_ptr, &dataSet,
                             sizeof(SPH::NeighborsSearchDataSet), cudaMemcpyHostToDevice));
    }

    gpuErrchk(cudaDeviceSynchronize());

}


void release_neighbors_search_data_set(SPH::NeighborsSearchDataSet& dataSet, bool keep_result_buffers, bool keep_grid_related,
                                       bool keep_gpu) {
    //allocatethe mme for fluid particles
    cudaFree(dataSet.cell_id); dataSet.cell_id = NULL;
    cudaFree(dataSet.local_id); dataSet.local_id = NULL;
    cudaFree(dataSet.p_id); dataSet.p_id = NULL;
    cudaFree(dataSet.cell_id_sorted); dataSet.cell_id_sorted = NULL;

    cudaFree(dataSet.d_temp_storage_pair_sort); dataSet.d_temp_storage_pair_sort = NULL;
    dataSet.temp_storage_bytes_pair_sort = 0;

    cudaFree(dataSet.intermediate_buffer_v3d); dataSet.intermediate_buffer_v3d = NULL;
    cudaFree(dataSet.intermediate_buffer_real); dataSet.intermediate_buffer_real = NULL;

    if(!keep_grid_related){
        cudaFree(dataSet.hist); dataSet.hist = NULL;
        cudaFree(dataSet.d_temp_storage_cumul_hist); dataSet.d_temp_storage_cumul_hist = NULL;
        dataSet.temp_storage_bytes_cumul_hist = 0;
    }

    dataSet.internal_buffers_allocated = false;

    if (!keep_result_buffers) {
        cudaFree(dataSet.p_id_sorted); dataSet.p_id_sorted = NULL;

        if(!keep_grid_related){
            cudaFree(dataSet.cell_start_end); dataSet.cell_start_end = NULL;
        }
    }

    if(!keep_gpu){
        CUDA_FREE_PTR(dataSet.gpu_ptr);
    }
}























/*
AFTER THIS ARE ONLY THE TEST FUNCTION TO HAVE CUDA WORKING ...
*/


inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}

inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}

inline __host__ __device__ float4 operator*(float4& a, RealCuda b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, 0);
}

inline __host__ __device__ float3 operator*(float3& a, RealCuda b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}



template<typename T>
__global__ void test_vector_type_kernel(T* v1, T* v2, RealCuda factor, int count_elem) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count_elem) { return; }

    v1[i]+=v2[i]*factor;
}

void compare_vector3_struct_speed(){
    RealCuda factor=0.001;
    int count_elem= 1000000;
    Vector3d* v1_v3d;
    Vector3d* v2_v3d;
    float3* v1_f3;
    float3* v2_f3;
    float4* v1_f4;
    float4* v2_f4;
    Vector3d* v1_v3d_2=new Vector3d[count_elem];
    Vector3d* v2_v3d_2=new Vector3d[count_elem];
    float3* v1_f3_2=new float3[count_elem];
    float3* v2_f3_2=new float3[count_elem];
    float4* v1_f4_2=new float4[count_elem];
    float4* v2_f4_2=new float4[count_elem];
    cudaMalloc(&(v1_v3d), count_elem * sizeof(Vector3d));
    cudaMalloc(&(v2_v3d), count_elem * sizeof(Vector3d));

    cudaMalloc(&(v1_f3), count_elem * sizeof(float3));
    cudaMalloc(&(v2_f3), count_elem * sizeof(float3));

    cudaMalloc(&(v1_f4), count_elem * sizeof(float4));
    cudaMalloc(&(v2_f4), count_elem * sizeof(float4));

    for (int i=0;i<count_elem;++i){
        v1_v3d_2[i]=i;
        v2_v3d_2[i]=i;
        v1_f3_2[i]=make_float3(i);
        v2_f3_2[i]=make_float3(i);
        v1_f4_2[i]=make_float4(i);
        v2_f4_2[i]=make_float4(i);
    }

    gpuErrchk(cudaMemcpy(v1_v3d, v1_v3d_2,count_elem * sizeof(Vector3d),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v2_v3d, v2_v3d_2,count_elem * sizeof(Vector3d),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v1_f3, v1_f3_2,count_elem * sizeof(float3),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v2_f3, v2_f3_2,count_elem * sizeof(float3),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v1_f4, v1_f4_2,count_elem * sizeof(float4),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v2_f4, v2_f4_2,count_elem * sizeof(float4),cudaMemcpyHostToDevice));

    int numBlocks = (count_elem + BLOCKSIZE - 1) / BLOCKSIZE;
    gpuErrchk(cudaDeviceSynchronize());

    float avg0=0;
    float avg1=0;
    float avg2=0;

    int iter=10;
    for (int i=0;i<iter;++i){
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        test_vector_type_kernel<float3> << <numBlocks, BLOCKSIZE >> > (v1_f3, v2_f3, factor, count_elem);
        gpuErrchk(cudaDeviceSynchronize());

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        test_vector_type_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (v1_v3d, v2_v3d, factor, count_elem);
        gpuErrchk(cudaDeviceSynchronize());

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        test_vector_type_kernel<float4> << <numBlocks, BLOCKSIZE >> > (v1_f4, v2_f4, factor, count_elem);
        gpuErrchk(cudaDeviceSynchronize());

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

        float time_0 = std::chrono::duration_cast<std::chrono::nanoseconds> (t1 - t0).count() / 1000000.0f;
        float time_1 = std::chrono::duration_cast<std::chrono::nanoseconds> (t2 - t1).count() / 1000000.0f;
        float time_2 = std::chrono::duration_cast<std::chrono::nanoseconds> (t3 - t2).count() / 1000000.0f;

        printf("comparison between vector data struct  (float3, Vector3d, float4): %f   %f   %f\n", time_0, time_1, time_2);

        if (iter>0){
            avg0+=time_0;
            avg1+=time_1;
            avg2+=time_2;
        }
    }
    iter--;

    printf("comparison between vector data struct Global (float3, Vector3d, float4): %f   %f   %f\n",
           avg0/iter, avg1/iter, avg2/iter);

}







cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
//*
__global__ void addKernel(Vector3d* vect)
{
    int i = threadIdx.x;
    vect[i].z = vect[i].x + vect[i].y;
}

__global__ void setVectkernel(Vector3d& vect)
{
    vect.x = 5;
    vect.y = 6;
    vect.z = 7;
}
//*/
int test_cuda()
{
    //DFSPHCData* data;
    std::cout << "start cuda test basic" << std::endl;

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    //*
    Vector3d* vect;
    cudaMallocManaged(&vect, arraySize * sizeof(Vector3d));
    for (int i = 0; i < arraySize; ++i) {
        vect[i].x = a[i];
        vect[i].y = b[i];
    }
    //*/*

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }


    printf("macro val: %d, %d, %d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
           c[0], c[1], c[2], c[3], c[4]);

    for (int i = 0; i < arraySize; ++i) {
        c[i] = 0;
    }


    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, arraySize >> > (vect);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    printf("with vects {1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
           (int)(vect[0].z), (int)(vect[1].z), (int)(vect[2].z), (int)(vect[3].z), (int)(vect[4].z));

    cudaFree(vect);



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    /*
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    return 1;
    }
    //*/

    printf("Finished test cuda\n");


    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
