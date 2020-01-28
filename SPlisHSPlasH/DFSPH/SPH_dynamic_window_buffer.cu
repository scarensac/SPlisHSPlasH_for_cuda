#include "SPH_dynamic_window_buffer.h"
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



//NOTE1:	seems that virtual function can't be used with managed allocation
//			so I'll use a template to have an equivalent solution
//			0 ==> plane, 
//this is a variant of the surface class to define an area by one of multiples planes
//the given normal must point otward the inside of the fluid
//NOTE can't use the stl on every cuda version so for now I'll do it with static arrays
template<int type>
class BufferFluidSurfaceBase
{
	int count_planes;
	Vector3d* o;
	Vector3d* n;
public:

	inline BufferFluidSurfaceBase() {
		count_planes = 0;
		cudaMallocManaged(&(o), sizeof(Vector3d) * 16);
		cudaMallocManaged(&(n), sizeof(Vector3d) * 16);
	}


	inline ~BufferFluidSurfaceBase() {
		CUDA_FREE_PTR(o);
		CUDA_FREE_PTR(n);
	}


	FUNCTION inline void addPlane(Vector3d o_i, Vector3d n_i) {
		o[count_planes]=o_i;
		n[count_planes]=n_i;
		count_planes++;
	}

	//to know if we are on the inside of each plane we can simply use the dot product*
	FUNCTION inline bool isInsideFluid(Vector3d p) {
		for (int i = 0; i < count_planes; ++i) {
			Vector3d v = p - o[i];
			if (v.dot(n[i]) < 0) {
				return false;
			}
		}
		return true;
	}

	FUNCTION inline RealCuda distanceToSurface(Vector3d p) {
		RealCuda dist = abs((p - o[0]).dot(n[0]));
		for (int i = 1; i < count_planes; ++i) {
			Vector3d v = p - o[i];
			RealCuda l = abs(v.dot(n[i]));
			dist = MIN_MACRO_CUDA(dist, l);
		}
		return dist;
	}

	FUNCTION inline RealCuda distanceToSurfaceSigned(Vector3d p) {
		int plane_id = 0;
		RealCuda dist = abs((p - o[0]).dot(n[0]));
		for (int i = 1; i < count_planes; ++i) {
			Vector3d v = p - o[i];
			RealCuda l = abs(v.dot(n[i]));
			if (l < dist) {
				dist = 0;
				plane_id = i;
			}
		}
		return (p - o[plane_id]).dot(n[plane_id]);
	}
};

using BufferFluidSurface = BufferFluidSurfaceBase<0>;


//this macro is juste so that the expression get optimized at the compilation 
//x_motion should be a bollean comming from a template configuration of the function where this macro is used
#define VECTOR_X_MOTION(pos,x_motion) ((x_motion)?pos.x:pos.z)

namespace DynamicWindowBuffer
{
	__global__ void init_buffer_kernel(Vector3d* buff, unsigned int size, Vector3d val) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= size) { return; }

		buff[i] = val;
	}
}


#define GAP_PLANE_POS 3


__global__ void DFSPH_init_buffer_velocity_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, 
													Vector3d* pos, Vector3d* vel, int numParticles) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numParticles) { return; }

	Vector3d pos_i = pos[i];

	//let's brute force it for now, I technicaly should be able to use the neighbor structure to accelerate it
#define num_neighbors 3
	Vector3d vel_neighbors[num_neighbors];
	RealCuda dist_neighbors[num_neighbors];

	for (int j = 0; j < num_neighbors; ++j) {
		dist_neighbors[j] = 1000000;
	}

	//we save the velocities and distance of the n closests
	for (int j = 0; j < particleSet->numParticles; ++j) {
		RealCuda cur_dist = (pos_i - particleSet->pos[j]).norm();
		

		int k = num_neighbors-1;
		while ((k > 0) && (cur_dist < dist_neighbors[k-1])) {
			dist_neighbors[k] = dist_neighbors[k - 1];
			vel_neighbors[k] = vel_neighbors[k - 1];
			k--;
		}
		if (cur_dist < dist_neighbors[k]) {
			dist_neighbors[k] = cur_dist;
			vel_neighbors[k] = particleSet->vel[j];
		}
	}

	//and now we can set the velocity to the averare of the closests
	RealCuda sum_dist = 0;
	Vector3d weighted_vel(0, 0, 0);
	for (int j = 0; j < num_neighbors; ++j) {
		sum_dist += dist_neighbors[j];
		weighted_vel += dist_neighbors[j]* vel_neighbors[j];
	}
	vel[i] = weighted_vel / sum_dist;

#undef num_neighbors
}

template<bool x_motion>
__global__ void DFSPH_reset_fluid_boundaries_remove_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, int* countRmv,
	RealCuda plane_pos_inf, RealCuda plane_pos_sup) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	//*

	if ((VECTOR_X_MOTION(particleSet->pos[i], x_motion) <= plane_pos_inf)|| (VECTOR_X_MOTION(particleSet->pos[i], x_motion) >= plane_pos_sup)) {
		atomicAdd(countRmv, 1);
		particleSet->neighborsDataSet->cell_id[i] = 25000000;
	}
	//*/

}

__global__ void DFSPH_reset_fluid_boundaries_add_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, SPH::UnifiedParticleSet* fluidBufferSet){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= fluidBufferSet->numParticles) { return; }

	particleSet->pos[particleSet->numParticles + i] = fluidBufferSet->pos[i];
	particleSet->vel[particleSet->numParticles + i] = fluidBufferSet->vel[i];
	particleSet->mass[particleSet->numParticles + i] = fluidBufferSet->mass[i];
	particleSet->color[particleSet->numParticles + i] = fluidBufferSet->color[i];

}

__device__ void atomicToMin(float* addr, float value)
{
	float old = *addr;
	if (old <= value) return;
	for (;;) {
		old = atomicExch(addr, value);
		if (old < value) {
			value = old;
		}
		else {
			return;
		}
	}
}

__device__ void atomicToMax(float* addr, float value)
{
	float old = *addr;
	if (old >= value) return;
	for (;;) {
		old = atomicExch(addr, value);
		if (old > value) {
			value = old;
		}
		else {
			return;
		}
	}
}

__global__ void DFSPH_compute_gap_length_fluid_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, RealCuda* gap) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	RealCuda gap_pos = (-2.0 + GAP_PLANE_POS * data.getKernelRadius());
	if (particleSet->pos[i].x >= gap_pos) {
		//fluid side
		RealCuda dist = abs(particleSet->pos[i].x - gap_pos);
		atomicToMin(gap, dist);
	}
}

__global__ void DFSPH_compute_gap_length_buffer_kernel(SPH::DFSPHCData data, Vector3d* pos, int numParticles, RealCuda* gap) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numParticles) { return; }
	
	RealCuda gap_pos = (-2.0 + GAP_PLANE_POS * data.getKernelRadius());
	RealCuda dist = abs(pos[i].x - gap_pos);
	atomicToMin(gap, dist);
	
}

__global__ void DFSPH_reduce_gap_length_kernel(SPH::DFSPHCData data, Vector3d* pos, int numParticles,
	RealCuda gap_length, RealCuda buffer_closest, RealCuda buffer_furthest) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numParticles) { return; }

	RealCuda gap_pos = (-2.0 + GAP_PLANE_POS * data.getKernelRadius());
	
	//do a linear displacment to distribute the particles
	RealCuda dist = abs(pos[i].x - gap_pos);
	RealCuda displacement = (buffer_furthest - dist) / (buffer_furthest - buffer_closest) * gap_length;
	pos[i].x += displacement;
	
}

__global__ void DFSPH_get_fluid_particles_near_plane_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet,
	int* count_particles_fluid, int* ids_near_plane_fluid) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }
	RealCuda gap_pos = (-2.0 + GAP_PLANE_POS * data.getKernelRadius());
	if (particleSet->pos[i].x >= gap_pos) {
		if (particleSet->pos[i].x < (gap_pos + 0.5 * data.getKernelRadius())) {
			int id = atomicAdd(count_particles_fluid, 1);
			ids_near_plane_fluid[id] = i;
			particleSet->color[i] = Vector3d(1, 0, 0);
		}
	}
}

__global__ void DFSPH_get_buffer_particles_near_plane_kernel(SPH::DFSPHCData data, Vector3d* pos, int numParticles,
	int* count_particles_buffer, int* ids_near_plane_buffer) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numParticles) { return; }

	RealCuda gap_pos = (-2.0 + GAP_PLANE_POS * data.getKernelRadius());
	if (pos[i].x > (gap_pos - data.getKernelRadius())) {
		int id = atomicAdd(count_particles_buffer, 1);
		ids_near_plane_buffer[id] = i;
	}
	
}

__global__ void DFSPH_fit_particles_simple_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, Vector3d* pos_buffer,
	int count_particles_buffer, int count_particles_fluid, int* ids_near_plane_buffer, int* ids_near_plane_fluid,
	int count_buffer_displaced, int* ids_buffer_displaced,
	int* nbr_displaced, RealCuda* amount_displaced, int iter_nb, Vector3d* color) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (count_particles_buffer)) { return; }
	Vector3d pos_i = pos_buffer[ids_near_plane_buffer[i]];
	color[ids_near_plane_buffer[i]] = Vector3d(0, 0, 0); 
	
	//first I'll only treat the rightmost particles so for anyother I just end the function
	RealCuda dist_limit = data.particleRadius * 2 * data.particleRadius * 2;
	for (int k = 0; k < count_particles_buffer; ++k) {
		if (k != i) {
			Vector3d pos_k = pos_buffer[ids_near_plane_buffer[k]];
			if (pos_i.x < pos_k.x) {
				Vector3d d = pos_i - pos_k;
				d.x = 0;
				RealCuda dist = d.squaredNorm();
				if (dist < dist_limit) {
					return;
				}
			}
		}
	}


	color[ids_near_plane_buffer[i]] = Vector3d(0.5, 0.5, 0.5);


	//since we are he rightmost we can look for the closest particle in the fluid and put us tangent to it
	RealCuda dist = (pos_i - particleSet->pos[ids_near_plane_fluid[0]]).squaredNorm();
	int id_closest = 0;
	for (int k = 1; k < count_particles_fluid; ++k) {
		RealCuda d = (pos_i - particleSet->pos[ids_near_plane_fluid[k]]).squaredNorm();
		if (d < dist) {
			dist = d;
			id_closest = k;
		}
	}

	//also look in the particles that were previouslys displaced
	bool closest_was_displaced = false;
	for (int k = 0; k < count_buffer_displaced; ++k) {
		RealCuda d = (pos_i - pos_buffer[ids_buffer_displaced[k]]).squaredNorm();
		if (d < dist) {
			dist = d;
			id_closest = k;
			closest_was_displaced = true;
		}
	}

	//now that we have the closest fluid put ourselve tangent to it by X-axis movement
	if (dist > dist_limit) {
		Vector3d pos_closest = closest_was_displaced?pos_buffer[ids_buffer_displaced[id_closest]]:particleSet->pos[ids_near_plane_fluid[id_closest]];
		Vector3d d_pos = pos_closest - pos_i;

		//remove the case of particles of to far in the tangent plane
		Vector3d d_pos_temp = d_pos;
		d_pos_temp.x = 0;
		RealCuda sq_tangent_plane_dist = d_pos_temp.squaredNorm();
		if (sq_tangent_plane_dist > dist_limit) {
			return;
		}

		//particleSet->vel[ids_near_plane_buffer[i]] = Vector3d(100);
		//displace the particle
		RealCuda x_displaced = sqrtf(dist_limit - sq_tangent_plane_dist);
		RealCuda d_x = d_pos.x - x_displaced;
		d_x *= 0.75;
		//pos_buffer[ids_near_plane_buffer[i]].x += d_x;

		atomicAdd(amount_displaced, d_x);
		int id_displaced=atomicAdd(nbr_displaced, 1);

		ids_buffer_displaced[count_buffer_displaced + id_displaced] = i;

		color[ids_near_plane_buffer[i]] = Vector3d(0, 1, 0); 
			/*
		switch (iter_nb) {
		case 0: color[ids_near_plane_buffer[i]] = Vector3d(0, 0, 1); break;
		case 1: color[ids_near_plane_buffer[i]] = Vector3d(0, 0, 0); break;
		case 2: color[ids_near_plane_buffer[i]] = Vector3d(0, 0, 0); break;
		case 3: color[ids_near_plane_buffer[i]] = Vector3d(0, 0, 0); break;
		case 4: color[ids_near_plane_buffer[i]] = Vector3d(0, 0, 0); break;
		}
		//*/

	}

}


__global__ void DFSPH_move_to_displaced_kernel(int* ids_near_plane_buffer, int count_particles_buffer, int* ids_buffer_displaced, 
	int count_buffer_displaced,	int nbr_to_move) {
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= 1) { return; }
	

	for (int i = 0; i < nbr_to_move; ++i) {
		int id_in_buffer = ids_buffer_displaced[count_buffer_displaced + i];

		//store the actual particle id in the displaced buffer
		ids_buffer_displaced[count_buffer_displaced + i] = ids_near_plane_buffer[id_in_buffer];

		//and now we need to sort it to remove it so I switch it with the last particle
		int last_particle_id = ids_near_plane_buffer[count_particles_buffer-1];
		ids_near_plane_buffer[id_in_buffer] = last_particle_id;
		

		for (int k = i + 1; k < nbr_to_move; ++k) {
			if (ids_buffer_displaced[count_buffer_displaced + k] == count_particles_buffer - 1) {
				ids_buffer_displaced[count_buffer_displaced + k] = id_in_buffer;
				break;
			}
		}

		count_particles_buffer--;

	}
}

__global__ void DFSPH_find_min_dist_near_plane_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, Vector3d* pos_buffer, 
	int* ids_near_plane_buffer, int count_particles_buffer, int* ids_buffer_displaced,
	int count_buffer_displaced, int* ids_near_plane_fluid, int count_particles_fluid) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (count_particles_buffer + count_buffer_displaced)) { return; }

	RealCuda min_dist = 10000000;
	Vector3d pos_i;
	if (i < count_particles_buffer) {
		pos_i = pos_buffer[ids_near_plane_buffer[i]];
		particleSet->densityAdv[i] = min_dist; return;
	}
	else {
		pos_i = pos_buffer[ids_buffer_displaced[i]];
	}

	//handle the buffer side
	/*
	for (int j = i + 1; j < (count_particles_buffer+count_buffer_displaced); ++j) {
		Vector3d pos_j;
		if (j < count_particles_buffer) {
			pos_j = pos_buffer[ids_near_plane_buffer[j]];
		}
		else {
			pos_j = pos_buffer[ids_buffer_displaced[j]];
		}

		RealCuda dist = (pos_j - pos_i).norm();
		if (dist < min_dist) {
			min_dist = dist;
		}
	}
	//*/

	//handle the fluid side
	//*
	for (int j = 0; j < count_particles_fluid; ++j) {
		Vector3d pos_j= particleSet->pos[ids_near_plane_fluid[j]];
		
		RealCuda dist = (pos_j - pos_i).norm();
		if (min_dist > dist ) {
			min_dist = dist;
		}
	}
	//*/

	//I'll just use a curerntly unused buffer to get the valueback
	particleSet->densityAdv[i] = min_dist;
}

__global__ void get_buffer_min_kernel(RealCuda* buffer, RealCuda* result, int nb_elem) {
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= 1) { return; }

	RealCuda min = buffer[0];
	//*
	for (int i = 1; i < nb_elem; ++i) {
		if (min > buffer[i]) {
			min = buffer[i];
		}
	}
	//*/
	*result = min;
}

template<int nbr_layers, int nbr_layers_in_buffer>
__global__ void DFSPH_evaluate_density_field_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
	SPH::UnifiedParticleSet* bufferSet, Vector3d min, Vector3d max,
	Vector3i vec_count_samples, RealCuda* samples, RealCuda* samples_after_buffer,
	int count_samples, RealCuda sampling_spacing, Vector3d* sample_pos) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count_samples) { return; }

	//find the layer
	int nb_particle_in_layer = vec_count_samples.y * vec_count_samples.z;
	int layer_id = 0;
	int i_local = i;
	while (i_local >= nb_particle_in_layer) {
		i_local -= nb_particle_in_layer;
		layer_id++;
	}

	//it's a fail safe but should never be triggered unless something realy bad happened
	if (layer_id >= nbr_layers) {
		return;
	}


	//let's have 3 layers in in the buffer
	layer_id -= nbr_layers_in_buffer;


	Vector3d sampling_point;
	sampling_point.x = 0;
	sampling_point.y = (int)(i_local / vec_count_samples.z);
	sampling_point.z = (int)(i_local - (sampling_point.y) * vec_count_samples.z);
	sampling_point *= sampling_spacing;
	//put the coordinate to absolute
	sampling_point.y += min.y - sampling_spacing * 5;
	sampling_point.z += min.z - sampling_spacing * 5;

	//add the gap plane position
	RealCuda plane_pos = (-2.0 + GAP_PLANE_POS * data.getKernelRadius()) + sampling_spacing;
	sampling_point.x = plane_pos + layer_id * sampling_spacing;
	sample_pos[i] = sampling_point;//Vector3d(i_local, min.y, min.z);

	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(sampling_point, data.getKernelRadius(), data.gridOffset);


	RealCuda density = 0;
	RealCuda density_after_buffer = 0;



	bool near_fluid = false;
	bool near_buffer = false;

	//*
	//compute the fluid contribution
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
		RealCuda density_delta = fluidSet->mass[j] * KERNEL_W(data, sampling_point - fluidSet->pos[j]);
		if (density_delta>0){
			if (fluidSet->pos[j].x > plane_pos) {
				density_after_buffer += density_delta;
				if (sampling_point.x > plane_pos) {
					if (fluidSet->pos[j].y > (sampling_point.y)){
						near_fluid = true;
					}
				}
				else {
					if (fluidSet->pos[j].y > (sampling_point.y - sampling_spacing)) {
						near_fluid = true;
					}
				}
			}
			density += density_delta;
		}
	);

	//*
	//compute the buffer contribution
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
		RealCuda density_delta = bufferSet->mass[j] * KERNEL_W(data, sampling_point - bufferSet->pos[j]);
		if (density_delta > 0) {
			density_after_buffer += density_delta;
			near_buffer = true;
		}

	);
	//*/

	//compute the boundaries contribution only if there is a fluid particle anywhere near
	//*
	if ((density > 100) || (density_after_buffer > 100)) {
		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
			RealCuda density_delta = data.boundaries_data_cuda->mass[j] * KERNEL_W(data, sampling_point - data.boundaries_data_cuda->pos[j]);
			density_after_buffer += density_delta;
			density += density_delta;
		);
	}
	//*/

	if (near_buffer && near_fluid) {
		samples[i] = density;
		samples_after_buffer[i] = density_after_buffer ;
		sample_pos[i] = sampling_point;

		//that line is just an easy way to recognise the plane
		//samples_after_buffer[i] *= (((layer_id == 0)&&(density_after_buffer>500)) ? -1 : 1);
	}
	else {
		samples[i] = 0;
		samples_after_buffer[i] = 0;
	}
}

//technically only the particles aorund the plane have any importance
//so no need to compute the density for any particles far from the plances
//also unless I want to save it I don't actually need to outpu the density
// I set those optimisations statically so that the related ifs get optimized
template<bool x_motion>
__global__ void DFSPH_evaluate_density_in_buffer_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
	SPH::UnifiedParticleSet* backgroundBufferSet, SPH::UnifiedParticleSet* bufferSet, int* countRmv, 
	RealCuda plane_pos_inf, RealCuda plane_pos_sup, int iter=0) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }

	Vector3d p_i=bufferSet->pos[i];


	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);


	RealCuda density_after_buffer = bufferSet->mass[i] * data.W_zero;
	RealCuda density = 0;
	
	int count_neighbors = 0;
	bool keep_particle = true;

	//*
	keep_particle = false;

	if ((VECTOR_X_MOTION(p_i, x_motion) < (plane_pos_inf+ data.particleRadius))||(VECTOR_X_MOTION(p_i, x_motion) > (plane_pos_sup - data.particleRadius))) {
		//compute the fluid contribution
		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
			RealCuda density_delta = fluidSet->mass[j] * KERNEL_W(data, p_i - fluidSet->pos[j]);
		if (density_delta > 0) {
			if ((VECTOR_X_MOTION(fluidSet->pos[j],x_motion) > plane_pos_inf)&& (VECTOR_X_MOTION(fluidSet->pos[j], x_motion) < plane_pos_sup)) {
				density += density_delta;
				count_neighbors++;
				int nb_neighbors = fluidSet->getNumberOfNeighbourgs(0) + fluidSet->getNumberOfNeighbourgs(1);
				if ((nb_neighbors > 15)) {
					if (fluidSet->pos[j].y > (p_i.y)) {
						keep_particle = true;
					}
				}
			}
		}
		);
	}

	keep_particle = keep_particle || (VECTOR_X_MOTION(p_i, x_motion) < plane_pos_inf) || (VECTOR_X_MOTION(p_i, x_motion) > plane_pos_sup);
	

	//keep_particle = true;
	if (keep_particle) {
		//*
		//compute the buffer contribution
		if (iter == 0) {
			//Note: the strange if is just here to check if p_i!=p_j but since doing this kind of operations on float
			//		that are read from 2 files might not be that smat I did some bastard check verifying if the particle are pretty close
			//		since the buffer we are evaluating is a subset of the background buffer, the only possibility to have particle 
			//		that close is that they are the same particle.
			float limit = data.particleRadius / 10.0f;
			limit *= limit;
			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(backgroundBufferSet->neighborsDataSet, backgroundBufferSet->pos,
				if ((p_i - backgroundBufferSet->pos[j]).squaredNorm()> (limit)) {
					RealCuda density_delta = backgroundBufferSet->mass[j] * KERNEL_W(data, p_i - backgroundBufferSet->pos[j]);
					density_after_buffer += density_delta;
					count_neighbors++;
				}
			);
		}
		else {
			//on the following passes I do the computations using the neighbors from the buffer
			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
				if (i!=j) {
					RealCuda density_delta = bufferSet->mass[j] * KERNEL_W(data, p_i - bufferSet->pos[j]);
					density_after_buffer += density_delta;
					count_neighbors++;
				}
			);

			//also has to iterate over the background buffer that now represent the air
			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(backgroundBufferSet->neighborsDataSet, backgroundBufferSet->pos,
				RealCuda density_delta = backgroundBufferSet->mass[j] * KERNEL_W(data, p_i - backgroundBufferSet->pos[j]);
				density_after_buffer += density_delta;
				count_neighbors++;
			);
		}
		//*/

		//compute the boundaries contribution only if there is a fluid particle anywhere near
		//*
		if ((density > 100) || (density_after_buffer > 100)) {
			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
				RealCuda density_delta = data.boundaries_data_cuda->mass[j] * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
			density += density_delta;
			);
		}
		//*/
	}

	if (keep_particle) {
		density_after_buffer += density;
		if (true) {
			bufferSet->density[i] = density_after_buffer;
			bufferSet->densityAdv[i] = density;
			bufferSet->color[i] = Vector3d(1, 0, 0);//Vector3d((count_neighbors * 3)/255.0f, 0, 0);
		}
		
		//that line is just an easy way to recognise the plane
		//samples_after_buffer[i] *= (((layer_id == 0)&&(density_after_buffer>500)) ? -1 : 1);

		int limit_density = 1500 -50 * iter;


		keep_particle = (density_after_buffer) < limit_density;
		
		
		if (!keep_particle) {
			atomicAdd(countRmv, 1);
			bufferSet->neighborsDataSet->cell_id[i] = 25000000;
		}
		else {
			bufferSet->neighborsDataSet->cell_id[i] = 0;
		}
	}
	else {
		//here do a complete tag of the particles to remove them in the future
		if (true) {
			bufferSet->density[i] = 100000;
			bufferSet->densityAdv[i] = 100000;
		}

		atomicAdd(countRmv, 1);
		bufferSet->neighborsDataSet->cell_id[i] = 25000000;
	}
}

template<bool x_motion>
__global__ void DFSPH_compress_fluid_buffer_kernel(SPH::UnifiedParticleSet* particleSet, float compression_coefficient,
	Vector3d min, Vector3d max, RealCuda plane_inf, RealCuda plane_sup) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	float pos = VECTOR_X_MOTION(particleSet->pos[i], x_motion);
	//I must compre each side superior/inferior toward it's border
	float extremity = VECTOR_X_MOTION(((pos < 0) ? min : max), x_motion);

	float plane_pos = ((extremity < 0) ? plane_inf : plane_sup);


	pos -= extremity;
	if (abs(pos) > (abs(plane_pos - extremity) / 2)) {
		pos *= compression_coefficient;
		pos += extremity;
		VECTOR_X_MOTION(particleSet->pos[i], x_motion) = pos;
	}

}

__global__ void DFSPH_generate_buffer_from_surface_count_particles_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* backgroundSet,
	BufferFluidSurface S, int* count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= backgroundSet->numParticles) { return; }

	//ok so I'll explain that
	//the idea is that I want the position of the particle from inside the back ground and inside the buffer to be the same
	//for an easy handling when applying the buffer as a mask
	//BUT I also would like that the particles stay relatively ordoned to still have pretty good memory coherency 
	//for faster access
	//So I manipulate the cell_id (which here is only an index for sorting the particles) to put every particle that are not
	//inside the buffer at the end of the buffer but with a linear addition so that the relative order of the particle
	//in each suggroups stays the same
	//also we need to do the height with a height map but for now it wil just be a fixe height
	//*
	RealCuda dist = S.distanceToSurfaceSigned(backgroundSet->pos[i]);
	if ((dist<data.getKernelRadius()&&(backgroundSet->pos[i].y<1.2))) {
		atomicAdd(count, 1);
	}
	else {
		backgroundSet->neighborsDataSet->cell_id[i] += CELL_COUNT;
	}
}


//I want to only keep particles that are above the fluid or above the buffer
template<bool x_motion>
__global__ void DFSPH_lighten_background_buffer_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
	SPH::UnifiedParticleSet* backgroundBufferSet, SPH::UnifiedParticleSet* bufferSet, int* countRmv,
	RealCuda plane_pos_inf, RealCuda plane_pos_sup) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= backgroundBufferSet->numParticles) { return; }


	Vector3d p_i = backgroundBufferSet->pos[i];


	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);


	bool keep_particle = true;
	//*
	RealCuda sq_diameter = data.particleRadius * 2;
	sq_diameter *= sq_diameter;
	
	//check if there is any fluid particle above us
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
		if (keep_particle&&(VECTOR_X_MOTION(fluidSet->pos[j], x_motion) > plane_pos_inf) && (VECTOR_X_MOTION(fluidSet->pos[j], x_motion) < plane_pos_sup)) {
			//*
			Vector3d delta = fluidSet->pos[j] - p_i;
			//if (delta.squaredNorm() < sq_diameter) 
			int nb_neighbors = fluidSet->getNumberOfNeighbourgs(0) + fluidSet->getNumberOfNeighbourgs(1);
			if ((nb_neighbors > 15)){
				if (delta.y > 2*data.particleRadius) {
					keep_particle = false;
				}
			}
			//*/
		}
	);
	

	keep_particle = keep_particle || (VECTOR_X_MOTION(p_i, x_motion) < plane_pos_inf) || (VECTOR_X_MOTION(p_i, x_motion) > plane_pos_sup);
	//*/
	if(keep_particle){
		//Note: the strange if is just here to check if p_i!=p_j but since doing this kind of operations on float
		//		that are read from 2 files might not be that smat I did some bastard check verifying if the particle are pretty close
		//		since the buffer we are evaluating is a subset of the background buffer, the only possibility to have particle 
		//		that close is that they are the same particle.
		//Note2:and the logic behind this is that there are no hles in the buffer so particles above the buffer are any
		//		particle that is not part of the buffer
		//*
		float limit = data.particleRadius / 10.0f;
		limit *= limit;
		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
			if ((p_i - bufferSet->pos[j]).squaredNorm() < (limit)) {
				keep_particle = false;
			}
		);
		//*/
	}

	if (!keep_particle) {
		atomicAdd(countRmv, 1);
		backgroundBufferSet->neighborsDataSet->cell_id[i] = 25000000;
	}
	else {
		backgroundBufferSet->neighborsDataSet->cell_id[i] = 0;
	}
}


 
__global__ void DFSPH_evaluate_density_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
	RealCuda plane_pos_inf, RealCuda plane_pos_sup) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= fluidSet->numParticles) { return; }


	Vector3d p_i = fluidSet->pos[i];


	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);


	bool keep_particle = true;
	//*
	RealCuda sq_diameter = data.particleRadius * 2;
	sq_diameter *= sq_diameter;
	int count_neighbors = 0;
	RealCuda density = fluidSet->mass[i] * data.W_zero;

	//check if there is any fluid particle above us
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
		if (i != j) {
			//RealCuda density_delta = (fluidSet->pos[j]-p_i).norm();
			RealCuda density_delta = fluidSet->mass[j] * KERNEL_W(data, p_i - fluidSet->pos[j]);
			density += density_delta;
			count_neighbors++;
		}
	);

	/*
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		RealCuda density_delta = data.boundaries_data_cuda->mass[j] * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
		density += density_delta;
		count_neighbors++;
	);
	//*/

	fluidSet->density[i] = density;
}


__global__ void DFSPH_evaluate_particle_concentration_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
	RealCuda plane_pos_inf, RealCuda plane_pos_sup) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= fluidSet->numParticles) { return; }


	Vector3d p_i = fluidSet->pos[i];


	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);


	bool keep_particle = true;
	//*
	RealCuda sq_diameter = data.particleRadius * 2;
	sq_diameter *= sq_diameter;
	int count_neighbors = 0;
	//we cna start at 0 and ignire the i contribution because we will do a sustracction when computing the concentration gradiant
	RealCuda concentration = 0;

	//check if there is any fluid particle above us
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
		if (i != j) {
			RealCuda concentration_delta = (fluidSet->mass[j]/ fluidSet->density[j])* KERNEL_W(data, p_i - fluidSet->pos[j]);
			concentration += concentration_delta;
			count_neighbors++;
		}
	);

	fluidSet->densityAdv[i] = concentration;
}

__global__ void DFSPH_particle_shifting_base_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet, RealCuda displacement_coefficient,
	RealCuda plane_pos_inf, RealCuda plane_pos_sup) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= fluidSet->numParticles) { return; }

	bool x_motion = true;
	Vector3d p_i = fluidSet->pos[i];

	//only move particles that are close to the planes
	RealCuda affected_range = data.getKernelRadius();
	if (!(((VECTOR_X_MOTION(p_i, x_motion) < (plane_pos_inf + affected_range)) && (VECTOR_X_MOTION(p_i, x_motion) > (plane_pos_inf - affected_range))) ||
		((VECTOR_X_MOTION(p_i, x_motion) < (plane_pos_sup + affected_range)) && (VECTOR_X_MOTION(p_i, x_motion) > (plane_pos_sup - affected_range))))) {
		return;
	}

	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);

	RealCuda scaling = 1 - MIN_MACRO_CUDA(abs((VECTOR_X_MOTION(p_i, x_motion) - plane_pos_inf)), abs((VECTOR_X_MOTION(p_i, x_motion) - plane_pos_sup))) / affected_range;

	bool keep_particle = true;
	//*
	RealCuda sq_diameter = data.particleRadius * 2;
	sq_diameter *= sq_diameter;
	int count_neighbors = 0;
	//we cna start at 0 and ignire the i contribution because we will do a sustracction when computing the concentration gradiant
	Vector3d displacement = Vector3d(0,0,0);

	//check if there is any fluid particle above us
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
		if (i != j) {
			Vector3d displacement_delta = (fluidSet->densityAdv[j]- fluidSet->densityAdv[i])* (fluidSet->mass[j] / fluidSet->density[j]) * KERNEL_GRAD_W(data, p_i - fluidSet->pos[j]);
			/*
			Vector3d nj = fluidSet->pos[j] - p_i;
			nj.toUnit();
			Vector3d displacement_delta = fluidSet->density[j]*KERNEL_W(data, p_i - fluidSet->pos[j])*nj;
			//*/
			displacement += displacement_delta;
			count_neighbors++;
		}
	);

	int count_neighbors_b = 0;
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		count_neighbors_b++;
	);

	displacement *= -displacement_coefficient;
	displacement *= scaling * scaling;

	if (count_neighbors_b ==0) {
		fluidSet->pos[i]+=displacement;
		//fluidSet->color[i] = Vector3d(1, 1, 0.2);
	}
}


void handle_fluid_boundries_cuda(SPH::DFSPHCData& data, bool loading) {
	SPH::UnifiedParticleSet* particleSet = data.fluid_data;

	//the structture containing the fluid buffer x and z axis
	static SPH::UnifiedParticleSet* fluidBufferXSet = NULL;
	static Vector3d* pos_base_x = NULL;
	static int numParticles_base_x = 0;
	Vector3d min_fluid_buffer, max_fluid_buffer;
	//the structture containing the fluid buffer
	static SPH::UnifiedParticleSet* fluidBufferZSet = NULL;
	static Vector3d* pos_base_z = NULL;
	static int numParticles_base_z = 0;

	//this buffer contains a set a particle corresponding to a fluid at rest covering the whole simulation space
	static SPH::UnifiedParticleSet* backgroundFluidBufferSet = NULL;
	static Vector3d* pos_background = NULL;
	static int numParticles_background = 0;

	//the object for the surface
	static BufferFluidSurface S;
	static SPH::UnifiedParticleSet* fluidBufferSetFromSurface = NULL;
	static Vector3d* pos_base_from_surface = NULL;
	static int numParticles_base_from_surface = 0;

	//some variable defining the desired fluid buffer (axis and motion)
	//note those thing should be obtained from arguments at some point
	bool displace_windows = true;
	Vector3d movement(1, 0, 0);
	bool x_motion = (movement.x > 0.01) ? true : false;
	//compute the movement on the position and the axis
	Vector3d mov_pos = movement * data.getKernelRadius();
	Vector3d mov_axis = (movement.abs()) / movement.norm();

	if (displace_windows&&(!loading)) {
		//update the displacement offset
		data.dynamicWindowTotalDisplacement += mov_pos;
		data.gridOffset -= movement;
	}


	//define the jonction planes
	RealCuda plane_pos_inf = -2.0;
	RealCuda plane_pos_sup = 2.0;
	if (!x_motion) {
		plane_pos_inf = -0.7;
		plane_pos_sup = 0.7;
	}
	plane_pos_inf += (GAP_PLANE_POS)*data.getKernelRadius() + VECTOR_X_MOTION(data.dynamicWindowTotalDisplacement, x_motion);
	plane_pos_sup += -(GAP_PLANE_POS)*data.getKernelRadius() + VECTOR_X_MOTION(data.dynamicWindowTotalDisplacement, x_motion);




	if (loading) {
		if (fluidBufferXSet) {
			std::string msg = "handle_fluid_boundries_cuda: trying to change the boundary buffer size";
			std::cout << msg << std::endl;
			throw(msg);
		}
		//just activate that to back the current fluid for buffer usage
		bool save_to_buffer = false;
		if (save_to_buffer) {
			particleSet->write_to_file(data.fluid_files_folder + "fluid_buffer_file.txt");
		}

		//define the surface
		int surfaceType = 0;
		if(surfaceType==0){
			if (x_motion) {
				S.addPlane(Vector3d(plane_pos_inf, 0, 0), Vector3d(1, 0, 0));
				S.addPlane(Vector3d(plane_pos_sup, 0, 0), Vector3d(-1, 0, 0));
			}
			else {
				S.addPlane(Vector3d(0, 0, plane_pos_inf), Vector3d(0, 0, 1));
				S.addPlane(Vector3d(0, 0, plane_pos_sup), Vector3d(0, 0, -1));
			}
		}
		else {
			throw("the surface type need to be defined for that test");
		}


		//coefficient if we want to compress the buffers
		float buffer_compression_coefficient = -1.0f;
		//load the backgroundset
		{
			SPH::UnifiedParticleSet* dummy = NULL;
			backgroundFluidBufferSet = new SPH::UnifiedParticleSet();
			backgroundFluidBufferSet->load_from_file(data.fluid_files_folder + "background_buffer_file.txt", false, &min_fluid_buffer, &max_fluid_buffer, false);
			//fluidBufferSet->write_to_file(data.fluid_files_folder + "fluid_buffer_file.txt");
			allocate_and_copy_UnifiedParticleSet_vector_cuda(&dummy, backgroundFluidBufferSet, 1);
			if (buffer_compression_coefficient > 0.0f) {
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				if (x_motion) {
					DFSPH_compress_fluid_buffer_kernel<true> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, buffer_compression_coefficient, min_fluid_buffer, max_fluid_buffer
						, plane_pos_inf, plane_pos_sup);
				}
				else {
					DFSPH_compress_fluid_buffer_kernel<false> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, buffer_compression_coefficient, min_fluid_buffer, max_fluid_buffer
						, plane_pos_inf, plane_pos_sup);
				}
				gpuErrchk(cudaDeviceSynchronize());
			}
			backgroundFluidBufferSet->initNeighborsSearchData(data, true);
			backgroundFluidBufferSet->resetColor();

			numParticles_background = backgroundFluidBufferSet->numParticles;
			cudaMallocManaged(&(pos_background), sizeof(Vector3d) * numParticles_background);
			gpuErrchk(cudaMemcpy(pos_background, backgroundFluidBufferSet->pos, numParticles_background * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

			if (false){
				Vector3d* pos = pos_background;
				int numParticles = numParticles_background;
				std::ostringstream oss;
				int effective_count = 0;
				for (int i = 0; i < numParticles; ++i) {
					uint8_t density = 255;
					uint8_t alpha = 255;
					uint32_t txt = (((alpha << 8) + density << 8) + 0 << 8) + 0;
					

					oss << pos[i].x << " " << pos[i].y << " " << pos[i].z << " "
						<< txt << std::endl;
					effective_count++;
				}

				std::ofstream myfile("densityCloud.pcd", std::ofstream::trunc);
				if (myfile.is_open())
				{
					myfile << "VERSION 0.7" << std::endl
						<< "FIELDS x y z rgb" << std::endl
						<< "SIZE 4 4 4 4" << std::endl
						<< "TYPE F F F U" << std::endl
						<< "COUNT 1 1 1 1" << std::endl
						<< "WIDTH " << effective_count << std::endl
						<< "HEIGHT " << 1 << std::endl
						<< "VIEWPOINT 0 0 0 1 0 0 0" << std::endl
						<< "POINTS " << effective_count << std::endl
						<< "DATA ascii" << std::endl;
					myfile << oss.str();
					myfile.close();
				}
			}
		}

		
		//create the buffer from the background and the surface
		{
			std::cout << "generating fluid buffer from background start" << std::endl;
			//first we count the nbr of particles and attribute the index for sorting
			//als we will reorder the background buffer so we need to resave its state
			int* out_int = NULL;
			cudaMallocManaged(&(out_int), sizeof(int));
			*out_int = 0;
			{
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				DFSPH_generate_buffer_from_surface_count_particles_kernel << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, S, out_int);
				gpuErrchk(cudaDeviceSynchronize());
			}
			int count_inside_buffer = *out_int;
			CUDA_FREE_PTR(out_int);


			std::cout << "reorganise background" << std::endl;

			//sort the buffer
			cub::DeviceRadixSort::SortPairs(backgroundFluidBufferSet->neighborsDataSet->d_temp_storage_pair_sort, backgroundFluidBufferSet->neighborsDataSet->temp_storage_bytes_pair_sort,
				backgroundFluidBufferSet->neighborsDataSet->cell_id, backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted,
				backgroundFluidBufferSet->neighborsDataSet->p_id, backgroundFluidBufferSet->neighborsDataSet->p_id_sorted, backgroundFluidBufferSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());

			cuda_sortData(*backgroundFluidBufferSet, backgroundFluidBufferSet->neighborsDataSet->p_id_sorted);
			gpuErrchk(cudaDeviceSynchronize());

			//resave the background state
			gpuErrchk(cudaMemcpy(pos_background, backgroundFluidBufferSet->pos, numParticles_background * sizeof(Vector3d), cudaMemcpyDeviceToDevice));


			std::cout << "creating buffer" << std::endl;

			//and now we can create the buffer and save the positions
			SPH::UnifiedParticleSet* dummy = NULL;
			fluidBufferSetFromSurface = new SPH::UnifiedParticleSet();
			fluidBufferSetFromSurface->init(count_inside_buffer, true, true, false, true);
			allocate_and_copy_UnifiedParticleSet_vector_cuda(&dummy, fluidBufferSetFromSurface, 1);

			std::cout << "copy to bbuffer" << std::endl;

			numParticles_base_from_surface = fluidBufferSetFromSurface->numParticles;
			cudaMallocManaged(&(pos_base_from_surface), sizeof(Vector3d) * numParticles_base_from_surface);
			gpuErrchk(cudaMemcpy(pos_base_from_surface, pos_background, numParticles_base_from_surface * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(fluidBufferSetFromSurface->mass, backgroundFluidBufferSet->mass, numParticles_base_from_surface * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

			fluidBufferSetFromSurface->resetColor();

			std::cout << "generating fluid buffer from background end" << std::endl;



			if (false) {
				Vector3d* pos = pos_base_from_surface;
				int numParticles = numParticles_base_from_surface;
				std::ostringstream oss;
				int effective_count = 0;
				for (int i = 0; i < numParticles; ++i) {
					uint8_t density = 255;
					uint8_t alpha = 255;
					uint32_t txt = (((alpha << 8) + density << 8) + 0 << 8) + 0;


					oss << pos[i].x << " " << pos[i].y << " " << pos[i].z << " "
						<< txt << std::endl;
					effective_count++;
				}

				std::ofstream myfile("densityCloud.pcd", std::ofstream::trunc);
				if (myfile.is_open())
				{
					myfile << "VERSION 0.7" << std::endl
						<< "FIELDS x y z rgb" << std::endl
						<< "SIZE 4 4 4 4" << std::endl
						<< "TYPE F F F U" << std::endl
						<< "COUNT 1 1 1 1" << std::endl
						<< "WIDTH " << effective_count << std::endl
						<< "HEIGHT " << 1 << std::endl
						<< "VIEWPOINT 0 0 0 1 0 0 0" << std::endl
						<< "POINTS " << effective_count << std::endl
						<< "DATA ascii" << std::endl;
					myfile << oss.str();
					myfile.close();
				}

				std::cout << "writting buffer to file end" << std::endl;

			}
		}


		return;
	}

	std::vector<std::string> timing_names{"color_reset","reset pos","apply displacement to buffer","init neightbors","compute density",
		"reduce buffer","cpy_velocity","reduce fluid", "apply buffer"};
	static SPH::SegmentedTiming timings("handle_fluid_boundries_cuda",timing_names,false);
	timings.init_step();



	SPH::UnifiedParticleSet* fluidBufferSet = fluidBufferSetFromSurface;
	Vector3d* pos_base = pos_base_from_surface;
	int numParticles_base = numParticles_base_from_surface;

	if (!fluidBufferSet) {
		std::string msg = "handle_fluid_boundries_cuda: you have to load a buffer first";
		std::cout << msg << std::endl;
		throw(msg);
	}


	//now when loading the boundary buffer i need first to rmv the fluid particles that are on the buffer then load the buffer
	particleSet->resetColor();
	fluidBufferSet->resetColor();

	timings.time_next_point();

	{
		//two problems I need a measure and I need a spacing algorithm.
		//first the measure, I know the plane of the gap
		//so I need to compute the disance between the particles from the buffer and that plane 
		//and add it to the distance between the fluid particles and the plane

		//we could use the neighbo structure to have a faster access but for now let's just brute force it	
		static int* countRmv = NULL;

		if (!countRmv) {
			cudaMallocManaged(&(countRmv), sizeof(int));
		}

		//make a copy of the buffer to be able to modify the opstions before setting it into the simulation
		//not since the algorithm can only add particles and not remove them I don't have to check that I don't croos the max particle number
		if (fluidBufferSet->numParticles != numParticles_base) {
			fluidBufferSet->updateActiveParticleNumber(numParticles_base);
		}

		//reset the buffers
		fluidBufferSet->updateActiveParticleNumber(numParticles_base);
		backgroundFluidBufferSet->updateActiveParticleNumber(numParticles_background);
		gpuErrchk(cudaMemcpy(fluidBufferSet->pos, pos_base, numParticles_base * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(backgroundFluidBufferSet->pos, pos_background, numParticles_background * sizeof(Vector3d), cudaMemcpyDeviceToDevice));


		timings.time_next_point();

		//we need to move the positions of the particles inside the buffer if we want the window to be dynamic
		if (displace_windows) {

			//first displace the boundaries
			SPH::UnifiedParticleSet* particleSetMove = data.boundaries_data;
			unsigned int numParticles = particleSetMove->numParticles;
			int numBlocks = calculateNumBlocks(numParticles);
			apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (particleSetMove->pos, mov_pos, numParticles);

			//and the buffers
			//carefull since they are reset you must displace them for the full displacment since the start of the simulation
			particleSetMove = backgroundFluidBufferSet;
			numParticles = particleSetMove->numParticles;
			numBlocks = calculateNumBlocks(numParticles);
			apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (particleSetMove->pos, data.dynamicWindowTotalDisplacement, numParticles);


			particleSetMove = fluidBufferSet;
			numParticles = particleSetMove->numParticles;
			numBlocks = calculateNumBlocks(numParticles);
			apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (particleSetMove->pos, data.dynamicWindowTotalDisplacement, numParticles);

			gpuErrchk(cudaDeviceSynchronize());

			//update the boundaries neighbors
			data.boundaries_data->initNeighborsSearchData(data, false);
		}

		//update the neighbors structures for the buffers
		fluidBufferSet->initNeighborsSearchData(data, false);
		backgroundFluidBufferSet->initNeighborsSearchData(data, false);

		//update the neighbor structure for the fluid
		particleSet->initNeighborsSearchData(data, false);


		timings.time_next_point();

		//ok since sampling the space regularly with particles to close the gap between the fluid and the buffer is realy f-ing hard
		//let's go the oposite way
		//I'll use a buffer too large,  evaluate the density on the buffer particles and remove the particle with density too large
		//also no need to do it on all partilce only those close enught from the plane with the fluid end
		{

			*countRmv = 0;
			{
				int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
				if (x_motion) {
					DFSPH_evaluate_density_in_buffer_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
						countRmv, plane_pos_inf, plane_pos_sup);
				}
				else {
					DFSPH_evaluate_density_in_buffer_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
						countRmv, plane_pos_inf, plane_pos_sup);
				}

				gpuErrchk(cudaDeviceSynchronize());
			}


			timings.time_next_point();
			//write it forthe viewer
			if (false) {
				std::ostringstream oss;
				int effective_count = 0;
				for (int i = 0; i < fluidBufferSet->numParticles; ++i) {
					uint8_t density = fminf(1.0f, (fluidBufferSet->density[i] / 1500.0f)) * 255;
					uint8_t alpha = 255;
					if (fluidBufferSet->density[i] >= 5000) {
						//continue;

					}
					/*
					if (density_field_after_buffer[i] >= 0) {
						if (density == 0) {
							continue;
						}
						if (density > 245) {
							continue;
						}
					}
					//*/
					//uint8_t density = (density_field[i] > 500) ? 255 : 0;
					uint32_t txt = (((alpha << 8) + density << 8) + density << 8) + density;
					//*
					if (fluidBufferSet->density[i] < 1000) {
						txt = (((alpha << 8) + density << 8) + 0 << 8) + 0;
					}
					if (fluidBufferSet->density[i] > 1000) {
						txt = (((alpha << 8) + 0 << 8) + density << 8) + 0;
					}
					if (fluidBufferSet->density[i] > 1100) {
						txt = (((alpha << 8) + 0 << 8) + 0 << 8) + density;
					}
					if (fluidBufferSet->density[i] > 1150) {
						txt = (((alpha << 8) + 255 << 8) + 0 << 8) + 144;
					}
					//*/

					oss << pos_base[i].x << " " << pos_base[i].y << " " << pos_base[i].z << " "
						<< txt << std::endl;
					effective_count++;
				}

				std::ofstream myfile("densityCloud.pcd", std::ofstream::trunc);
				if (myfile.is_open())
				{
					myfile << "VERSION 0.7" << std::endl
						<< "FIELDS x y z rgb" << std::endl
						<< "SIZE 4 4 4 4" << std::endl
						<< "TYPE F F F U" << std::endl
						<< "COUNT 1 1 1 1" << std::endl
						<< "WIDTH " << effective_count << std::endl
						<< "HEIGHT " << 1 << std::endl
						<< "VIEWPOINT 0 0 0 1 0 0 0" << std::endl
						<< "POINTS " << effective_count << std::endl
						<< "DATA ascii" << std::endl;
					myfile << oss.str();
					myfile.close();
				}
			}

			//remove the tagged particle from the buffer (all yhe ones that have a density that is too high)
			//now we can remove the partices from the simulation
			// use the same process as when creating the neighbors structure to put the particles to be removed at the end
			cub::DeviceRadixSort::SortPairs(fluidBufferSet->neighborsDataSet->d_temp_storage_pair_sort, fluidBufferSet->neighborsDataSet->temp_storage_bytes_pair_sort,
				fluidBufferSet->neighborsDataSet->cell_id, fluidBufferSet->neighborsDataSet->cell_id_sorted,
				fluidBufferSet->neighborsDataSet->p_id, fluidBufferSet->neighborsDataSet->p_id_sorted, fluidBufferSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());

			cuda_sortData(*fluidBufferSet, fluidBufferSet->neighborsDataSet->p_id_sorted);
			gpuErrchk(cudaDeviceSynchronize());

			int new_num_particles = fluidBufferSet->numParticles - *countRmv;
			std::cout << "handle_fluid_boundries_cuda: changing num particles in the buffer: " << new_num_particles << "   nb removed : " << *countRmv << std::endl;
			fluidBufferSet->updateActiveParticleNumber(new_num_particles);

			//we need to reinit the neighbors struct for the fluidbuffer since we removed some particles
			fluidBufferSet->initNeighborsSearchData(data, false);

			timings.time_next_point();

			//we now have to lighten the background buffer so we can continue the computations near the surface
			{
				*countRmv = 0;
				{
					int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
					if (x_motion) {
						DFSPH_lighten_background_buffer_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
							countRmv, plane_pos_inf, plane_pos_sup);
					}
					else {
						DFSPH_lighten_background_buffer_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
							countRmv, plane_pos_inf, plane_pos_sup);
					}
					gpuErrchk(cudaDeviceSynchronize());
				}


				if (false) {
					Vector3d* pos = pos_background;
					int numParticles = numParticles_background;
					std::ostringstream oss;
					int effective_count = 0;
					for (int i = 0; i < numParticles; ++i) {
						if (backgroundFluidBufferSet->neighborsDataSet->cell_id[i] != 0) {
							continue;
						}
						uint8_t density = 255;
						uint8_t alpha = 255;
						uint32_t txt = (((alpha << 8) + density << 8) + 0 << 8) + 0;


						oss << pos[i].x << " " << pos[i].y << " " << pos[i].z << " "
							<< txt << std::endl;
						effective_count++;
					}
					std::cout << "effcount: " << effective_count << "from:  " << numParticles<<std::endl;

					pos = pos_base;
					numParticles = numParticles_base;
					for (int i = 0; i < numParticles; ++i) {
						
						uint8_t density = 255;
						uint8_t alpha = 255;
						uint32_t txt = (((alpha << 8) + 0 << 8) + density << 8) + 0;


						oss << pos[i].x << " " << pos[i].y << " " << pos[i].z << " "
							<< txt << std::endl;
						effective_count++;
					}

					Vector3d* pos_cpu = new Vector3d[particleSet->numParticles];
					read_UnifiedParticleSet_cuda(*particleSet, pos_cpu, NULL, NULL,  NULL);
					pos = pos_cpu;
					numParticles = particleSet->numParticles;
					for (int i = 0; i < numParticles; ++i) {

						uint8_t density = 255;
						uint8_t alpha = 255;
						uint32_t txt = (((alpha << 8) + 0 << 8) + 0 << 8) + 8;


						oss << pos[i].x << " " << pos[i].y << " " << pos[i].z << " "
							<< txt << std::endl;
						effective_count++;
					}
					delete[] pos_cpu;

					std::ofstream myfile("densityCloud.pcd", std::ofstream::trunc);
					if (myfile.is_open())
					{
						myfile << "VERSION 0.7" << std::endl
							<< "FIELDS x y z rgb" << std::endl
							<< "SIZE 4 4 4 4" << std::endl
							<< "TYPE F F F U" << std::endl
							<< "COUNT 1 1 1 1" << std::endl
							<< "WIDTH " << effective_count << std::endl
							<< "HEIGHT " << 1 << std::endl
							<< "VIEWPOINT 0 0 0 1 0 0 0" << std::endl
							<< "POINTS " << effective_count << std::endl
							<< "DATA ascii" << std::endl;
						myfile << oss.str();
						myfile.close();
					}
				}

				
				//remove the tagged particle from the background
				//now we can remove the partices from the simulation
				// use the same process as when creating the neighbors structure to put the particles to be removed at the end
				cub::DeviceRadixSort::SortPairs(backgroundFluidBufferSet->neighborsDataSet->d_temp_storage_pair_sort, backgroundFluidBufferSet->neighborsDataSet->temp_storage_bytes_pair_sort,
					backgroundFluidBufferSet->neighborsDataSet->cell_id, backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted,
					backgroundFluidBufferSet->neighborsDataSet->p_id, backgroundFluidBufferSet->neighborsDataSet->p_id_sorted, backgroundFluidBufferSet->numParticles);
				gpuErrchk(cudaDeviceSynchronize());

				cuda_sortData(*backgroundFluidBufferSet, backgroundFluidBufferSet->neighborsDataSet->p_id_sorted);
				gpuErrchk(cudaDeviceSynchronize());

				int new_num_particles = backgroundFluidBufferSet->numParticles - *countRmv;
				std::cout << "handle_fluid_boundries_cuda: changing num particles in the background: " << new_num_particles << "   nb removed : " << *countRmv << std::endl;
				backgroundFluidBufferSet->updateActiveParticleNumber(new_num_particles);
			
				backgroundFluidBufferSet->initNeighborsSearchData(data, false);

			}


			for (int i = 4; i < 8;++i){
				//we need to reinit the neighbors struct for the fluidbuffer since we removed some particles
				fluidBufferSet->initNeighborsSearchData(data, false);

				*countRmv = 0;
				{
					int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
					if (x_motion) {
						DFSPH_evaluate_density_in_buffer_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
							countRmv, plane_pos_inf, plane_pos_sup, i);
					}
					else {
						DFSPH_evaluate_density_in_buffer_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
							countRmv, plane_pos_inf, plane_pos_sup, i);
					}

					gpuErrchk(cudaDeviceSynchronize());
				}
			
				//remove the tagged particle from the buffer (all yhe ones that have a density that is too high)
				//now we can remove the partices from the simulation
				// use the same process as when creating the neighbors structure to put the particles to be removed at the end
				cub::DeviceRadixSort::SortPairs(fluidBufferSet->neighborsDataSet->d_temp_storage_pair_sort, fluidBufferSet->neighborsDataSet->temp_storage_bytes_pair_sort,
					fluidBufferSet->neighborsDataSet->cell_id, fluidBufferSet->neighborsDataSet->cell_id_sorted,
					fluidBufferSet->neighborsDataSet->p_id, fluidBufferSet->neighborsDataSet->p_id_sorted, fluidBufferSet->numParticles);
				gpuErrchk(cudaDeviceSynchronize());

				cuda_sortData(*fluidBufferSet, fluidBufferSet->neighborsDataSet->p_id_sorted);
				gpuErrchk(cudaDeviceSynchronize());

				int new_num_particles = fluidBufferSet->numParticles - *countRmv;
				std::cout << "handle_fluid_boundries_cuda: (iter: "<<i<<") changing num particles in the buffer: " << new_num_particles << "   nb removed : " << *countRmv << std::endl;
				fluidBufferSet->updateActiveParticleNumber(new_num_particles);

			}





		}


		//I'll save the velocity field by setting the velocity of each particle to the weighted average of the three nearest
		//or set it to 0, maybe I need to do smth intermediary
		{
			int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
			//DFSPH_init_buffer_velocity_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->pos, fluidBufferSet->vel, fluidBufferSet->numParticles);
			DynamicWindowBuffer::init_buffer_kernel << <numBlocks, BLOCKSIZE >> > (fluidBufferSet->vel, fluidBufferSet->numParticles, Vector3d(0, 0, 0));
			gpuErrchk(cudaDeviceSynchronize());
		}

		timings.time_next_point();

		//now we can remove the partices from the simulation
		{

			*countRmv = 0;
			int numBlocks = calculateNumBlocks(particleSet->numParticles);
			if (x_motion) {
				DFSPH_reset_fluid_boundaries_remove_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, countRmv, plane_pos_inf, plane_pos_sup);
			}
			else {
				DFSPH_reset_fluid_boundaries_remove_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, countRmv, plane_pos_inf, plane_pos_sup);
			}
			gpuErrchk(cudaDeviceSynchronize());


			timings.time_next_point();

			//*
			//now use the same process as when creating the neighbors structure to put the particles to be removed at the end
			cub::DeviceRadixSort::SortPairs(particleSet->neighborsDataSet->d_temp_storage_pair_sort, particleSet->neighborsDataSet->temp_storage_bytes_pair_sort,
				particleSet->neighborsDataSet->cell_id, particleSet->neighborsDataSet->cell_id_sorted,
				particleSet->neighborsDataSet->p_id, particleSet->neighborsDataSet->p_id_sorted, particleSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());

			cuda_sortData(*particleSet, particleSet->neighborsDataSet->p_id_sorted);
			gpuErrchk(cudaDeviceSynchronize());

			//and now you can update the number of particles


			int new_num_particles = particleSet->numParticles - *countRmv;
			std::cout << "handle_fluid_boundries_cuda: changing num particles: " << new_num_particles << "   nb removed : " << *countRmv << std::endl;
			particleSet->updateActiveParticleNumber(new_num_particles);
			//*/

			timings.time_next_point();
		}

		{
			//and now add the buffer back into the simulation
			//check there is enougth space for the particles and make some if necessary
			int new_num_particles = particleSet->numParticles + fluidBufferSet->numParticles;
			if (new_num_particles > particleSet->numParticlesMax) {
				particleSet->changeMaxParticleNumber(new_num_particles * 1.25);
			}

			//add the particle in the simulation
			int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
			DFSPH_reset_fluid_boundaries_add_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->gpu_ptr);
			gpuErrchk(cudaDeviceSynchronize());

			//and change the number
			std::cout << "handle_fluid_boundries_cuda: changing num particles: " << new_num_particles << "   nb added : " << fluidBufferSet->numParticles << std::endl;
			particleSet->updateActiveParticleNumber(new_num_particles);

		}

		//now we need to use a particle shifting to remove the density spikes and gap
		{
			RealCuda diffusion_coefficient = 1000*0.5* data.getKernelRadius()* data.getKernelRadius()/data.density0;
			particleSet->initNeighborsSearchData(data, false);

			{
				int numBlocks = calculateNumBlocks(particleSet->numParticles);
				DFSPH_evaluate_density_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, plane_pos_inf, plane_pos_sup);
				gpuErrchk(cudaDeviceSynchronize());
			}

			{
				int numBlocks = calculateNumBlocks(particleSet->numParticles);
				DFSPH_evaluate_particle_concentration_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, plane_pos_inf, plane_pos_sup);
				gpuErrchk(cudaDeviceSynchronize());
			}

			{
				int numBlocks = calculateNumBlocks(particleSet->numParticles);
				DFSPH_particle_shifting_base_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, diffusion_coefficient, plane_pos_inf, plane_pos_sup);
				gpuErrchk(cudaDeviceSynchronize());
			}


			//	__global__ void DFSPH_evaluate_particle_concentration_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
			//DFSPH_particle_shifting_base_kernel
		}

		timings.time_next_point();
		timings.end_step();
		timings.recap_timings();

		//still need the damping near the borders as long as we don't implement the implicit borders
		//with paricle boundaries 3 is the lowest number of steps that absorb nearly any visible perturbations
		add_border_to_damp_planes_cuda(data, x_motion, !x_motion);
		//data.damp_planes[data.damp_planes_count++] = Vector3d( plane_pos_inf, 0, 0);
		//data.damp_planes[data.damp_planes_count++] = Vector3d(plane_pos_sup, 0, 0);
		data.damp_borders_steps_count = 3;
		data.damp_borders = true;
	}

	//here is a test to see what does the density filed looks like at the interface
		//first let's do a test
		//let's compute the density on the transition plane at regular interval in the fluid 
		//then compare with the values obtained when I add the buffer back in the simulation
	if (false) {

		//get the min for further calculations
		Vector3d min, max;
		get_UnifiedParticleSet_min_max_naive_cuda(*particleSet, min, max);
		//first define the structure that will hold the density values
#define NBR_LAYERS 10
#define NBR_LAYERS_IN_BUFFER 7
		RealCuda sampling_spacing = data.particleRadius / 2;
		Vector3i vec_count_samples(0, (max.y - min.y) / sampling_spacing + 1 + 10, (max.z - min.z) / sampling_spacing + 1 + 10);
		int count_samples = vec_count_samples.y * vec_count_samples.z;
		count_samples *= NBR_LAYERS;

		static RealCuda* density_field = NULL;
		static RealCuda* density_field_after_buffer = NULL;
		static Vector3d* sample_pos = NULL;
		static int density_field_size = 0;

		if (count_samples > density_field_size) {
			CUDA_FREE_PTR(density_field);

			density_field_size = count_samples * 1.5;
			cudaMallocManaged(&(density_field), density_field_size * sizeof(RealCuda));
			cudaMallocManaged(&(density_field_after_buffer), density_field_size * sizeof(RealCuda));
			cudaMallocManaged(&(sample_pos), density_field_size * sizeof(Vector3d));

		}

		//re-init the neighbor structure to be able to compute the density
		particleSet->initNeighborsSearchData(data, false);
		fluidBufferSet->initNeighborsSearchData(data, false);

		{
			int numBlocks = calculateNumBlocks(count_samples);
			DFSPH_evaluate_density_field_kernel<NBR_LAYERS, NBR_LAYERS_IN_BUFFER> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->gpu_ptr,
				min, max, vec_count_samples, density_field, density_field_after_buffer, count_samples, sampling_spacing, sample_pos);
			gpuErrchk(cudaDeviceSynchronize());
		}

		//write it forthe viewer
		if (true) {
			std::ostringstream oss;
			int effective_count = 0;
			for (int i = 0; i < count_samples; ++i) {
				uint8_t density = fminf(1.0f, (density_field[i] / 1300.0f)) * 255;
				uint8_t alpha = 255;
				/*
				if (density_field[i] >= 0) {
					if (density == 0) {
						continue;
					}
					if (density > 245) {
						continue;
					}
				}
				//*/

				//uint8_t density = (density_field[i] > 500) ? 255 : 0;
				uint32_t txt = (((alpha << 8) + density << 8) + density << 8) + density;

				if (density_field_after_buffer[i] < 1000) {
					txt = (((alpha << 8) + density << 8) + 0 << 8) + 0;
				}
				if (density_field_after_buffer[i] < 800) {
					txt = (((alpha << 8) + 0 << 8) + density << 8) + 0;
				}
				if (density_field_after_buffer[i] < 600) {
					txt = (((alpha << 8) + 0 << 8) + 0 << 8) + density;
				}
				//if (density_field_after_buffer[i] > 950) {
				//	txt = (((alpha << 8) + 255 << 8) + 0 << 8) + 144;
				//}

				oss << sample_pos[i].x << " " << sample_pos[i].y << " " << sample_pos[i].z << " "
					<< txt << std::endl;
				effective_count++;
			}

			std::ofstream myfile("densityCloud.pcd", std::ofstream::trunc);
			if (myfile.is_open())
			{
				myfile << "VERSION 0.7" << std::endl
					<< "FIELDS x y z rgb" << std::endl
					<< "SIZE 4 4 4 4" << std::endl
					<< "TYPE F F F U" << std::endl
					<< "COUNT 1 1 1 1" << std::endl
					<< "WIDTH " << effective_count << std::endl
					<< "HEIGHT " << 1 << std::endl
					<< "VIEWPOINT 0 0 0 1 0 0 0" << std::endl
					<< "POINTS " << effective_count << std::endl
					<< "DATA ascii" << std::endl;
				myfile << oss.str();
				myfile.close();
			}
		}
#undef NBR_LAYERS
#undef NBR_LAYERS_IN_BUFFER
	}

	std::cout << "handling lfuid boundaries finished" << std::endl;
	
}

void handle_fluid_boundries_experimental_cuda(SPH::DFSPHCData& data, bool loading) {
	SPH::UnifiedParticleSet* particleSet = data.fluid_data;

	//the structture containing the fluid buffer x and z axis
	static SPH::UnifiedParticleSet* fluidBufferXSet = NULL;
	static Vector3d* pos_base_x = NULL;
	static int numParticles_base_x = 0;
	Vector3d min_fluid_buffer, max_fluid_buffer;
	//the structture containing the fluid buffer
	static SPH::UnifiedParticleSet* fluidBufferZSet = NULL;
	static Vector3d* pos_base_z = NULL;
	static int numParticles_base_z = 0;

	//this buffer contains a set a particle corresponding to a fluid at rest covering the whole simulation space
	static SPH::UnifiedParticleSet* backgroundFluidBufferSet = NULL;
	static Vector3d* pos_background = NULL;
	static int numParticles_background = 0;

	if (!loading) {
		if (fluidBufferXSet) {
			std::string msg = "handle_fluid_boundries_cuda: trying to change the boundary buffer size";
			std::cout << msg << std::endl;
			throw(msg);
		}
		//just activate that to back the current fluid for buffer usage
		bool save_to_buffer = false;
		if (save_to_buffer) {
			particleSet->write_to_file(data.fluid_files_folder + "fluid_buffer_file.txt");
		}

		//load the backgroundset
		{
			SPH::UnifiedParticleSet* dummy = NULL;
			backgroundFluidBufferSet = new SPH::UnifiedParticleSet();
			backgroundFluidBufferSet->load_from_file(data.fluid_files_folder + "background_buffer_file.txt", false, &min_fluid_buffer, &max_fluid_buffer, false);
			//fluidBufferSet->write_to_file(data.fluid_files_folder + "fluid_buffer_file.txt");
			allocate_and_copy_UnifiedParticleSet_vector_cuda(&dummy, backgroundFluidBufferSet, 1);
			backgroundFluidBufferSet->initNeighborsSearchData(data, true);
			backgroundFluidBufferSet->resetColor();
			std::cout << "retard test: " << backgroundFluidBufferSet->numParticles << std::endl;

			numParticles_background = backgroundFluidBufferSet->numParticles;
			cudaMallocManaged(&(pos_background), sizeof(Vector3d) * numParticles_background);
			gpuErrchk(cudaMemcpy(pos_background, backgroundFluidBufferSet->pos, numParticles_background * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		}

		//create the fluid buffers
		//it must be a sub set of the background set (enougth to dodge the kernel sampling problem at the fluid/air interface)
		{
			SPH::UnifiedParticleSet* dummy = NULL;
			fluidBufferXSet = new SPH::UnifiedParticleSet();
			fluidBufferXSet->load_from_file(data.fluid_files_folder + "fluid_buffer_x_file.txt", false, &min_fluid_buffer, &max_fluid_buffer);
			allocate_and_copy_UnifiedParticleSet_vector_cuda(&dummy, fluidBufferXSet, 1);
			fluidBufferXSet->initNeighborsSearchData(data, true);
			fluidBufferXSet->resetColor();

			numParticles_base_x = fluidBufferXSet->numParticles;
			cudaMallocManaged(&(pos_base_x), sizeof(Vector3d) * numParticles_base_x);
			gpuErrchk(cudaMemcpy(pos_base_x, fluidBufferXSet->pos, numParticles_base_x * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		}

		{
			SPH::UnifiedParticleSet* dummy = NULL;
			fluidBufferZSet = new SPH::UnifiedParticleSet();
			fluidBufferZSet->load_from_file(data.fluid_files_folder + "fluid_buffer_z_file.txt", false, &min_fluid_buffer, &max_fluid_buffer);
			allocate_and_copy_UnifiedParticleSet_vector_cuda(&dummy, fluidBufferZSet, 1);
			fluidBufferZSet->initNeighborsSearchData(data, true);
			fluidBufferZSet->resetColor();

			numParticles_base_z = fluidBufferZSet->numParticles;
			cudaMallocManaged(&(pos_base_z), sizeof(Vector3d) * numParticles_base_z);
			gpuErrchk(cudaMemcpy(pos_base_z, fluidBufferZSet->pos, numParticles_base_z * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		}


		return;
	}

	bool displace_windows = true;
	Vector3d movement(0, 0, 1);
	bool x_motion = (movement.x > 0.01) ? true : false;
	//compute the movement on the position and the axis
	Vector3d mov_pos = movement * data.getKernelRadius();
	Vector3d mov_axis = (movement.abs()) / movement.norm();

	SPH::UnifiedParticleSet* fluidBufferSet = fluidBufferXSet;
	Vector3d* pos_base = pos_base_x;
	int numParticles_base = numParticles_base_x;
	if (!x_motion) {
		fluidBufferSet = fluidBufferZSet;
		pos_base = pos_base_z;
		numParticles_base = numParticles_base_z;
	}

	if (!fluidBufferSet) {
		std::string msg = "handle_fluid_boundries_cuda: you have to load a buffer first";
		std::cout << msg << std::endl;
		throw(msg);
	}


	//now when loading the boundary buffer i need first to rmv the fluid particles that are on the buffer then load the buffer
	particleSet->resetColor();
	fluidBufferSet->resetColor();

	//ok so now we have a gap between the fluid boundary buffer and the actual fluid
	//what I wouls like is to reduce it
	//the easiest way would be to slightly increase the spacing between the particles inside the buffer
	//in the horizontal plane in the direction of the fluid untill we have an acceptable gap between the fluid and the buffer
	//also I need a dynamic computation of that offset so it adapt itself to the disposition of the fluid particles

	///TODO maybe an iterative process that cherche the completude of the neighborhood of the fluid particles closest 
	//to the gap would be the best way to deal with that problem
	{
		//two problems I need a measure and I need a spacing algorithm.
		//first the measure, I know the plane of the gap
		//so I need to compute the disance between the particles from the buffer and that plane 
		//and add it to the distance between the fluid particles and the plane

		//we could use the neighbo structure to have a faster access but for now let's just brute force it	
		static int* countRmv = NULL;
		static RealCuda* min_distance_fluid = NULL;
		static RealCuda* min_distance_buffer = NULL;
		static int* count_near_plane_fluid = NULL;
		static int* count_near_plane_buffer = NULL;
		static int* ids_near_plane_fluid = NULL;
		static int* ids_near_plane_buffer = NULL;
		static int* count_buffer_displaced = NULL;
		static int* ids_buffer_displaced = NULL;
		static RealCuda* test_out_real = NULL;
		static int* test_out_int = NULL;
		const int ids_count_size = 2000;

		if (!min_distance_buffer) {
			cudaMallocManaged(&(countRmv), sizeof(int));
			cudaMallocManaged(&(min_distance_fluid), sizeof(RealCuda));
			cudaMallocManaged(&(min_distance_buffer), sizeof(RealCuda));
			cudaMallocManaged(&(count_near_plane_fluid), sizeof(int));
			cudaMallocManaged(&(count_near_plane_buffer), sizeof(int));
			cudaMallocManaged(&(ids_near_plane_fluid), sizeof(int) * ids_count_size);
			cudaMallocManaged(&(ids_near_plane_buffer), sizeof(int) * ids_count_size);
			cudaMallocManaged(&(count_buffer_displaced), sizeof(int));
			cudaMallocManaged(&(ids_buffer_displaced), sizeof(int) * ids_count_size);
			cudaMallocManaged(&(test_out_real), sizeof(RealCuda));
			cudaMallocManaged(&(test_out_int), sizeof(int));
		}

		//make a copy of the buffer to be able to modify the opstions before setting it into the simulation
		//not since the algorithm can only add particles and not remove them I don't have to check that I don't croos the max particle number
		if (fluidBufferSet->numParticles != numParticles_base) {
			fluidBufferSet->updateActiveParticleNumber(numParticles_base);
		}

		//reset the buffers
		gpuErrchk(cudaMemcpy(fluidBufferSet->pos, pos_base, numParticles_base * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(backgroundFluidBufferSet->pos, pos_background, numParticles_background * sizeof(Vector3d), cudaMemcpyDeviceToDevice));


		if (displace_windows) {
			//update the displacement offsets
			data.dynamicWindowTotalDisplacement += mov_pos;
			data.gridOffset -= movement;

			//first displace the boundaries
			SPH::UnifiedParticleSet* particleSetMove = data.boundaries_data;
			unsigned int numParticles = particleSetMove->numParticles;
			int numBlocks = calculateNumBlocks(numParticles);
			apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (particleSetMove->pos, mov_pos, numParticles);

			//and the buffers
			//carefull since they are reset you must displace them for the full displacment since the start of the simulation
			particleSetMove = backgroundFluidBufferSet;
			numParticles = particleSetMove->numParticles;
			numBlocks = calculateNumBlocks(numParticles);
			apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (particleSetMove->pos, data.dynamicWindowTotalDisplacement, numParticles);


			particleSetMove = fluidBufferSet;
			numParticles = particleSetMove->numParticles;
			numBlocks = calculateNumBlocks(numParticles);
			apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (particleSetMove->pos, data.dynamicWindowTotalDisplacement, numParticles);

			gpuErrchk(cudaDeviceSynchronize());

			//update the boundaries neighbors
			data.boundaries_data->initNeighborsSearchData(data, false);
		}

		//update the neighbors structures for the buffers
		fluidBufferSet->initNeighborsSearchData(data, false);
		backgroundFluidBufferSet->initNeighborsSearchData(data, false);

		//update the neighbor structure for the fluid
		particleSet->initNeighborsSearchData(data, false);

		//get the min for further calculations
		Vector3d min, max;
		get_UnifiedParticleSet_min_max_naive_cuda(*particleSet, min, max);

		//define the jonction planes
		RealCuda plane_pos_inf = -2.0;
		RealCuda plane_pos_sup = 2.0;
		if (!x_motion) {
			plane_pos_inf = -0.7;
			plane_pos_sup = 0.7;
		}
		plane_pos_inf += (GAP_PLANE_POS)*data.getKernelRadius() + VECTOR_X_MOTION(data.dynamicWindowTotalDisplacement, x_motion);
		plane_pos_sup += -(GAP_PLANE_POS)*data.getKernelRadius() + VECTOR_X_MOTION(data.dynamicWindowTotalDisplacement, x_motion);


		//first let's do a test
		//let's compute the density on the transition plane at regular interval in the fluid 
		//then compare with the values obtained when I add the buffer back in the simulation
		if (false) {


			//first define the structure that will hold the density values
#define NBR_LAYERS 10
#define NBR_LAYERS_IN_BUFFER 7
			RealCuda sampling_spacing = data.particleRadius / 2;
			Vector3i vec_count_samples(0, (max.y - min.y) / sampling_spacing + 1 + 10, (max.z - min.z) / sampling_spacing + 1 + 10);
			int count_samples = vec_count_samples.y * vec_count_samples.z;
			count_samples *= NBR_LAYERS;

			static RealCuda* density_field = NULL;
			static RealCuda* density_field_after_buffer = NULL;
			static Vector3d* sample_pos = NULL;
			static int density_field_size = 0;

			if (count_samples > density_field_size) {
				CUDA_FREE_PTR(density_field);

				density_field_size = count_samples * 1.5;
				cudaMallocManaged(&(density_field), density_field_size * sizeof(RealCuda));
				cudaMallocManaged(&(density_field_after_buffer), density_field_size * sizeof(RealCuda));
				cudaMallocManaged(&(sample_pos), density_field_size * sizeof(Vector3d));

			}

			//re-init the neighbor structure to be able to compute the density
			particleSet->initNeighborsSearchData(data, false);
			fluidBufferSet->initNeighborsSearchData(data, false);

			{
				int numBlocks = calculateNumBlocks(count_samples);
				DFSPH_evaluate_density_field_kernel<NBR_LAYERS, NBR_LAYERS_IN_BUFFER> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->gpu_ptr,
					min, max, vec_count_samples, density_field, density_field_after_buffer, count_samples, sampling_spacing, sample_pos);
				gpuErrchk(cudaDeviceSynchronize());
			}

			//write it forthe viewer
			if (false) {
				std::ostringstream oss;
				int effective_count = 0;
				for (int i = 0; i < count_samples; ++i) {
					uint8_t density = fminf(1.0f, (density_field_after_buffer[i] / 1000.0f)) * 255;
					uint8_t alpha = 255;
					if (density_field_after_buffer[i] >= 0) {
						if (density == 0) {
							continue;
						}
						if (density > 245) {
							continue;
						}
					}

					//uint8_t density = (density_field[i] > 500) ? 255 : 0;
					uint32_t txt = (((alpha << 8) + density << 8) + density << 8) + density;

					if (density_field_after_buffer[i] < 600) {
						txt = (((alpha << 8) + density << 8) + 0 << 8) + 0;
					}
					if (density_field_after_buffer[i] < 400) {
						txt = (((alpha << 8) + 0 << 8) + density << 8) + 0;
					}
					if (density_field_after_buffer[i] < 300) {
						txt = (((alpha << 8) + 0 << 8) + 0 << 8) + density;
					}
					//if (density_field_after_buffer[i] > 950) {
					//	txt = (((alpha << 8) + 255 << 8) + 0 << 8) + 144;
					//}

					oss << sample_pos[i].x << " " << sample_pos[i].y << " " << sample_pos[i].z << " "
						<< txt << std::endl;
					effective_count++;
				}

				std::ofstream myfile("densityCloud.pcd", std::ofstream::trunc);
				if (myfile.is_open())
				{
					myfile << "VERSION 0.7" << std::endl
						<< "FIELDS x y z rgb" << std::endl
						<< "SIZE 4 4 4 4" << std::endl
						<< "TYPE F F F U" << std::endl
						<< "COUNT 1 1 1 1" << std::endl
						<< "WIDTH " << effective_count << std::endl
						<< "HEIGHT " << 1 << std::endl
						<< "VIEWPOINT 0 0 0 1 0 0 0" << std::endl
						<< "POINTS " << effective_count << std::endl
						<< "DATA ascii" << std::endl;
					myfile << oss.str();
					myfile.close();
				}
			}
#undef NBR_LAYERS
#undef NBR_LAYERS_IN_BUFFER
		}

		//ok sinece sampling the space regularly with particles to close the gap between the fluid and the buffer is realy f-ing hard
		//let's go the oposite way
		//I'll use a buffer too large,  évaluate the density on the buffer particles and remove the particle with density too large
		//also no need to do it on all partilce only those close enught from the plane with the fluid end
		{

			*countRmv = 0;
			{
				int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
				if (x_motion) {
					DFSPH_evaluate_density_in_buffer_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
						countRmv, plane_pos_inf, plane_pos_sup);
				}
				else {
					DFSPH_evaluate_density_in_buffer_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
						countRmv, plane_pos_inf, plane_pos_sup);
				}

				gpuErrchk(cudaDeviceSynchronize());
			}
			//write it forthe viewer
			if (false) {
				std::ostringstream oss;
				int effective_count = 0;
				for (int i = 0; i < fluidBufferSet->numParticles; ++i) {
					uint8_t density = fminf(1.0f, (fluidBufferSet->densityAdv[i] / 1500.0f)) * 255;
					uint8_t alpha = 255;
					if (fluidBufferSet->densityAdv[i] >= 5000) {
						continue;

					}
					/*
					if (density_field_after_buffer[i] >= 0) {
						if (density == 0) {
							continue;
						}
						if (density > 245) {
							continue;
						}
					}
					//*/
					//uint8_t density = (density_field[i] > 500) ? 255 : 0;
					uint32_t txt = (((alpha << 8) + density << 8) + density << 8) + density;
					//*
					if (fluidBufferSet->densityAdv[i] < 1000) {
						txt = (((alpha << 8) + density << 8) + 0 << 8) + 0;
					}
					if (fluidBufferSet->densityAdv[i] > 1000) {
						txt = (((alpha << 8) + 0 << 8) + density << 8) + 0;
					}
					if (fluidBufferSet->densityAdv[i] > 1100) {
						txt = (((alpha << 8) + 0 << 8) + 0 << 8) + density;
					}
					if (fluidBufferSet->densityAdv[i] > 1150) {
						txt = (((alpha << 8) + 255 << 8) + 0 << 8) + 144;
					}
					//*/

					oss << pos_base[i].x << " " << pos_base[i].y << " " << pos_base[i].z << " "
						<< txt << std::endl;
					effective_count++;
				}

				std::ofstream myfile("densityCloud.pcd", std::ofstream::trunc);
				if (myfile.is_open())
				{
					myfile << "VERSION 0.7" << std::endl
						<< "FIELDS x y z rgb" << std::endl
						<< "SIZE 4 4 4 4" << std::endl
						<< "TYPE F F F U" << std::endl
						<< "COUNT 1 1 1 1" << std::endl
						<< "WIDTH " << effective_count << std::endl
						<< "HEIGHT " << 1 << std::endl
						<< "VIEWPOINT 0 0 0 1 0 0 0" << std::endl
						<< "POINTS " << effective_count << std::endl
						<< "DATA ascii" << std::endl;
					myfile << oss.str();
					myfile.close();
				}
			}

			//remove the tagged particle from the buffer (all yhe ones that have a density that is too high)
			//now we can remove the partices from the simulation
			// use the same process as when creating the neighbors structure to put the particles to be removed at the end
			cub::DeviceRadixSort::SortPairs(fluidBufferSet->neighborsDataSet->d_temp_storage_pair_sort, fluidBufferSet->neighborsDataSet->temp_storage_bytes_pair_sort,
				fluidBufferSet->neighborsDataSet->cell_id, fluidBufferSet->neighborsDataSet->cell_id_sorted,
				fluidBufferSet->neighborsDataSet->p_id, fluidBufferSet->neighborsDataSet->p_id_sorted, fluidBufferSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());

			cuda_sortData(*fluidBufferSet, fluidBufferSet->neighborsDataSet->p_id_sorted);
			gpuErrchk(cudaDeviceSynchronize());

			int new_num_particles = fluidBufferSet->numParticles - *countRmv;
			std::cout << "handle_fluid_boundries_cuda: changing num particles in the buffer: " << new_num_particles << "   nb removed : " << *countRmv << std::endl;
			fluidBufferSet->updateActiveParticleNumber(new_num_particles);
		}


		//a first version that is a simule mouvement of the particle that are inside the buffer so that 
		//they are slightlymore spread out to compensate the gap
		//the problem is that it results in a too low density for the buffer which iduce motion in the simulation
		if (false) {

			//first do a global reduction fo the gap by spreading the particles horizontally
			{
				*min_distance_buffer = 1;
				*min_distance_fluid = 1;
				gpuErrchk(cudaDeviceSynchronize());
				{
					int numBlocks = calculateNumBlocks(particleSet->numParticles);
					DFSPH_compute_gap_length_fluid_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, min_distance_fluid);
				}
				{
					int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
					DFSPH_compute_gap_length_buffer_kernel << <numBlocks, BLOCKSIZE >> > (data, fluidBufferSet->pos, fluidBufferSet->numParticles, min_distance_buffer);
				}
				gpuErrchk(cudaDeviceSynchronize());

				RealCuda gap_length = (*min_distance_buffer) + (*min_distance_fluid);
				std::cout << "the detected distance is total(bufer/fluid): " << gap_length << "  ( " << (*min_distance_buffer) <<
					"  //  " << (*min_distance_fluid) << " ) " << std::endl;

				//we want them tengent not having the particles overlapping
				gap_length -= data.particleRadius * 2;
				std::cout << "the detected gap after reduction: " << gap_length << std::endl;

				//and now reducce the gap
				RealCuda gap_pos = (-2.0 + GAP_PLANE_POS * data.getKernelRadius());
				std::cout << "dsfgdfg:  " << (gap_pos - min.x) << std::endl;
				{
					int numBlocks = calculateNumBlocks(particleSet->numParticles);
					DFSPH_reduce_gap_length_kernel << <numBlocks, BLOCKSIZE >> > (data, fluidBufferSet->pos, fluidBufferSet->numParticles,
						gap_length, *min_distance_buffer, (gap_pos - min.x));
				}
				gpuErrchk(cudaDeviceSynchronize());

			}

			//a test to see if wee indeed corected the gap
			{
				*min_distance_buffer = 1;
				*min_distance_fluid = 1;
				gpuErrchk(cudaDeviceSynchronize());
				{
					int numBlocks = calculateNumBlocks(particleSet->numParticles);
					DFSPH_compute_gap_length_fluid_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, min_distance_fluid);
				}
				{
					int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
					DFSPH_compute_gap_length_buffer_kernel << <numBlocks, BLOCKSIZE >> > (data, fluidBufferSet->pos, fluidBufferSet->numParticles, min_distance_buffer);
				}
				gpuErrchk(cudaDeviceSynchronize());
				RealCuda gap_length = (*min_distance_buffer) + (*min_distance_fluid);
				std::cout << "post test: the detected distance is total(buffer/fluid): " << gap_length << "  ( " << (*min_distance_buffer) <<
					"  //  " << (*min_distance_fluid) << " ) " << std::endl;

				//we want them tengent not having the particles overlapping
				gap_length -= data.particleRadius * 2;
				std::cout << "post test: the detected gap after reduction: " << gap_length << std::endl;

			}


			if (false) {
				//find the particle id for all particles close to the plane
				*count_near_plane_fluid = 0;
				*count_near_plane_buffer = 0;
				*count_buffer_displaced = 0;
				gpuErrchk(cudaDeviceSynchronize());
				{
					int numBlocks = calculateNumBlocks(particleSet->numParticles);
					DFSPH_get_fluid_particles_near_plane_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, count_near_plane_fluid, ids_near_plane_fluid);
				}
				{
					int numBlocks = calculateNumBlocks(particleSet->numParticles);
					DFSPH_get_buffer_particles_near_plane_kernel << <numBlocks, BLOCKSIZE >> > (data, fluidBufferSet->pos, fluidBufferSet->numParticles,
						count_near_plane_buffer, ids_near_plane_buffer);
				}
				gpuErrchk(cudaDeviceSynchronize());
				std::cout << "count particles near plane (buffer/fluid): " << "  ( " << (*count_near_plane_buffer) <<
					"  //  " << (*count_near_plane_fluid) << " ) " << std::endl;

				//a check to see the min distance
				{
					int numBlocks = calculateNumBlocks(*count_near_plane_buffer + *count_buffer_displaced);
					DFSPH_find_min_dist_near_plane_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->pos,
						ids_near_plane_buffer, *count_near_plane_buffer, ids_buffer_displaced, *count_buffer_displaced,
						ids_near_plane_fluid, *count_near_plane_fluid);
					gpuErrchk(cudaDeviceSynchronize());
					get_buffer_min_kernel << <1, 1 >> > (particleSet->densityAdv, test_out_real, (*count_near_plane_buffer + *count_buffer_displaced));
					gpuErrchk(cudaDeviceSynchronize());

					RealCuda min_dist = (*test_out_real) / data.particleRadius;

					std::cout << "detected min distance: " << min_dist << std::endl;

				}

				//now refit the position of the articles from the buffer that are close to the plane to fit the gap
				//first with a simple rule that move the particle left to fit it to the closest left particles from the right side of the plane
				//I'll use a loop to have and iterative process (maybe use a convergence condition for now I'll only use an iteration count)
				//*
				for (int i = 0; i < 1; ++i) {
					*test_out_int = 0;
					*test_out_real = 0;
					gpuErrchk(cudaDeviceSynchronize());
					{
						int numBlocks = calculateNumBlocks(*count_near_plane_buffer);
						DFSPH_fit_particles_simple_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->pos, *count_near_plane_buffer, *count_near_plane_fluid,
							ids_near_plane_buffer, ids_near_plane_fluid, *count_buffer_displaced, ids_buffer_displaced, test_out_int, test_out_real, i, fluidBufferSet->color);
					}
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "iteration " << i << " checking actual correction total (nbr corected/ sum_dist_corrected): " << "  ( " << (*test_out_int) <<
						"  //  " << (*test_out_real) / data.particleRadius << " ) " << std::endl;

					//now switch to particle that I moved to the displaced buffer so I can ajust the next ones
					int displacement_start = *count_buffer_displaced;
					int count_to_move = *test_out_int;
					{

						DFSPH_move_to_displaced_kernel << <1, 1 >> > (ids_near_plane_buffer, *count_near_plane_buffer, ids_buffer_displaced, displacement_start, count_to_move);
					}
					gpuErrchk(cudaDeviceSynchronize());
					*count_near_plane_buffer -= count_to_move;
					*count_buffer_displaced += count_to_move;

					//a check to see the min distance
					{
						int numBlocks = calculateNumBlocks(*count_near_plane_buffer);
						DFSPH_find_min_dist_near_plane_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->pos,
							ids_near_plane_buffer, *count_near_plane_buffer, ids_buffer_displaced, *count_buffer_displaced,
							ids_near_plane_fluid, *count_near_plane_fluid);
						gpuErrchk(cudaDeviceSynchronize());
						get_buffer_min_kernel << <1, 1 >> > (particleSet->densityAdv, test_out_real, (*count_near_plane_buffer + *count_buffer_displaced));
						gpuErrchk(cudaDeviceSynchronize());

						RealCuda min_dist = *test_out_real;

						std::cout << "detected min distance: " << min_dist / data.particleRadius << std::endl;

					}
				}
				//*/
			}
		}

		//I'll save the velocity field by setting the velocity of each particle to the weighted average of the three nearest
		{
			int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
			//DFSPH_init_buffer_velocity_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->pos, fluidBufferSet->vel, fluidBufferSet->numParticles);
			DynamicWindowBuffer::init_buffer_kernel << <numBlocks, BLOCKSIZE >> > (fluidBufferSet->vel, fluidBufferSet->numParticles, Vector3d(0, 0, 0));
			gpuErrchk(cudaDeviceSynchronize());
		}

		//now we can remove the partices from the simulation
		{

			*countRmv = 0;
			int numBlocks = calculateNumBlocks(particleSet->numParticles);
			if (x_motion) {
				DFSPH_reset_fluid_boundaries_remove_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, countRmv, plane_pos_inf, plane_pos_sup);
			}
			else {
				DFSPH_reset_fluid_boundaries_remove_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, countRmv, plane_pos_inf, plane_pos_sup);
			}
			gpuErrchk(cudaDeviceSynchronize());



			//*
			//now use the same process as when creating the neighbors structure to put the particles to be removed at the end
			cub::DeviceRadixSort::SortPairs(particleSet->neighborsDataSet->d_temp_storage_pair_sort, particleSet->neighborsDataSet->temp_storage_bytes_pair_sort,
				particleSet->neighborsDataSet->cell_id, particleSet->neighborsDataSet->cell_id_sorted,
				particleSet->neighborsDataSet->p_id, particleSet->neighborsDataSet->p_id_sorted, particleSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());

			cuda_sortData(*particleSet, particleSet->neighborsDataSet->p_id_sorted);
			gpuErrchk(cudaDeviceSynchronize());

			//and now you can update the number of particles


			int new_num_particles = particleSet->numParticles - *countRmv;
			std::cout << "handle_fluid_boundries_cuda: changing num particles: " << new_num_particles << "   nb removed : " << *countRmv << std::endl;
			particleSet->updateActiveParticleNumber(new_num_particles);
			//*/
		}

		{//and now add the buffer back into the simulation
			//check there is enougth space for the particles and make some if necessary
			int new_num_particles = particleSet->numParticles + fluidBufferSet->numParticles;
			if (new_num_particles > particleSet->numParticlesMax) {
				particleSet->changeMaxParticleNumber(new_num_particles * 1.25);
			}

			//add the particle in the simulation
			int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
			DFSPH_reset_fluid_boundaries_add_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->gpu_ptr);
			gpuErrchk(cudaDeviceSynchronize());

			//and change the number
			std::cout << "handle_fluid_boundries_cuda: changing num particles: " << new_num_particles << "   nb added : " << fluidBufferSet->numParticles << std::endl;
			particleSet->updateActiveParticleNumber(new_num_particles);

		}

		//still need the damping near the borders as long as we don't implement the implicit borders
		//with paricle boundaries 3 is the lowest number of steps that absorb nearly any visible perturbations
		add_border_to_damp_planes_cuda(data, x_motion, !x_motion);
		data.damp_borders_steps_count = 3;
		data.damp_borders = true;
	}
}

