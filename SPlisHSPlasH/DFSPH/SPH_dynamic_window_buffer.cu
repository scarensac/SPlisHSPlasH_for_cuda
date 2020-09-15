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





using namespace SPH;

//this macro is juste so that the expression get optimized at the compilation 
//x_motion should be a bollean comming from a template configuration of the function where this macro is used
#define VECTOR_X_MOTION(pos,x_motion) ((x_motion)?pos.x:pos.z)


#define GAP_PLANE_POS 3



__device__ void atomicToMin(RealCuda* addr, RealCuda value)
{
#ifndef USE_DOUBLE_CUDA
	RealCuda old = *addr;
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
#else
	asm("trap;");
#endif
}

__device__ void atomicToMax(RealCuda* addr, RealCuda value)
{
#ifndef USE_DOUBLE_CUDA
	RealCuda old = *addr;
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
#else
	asm("trap;");
#endif
}





namespace SPH {
	class DynamicWindow {

	public:
		bool initialized;

		//this buffer contains a set a particle corresponding to a fluid at rest covering the whole simulation space
		UnifiedParticleSet* backgroundFluidBufferSet ;
		Vector3d* pos_background ;
		int numParticles_background;

		//the particle from the fluid buffer
		UnifiedParticleSet* fluidBufferSetFromSurface ;
		Vector3d* pos_base_from_surface ;
		int numParticles_base_from_surface;
		//this is another formalim that I need while transfering the code (in the end I'll only kee this one)
		SPH::UnifiedParticleSet* fluidBufferSet;
		Vector3d* pos_base;
		int numParticles_base;


		//the object for the surface
		BufferFluidSurface S_initial;
		BufferFluidSurface S;

		//to represent the state of the ocean
		BorderHeightMap borderHeightMap;
	
	public:
		DynamicWindow() {
			initialized = false;

			backgroundFluidBufferSet = NULL;
			pos_background = NULL;
			numParticles_background = 0;

			fluidBufferSetFromSurface = NULL;
			pos_base_from_surface = NULL;
			numParticles_base_from_surface = 0;
		}

		~DynamicWindow() {

		}

		static DynamicWindow& getStructure() {
			static DynamicWindow dwb;
			return dwb;
		}

		bool isInitialized() { return initialized; }

		void init(DFSPHCData& data);

		//reset and move the structure depending on the displacement
		void initStep(DFSPHCData& data, Vector3d movement, bool init_buffers_neighbors = true, bool init_fluid_neighbors = true);

		//this function remove the unused particles from the background and fluid buffers
		//It also make sure that the back ground and the fluid buffer have no particle in common which is necessary for further computations
		void lightenBuffers(DFSPHCData& data);


		//this function remove some particles from the flui buffer so that it fit in the space where we will remoe fluid particles
		void fitFluidBuffer(DFSPHCData& data);


		//set the velocites of the particles inside the fluid buffer
		void computeFluidBufferVelocities(DFSPHCData& data);

		//place the fluid buffer paticles inside the simulation
		//this fnction also remove any existing fluid particle that are at the buffer location
		void addFluidBufferToSimulation(DFSPHCData& data);

		//applys some particle shifting near the transition surface
		// particle shifting from particle concentration
		// Formula found in the fllowing paper (though it's when they explain the earlier works)
		// A multi - phase particle shifting algorithm for SPHsimulations of violent hydrodynamics with a largenumber of particles
		// with cancelling of the shifting near the fluid- air surface
		void applyParticleShiftNearSurface(DFSPHCData& data);

		void clear() {
			///TODO: CLEAR all the buffers


			initialized = false;
		}

		// this fnction hadnle the operations needed we I ask to move or reset the dynamic buffer (for a reset just specify no movement 
		void handleFluidBoundaries(SPH::DFSPHCData& data, Vector3d movement);

		//this function is here to handle a simplified version of the open boundaries
		//the goal is to be completely separated from the simulation step
		void handleOceanBoundariesTestCurrent(SPH::DFSPHCData& data);

		//ok here I'll test a system to initialize a volume of fluid from
		//a large wolume of fluid (IE a technique to iinit the fluid at rest)
		void initializeFluidToSurface(SPH::DFSPHCData& data);
		


		BufferFluidSurface& getSurface() {
			return S;
		}

		//This function set the surface position ralative to the initial surface position 
		//WARNING: if you have for some strange reason modified the surface this function will erase all your modification
		void setSurfacePositionRelativeToInitial(Vector3d pos) {
			S_initial.copy(S);
			S.move(pos);
		}

	};

}



void DynamicWindowInterface::initDynamicWindow(DFSPHCData& data) {
	DynamicWindow::getStructure().init(data);
}

bool DynamicWindowInterface::isInitialized() {
	return DynamicWindow::getStructure().isInitialized();
}

void DynamicWindowInterface::handleFluidBoundaries(SPH::DFSPHCData& data, Vector3d movement ) {
	DynamicWindow::getStructure().handleFluidBoundaries(data, movement);
}

void DynamicWindowInterface::clearDynamicWindow() {
	DynamicWindow::getStructure().clear();
}


void DynamicWindowInterface::handleOceanBoundariesTest(SPH::DFSPHCData& data) {
	//DynamicWindow::getStructure().handleOceanBoundariesTest(data); 
	DynamicWindow::getStructure().handleOceanBoundariesTestCurrent(data);
}

/*
void DynamicWindowInterface::initializeFluidToSurface(SPH::DFSPHCData& data) {
	DynamicWindow::getStructure().initializeFluidToSurface(data);
}
//*/





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

	//this line add some decay to the velocity so taht It will converge to 0 at rest
	sum_dist *= 1.1;

	vel[i] = weighted_vel / sum_dist;

#undef num_neighbors
}

__global__ void DFSPH_reset_fluid_boundaries_remove_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, int* countRmv,
	BufferFluidSurface S, bool keep_inside) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	//*
	particleSet->neighborsDataSet->cell_id[i] = 0;
	bool keep = S.isinside(particleSet->pos[i]);
	keep = (keep_inside) ? keep : (!keep);
	if (!keep) {
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
	particleSet->mass[particleSet->numParticles + i] = fluidBufferSet->getMass(i);
	particleSet->color[particleSet->numParticles + i] = fluidBufferSet->color[i];

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
		RealCuda density_delta = fluidSet->getMass(j) * KERNEL_W(data, sampling_point - fluidSet->pos[j]);
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
		RealCuda density_delta = bufferSet->getMass(j) * KERNEL_W(data, sampling_point - bufferSet->pos[j]);
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
			RealCuda density_delta = data.boundaries_data_cuda->getMass(j) * KERNEL_W(data, sampling_point - data.boundaries_data_cuda->pos[j]);
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


//run the one below at least once before this one (to rmv the buffer that would be above the fluid)
__global__ void DFSPH_evaluate_density_from_buffer_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
	SPH::UnifiedParticleSet* backgroundBufferSet, SPH::UnifiedParticleSet* bufferSet, BufferFluidSurface S) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }

	//do not do useless computation on particles that have already been taged for removal
	int removal_tag = 25000000;
	if (bufferSet->neighborsDataSet->cell_id[i] == removal_tag) {
		return;
	}

	Vector3d p_i = bufferSet->pos[i];

	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);


	RealCuda density_after_buffer = bufferSet->getMass(i) * data.W_zero;
	RealCuda density = 0;

	int count_neighbors = 0;


	//compute the fluid contribution
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
		RealCuda density_delta = fluidSet->getMass(j) * KERNEL_W(data, p_i - fluidSet->pos[j]);
	if (density_delta > 0) {
		if (S.isinside(fluidSet->pos[j])) {
			density += density_delta;
			count_neighbors++;
		}
	}
	);


	//keep_particle = true;
	{
		//*
		//compute the buffer contribution
		{
			//on the following passes I do the computations using the neighbors from the buffer
			//ze need to ingore pqrticles that have been tagged for removal

			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
				if (i != j) {
					if (bufferSet->neighborsDataSet->cell_id[j] != removal_tag) {
						RealCuda density_delta = bufferSet->getMass(j) * KERNEL_W(data, p_i - bufferSet->pos[j]);
						density_after_buffer += density_delta;
						count_neighbors++;
					}
				}
			);

			//also has to iterate over the background buffer that now represent the air
			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(backgroundBufferSet->neighborsDataSet, backgroundBufferSet->pos,
				RealCuda density_delta = backgroundBufferSet->getMass(j) * KERNEL_W(data, p_i - backgroundBufferSet->pos[j]);
			density_after_buffer += density_delta;
			count_neighbors++;
			);
		}
		//*/

		//compute the boundaries contribution only if there is a fluid particle anywhere near
		//*
		if (count_neighbors > 0) {

			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
				RealCuda density_delta = data.boundaries_data_cuda->getMass(j) * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
			density += density_delta;
			);
		}
		//*/
	}

	density_after_buffer += density;
	if (true) {
		bufferSet->density[i] = density_after_buffer;
		bufferSet->densityAdv[i] = density;
		//bufferSet->color[i] = Vector3d(1, 0, 0);//Vector3d((count_neighbors * 3)/255.0f, 0, 0);
	}
}


//technically only the particles aorund the plane have any importance
//so no need to compute the density for any particles far from the plances
//also unless I want to save it I don't actually need to outpu the density
// I set those optimisations statically so that the related ifs get optimized
__global__ void DFSPH_evaluate_and_tag_high_density_from_buffer_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
	SPH::UnifiedParticleSet* backgroundBufferSet, SPH::UnifiedParticleSet* bufferSet, int* countRmv,
	BufferFluidSurface S, RealCuda limit_density) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }

	//do not do useless computation on particles that have already been taged for removal
	int removal_tag = 25000000;
	if (bufferSet->neighborsDataSet->cell_id[i] == removal_tag) {
		return;
	}

	Vector3d p_i = bufferSet->pos[i];

	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);


	RealCuda density_after_buffer = bufferSet->getMass(i) * data.W_zero;
	RealCuda density = 0;

	int count_neighbors = 0;
	bool keep_particle = true;

	//*
	keep_particle = false;


	//compute the fluid contribution
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
		RealCuda density_delta = fluidSet->getMass(j) * KERNEL_W(data, p_i - fluidSet->pos[j]);
	if (density_delta > 0) {
		if (S.isinside(fluidSet->pos[j])) {
			density += density_delta;
			count_neighbors++;
			if (fluidSet->pos[j].y > (p_i.y)) {
				keep_particle = true;
			}
		}
	}
	);

	keep_particle = keep_particle || (!S.isinside(p_i));


	//keep_particle = true;
	if (keep_particle) {
		//*
		//compute the buffer contribution
		//if (iter == 0)
		if (false) {
			//Note: the strange if is just here to check if p_i!=p_j but since doing this kind of operations on float
			//		that are read from 2 files might not be that smat I did some bastard check verifying if the particle are pretty close
			//		since the buffer we are evaluating is a subset of the background buffer, the only possibility to have particle 
			//		that close is that they are the same particle.
			//*
			//no need for that since I lighten the buffers before now
			float limit = data.particleRadius / 10.0f;
			limit *= limit;

			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(backgroundBufferSet->neighborsDataSet, backgroundBufferSet->pos,
				if ((p_i - backgroundBufferSet->pos[j]).squaredNorm() > (limit)) {
					RealCuda density_delta = backgroundBufferSet->getMass(j) * KERNEL_W(data, p_i - backgroundBufferSet->pos[j]);
					density_after_buffer += density_delta;
					count_neighbors++;
				}
			);
			//*/
		}
		else {
			//on the following passes I do the computations using the neighbors from the buffer
			//ze need to ingore pqrticles that have been tagged for removal

			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
				if (i != j) {
					if (bufferSet->neighborsDataSet->cell_id[j] != removal_tag) {
						RealCuda density_delta = bufferSet->getMass(j) * KERNEL_W(data, p_i - bufferSet->pos[j]);
						density_after_buffer += density_delta;
						count_neighbors++;
					}
				}
			);

			//also has to iterate over the background buffer that now represent the air
			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(backgroundBufferSet->neighborsDataSet, backgroundBufferSet->pos,
				RealCuda density_delta = backgroundBufferSet->getMass(j) * KERNEL_W(data, p_i - backgroundBufferSet->pos[j]);
			density_after_buffer += density_delta;
			count_neighbors++;
			);
		}
		//*/

		//compute the boundaries contribution only if there is a fluid particle anywhere near
		//*
		if (count_neighbors > 0) {

			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
				RealCuda density_delta = data.boundaries_data_cuda->getMass(j) * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
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
			//bufferSet->color[i] = Vector3d(1, 0, 0);//Vector3d((count_neighbors * 3)/255.0f, 0, 0);
		}

		//that line is just an easy way to recognise the plane
		//samples_after_buffer[i] *= (((layer_id == 0)&&(density_after_buffer>500)) ? -1 : 1);


		keep_particle = (density_after_buffer) < limit_density;


		if (!keep_particle) {
			atomicAdd(countRmv, 1);
			bufferSet->neighborsDataSet->cell_id[i] = removal_tag;
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
		bufferSet->neighborsDataSet->cell_id[i] = removal_tag;
	}
}

__global__ void DFSPH_save_low_density_neighbors_from_buffer_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
	SPH::UnifiedParticleSet* backgroundBufferSet, SPH::UnifiedParticleSet* bufferSet, int* countRmv,
	BufferFluidSurface S, RealCuda limit_density) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }

	if(false){
		//this try to save the neighborhood a low densities particles but it does not realy work 
		//partly due to the fact I do not actually exactly evaluate the densities
		if (bufferSet->density[i] > limit_density) { return; }

		int removal_tag = 25000000;
		Vector3d p_i = bufferSet->pos[i];
		ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);

		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
			if (i != j) {
				if (bufferSet->neighborsDataSet->cell_id[j] == removal_tag) {
					//RealCuda density_delta = bufferSet->getMass(j) * KERNEL_W(data, p_i - bufferSet->pos[j]);
					bufferSet->neighborsDataSet->cell_id[j] = 0;
					atomicAdd(countRmv, 1);
				}
			}
		);
	}

	
	if(false){	
		//here I'll re�valuate the density of removed particles to see i I can reinstate it
		//this case never happens
		int removal_tag = 25000000;
		if (bufferSet->neighborsDataSet->cell_id[i] != removal_tag) { return; }
		
		
		Vector3d p_i = bufferSet->pos[i];
		int count_neighbors = 0;
		RealCuda density = 0;
		RealCuda density_after_buffer = bufferSet->getMass(i) * data.W_zero;
		bool keep_particle = false;
		ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);

		//compute the fluid contribution
		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
			RealCuda density_delta = fluidSet->getMass(j) * KERNEL_W(data, p_i - fluidSet->pos[j]);
		if (density_delta > 0) {
			if (S.isinside(fluidSet->pos[j])) {
				density += density_delta;
				count_neighbors++;
				if (fluidSet->pos[j].y > (p_i.y)) {
					keep_particle = true;
				}
			}
		}
		);

		if (keep_particle){
			//on the following passes I do the computations using the neighbors from the buffer
			//ze need to ingore pqrticles that have been tagged for removal

			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
				if (i != j) {
					if (bufferSet->neighborsDataSet->cell_id[j] != removal_tag) {
						RealCuda density_delta = bufferSet->getMass(j) * KERNEL_W(data, p_i - bufferSet->pos[j]);
						density_after_buffer += density_delta;
						count_neighbors++;
					}
				}
			);

			//also has to iterate over the background buffer that now represent the air
			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(backgroundBufferSet->neighborsDataSet, backgroundBufferSet->pos,
				RealCuda density_delta = backgroundBufferSet->getMass(j) * KERNEL_W(data, p_i - backgroundBufferSet->pos[j]);
			density_after_buffer += density_delta;
			count_neighbors++;
			);

			//compute the boundaries contribution only if there is a fluid particle anywhere near
			//*
			if (count_neighbors > 0) {

				ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
					RealCuda density_delta = data.boundaries_data_cuda->getMass(j) * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
				density += density_delta;
				);
			}

			if ((density + density_after_buffer) < limit_density) {
				bufferSet->neighborsDataSet->cell_id[i] = 0;
				atomicAdd(countRmv, 1);
			}
		}
		//*/

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
	if ((dist<data.particleRadius)&&(backgroundSet->pos[i].y<0.8)) {
		atomicAdd(count, 1);
	}
	else {
		backgroundSet->neighborsDataSet->cell_id[i] += CELL_COUNT;
	}
}



//I want to only keep particles that are above the fluid or above the buffer
//also remove the buffer particles that are above the fluid
//WARNING:	this function make full use of the fact that the fluid buffer is a subset of the background
//			Specificaly it needs to be composed of the first particles of the background and the orders of the particles must be the same 
__global__ void DFSPH_lighten_buffers_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
	SPH::UnifiedParticleSet* backgroundBufferSet, SPH::UnifiedParticleSet* bufferSet, int* countRmvBackground,
	int* countRmvBuffer, BufferFluidSurface S, BorderHeightMap borderHeightMap) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= backgroundBufferSet->numParticles) { return; }


	Vector3d p_i = backgroundBufferSet->pos[i];


	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);


	bool keep_particle_background = true;
	bool keep_particle_buffer = (i<bufferSet->numParticles);//since the buffer is subset of the background
	//*
	RealCuda dist = S.distanceToSurfaceSigned(p_i);
	
	/*
	//quick code that verify that the buffer is a correct subset of the background
	dist = -1;
	if (keep_particle_buffer) {
		Vector3d v = p_i - bufferSet->pos[i];
		if (v.norm() > (data.particleRadius / 10.0)) {
			keep_particle_buffer = false;
		}
	}
	//*/

	if (dist>0) {
		//if the particle is too far inside the fluid we can discard it
		//the bufffer currently expends 1 kernel radius inside the fluid so anything further than 2 kernel radius
		//can be removed from the background
		if (dist>data.getKernelRadius()*3) {
			keep_particle_background = false;
		}

		bool is_buffer_particle_under_fluid = false;
		
		if (keep_particle_background) {
			//for the buffer we want the particle to be under the fluid
			//for the back ground we want only above the fluid (or at least realyc close from the fluid surface
			//note:	if it is part of the buffer and under the fluid we can stop the computation because it mean we have to discard
			//		it from the background

			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
				if (!is_buffer_particle_under_fluid) {
					if (S.isinside(fluidSet->pos[j])) {
						if (keep_particle_buffer) {
							if (fluidSet->pos[j].y > (p_i.y)) {
								is_buffer_particle_under_fluid = true;
							}
						}
					
						//*
						if (keep_particle_background) {
							int nb_neighbors = fluidSet->getNumberOfNeighbourgs(0) + fluidSet->getNumberOfNeighbourgs(1);
							if ((nb_neighbors > 15)) {
								Vector3d delta = fluidSet->pos[j] - p_i;
								RealCuda dh = delta.y;
								delta.y = 0;
								if (dh > (2 * data.particleRadius)) {
								//if ((delta.norm() < 2 * data.particleRadius) && (dh > (2 * data.particleRadius))) {
										keep_particle_background = false;
								}
							}
						}
						//*/
					}
				}
			);
			
			//if it's a buffer particle only keep it if it  is under the fluid
			keep_particle_buffer &= is_buffer_particle_under_fluid;

		}

	}
	else {
		//if the particle position is inside the area that is occupied by buffer particle at the end of the procedure
		//we need to know if it is above the ocean elevation(in wich case it will be removed)
		if (keep_particle_buffer) {
			if (p_i.y > borderHeightMap.getHeight(p_i)) {
				keep_particle_buffer = false;
			}
		}
	}

	//if we keep it in the buffer it must be removed from the background
	keep_particle_background &= (!keep_particle_buffer);

	if (!keep_particle_buffer) {
		if (i < bufferSet->numParticles) {
			atomicAdd(countRmvBuffer, 1);
			bufferSet->neighborsDataSet->cell_id[i] = 25000000;
		}
	}
	else {
		bufferSet->neighborsDataSet->cell_id[i] = 0;
	}

	if (!keep_particle_background) {
		atomicAdd(countRmvBackground, 1);
		backgroundBufferSet->neighborsDataSet->cell_id[i] = 25000000;
	}
	else {
		backgroundBufferSet->neighborsDataSet->cell_id[i] = 0;
	}
}


 
__global__ void DFSPH_evaluate_density_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet, BufferFluidSurface S) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= fluidSet->numParticles) { return; }


	Vector3d p_i = fluidSet->pos[i];


	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);


	bool keep_particle = true;
	//*
	RealCuda sq_diameter = data.particleRadius * 2;
	sq_diameter *= sq_diameter;
	int count_neighbors = 0;
	RealCuda density = fluidSet->getMass(i) * data.W_zero;


	//check if there is any fluid particle above us
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
		if (i != j) {
			//RealCuda density_delta = (fluidSet->pos[j]-p_i).norm();
			RealCuda density_delta = fluidSet->getMass(j) * KERNEL_W(data, p_i - fluidSet->pos[j]);
			density += density_delta;
			count_neighbors++;
		}
	);

	//*
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		RealCuda density_delta = data.boundaries_data_cuda->getMass(j) * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
		density += density_delta;
		count_neighbors++;
	);
	//*/
	//tag the surface
	fluidSet->neighborsDataSet->cell_id[i] = 0;
	if ((count_neighbors) < 30) {
		fluidSet->neighborsDataSet->cell_id[i] = 100;

	}

	fluidSet->density[i] = density;
}


__global__ void DFSPH_evaluate_particle_concentration_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
	BufferFluidSurface S) {
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
			RealCuda concentration_delta = (fluidSet->getMass(j)/ fluidSet->density[j])* KERNEL_W(data, p_i - fluidSet->pos[j]);
			concentration += concentration_delta;
			count_neighbors++;
		}
	);

	//supose that the density of boundaries particles is the rest density
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		RealCuda concentration_delta = (data.boundaries_data_cuda->getMass(j) / data.density0) * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
	concentration += concentration_delta;
	count_neighbors++;
	);

	

	fluidSet->densityAdv[i] = concentration;
}

__global__ void DFSPH_particle_shifting_base_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet, RealCuda displacement_coefficient,
	BufferFluidSurface S, int* count_affected, RealCuda* total_abs_displacement) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= fluidSet->numParticles) { return; }



	bool x_motion = true;
	Vector3d p_i = fluidSet->pos[i];

	//only move particles that are close to the planes
	RealCuda affected_range = data.getKernelRadius();
	RealCuda dist_to_surface = S.distanceToSurface(p_i);
	if (dist_to_surface>affected_range) {
		return;
	}

	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);

	RealCuda scaling = 1 - dist_to_surface / affected_range;

	//*
	RealCuda sq_diameter = data.particleRadius * 2;
	sq_diameter *= sq_diameter;
	int count_neighbors = 0;
	//we cna start at 0 and ignire the i contribution because we will do a sustracction when computing the concentration gradiant
	Vector3d displacement = Vector3d(0,0,0);

	RealCuda surface_factor = 0;
	if (fluidSet->neighborsDataSet->cell_id[i] > 50) {
		surface_factor += data.W_zero;
	}


	//check if there is any fluid particle above us
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
		if (i != j) {
			Vector3d displacement_delta = (fluidSet->densityAdv[j]- fluidSet->densityAdv[i])* (fluidSet->getMass(j) / fluidSet->density[j]) * KERNEL_GRAD_W(data, p_i - fluidSet->pos[j]);
			/*
			Vector3d nj = fluidSet->pos[j] - p_i;
			nj.toUnit();
			Vector3d displacement_delta = fluidSet->density[j]*KERNEL_W(data, p_i - fluidSet->pos[j])*nj;
			//*/
			displacement += displacement_delta;
			count_neighbors++;

			if (fluidSet->neighborsDataSet->cell_id[j] > 50) {
				surface_factor += KERNEL_W(data, p_i - fluidSet->pos[j]);
			}
		}
	);

	//as long as I make so that the surface is more than 1 kernel radius from the boundaries those computation are fine no need to iterate on the boundaries
	//so I'll prevent shifting the particles that are even remotely clse from the boundary
	int count_neighbors_b = 0;
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		count_neighbors_b++;

	);

	displacement *= -displacement_coefficient;
	//we cap the displacement like in the papers
	RealCuda disp_norm = displacement.norm();
	if (disp_norm > (0.2 * data.getKernelRadius())) {
		displacement *= (0.2 * data.getKernelRadius()) / disp_norm;
	}

	surface_factor /= data.W_zero;
	surface_factor = MAX_MACRO_CUDA(surface_factor, 1);
	surface_factor = 1 - surface_factor;


	//a scaling so that the particle that are the most displaced are those near the plane
	displacement *= scaling * scaling;
	//displacement.y *= surface_factor ;
	
	atomicAdd(count_affected, 1);
	atomicAdd(total_abs_displacement, displacement.norm());

	if (count_neighbors_b ==0) {
		fluidSet->pos[i]+=displacement;
	}
}

__global__ void DFSPH_particle_shifting_density_grid_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet, 
	BufferFluidSurface S, Vector3d* positions, RealCuda* densities, int count_samples) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= fluidSet->numParticles) { return; }

	Vector3d p_i = fluidSet->pos[i];
	Vector3d den_i = fluidSet->density[i];

	//only move particles that are close to the planes
	RealCuda affected_range = data.getKernelRadius();
	RealCuda dist_to_surface = S.distanceToSurface(p_i);
	if (dist_to_surface > affected_range) {
		return;
	}

	RealCuda radius_sq = data.getKernelRadius(); 
	radius_sq *= radius_sq;
	
	int target = -1;
	int den_target = fluidSet->density[i];
	for (int j = 0; j < count_samples; ++j) {
		if ((positions[j] - p_i).squaredNorm() < radius_sq) {
			if (densities[j] < den_target) {
				den_target = densities[j];
				target = j;
			}
		}
	}

	

	if (target >= 0) {
		Vector3d dp= (positions[target] - p_i);
		//*
		RealCuda coef = 1.0;
		//coef=KERNEL_W(data, dp) / data.W_zero * data.particleRadius;
		//coef *= 1000000;
		coef = data.particleRadius/8;
		coef = MIN_MACRO_CUDA(coef, dp.norm()/2);
		dp.toUnit();
		dp *= coef;
		//*/
		//dp /= 2;

		fluidSet->pos[i] += dp;
	}
}


void DynamicWindow::init(DFSPHCData& data) {
	if (initialized) {
		throw("DynamicWindow::init the structure has already been initialized");
	}

	int surfaceType = 1;
	//define the surface
	if (surfaceType == 0) {
		/*
		std::ostringstream oss;
		oss << "DynamicWindow::init The initialization for this type of surface has not been defined (type=" << surfaceType << std::endl;
		throw(oss.str());
		//*/
		//S_initial.setPlane(Vector3d(-1.8, 0, 0), Vector3d(1, 0, 0));
		S_initial.setPlane(Vector3d(-7.8, 0, 0), Vector3d(1, 0, 0));
		
	}
	else if (surfaceType == 1) {

		//*
		Vector3d center(0, 0, 0);
		Vector3d halfLength(100);
		halfLength.x = 1.6;
		//halfLength.z = 0.4;
		S_initial.setCuboid(center, halfLength);
		//*/
		/*
		Vector3d center(0, 0.3, 0);
		Vector3d halfLength(100);
		halfLength.x = 0.6;
		halfLength.z = 0.4;
		halfLength.y = 0.3;
		//halfLength.z = 0.4;
		S_initial.setCuboid(center, halfLength);
		//*/

	}
	else if (surfaceType == 2) {
		Vector3d center(0, 0, 0);
		S_initial.setCylinder(center, 10, 2-0.2);
	}
	else {
		std::ostringstream oss;
		oss << "DynamicWindow::init The initialization for this type of surface has not been defined (type=" << surfaceType << std::endl;
		throw(oss.str());
	}
	std::cout << "Initial surface description: " << S_initial.toString() << std::endl;

	//test the distance computation
	if (false) {
		Vector3d test_pt(0, 0, 0);
		std::cout << "distance to surface: " << S_initial.distanceToSurface(test_pt) << "   " << S_initial.distanceToSurfaceSigned(test_pt) << std::endl;
		test_pt = Vector3d(40, 40, 40);
		std::cout << "distance to surface: " << S_initial.distanceToSurface(test_pt) << "   " << S_initial.distanceToSurfaceSigned(test_pt) << std::endl;
	}


	Vector3d min_fluid_buffer, max_fluid_buffer;
	//load the backgroundset
	{
		SPH::UnifiedParticleSet* dummy = NULL;
		backgroundFluidBufferSet = new SPH::UnifiedParticleSet();
		backgroundFluidBufferSet->load_from_file(data.fluid_files_folder + "background_buffer_file.txt", false, &min_fluid_buffer, &max_fluid_buffer, false);
		allocate_and_copy_UnifiedParticleSet_vector_cuda(&dummy, backgroundFluidBufferSet, 1);

		backgroundFluidBufferSet->initNeighborsSearchData(data, true);
		backgroundFluidBufferSet->resetColor();

		numParticles_background = backgroundFluidBufferSet->numParticles;
		cudaMallocManaged(&(pos_background), sizeof(Vector3d) * numParticles_background);
		gpuErrchk(cudaMemcpy(pos_background, backgroundFluidBufferSet->pos, numParticles_background * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

		if (false) {
			//save the background as a point cloud for debug
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
		//first we count the nbr of particles and attribute the index for sorting
		//also we will reorder the background buffer so we need to resave its state
		int* out_int = NULL;
		cudaMallocManaged(&(out_int), sizeof(int));
		*out_int = 0;
		{
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			DFSPH_generate_buffer_from_surface_count_particles_kernel << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, S_initial, out_int);
			gpuErrchk(cudaDeviceSynchronize());
		}
		int count_inside_buffer = *out_int;
		CUDA_FREE_PTR(out_int);



		//sort the buffer
		cub::DeviceRadixSort::SortPairs(backgroundFluidBufferSet->neighborsDataSet->d_temp_storage_pair_sort, backgroundFluidBufferSet->neighborsDataSet->temp_storage_bytes_pair_sort,
			backgroundFluidBufferSet->neighborsDataSet->cell_id, backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted,
			backgroundFluidBufferSet->neighborsDataSet->p_id, backgroundFluidBufferSet->neighborsDataSet->p_id_sorted, backgroundFluidBufferSet->numParticles);
		gpuErrchk(cudaDeviceSynchronize());

		cuda_sortData(*backgroundFluidBufferSet, backgroundFluidBufferSet->neighborsDataSet->p_id_sorted);
		gpuErrchk(cudaDeviceSynchronize());

		//resave the background state
		gpuErrchk(cudaMemcpy(pos_background, backgroundFluidBufferSet->pos, numParticles_background * sizeof(Vector3d), cudaMemcpyDeviceToDevice));



		//and now we can create the buffer and save the positions
		SPH::UnifiedParticleSet* dummy = NULL;
		fluidBufferSetFromSurface = new SPH::UnifiedParticleSet();
		fluidBufferSetFromSurface->init(count_inside_buffer, true, true, false, true);
		allocate_and_copy_UnifiedParticleSet_vector_cuda(&dummy, fluidBufferSetFromSurface, 1);


		numParticles_base_from_surface = fluidBufferSetFromSurface->numParticles;
		cudaMallocManaged(&(pos_base_from_surface), sizeof(Vector3d) * numParticles_base_from_surface);
		gpuErrchk(cudaMemcpy(pos_base_from_surface, pos_background, numParticles_base_from_surface * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(fluidBufferSetFromSurface->mass, backgroundFluidBufferSet->mass, numParticles_base_from_surface * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

		fluidBufferSetFromSurface->resetColor();


		//the other formalism
		fluidBufferSet = fluidBufferSetFromSurface;
		pos_base = pos_base_from_surface;
		numParticles_base = numParticles_base_from_surface;


		//save the buffer to a point cloud for debug
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

	//init the struture used to define the elevation of the ocean
	{
		borderHeightMap.init(data);
	}


	initialized = true;
}


void DynamicWindow::initStep(DFSPHCData& data, Vector3d movement, bool init_buffers_neighbors, bool init_fluid_neighbors) {
	UnifiedParticleSet* particleSet = data.fluid_data;


	//reset the buffers particle set
	fluidBufferSet->updateActiveParticleNumber(numParticles_base);
	backgroundFluidBufferSet->updateActiveParticleNumber(numParticles_background);
	gpuErrchk(cudaMemcpy(fluidBufferSet->pos, pos_base, numParticles_base * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(backgroundFluidBufferSet->pos, pos_background, numParticles_background * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
	//particleSet->resetColor();
	//fluidBufferSet->resetColor();
	S.copy(S_initial);


	//only bother displacing the elements if there is an acutal displacement
	if (movement.norm() > 0.0005) {
		//update the displacement offset
		Vector3d mov_pos = movement * data.getKernelRadius();
		data.dynamicWindowTotalDisplacement += mov_pos;
		data.gridOffset -= movement;

		//first displace the boundaries
		SPH::UnifiedParticleSet* particleSetMove = data.boundaries_data;
		unsigned int numParticles = particleSetMove->numParticles;
		int numBlocks = calculateNumBlocks(numParticles);
		apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (particleSetMove->pos, mov_pos, numParticles);
		

	}

	//thoise some buffers depends on the totla displacement and might have to be displaced even though there is no displacement for this step
	if (data.dynamicWindowTotalDisplacement.norm() > 0.0005) {
		std::cout << "Current total displacement since start: " << data.dynamicWindowTotalDisplacement.toString() << std::endl;

		//move the surface
		S.move(data.dynamicWindowTotalDisplacement);
		std::cout << S.toString() << std::endl;

		//and the buffers
		//carefull since they are reset you must displace them for the full displacment since the start of the simulation
		SPH::UnifiedParticleSet* particleSetMove = backgroundFluidBufferSet;
		unsigned int numParticles = particleSetMove->numParticles;
		int numBlocks = calculateNumBlocks(numParticles);
		apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (particleSetMove->pos, data.dynamicWindowTotalDisplacement, numParticles);


		particleSetMove = fluidBufferSet;
		numParticles = particleSetMove->numParticles;
		numBlocks = calculateNumBlocks(numParticles);
		apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (particleSetMove->pos, data.dynamicWindowTotalDisplacement, numParticles);
		gpuErrchk(cudaDeviceSynchronize());
	}

	//we now reinitialize the neighbor structures
	//update the neighbors structures for the buffers
	if (init_buffers_neighbors) {
		fluidBufferSet->initNeighborsSearchData(data, false);
		backgroundFluidBufferSet->initNeighborsSearchData(data, false);
	}

	//update the neighbor structure for the fluid
	if (init_fluid_neighbors) {
		particleSet->initNeighborsSearchData(data, false);
	}

	//update the boundaries neighbors
	//I don't do it earlier to be able to run every movement kernel in parallel
	if (movement.norm() > 0.0005) {
		data.boundaries_data->initNeighborsSearchData(data, false);
	}

}


void DynamicWindow::lightenBuffers(DFSPHCData& data) {
	UnifiedParticleSet* particleSet = data.fluid_data;

	//first let's lighten the buffers to reduce the computation times
	static int* countRmv = NULL;
	static int* countRmv2 = NULL;

	if (!countRmv) {
		cudaMallocManaged(&(countRmv), sizeof(int));
		cudaMallocManaged(&(countRmv2), sizeof(int));
	}


	*countRmv = 0;
	*countRmv2 = 0;
	
	int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
	DFSPH_lighten_buffers_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
		countRmv, countRmv2, S, borderHeightMap);

	gpuErrchk(cudaDeviceSynchronize());

	std::cout << "ligthen estimation (background // buffer): " << *countRmv << "  //  " << *countRmv2 << std::endl;

	if (false) {
		Vector3d* pos_cpu = new Vector3d[static_cast<int>(MAX_MACRO_CUDA(backgroundFluidBufferSet->numParticles, MAX_MACRO_CUDA(fluidBufferSet->numParticles,particleSet->numParticles)))];

		Vector3d min = Vector3d(15000);
		Vector3d max = Vector3d(-15000);
		read_UnifiedParticleSet_cuda(*backgroundFluidBufferSet, pos_cpu, NULL, NULL, NULL);
		Vector3d* pos = pos_cpu;
		int numParticles = backgroundFluidBufferSet->numParticles;
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

			max.toMax(pos[i]);
			min.toMin(pos[i]);
		}
		//std::cout << "effcount: " << effective_count << "from:  " << numParticles << std::endl;
		std::cout << "Background min/max: " << min.toString() << "  //  " << max.toString() << std::endl;

		min = Vector3d(15000);
		max = Vector3d(-15000);
		read_UnifiedParticleSet_cuda(*fluidBufferSet, pos_cpu, NULL, NULL, NULL);
		pos = pos_cpu;
		numParticles = fluidBufferSet->numParticles;
		for (int i = 0; i < numParticles; ++i) {
			if (fluidBufferSet->neighborsDataSet->cell_id[i] != 0) {
				continue;
			}

			uint8_t density = 255;
			uint8_t alpha = 255;
			uint32_t txt = (((alpha << 8) + 0 << 8) + density << 8) + 0;


			oss << pos[i].x << " " << pos[i].y << " " << pos[i].z << " "
				<< txt << std::endl;
			effective_count++;

			max.toMax(pos[i]);
			min.toMin(pos[i]);
		}
		std::cout << "Buffer min/max: " << min.toString() << "  //  " << max.toString() << std::endl;

		min = Vector3d(15000);
		max = Vector3d(-15000);
		read_UnifiedParticleSet_cuda(*particleSet, pos_cpu, NULL, NULL, NULL);
		pos = pos_cpu;
		numParticles = particleSet->numParticles;
		for (int i = 0; i < numParticles; ++i) {

			uint8_t density = 255;
			uint8_t alpha = 255;
			uint32_t txt = (((alpha << 8) + 0 << 8) + 0 << 8) + 8;


			oss << pos[i].x << " " << pos[i].y << " " << pos[i].z << " "
				<< txt << std::endl;
			effective_count++;

			max.toMax(pos[i]);
			min.toMin(pos[i]);
		}
		delete[] pos_cpu;
		std::cout << "Fluid min/max: " << min.toString() << "  //  " << max.toString() << std::endl;

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

	//remove the tagged particle from the buffer and the background 
	// use the same process as when creating the neighbors structure to put the particles to be removed at the end
	//from the background
	{
		remove_particles(backgroundFluidBufferSet, backgroundFluidBufferSet->neighborsDataSet->cell_id, *countRmv);


		std::cout << "handle_fluid_boundries_cuda: ligthening the background to: " << backgroundFluidBufferSet->numParticles << "   nb removed : " << *countRmv << std::endl;

		//we need to reinit the neighbors struct for the fluidbuffer since we removed some particles
		backgroundFluidBufferSet->initNeighborsSearchData(data, false);
		
	}


	//now the buffer
	{
		remove_particles(fluidBufferSet, fluidBufferSet->neighborsDataSet->cell_id, *countRmv2);

		std::cout << "handle_fluid_boundries_cuda: ligthening the fluid buffer to: " << fluidBufferSet->numParticles << "   nb removed : " << *countRmv2 << std::endl;

		//we need to reinit the neighbors struct for the fluidbuffer since we removed some particles
		fluidBufferSet->initNeighborsSearchData(data, false);
		
	}

	


}

void DynamicWindow::fitFluidBuffer(DFSPHCData& data) {
	UnifiedParticleSet* particleSet = data.fluid_data;

	//ok since sampling the space regularly with particles to close the gap between the fluid and the buffer is realy f-ing hard
	//let's go the oposite way
	//I'll use a buffer too large,  evaluate the density on the buffer particles and remove the particle with density too large
	//also no need to do it on all partilce only those close enught from the plane with the fluid end
	static int* countRmv = NULL;
	
	if (!countRmv) {
		cudaMallocManaged(&(countRmv), sizeof(int));
	}

	Vector3d* pos_temp = new Vector3d[fluidBufferSet->numParticles];
	read_UnifiedParticleSet_cuda(*fluidBufferSet, pos_temp, NULL, NULL, NULL);

	//we need to reinit the neighbors struct for the fluidbuffer since we removed some particles
	fluidBufferSet->initNeighborsSearchData(data, false);
	int cumulative_countRmv = 0;

	{
		RealCuda target_density = 100000;

		*countRmv = 0;
		{
			int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
			DFSPH_evaluate_and_tag_high_density_from_buffer_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
				countRmv, S, target_density);

			gpuErrchk(cudaDeviceSynchronize());
		}
		std::cout << "starting info: " << std::endl;
		std::cout << "count rmv: " <<*countRmv <<std::endl;
		RealCuda min_den = 100000;
		RealCuda max_den = 0;
		RealCuda sum_den = 0;
		for (int j = 0; j < fluidBufferSet->numParticles; ++j) {
			if (pos_temp[j].y > 0 && pos_temp[j].y < 0.6) {
				if (fluidBufferSet->neighborsDataSet->cell_id[j] != 25000000) {
					sum_den += fluidBufferSet->density[j];
					min_den = MIN_MACRO_CUDA(min_den, fluidBufferSet->density[j]);
					max_den = MAX_MACRO_CUDA(max_den, fluidBufferSet->density[j]);
				}
			}
		}
		std::cout << "sum/min/max density in buffer this iter: " << sum_den << "   " << min_den << "   " << max_den << std::endl;
	}

	for (int i = 4; i < 18; ++i) {

		RealCuda target_density = 1500 - i * 25;

		*countRmv = 0;
		{
			int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
			DFSPH_evaluate_and_tag_high_density_from_buffer_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
				countRmv, S, target_density);

			gpuErrchk(cudaDeviceSynchronize());
		}
		cumulative_countRmv += *countRmv;

		if (false){
			*countRmv = 0;

			int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
			DFSPH_save_low_density_neighbors_from_buffer_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
				countRmv, S, 950);

			gpuErrchk(cudaDeviceSynchronize());

			std::cout << "nbr saved: " << *countRmv << std::endl;
		}


		/*
		RealCuda avg = 0;
		int count = 0;
		for (int j = 0; j < fluidBufferSet->numParticles; ++j) {
			if (pos_temp[j].y > 0) {
				if (fluidBufferSet->neighborsDataSet->cell_id[j] == 25000000) {	
					avg+=S.distanceToSurface(pos_temp[j]);
					pos_temp[j].y = -1;
					count++;
				}
			}
		}
		std::cout << "avg removal distance this iter: " <<avg/count <<std::endl;
		//*/

		//a test to see the minimum and maximum densities left in the buffer
		RealCuda temp;
		{
			RealCuda min_den = 100000;
			RealCuda max_den = 0;
			RealCuda sum_den = 0;
			for (int j = 0; j < fluidBufferSet->numParticles; ++j) {
				if (pos_temp[j].y > 0 && pos_temp[j].y < 0.6) {
					if (fluidBufferSet->neighborsDataSet->cell_id[j] != 25000000) {
						sum_den += fluidBufferSet->density[j];
						min_den = MIN_MACRO_CUDA(min_den, fluidBufferSet->density[j]);
						max_den = MAX_MACRO_CUDA(max_den, fluidBufferSet->density[j]);
					}
				}
			}
			std::cout << "sum/min/max density in buffer this iter: " << sum_den << "   " << min_den << "   " << max_den << std::endl;
			temp = sum_den;
		}


		if (true) {

			int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
			DFSPH_evaluate_density_from_buffer_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,S);

			gpuErrchk(cudaDeviceSynchronize());

			RealCuda min_den = 100000;
			RealCuda max_den = 0;
			RealCuda sum_den = 0;
			for (int j = 0; j < fluidBufferSet->numParticles; ++j) {
				if (pos_temp[j].y > 0 && pos_temp[j].y < 0.6) {
					if (fluidBufferSet->neighborsDataSet->cell_id[j] != 25000000) {
						sum_den += fluidBufferSet->density[j];
						min_den = MIN_MACRO_CUDA(min_den, fluidBufferSet->density[j]);
						max_den = MAX_MACRO_CUDA(max_den, fluidBufferSet->density[j]);
					}
				}
			}
			std::cout << "sum/min/max density in buffer reevaluate: " << sum_den << "   " << min_den << "   " << max_den << std::endl;
			if (temp == sum_den) {
				std::cout << "no difference detected at all" << std::endl;
			}
		}
		

		std::cout << "handle_fluid_boundries_cuda: (iter: " << i << ") limit density : " << target_density<<
			"   nb removed total (this step): " << cumulative_countRmv << "  (" << *countRmv << ")" << std::endl;

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



	}
		//remove the tagged particle from the buffer (all yhe ones that have a density that is too high)
		//now we can remove the partices from the simulation
		// use the same process as when creating the neighbors structure to put the particles to be removed at the end
		remove_particles(fluidBufferSet, fluidBufferSet->neighborsDataSet->cell_id, cumulative_countRmv);

}

void DynamicWindow::computeFluidBufferVelocities(DFSPHCData& data) {
	//I'll save the velocity field by setting the velocity of each particle to the weighted average of the three nearest
		//or set it to 0, maybe I need to do smth intermediary
	{
		int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
		//DFSPH_init_buffer_velocity_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->pos, fluidBufferSet->vel, fluidBufferSet->numParticles);
		cuda_setBufferToValue_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (fluidBufferSet->vel, Vector3d(0, 0, 0), fluidBufferSet->numParticles);
		gpuErrchk(cudaDeviceSynchronize());
	}
}

void DynamicWindow::addFluidBufferToSimulation(DFSPHCData& data) {

	UnifiedParticleSet* particleSet = data.fluid_data;
	static int* countRmv = NULL;

	if (!countRmv) {
		cudaMallocManaged(&(countRmv), sizeof(int));
	}

	//now we can remove the partices from the simulation
	{

		*countRmv = 0;
		int numBlocks = calculateNumBlocks(particleSet->numParticles);
		DFSPH_reset_fluid_boundaries_remove_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, countRmv, S, true);

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
		std::cout << "handle_fluid_boundries_cuda: removing fluid inside buffer area changing num particles: " << new_num_particles << "   nb removed : " << *countRmv << std::endl;
		particleSet->updateActiveParticleNumber(new_num_particles);
		//*/

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
		std::cout << "handle_fluid_boundries_cuda: adding buffer particles to fluid changing num particles: " << new_num_particles << "   nb added : " << fluidBufferSet->numParticles << std::endl;
		particleSet->updateActiveParticleNumber(new_num_particles);

	}

}



void DynamicWindow::applyParticleShiftNearSurface(DFSPHCData& data) {
	UnifiedParticleSet* particleSet = data.fluid_data;
	static int* outInt = NULL;
	static RealCuda* outReal = NULL;

	if (!outInt) {
		cudaMallocManaged(&(outInt), sizeof(int));
		cudaMallocManaged(&(outReal), sizeof(RealCuda));
	}
	 
	//initi neighbor structure
	particleSet->initNeighborsSearchData(data, false);

	//evaluate the density first since anyway we are gonna need it
	{
		int numBlocks = calculateNumBlocks(particleSet->numParticles);
		DFSPH_evaluate_density_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, S);
		gpuErrchk(cudaDeviceSynchronize());
	}




	int count_samples = 0;
	Vector3d* pos = NULL;
	RealCuda* den = NULL;

	bool use_fluid_particles = true;
	if (use_fluid_particles) {
		count_samples = particleSet->numParticles;
		cudaMallocManaged(&(pos), count_samples*sizeof(Vector3d));
		den = particleSet->density;
	}
	else {
		//I'll do a sampling on a regular grid
		RealCuda affected_range = data.getKernelRadius();
		RealCuda spacing = data.particleRadius;

		Vector3d min, max;
		get_UnifiedParticleSet_min_max_naive_cuda(*(data.boundaries_data), min, max);
		std::cout << "min/ max: " << min.toString() << " " << max.toString() << std::endl;
		min += 2*data.particleRadius;
		max -= 2*data.particleRadius;
		Vector3i count_dim = (max - min)/spacing;
		count_dim += 1;

		std::cout << "count samples base :" << count_dim.x * count_dim.y * count_dim.z << std::endl;
		
		//only keep the samples that are near the plane
		int real_count = 0;
		for (int i = 0; i < count_dim.x; ++i) {
			for (int j = 0; j < count_dim.y; ++j) {
				for (int k = 0; k < count_dim.z; ++k) {
					Vector3d p_i = min + Vector3d(i, j, k) * spacing;
					if (S.distanceToSurface(p_i) < affected_range*1.1){
						if (p_i.y < 0.7) {
							real_count++;
						}
					}
				}
			}
		}

		std::cout << "count samples near :" << real_count << std::endl;

		count_samples = real_count;
		cudaMallocManaged(&(pos), count_samples * sizeof(Vector3d));
		cudaMallocManaged(&(den), count_samples * sizeof(RealCuda));
		
		real_count = 0;
		for (int i = 0; i < count_dim.x; ++i) {
			for (int j = 0; j < count_dim.y; ++j) {
				for (int k = 0; k < count_dim.z; ++k) {
					Vector3d p_i = min + Vector3d(i, j, k) * spacing;
					if (S.distanceToSurface(p_i) < affected_range*1.1) {
						if (p_i.y < 0.7) {
							pos[real_count] = p_i;
							real_count++;
						}
					}
				}
			}
		}

		{
			int numBlocks = calculateNumBlocks(count_samples);
			DFSPH_evaluate_density_field_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, pos, den, count_samples);
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	if (true){
		
		if (use_fluid_particles) {
			read_UnifiedParticleSet_cuda(*particleSet, pos, NULL, NULL, NULL);
		}

		//density check
		//let's evaluate the average and std dev around the surface
		RealCuda avg = 0;
		RealCuda min = 1000000;
		RealCuda max = -1;
		int count_close = 0;
		RealCuda avg_global = 0;
		RealCuda min_global = 100000;
		RealCuda max_global = -1;
		RealCuda affectedRange = data.getKernelRadius();
		for (int i = 0; i < count_samples; ++i) {
			if (pos[i].y < 0.6) {


				if (S.distanceToSurface(pos[i]) < affectedRange) {
					avg += den[i];
					count_close++;
					min = MIN_MACRO_CUDA(min, den[i]);
					max = MAX_MACRO_CUDA(max, den[i]);
				}
				else {
					avg_global += den[i];
					min_global = MIN_MACRO_CUDA(min_global, den[i]);
					max_global = MAX_MACRO_CUDA(max_global, den[i]);
				}
			}
		}
		avg_global /= count_samples -count_close;
		avg /= count_close;


		RealCuda stddev = 0;
		RealCuda stddev_global = 0;
		for (int i = 0; i < count_samples; ++i) {
			if (pos[i].y < 0.6) {

				RealCuda delta = (den[i] - avg);
				delta *= delta;
				if (S.distanceToSurface(pos[i]) < affectedRange) {
					stddev += delta;
				}
				else {
					stddev_global += delta;
				}
			}
		}
		stddev = std::sqrtf(stddev);
		stddev /= count_close;
		stddev_global = std::sqrtf(stddev_global);
		stddev_global /= count_samples - count_close;

		std::cout << "global  average density/ deviation / count / min / max: " << avg_global << "   " << stddev_global << "   " <<count_samples-count_close<<"  "<<min_global << "   " << max_global << std::endl;
		std::cout << "Surface average density/ deviation / count / min / max: " << avg << "   " << stddev << "   " <<count_close <<"  " << min << "   " << max << std::endl;
	}



	//let's do a test where I remove every particles that have a density above 1050
	if (false){
		//tag every particles with a density above the decided limit
		RealCuda limit_density = 1050;
		*outInt = 0;
		for (int i = 0; i < particleSet->numParticles; ++i) {
			particleSet->neighborsDataSet->cell_id[i] = 0;
			if (particleSet->density[i] > limit_density) {
				particleSet->neighborsDataSet->cell_id[i] = REMOVAL_TAG;
				(*outInt)++;
			}
		}
		//*
		//now use the same process as when creating the neighbors structure to put the particles to be removed at the end
		cub::DeviceRadixSort::SortPairs(particleSet->neighborsDataSet->d_temp_storage_pair_sort, particleSet->neighborsDataSet->temp_storage_bytes_pair_sort,
			particleSet->neighborsDataSet->cell_id, particleSet->neighborsDataSet->cell_id_sorted,
			particleSet->neighborsDataSet->p_id, particleSet->neighborsDataSet->p_id_sorted, particleSet->numParticles);
		gpuErrchk(cudaDeviceSynchronize());

		cuda_sortData(*particleSet, particleSet->neighborsDataSet->p_id_sorted);
		gpuErrchk(cudaDeviceSynchronize());

		//and now you can update the number of particles


		int new_num_particles = particleSet->numParticles - *outInt;
		std::cout << "handle_fluid_boundries_cuda: particle shift, removing fluid above limit density changing num particles: " << new_num_particles << "   nb removed : " << *outInt << std::endl;
		particleSet->updateActiveParticleNumber(new_num_particles);
		//*/

		//we need to reinitialize the system
		//initi neighbor structure
		particleSet->initNeighborsSearchData(data, false);

		//evaluate the density first since anyway we are gonna need it
		{
			int numBlocks = calculateNumBlocks(particleSet->numParticles);
			DFSPH_evaluate_density_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, S);
			gpuErrchk(cudaDeviceSynchronize());
		}
	}


	RealCuda diffusion_coefficient = 1000 * 0.5 * data.getKernelRadius() * data.getKernelRadius() / data.density0;


	int shifting_type = 0;	
	if (shifting_type == 0) {
		std::cout << "Concentration particle shift "<< std::endl;
		//for the explanaition of eahc kernel please see the paper that is referenced in the function description
		//NOTE: I also use those kernel to do a simple surface detection with the following algorithm
		//		Use a low neighbors cap for sampling the surface
		//		then for each particel that I detected I increase a value in it's neighbor particles depending on the kernel value
		//		I then use that value to limit the vertical component of the particle shifting
		{
			int numBlocks = calculateNumBlocks(particleSet->numParticles);
			DFSPH_evaluate_particle_concentration_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, S);
			gpuErrchk(cudaDeviceSynchronize());
		}

		{
			*outInt = 0;
			*outReal = 0;
			int numBlocks = calculateNumBlocks(particleSet->numParticles);
			DFSPH_particle_shifting_base_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, diffusion_coefficient, S, outInt, outReal);
			gpuErrchk(cudaDeviceSynchronize());

			std::cout << "count particles shifted: " << *outInt << "   for a total displacement: " << *outReal << std::endl;
		}
	}
	else if (shifting_type == 1) {
		// this version will use the dircrete grid I created and will move the particles toward the closest minimum
		// the rule for the displacement will be 
		std::cout << "density grid " << std::endl;

		{
			int numBlocks = calculateNumBlocks(particleSet->numParticles);
			DFSPH_particle_shifting_density_grid_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, S, pos, den, count_samples);
			gpuErrchk(cudaDeviceSynchronize());

		}
	}


	if (true) {
		RealCuda affectedRange = data.getKernelRadius();


		std::cout << "<<<<<<<<<<<<<<< POST CORRECTION CHECK >>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;

		std::vector<RealCuda> vect_den;
		for (int i = 0; i < count_samples; ++i) {
			vect_den.push_back(den[i]);
		}

		if (use_fluid_particles) {
			int numBlocks = calculateNumBlocks(particleSet->numParticles);
			DFSPH_evaluate_density_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, S);
			gpuErrchk(cudaDeviceSynchronize());
			read_UnifiedParticleSet_cuda(*particleSet, pos, NULL, NULL, NULL);
		}
		else {
			int numBlocks = calculateNumBlocks(count_samples);
			DFSPH_evaluate_density_field_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, pos, den, count_samples);
			gpuErrchk(cudaDeviceSynchronize());

		}
		std::cout << "post" << std::endl;

		std::ofstream myfile("out_temp.txt", std::ofstream::trunc);
		if (myfile.is_open())
		{
			for (int i = 0; i < count_samples; ++i) {
				myfile << vect_den[i] << "  " << den[i] << "   " << den[i] - vect_den[i] << std::endl;
			}
			myfile.close();
		}

		//density check
		//let's evaluate the average and std dev around the surface
		RealCuda avg = 0;
		RealCuda min = 1000000;
		RealCuda max = -1;
		int count_close = 0;
		RealCuda avg_global = 0;
		RealCuda min_global = 100000;
		RealCuda max_global = -1;
		for (int i = 0; i < count_samples; ++i) {
			if (pos[i].y < 0.6) {

				if (S.distanceToSurface(pos[i]) < affectedRange) {
					avg += den[i];
					count_close++;
					min = MIN_MACRO_CUDA(min, den[i]);
					max = MAX_MACRO_CUDA(max, den[i]);
				}
				else {
					avg_global += den[i];
					min_global = MIN_MACRO_CUDA(min_global, den[i]);
					max_global = MAX_MACRO_CUDA(max_global, den[i]);
				}
			}
		}
		avg_global /= count_samples - count_close;
		avg /= count_close;


		RealCuda stddev = 0;
		RealCuda stddev_global = 0;
		for (int i = 0; i < count_samples; ++i) {
			if (pos[i].y < 0.6) {

				RealCuda delta = (den[i] - avg);
				delta *= delta;
				if (S.distanceToSurface(pos[i]) < affectedRange) {
					stddev += delta;
				}
				else {
					stddev_global += delta;
				}
			}
		}
		stddev = std::sqrtf(stddev);
		stddev /= count_close;
		stddev_global = std::sqrtf(stddev_global);
		stddev_global /= count_samples - count_close;


		std::cout << "global  average density/ deviation / count / min / max: " << avg_global << "   " << stddev_global << "   " << count_samples - count_close << "  " << min_global << "   " << max_global << std::endl;
		std::cout << "Surface average density/ deviation / count / min / max: " << avg << "   " << stddev << "   " << count_close << "  " << min << "   " << max << std::endl;

		if (true) {


			std::ostringstream oss;
			int effective_count = 0;
			for (int i = 0; i < count_samples; ++i) {
				if (S.distanceToSurface(pos[i]) > affectedRange) {
					//	continue;
				}


				uint8_t density = fminf(1.0f, (den[i] / 1300.0f)) * 255;
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
				//txt = (((255 << 8) + 255 << 8) + 0 << 8) + 0;
				//*
				if (den[i] > 1050) {
					txt = (((alpha << 8) + density << 8) + 0 << 8) + 0;
				}
				else if (den[i] < 900) {

					txt = (((alpha << 8) + 0 << 8) + density << 8) + 0;

					if (den[i] < 850) {

						txt = (((alpha << 8) + 0 << 8) + 0 << 8) + density;
					}

				}
				else {
					continue;
				}
				//*/
				//if (density_field_after_buffer[i] > 950) {
				//	txt = (((alpha << 8) + 255 << 8) + 0 << 8) + 144;
				//}

				oss << pos[i].x << " " << pos[i].y << " " << pos[i].z << " "
					<< txt << std::endl;
				effective_count++;
			}

			/*
			CUDA_FREE_PTR(pos);
			CUDA_FREE_PTR(den);
			//count_samples = data.boundaries_data->numParticles;// particleSet->numParticles;
			count_samples = particleSet->numParticles;
			cudaMallocManaged(&(pos), count_samples * sizeof(Vector3d));
			den = particleSet->density;

			//read_UnifiedParticleSet_cuda(*(data.boundaries_data), pos, NULL, NULL, NULL);
			read_UnifiedParticleSet_cuda(*particleSet, pos, NULL, NULL, NULL);

			for (int i = 0; i < count_samples; ++i) {
				if (S.distanceToSurface(pos[i]) > affectedRange) {
					continue;
				}
				uint32_t txt = (((255 << 8) + 0 << 8) + 0 << 8) + 255;

				oss << pos[i].x << " " << pos[i].y << " " << pos[i].z << " "
					<< txt << std::endl;
				effective_count++;
			}
			//*/

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

	CUDA_FREE_PTR(pos);
	if (!use_fluid_particles) {
		CUDA_FREE_PTR(den);
	}

}


void DynamicWindow::handleFluidBoundaries(SPH::DFSPHCData& data, Vector3d movement) {
	if (!isInitialized()) {
		throw("DynamicWindow::handleFluidBoundaries the structure must be initialized before calling this function");
	}


	std::vector<std::string> timing_names{"color_reset","reset pos","apply displacement to buffer","init neightbors","compute density",
		"reduce buffer","cpy_velocity","reduce fluid", "apply buffer"};
	static SPH::SegmentedTiming timings("handle_fluid_boundries_cuda",timing_names,false);
	timings.init_step();

	data.fluid_data->resetColor();

	timings.time_next_point();
	//movement = Vector3d(0, 0, 0);


	{
		timings.time_next_point();

		//temp test
		
		std::cout << "before movement" << std::endl;
		evaluate_density_field(data, data.fluid_data);
		
		//reset the buffers
		DynamicWindow::getStructure().initStep(data, movement, false);


		std::cout << "after boundary movement" << std::endl;
		evaluate_density_field(data, data.fluid_data);

		timings.time_next_point();
		
		//lighten the buffer to lower computation time
		DynamicWindow::getStructure().lightenBuffers(data);

		//remove particles from the buffer that are inside the fluid that we will kepp so that
		//the buffer will fit in the space where we will remove particles
		DynamicWindow::getStructure().fitFluidBuffer(data);

		timings.time_next_point();
		
		DynamicWindow::getStructure().computeFluidBufferVelocities(data);

		timings.time_next_point();

		DynamicWindow::getStructure().addFluidBufferToSimulation(data);

		timings.time_next_point();

	
		DynamicWindow::getStructure().applyParticleShiftNearSurface(data);


		timings.time_next_point();
		timings.end_step();
		timings.recap_timings();

		//still need the damping near the borders as long as we don't implement the implicit borders
		//with paricle boundaries 3 is the lowest number of steps that absorb nearly any visible perturbations
		//*
		if (S_initial.getType() < 2) {
			add_border_to_damp_planes_cuda(data, abs(movement.x)>0.5, abs(movement.z)>0.5);
			data.damp_borders_steps_count = 5;
			data.damp_borders = true;
		}
		//*/
	}

	data.fluid_data->resetColor();

	//here is a test to see what does the density filed looks like at the interface
		//first let's do a test
		//let's compute the density on the transition plane at regular interval in the fluid 
		//then compare with the values obtained when I add the buffer back in the simulation
	if (false) {

		SPH::UnifiedParticleSet* particleSet = data.fluid_data;

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

		SPH::UnifiedParticleSet* fluidBufferSet = DynamicWindow::getStructure().fluidBufferSet;

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




/*******************************************************************/
//	Function specific to the control of the height of the fluid
/*******************************************************************/




//this code the formula show in the paper: Fluxed Animated Boundary Method;  ALEXEY STOMAKHIN and ANDREW SELLE
class StokesWaveGenerator {
public:
	RealCuda* waveHeightField;
	Vector3d* waveVelocityField;
	int* waveVelocityFieldCount;
	RealCuda cellSize;
	Vector3i cellCount;
	Vector3d minPos;
	Vector3d maxPos;

	RealCuda dt;
	RealCuda samplingSpacing;
	Vector3i countSampling;
	int overSampling;//this is the number of samples I'll have before the min for the X axis

	RealCuda surfaceHeight;
	RealCuda a;
	RealCuda k;
	RealCuda omega;

	StokesWaveGenerator() {

	}

	void init(RealCuda cellSize_i, Vector3d min, Vector3d max) {
		cellSize = cellSize_i;
		minPos = min;
		maxPos = max;
		
		Vector3d length=(max - min).abs();
		length /= cellSize;
		cellCount.x = length.x;
		cellCount.y = length.y;

		cudaMallocManaged(&(waveHeightField), cellCount.x * sizeof(RealCuda));
		cudaMallocManaged(&(waveVelocityField), cellCount.x * cellCount.y * sizeof(Vector3d));
		cudaMallocManaged(&(waveVelocityFieldCount), cellCount.x * cellCount.y * sizeof(int));

		a = 0.15;
		k = 2 * CR_CUDART_PI / 4.0;
		omega = 2 * CR_CUDART_PI / 2.0;

		overSampling = 4;
	}

	//fusing the 2 getter may bring some benefits at some point (though it won't be that significative and there are way easier optimisation to do first
	//the first on beeing the fact that I know any particle with y<surface-a   are necessarily inside the fluid
	__device__ inline void getHeight(Vector3d pos, RealCuda& height) {
		if (!isInsideDomain(pos)) {
			height = -1;
		}

		int i_x = (pos.x - minPos.x) / cellSize;

		height = waveHeightField[i_x];
	}

	//the problem with the velocity is that I am no ensured that there was at least one particle in the cell to get a velocity
	//so I'll strat by using an average on the cross and if it is not enougth I'll add more cells untill I have at least one valid velocity
	//it will mush the result but I don't realy have the choice
	__device__  inline void getvel(Vector3d pos, Vector3d& vel) {
		if (!isInsideDomain(pos)) {
			vel = Vector3d(0,0,0);
		}

		int i_x = (pos.x - minPos.x) / cellSize;
		int i_y = (pos.y - minPos.y) / cellSize;

		int neighbors_version = 1;
		if (neighbors_version == 0) {
			//this version consider a square as the neighbor (it is unoptimized)
			//hard coding the possibilities for simple cases whould most likely improve the performances
			//and made to be easy to read
			int range = 0;
			//*
			Vector3d sum_vel(0, 0, 0);
			int count_samples = 0;
			do {

				sum_vel = Vector3d(0, 0, 0);
				count_samples = 0;
				for (int i = -range; i < range; ++i) {
					for (int j = -range; j < range; ++j) {
						//a short opti to not reexplore already explored cells
						if ((abs(i) != range) && (abs(j) != range)) {
							continue;
						}

						int lx = i_x + i;
						int ly = i_y + j;

						if ((lx < 0) || (ly < 0) || (lx >= cellCount.x) || (ly >= cellCount.y)) {
							continue;
						}


						int cell_id = lx + ly * cellCount.x;
						sum_vel += waveVelocityField[cell_id];
						count_samples += waveVelocityFieldCount[cell_id];
					}
				}
				range++;
			} while (count_samples == 0);
			vel = sum_vel / count_samples;
			//*/
			//a debug check
			//*
			if (range > 1) {
				asm("trap;");
			}
			//*/

		}
		else if (neighbors_version == 1) {
			//basic straight read
			int cell_id = i_x + i_y * cellCount.x;
			vel = waveVelocityField[cell_id];
			int count_samples = waveVelocityFieldCount[cell_id];
			if (count_samples > 0) {
				vel /= count_samples;
			}
		}
	}

	//NOTE: the pos must be given in global coordinate
	FUNCTION inline Vector3d computeSamplePos(Vector3d pos, RealCuda t) {
		Vector3d out;
		out.y = a * expf(k * (pos.y - surfaceHeight));
		
		out.x = -out.y * sinf(k * pos.x - omega * t) + pos.x;
		out.y = out.y * cosf(k * pos.x - omega * t) + (pos.y);
		return out;
	}


	//NOTE: the pos must be given in global coordinate
	FUNCTION inline void computeSampleProperties(Vector3d pos, RealCuda t, Vector3d& pos_out, Vector3d& vel_out) {
		pos_out = computeSamplePos(pos, t);

		//let's use the central finit difference for the velocity
		vel_out = (computeSamplePos(pos, t + dt / 2.0f) - computeSamplePos(pos, t - dt / 2.0f))/dt;
	}

	FUNCTION inline bool isInsideDomain(Vector3d pos) {
		return (pos.x >= minPos.x) && (pos.y >= minPos.y) && (pos.x <= maxPos.x) && (pos.y <= maxPos.y);
	}

	void computeWaveState(RealCuda t, RealCuda h0, RealCuda dt_i, RealCuda samplingSpacing_i);

};


__global__ void DFSPH_fill_stockes_waves_buffers_kernel(StokesWaveGenerator waveGenerator, RealCuda t) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (waveGenerator.countSampling.x*waveGenerator.countSampling.y)) { return; }

	int i_x = i % waveGenerator.countSampling.x;
	if (i < 0) {
		i += waveGenerator.countSampling.x;
	}
	int i_y = (i - i_x) / waveGenerator.countSampling.x;
	i_x -= waveGenerator.overSampling; //don't forget the oversampling
	Vector3d pos = waveGenerator.minPos;
	pos.x += i_x * waveGenerator.samplingSpacing;
	pos.y += waveGenerator.surfaceHeight - i_y * waveGenerator.samplingSpacing;

	Vector3d pos_out, vel_out;
	waveGenerator.computeSampleProperties(pos, t, pos_out, vel_out);

	//Ignore anything not inside the domain
	if (!waveGenerator.isInsideDomain(pos_out)) {
		return;
	}

	//go to local coordinates
	pos_out -= waveGenerator.minPos;

	//only keep the heighest height
	int i_x_out = pos_out.x / (waveGenerator.cellSize);
	atomicToMax(&(waveGenerator.waveHeightField[i_x_out]),pos_out.y);

	//but keep all the velocities
	int i_y_out = pos_out.y / (waveGenerator.cellSize);
	int cell_id = i_x_out + i_y_out * waveGenerator.cellCount.x;

	atomicAdd(&(waveGenerator.waveVelocityFieldCount[cell_id]), 1);
	atomicAdd(&(waveGenerator.waveVelocityField[cell_id].x), vel_out.x);
	atomicAdd(&(waveGenerator.waveVelocityField[cell_id].y), vel_out.y);


}

void StokesWaveGenerator::computeWaveState(RealCuda t, RealCuda h0, RealCuda dt_i, RealCuda samplingSpacing_i) {
	dt = dt_i;
	samplingSpacing = samplingSpacing_i;
	surfaceHeight = h0;

	//count the nbr of sampling we need
	//not I need ot oversample on X so that I don't have strange border effects, I'll add some samples on each side
	countSampling.x = (maxPos.x - minPos.x) / samplingSpacing + overSampling*2;
	countSampling.y = (surfaceHeight - minPos.y) / samplingSpacing + 1;

	//std::cout << "sampling count: (x y)" << countSampling.toString() << std::endl;

	//I need first to reset the structure
	{
		int numBlocks = calculateNumBlocks(cellCount.x * cellCount.y);
		cuda_setBufferToValue_kernel<int> << <numBlocks, BLOCKSIZE >> > (waveVelocityFieldCount, 0, cellCount.x * cellCount.y);
		cuda_setBufferToValue_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (waveVelocityField, Vector3d(0,0,0), cellCount.x * cellCount.y);
		numBlocks = calculateNumBlocks(cellCount.x);
		cuda_setBufferToValue_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (waveHeightField, -1, cellCount.x);
		gpuErrchk(cudaDeviceSynchronize());
	}

	//then compute the velocity and pos data
	{
		int numBlocks = calculateNumBlocks(countSampling.x * countSampling.y);
		DFSPH_fill_stockes_waves_buffers_kernel << <numBlocks, BLOCKSIZE >> > (*this, t);
		gpuErrchk(cudaDeviceSynchronize());
	}

}

template <bool use_neighbor_structure>
__global__ void DFSPH_tag_buffer_to_add_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, SPH::UnifiedParticleSet* bufferSet, 
	StokesWaveGenerator waveGenerator, BufferFluidSurface S_border, int* count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }

	Vector3d pos_i = bufferSet->pos[i];
	bufferSet->neighborsDataSet->cell_id[i] = 0;
	
	if (S_border.isinside(pos_i)) {
		return;
	}


	RealCuda h;
	waveGenerator.getHeight(pos_i, h);

	if (pos_i.y > h) {
		return;
	}

	//let's brute force it for now
	//and my condition will bee if we are far enougth from any fluid particle that mean i cna be added to the simulation
	//note since the buffer is as a configuration of a rest fluid that mean even I I add mutples particles at the same time I won't hav any problem
	//we save the velocities and distance of the n closests
	RealCuda limit_dist = data.particleRadius * 2;
	limit_dist *= limit_dist;
	if (use_neighbor_structure) {
		//*
		ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, bufferSet, i);
		ITER_NEIGHBORS_FROM_STRUCTURE(particleSet->neighborsDataSet, particleSet->pos,
			{
				RealCuda cur_dist = (pos - particleSet->pos[j]).squaredNorm();
				if (cur_dist < limit_dist) {
					return;
				}
			}
			);
			//*/
	}
	else {
		for (int j = 0; j < particleSet->numParticles; ++j) {
			RealCuda cur_dist = (pos_i - particleSet->pos[j]).squaredNorm();

			if (cur_dist < limit_dist) {
				return;
			}
		}
	}

	//reaching here means we have to add the particle
	//for now I'll simply tag it and count them

	atomicAdd(count, 1);
	bufferSet->neighborsDataSet->cell_id[i] = 25000000;


	//vel control here
	Vector3d vel;
	waveGenerator.getvel(pos_i, vel);
	bufferSet->vel[i] = vel;
}



__global__ void DFSPH_buffer_mimic_stockes_waves_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, StokesWaveGenerator waveGenerator, int* count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }

	Vector3d pos = bufferSet->pos[i];
	bufferSet->neighborsDataSet->cell_id[i] = 0;

	
	RealCuda h;
	waveGenerator.getHeight(pos, h);

	if (pos.y>h) {
		return;
	}

	//reaching here means we have to add the particle
	//for now I'll simply tag it and count them
	atomicAdd(count, 1);
	bufferSet->neighborsDataSet->cell_id[i] = 25000000;
}




__global__ void DFSPH_tag_above_desired_free_surface_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet,
	BufferFluidSurface S, StokesWaveGenerator waveGenerator, int* count, bool checkFreeSurfaceForAll=false) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	//*

	Vector3d pos = particleSet->pos[i];
	particleSet->neighborsDataSet->cell_id[i] = 0;

	if ((!S.isinside(particleSet->pos[i]))|| checkFreeSurfaceForAll) {

		RealCuda h;
		waveGenerator.getHeight(pos, h);

		if (pos.y > h) {
		//*
			atomicAdd(count, 1);
			particleSet->neighborsDataSet->cell_id[i] = 25000000;
			return;
		//*/
		}

		//I'll do the velocity control here
		//I need a smooth transition from the existing velocity to the wave one
		//it will depend on the size of the buffer so in all technicity I'll need to store that value in the structure, let's say 0.25 for now
		//if (!checkFreeSurfaceForAll) 
		{
			Vector3d vel;
			RealCuda buffer_length = 0.25;
			RealCuda coef= MIN_MACRO_CUDA(1.0, S.distanceToSurface(pos) / buffer_length);
			//coef = 0.005;
			waveGenerator.getvel(pos, vel);
			particleSet->vel[i]=vel*coef+ particleSet->vel[i]*(1-coef);
		}
	}
	//*/

}


__global__ void DFSPH_add_tagged_particles_to_fluid_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, SPH::UnifiedParticleSet* bufferSet, int* count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }

	if (bufferSet->neighborsDataSet->cell_id[i] == 25000000) {
		int id = atomicAdd(count, 1);;
		id += particleSet->numParticles;

		particleSet->pos[id] = bufferSet->pos[i];
		particleSet->vel[id] = bufferSet->vel[i];
		particleSet->mass[id] = bufferSet->mass[i];
		particleSet->color[id] = Vector3d(0, 1, 0);

	}


}





__global__ void DFSPH_handle_inflow_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, SPH::UnifiedParticleSet* bufferSet,
	BufferFluidSurface S_border, BorderHeightMap borderHeightMap, int* count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }


	if (!S_border.isinside(bufferSet->pos[i])) {
		return;
	}

	//only add particle if we are below the ocean level
	Vector3d p_temp = bufferSet->pos[i];
	if (p_temp.y > borderHeightMap.getHeight(p_temp)) {
		return;
	}

	
	//let's brute force it for now
	//and my condition will bee if we are far enougth from any fluid particle that mean i cna be added to the simulation
	//note since the buffer is as a configuration of a rest fluid that mean even I I add mutples particles at the same time I won't hav any problem
	//we save the velocities and distance of the n closests
	RealCuda limit_dist = data.particleRadius * 2 ;
	limit_dist *= limit_dist;
	//*

	int countn = 0;
	ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, bufferSet, i);
	ITER_NEIGHBORS_FROM_STRUCTURE(particleSet->neighborsDataSet, particleSet->pos,
		{
			countn++;
			RealCuda cur_dist = (pos - particleSet->pos[j]).squaredNorm();
	if (cur_dist < limit_dist) {
		return;
	}
		}
	);
	
//*/
	//if we reach here it means we need to add the particle into the simulation
	//we need a velocity forthe particle, so we will set it's velocity to the average velocity of neighbor particles
	//for that we will use the same approximation as ferrand 2012
	//though since all fluid particles have the same density if can be slightly simplified as simply a weigth equivalent to the kernel value
	RealCuda totalWeight = 0;
	Vector3d weightedSum(0,0,0);
	ITER_NEIGHBORS_FROM_STRUCTURE(particleSet->neighborsDataSet, particleSet->pos,
		{
			RealCuda weight =  KERNEL_W(data,pos - particleSet->pos[j]);
			weightedSum += particleSet->vel[j] * weight;
			totalWeight += weight;
		}
	);


	//don't create a particle if there is a flow toward the outside of the simulation space
	Vector3d normal(1, 0, 0);
	if (normal.dot(weightedSum) < 0) {
		return;
	}

	if (totalWeight != 0) {
		weightedSum /= totalWeight;
	}
	//we have to lower the vertical component or keeping the velocity will accumulate acceleration on the vertical component
	//I think we could lower it by the aceleration but for now I'll just won't consider the vertical velocity
	weightedSum.y = 0;
	if (pos.y < 0.1) {
	//	printf("test velocity: %f %f %f // %f %f %f // %d %f \n", pos.x, pos.y, pos.z, weightedSum.x, weightedSum.y, weightedSum.z, countn, totalWeight);
	}

//reaching here means we have to add the particle
int local_id = atomicAdd(count, 1);
particleSet->pos[particleSet->numParticles + local_id] = pos;
particleSet->mass[particleSet->numParticles + local_id] = particleSet->mass[0];
particleSet->vel[particleSet->numParticles + local_id] = weightedSum;//Vector3d(0, 0, 0);
particleSet->color[particleSet->numParticles + local_id] = Vector3d(1,0,0);
//we need to make sure they are not tagged for removal
particleSet->neighborsDataSet->cell_id[i] = 0;
}

__global__ void DFSPH_apply_current_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet,
	BufferFluidSurface S_current, Vector3d current_velocity, RealCuda transition_length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	//inside the surface I know the velocity 
	if (S_current.isinside(particleSet->pos[i])) {
		particleSet->vel[i].x = current_velocity.x;
	}
	/*
	//and around the surfa I'll set a transition area
	RealCuda dist = S_current.distanceToSurface(particleSet->pos[i]);

	if (dist > transition_length) {
		return;
	}

	RealCuda coef = dist / transition_length;
	particleSet->vel[i] = particleSet->vel[i] * coef + current_velocity * (1.0 - coef);
	//*/
}

__global__ void DFSPH_handle_outflow_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet,
	BufferFluidSurface S_fluidInterior, BorderHeightMap borderHeightMap, int* count,
	RealCuda* massDeltas) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	Vector3d pos_i = particleSet->pos[i];



	//only affect the border
	if (S_fluidInterior.isinside(pos_i)) {
		particleSet->neighborsDataSet->cell_id[i] = 0;
		return;
	}

	

	//by default a particles is tagged as to be removed
	int tag = 250000000;
	//the easy case is if there is no ocean reference I remove the particle
	RealCuda ocean_height = borderHeightMap.getHeight(pos_i);
	
	if (pos_i.y < ocean_height) 
	{
		//if we are below he ocean surface I do an estimation of the current to see if the particle should be removed
		//for now instead of taking the actual particle velocity,
		//I'll use an estimation of the velocity by taking a virtual position that is inside the fluid (similar to tafuni)
		//My reson is that I dont want that velocity estimation to be perturbated by the boundary particles
		Vector3d pos_virt = pos_i;
		///TODO you have to apply the normal of the surface to get the virtual position
		RealCuda factor = 1;
		if (pos_virt.x < 0) {
			factor = -1;
		}
		pos_virt.x -= factor*data.getKernelRadius();
		

		//estimate the velocity at that point in space
		Vector3d vel_virt(0, 0, 0);
		{
			//sadly my macros are not made to check the neighborhood of a random position in space so I have to do some manual initialisation
			ITER_NEIGHBORS_INIT_CELL_COMPUTATION(pos_virt, data.getKernelRadius(), data.gridOffset);
			unsigned int successive_cells_count = (x > 0) ? 3 : 2;
			x = (x > 0) ? x - 1 : x;

			//do the actual velocity estimation
			RealCuda  total_weight = 0;
			ITER_NEIGHBORS_FROM_STRUCTURE(particleSet->neighborsDataSet, particleSet->pos,
				{
					RealCuda weight = data.W(pos_virt - particleSet->pos[j]);
					total_weight += weight;
					vel_virt += weight * particleSet->vel[j];
				}
			);
			vel_virt /= total_weight;
		}
		///TODO you have to apply the normal of the surface to get the actual velocity
		RealCuda limit_vel = 0.4;
		RealCuda surface_vel = vel_virt.x*factor;
		if (surface_vel < limit_vel) {
			//change the tag to keep the particle
			tag = 0;
		}
		if (false) {
			surface_vel = particleSet->vel[i].x*factor;
			if (surface_vel < limit_vel) {
				//change the tag to keep the particle
				tag = 0;
			}
		}

		if (tag != 0) {
			//We need to distribute the mass to the neighboring boundaries to not have a "hole" in the fluid 

			//for that I'll distribute the wiegth between the 3 closest boundaries particles
			int idxs[3];
			RealCuda dists[3];
			for (int l = 0; (l < 3); ++l) {
				dists[l] = 1000;//just a huge base value so the algo is generic
			}

			ITER_NEIGHBORS_INIT(data, particleSet, i);
			neighbors_ptr += particleSet->getNumberOfNeighbourgs(i)*numParticles;
			end_ptr = neighbors_ptr;
			ITER_NEIGHBORS_BOUNDARIES(data, particleSet, i, 
				{
					//printf("test: %d,  %f , %f,  %f \n", i, pos_i.x , body.pos[neighborIndex].x ,data.boundaries_data_cuda->pos[neighborIndex].x);
					int idx = neighborIndex;
					RealCuda dist = (pos_i-body.pos[neighborIndex]).norm();
					if (dist < dists[2]) {

						idxs[2] = idx;
						dists[2] = dist;
						for (int l = 2; l>0; --l) {
							if (dists[l] < dists[l - 1]) {
								idx = idxs[l - 1];
								dist = dists[l - 1];
								idxs[l - 1] = idxs[l];
								dists[l - 1] = dists[l];
								idxs[l] = idx;
								dists[l] = dist;
							}
						}
					}

				}
			);

			//now we can distriute the mass, the weight I use are the kernel function distance
			RealCuda totalWeight = 0;
			for (int l = 0; (l < 3); ++l) {
				if (dists[l] < 999) {
					//printf("test: %d,  %f , %f,  %f \n", i,  dists[l], pos_i.x ,data.boundaries_data_cuda->pos[idxs[l]].x);
					dists[l]=data.W(dists[l]);
					totalWeight += dists[l];
				}
			}
			//and then add it
			RealCuda amplification_factor = 5;
			for (int l = 0; (l < 3); ++l) {
				if (dists[l] < 999) {
					//printf("test: %d,  %f * %f / %f = %f\n", i, particleSet->mass[i] , dists[l] , totalWeight,
					//	particleSet->mass[i] * dists[l] / totalWeight);
					atomicAdd(&(massDeltas[idxs[l]]), particleSet->mass[i] * amplification_factor * dists[l] / totalWeight);
				}
			}
		}

	}

	//and tag the particle if it needs to be removed
	particleSet->neighborsDataSet->cell_id[i] = tag;
	if (tag != 0) {
		atomicAdd(count, 1);
	}
}

__global__ void load_wave_to_height_map_kernel(BorderHeightMap borderHeightMap, StokesWaveGenerator waveGenerator) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= borderHeightMap.samplingCount) { return; }

	//only apply the wave on the left side
	Vector3d pos_i = borderHeightMap.samplingPositions[i];
	if(pos_i.x > 1.85){
		return;
	}

	RealCuda height = 0;
	waveGenerator.getHeight(pos_i, height);
	borderHeightMap.heights[i] = height;
}


__global__ void apply_wave_velocities_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, 
	StokesWaveGenerator waveGenerator, BufferFluidSurface S_oceanBorder, RealCuda transitionLength) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }
		
	Vector3d pos_i = particleSet->pos[i];
	RealCuda dist = S_oceanBorder.distanceToSurfaceSigned(pos_i);
	//if (dist<-transitionLength) {return;}
	if (!S_oceanBorder.isinside(pos_i)) { return; }

	//now we still have a huge problem: what happens if there is a particle above the procedural wave surface
	//I have no answer for that hopefully it doesn't realy happens but for now I'll just apply the velocity that is observed at
	//the wave surface verticaly
	//hum try to leacve the particles above the surface free
	RealCuda height;
	waveGenerator.getHeight(pos_i, height);
	if (height < pos_i.y) {
		pos_i.y = height;
		//return;
	}

	Vector3d vel;
	waveGenerator.getvel(pos_i, vel);

	//2 cases if we are inside the surface I apply the full velocity
	//but if it's on the other side I only apply part of it proportionaly to the distance to the surface
	if (dist < 0) {
		RealCuda coef = MIN_MACRO_CUDA(1.0, (-dist) / transitionLength);
		vel = vel * (1 - coef) + particleSet->vel[i] * coef;
	}

	//and apply
	particleSet->vel[i].x = vel.x;
	
	//particleSet->vel[i].z = vel.z;
}

__global__ void rmv_mass_delta_kernel(SPH::UnifiedParticleSet* particleSet, RealCuda* massDeltas) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	particleSet->mass[i] -= massDeltas[i];

	//let's do a 5% decay
	massDeltas[i] *= 0.99;
}

__global__ void add_mass_delta_kernel(SPH::UnifiedParticleSet* particleSet, RealCuda* massDeltas) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }


	particleSet->mass[i] += massDeltas[i];
}

void DynamicWindow::handleOceanBoundariesTestCurrent(SPH::DFSPHCData& data) {
	UnifiedParticleSet* particleSet = data.fluid_data;
	
	//just to be surre I have the necessary space to manipulate the particle number
	if (particleSet->numParticles > (particleSet->numParticlesMax*0.75)) {
		particleSet->changeMaxParticleNumber(particleSet->numParticlesMax*1.25);
	}

	//currents are less compliated
	//just define the input area and set the velocity
	static BufferFluidSurface S_inflow;
	static BufferFluidSurface S_current;
	//this is the lengtharound the current to do the velocity transition
	RealCuda transition_length = 0.10;
	Vector3d current_velocity = Vector3d(3.0,0,0);

	static int* int_ptr = NULL;

	//this is to handle the outflow
	//the first one define the area where particle are considerated to leave the simulation
	//And I need a fast structure to define the ocean profile
	static BufferFluidSurface S_fluidInterior;


	static RealCuda time = 0;
	time += 0.003;	
	static StokesWaveGenerator waveGenerator;

	//another surfece I use to apply the wave velocity
	static BufferFluidSurface S_borderWaveVelocity;

	//trying to handle the outflow kassotis style,
	//I'll need to store the mass deltas
	static RealCuda* massDeltas;

	Vector3d sim_min(-8, 0, 0.5);
	Vector3d sim_max(8, 7, 0.5);

	BorderHeightMap;

	static bool first_time = true;
	if (first_time) {
		first_time = false;

		/*
		for (int i = 0; i < 40; ++i) {
			RealCuda coef = 1 + i * 0.1;
			std::cout <<"kernel study: "<< coef<<"  //   " <<data.W((coef)*data.particleRadius) << std::endl;

		}//*/


		//define the border area that is used to add the particles
		S_inflow.setPlane(Vector3d(sim_min.x+0.075, 0, 0), Vector3d(-1, 0, 0));
		std::cout << "inflow surface: " << S_inflow.toString() << std::endl;

		//and the surface for the current
		S_current.setCuboid(Vector3d(sim_min.x, 0.0, 0),Vector3d(0.5,10,10));
		std::cout << "current surface: " << S_current.toString() << std::endl;


		S_fluidInterior.setCuboid(Vector3d(-1.0, 0.0, 0.0), Vector3d(sim_max.x +1 - 0.075, 100, 100));
		//S_fluidInterior.addPlane(Vector3d(1.9, 0, 0), Vector3d(-1, 0, 0));

		cudaMallocManaged(&(int_ptr), sizeof(int));

		//wave model (might need to be changed the current one is way slow
		waveGenerator.init(data.particleRadius, Vector3d(-sim_min.x, 0, 0), Vector3d(sim_min.x+2, 2.0, 0));

		//just some other parameters
		S_borderWaveVelocity.setPlane(Vector3d(sim_min.x-1,0,0), Vector3d(-1, 0, 0));

		cudaMallocManaged(&(massDeltas), sizeof(RealCuda)*data.boundaries_data->numParticles);
		{
			int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
			cuda_setBufferToValue_kernel<RealCuda> << <numBlocks, BLOCKSIZE >> > (massDeltas,0,data.boundaries_data->numParticles);
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	//init everything for now
	initStep(data, Vector3d(0, 0, 0), false, false);
	//particleSet->initNeighborsSearchData(data, false, false);
	//full neighbor search for now
	cuda_neighborsSearch(data,true);
	particleSet->resetColor();

	bool use_stokes_wave = false;

	//if we use a wave we have to initialize the system
	if (use_stokes_wave) {
		waveGenerator.computeWaveState(time, 0.47, 0.003, data.particleRadius / 4.0);
	}

	//first we have to define the ocean height profile at the fluid border
	if (use_stokes_wave) {
		
		int numBlocks = borderHeightMap.samplingCount;
		load_wave_to_height_map_kernel << <numBlocks, BLOCKSIZE >> > (borderHeightMap, waveGenerator);
		gpuErrchk(cudaDeviceSynchronize());
		

	}
	
	//we need to remove the mass deltas
	if(false){
		int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
		rmv_mass_delta_kernel << <numBlocks, BLOCKSIZE >> > (data.boundaries_data->gpu_ptr, massDeltas);
		gpuErrchk(cudaDeviceSynchronize());
	}

	//the order of operation is to firsttag the particle to be removed
	//the chekc if particles need to be added before actually removeing them
	//then removing the tagged particles
	//and we finish by controling the velocity where needed

	//so tag the particles to remove
	*int_ptr = 0;
	{
		int numBlocks = calculateNumBlocks(particleSet->numParticles);
		DFSPH_handle_outflow_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, S_fluidInterior, borderHeightMap, 
			int_ptr, massDeltas);
		gpuErrchk(cudaDeviceSynchronize());
	}
	int count_particles_to_remove = *int_ptr;
	

	

	//add the particles at inflow
	*int_ptr = 0;
	if(true){
		int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
		DFSPH_handle_inflow_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->gpu_ptr, S_inflow, 
			borderHeightMap,int_ptr);
		gpuErrchk(cudaDeviceSynchronize());
	}

	int count_new_particles = *int_ptr;

	if (count_new_particles > 0) {
		std::cout << "handleOceanBoundariesTestCurrent adding particles: " << count_new_particles << std::endl;
		particleSet->updateActiveParticleNumber(particleSet->numParticles+count_new_particles);
	}

	//remove the particles that have been tagged
	if (count_particles_to_remove > 0) {
		std::cout << "handleOceanBoundariesTestCurrent removing particles particles: " << count_particles_to_remove << std::endl;

		/*
		for (int i = 0; i < data.boundaries_data->numParticles; ++i) {
			if (massDeltas[i] > 0) {
				std::cout << "massdelta on particle: " << i << std::endl;
			}
			massDeltas[i] = 0;
		}//*/

		//*
		//now use the same process as when creating the neighbors structure to put the particles to be removed at the end
		cub::DeviceRadixSort::SortPairs(particleSet->neighborsDataSet->d_temp_storage_pair_sort, particleSet->neighborsDataSet->temp_storage_bytes_pair_sort,
			particleSet->neighborsDataSet->cell_id, particleSet->neighborsDataSet->cell_id_sorted,
			particleSet->neighborsDataSet->p_id, particleSet->neighborsDataSet->p_id_sorted, particleSet->numParticles);
		gpuErrchk(cudaDeviceSynchronize());

		cuda_sortData(*particleSet, particleSet->neighborsDataSet->p_id_sorted);
		gpuErrchk(cudaDeviceSynchronize());

		//and now you can update the number of particles
		int new_num_particles = particleSet->numParticles - count_particles_to_remove;
		particleSet->updateActiveParticleNumber(new_num_particles);
		//*/
	}



	//Now we have to set the velocities at to simulate the current
	if (time<0.5){
		int numBlocks = calculateNumBlocks(particleSet->numParticles);
		DFSPH_apply_current_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, S_current, current_velocity, transition_length);
		gpuErrchk(cudaDeviceSynchronize());
	}

	//we need to qpply the zqve velocities
	if (use_stokes_wave) {
		//load the information in the ocean height structure
		int numBlocks = calculateNumBlocks(particleSet->numParticles);
		apply_wave_velocities_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, waveGenerator,
			S_borderWaveVelocity, 0);

		gpuErrchk(cudaDeviceSynchronize());
	}

	//we apply the mass deltas
	if(false){
		int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
		add_mass_delta_kernel << <numBlocks, BLOCKSIZE >> > (data.boundaries_data->gpu_ptr, massDeltas);
		gpuErrchk(cudaDeviceSynchronize());
	}

}

//this will tag certain particles depending on the required restriction type
//0==> inside
//0==> outside
//0==> on the surface
template<int restrictionType, bool override_existing_tagging>
__global__ void surface_restrict_particleset_kernel(SPH::UnifiedParticleSet* particleSet, BufferFluidSurface S, RealCuda offset, int* countRmv) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (override_existing_tagging) {
		particleSet->neighborsDataSet->cell_id[i] = i;
	}
	if (restrictionType == 0) {
		if (S.distanceToSurfaceSigned(particleSet->pos[i]) < (-offset)) {
			particleSet->neighborsDataSet->cell_id[i] += particleSet->numParticles;
			atomicAdd(countRmv, 1);
		}
	}
	else if(restrictionType == 1) {
		if (S.distanceToSurfaceSigned(particleSet->pos[i]) > (-offset)) {
			particleSet->neighborsDataSet->cell_id[i] += particleSet->numParticles;
			atomicAdd(countRmv, 1);
		}

	}
	else if(restrictionType == 2) {
		if (S.distanceToSurface(particleSet->pos[i]) > (offset)) {
			particleSet->neighborsDataSet->cell_id[i] += particleSet->numParticles;
			atomicAdd(countRmv, 1);
		}
	}
	else {
		asm("trap;");
	}
	
}

//to do it I use the normal neighbor search process, altough it mean on important thing
//the limit_distance MUST be smaller or equals to the kernel radius
//btw no need to care about synchronization between threads
//at worst particles will be tagged multiples times which only waste computation time
template<bool tag_candidate_only>
__global__ void tag_neighborhood_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, SPH::UnifiedParticleSet* particleSetToTag, RealCuda limit_distance, int count_candidates) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(particleSet->pos[i], data.getKernelRadius(), data.gridOffset);


	//override the kernel distance
	radius_sq = limit_distance*limit_distance;

	if (tag_candidate_only) {
		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(particleSetToTag->neighborsDataSet, particleSetToTag->pos,
			if (j < count_candidates) {
				particleSetToTag->neighborsDataSet->cell_id[j] = TAG_ACTIVE;
			}
		);
	}
	else {
		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(particleSetToTag->neighborsDataSet, particleSetToTag->pos,
			particleSetToTag->neighborsDataSet->cell_id[j] = TAG_ACTIVE;
		);

	}
}

template<bool compute_active_only>
__global__ void evaluate_and_tag_high_density_from_buffer_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int* countRmv, RealCuda limit_density, int count_potential_fluid) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count_potential_fluid) { return; }

	//do not do useless computation on particles that have already been taged for removal
	if (compute_active_only) {
		if (bufferSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) {
			return;
		}
	}

	Vector3d p_i = bufferSet->pos[i];

	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);


	RealCuda density = bufferSet->getMass(i) * data.W_zero;
	RealCuda density_fluid = 0;
	RealCuda density_boundaries = 0;
	
	//also has to iterate over the background buffer that now represent the air
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
		if (bufferSet->neighborsDataSet->cell_id[j] != TAG_REMOVAL) {
			if (i != j) {
				RealCuda density_delta = bufferSet->getMass(j) * KERNEL_W(data, p_i - bufferSet->pos[j]);
				density += density_delta;
				density_fluid += density_delta;
			}
		}
	);
		
		//*/

	//compute the boundaries contribution only if there is a fluid particle anywhere near
	//*
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		RealCuda density_delta = data.boundaries_data_cuda->getMass(j) * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
		density += density_delta;
		density_boundaries += density_delta;
	);
		//*/
	
	if (density > limit_density) {
		atomicAdd(countRmv, 1);
		bufferSet->neighborsDataSet->cell_id[i] = TAG_REMOVAL;
		bufferSet->color[i] = Vector3d(0, 1, 0);

	}
	

	//only for debug it's useless in the final execution
	bufferSet->density[i] = density;


	bufferSet->kappa[i] = density_fluid;
	bufferSet->kappaV[i] = density_boundaries;

}


__global__ void evaluate_density_from_background_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* backgroundSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= data.fluid_data_cuda->numParticles) { return; }

	
	Vector3d p_i = data.fluid_data_cuda->pos[i];

	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);


	RealCuda density = data.fluid_data_cuda->getMass(i) * data.W_zero;


	//also has to iterate over the background buffer that now represent the air
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(backgroundSet->neighborsDataSet, backgroundSet->pos,
				RealCuda density_delta = backgroundSet->getMass(j) * KERNEL_W(data, p_i - backgroundSet->pos[j]);
				density += density_delta;
	);

	//*/

//compute the boundaries contribution only if there is a fluid particle anywhere near
//*
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		RealCuda density_delta = data.boundaries_data_cuda->getMass(j) * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
	density += density_delta;
	);
	//*/


	//only for debug it's useless in the final execution
	data.fluid_data_cuda->density[i] = density;

}

__global__ void particle_shift_test_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, Vector3d* shift_values) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }

	//do not do useless computation on particles that have already been taged for removal
	if (bufferSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) {
		return;
	}



	Vector3d p_i = bufferSet->pos[i];

	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);


	RealCuda density = data.fluid_data_cuda->getMass(i) * data.W_zero;
	Vector3d total_shifting(0, 0, 0);

	//also has to iterate over the background buffer that now represent the air
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
		if (bufferSet->neighborsDataSet->cell_id[j] != TAG_REMOVAL) {
			if (i != j) {
				Vector3d delta_disp = bufferSet->getMass(j)/ bufferSet->density[j] * KERNEL_GRAD_W(data, p_i - bufferSet->pos[j]);
				total_shifting += delta_disp;
			}
		}
	);

	//*/

//compute the boundaries contribution only if there is a fluid particle anywhere near
//*
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		Vector3d delta_disp = data.boundaries_data_cuda->getMass(j) / data.density0 * KERNEL_GRAD_W(data, p_i - data.boundaries_data_cuda->pos[j]);
		total_shifting += delta_disp;
	);
	//*/


	//only for debug it's useless in the final execution
	shift_values[i] = total_shifting;
	
	bufferSet->pos[i]-= total_shifting*0.001;
}

//related paper: An improved particle packing algorithm for complexgeometries
template<bool compute_active_only>
__global__ void particle_packing_negi_2019_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int count_potential_fluid, RealCuda delta_s,RealCuda p_b ,
	RealCuda k_r , RealCuda zeta, RealCuda coef_to_compare_v_sq_to, RealCuda c, RealCuda r_limit, RealCuda a_rf_r_limit, RealCuda* max_v_norm_sq){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count_potential_fluid) { return; }



	//do not do useless computation on particles that have already been taged for removal
	if (compute_active_only) {
		if (bufferSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) {
			return;
		}
	}
	else {
		if (bufferSet->neighborsDataSet->cell_id[i] == TAG_REMOVAL) {
			return;
		}
	}
	Vector3d a_i = Vector3d(0, 0, 0);

	
	Vector3d p_i = bufferSet->pos[i];
	RealCuda coef_ab = -p_b*bufferSet->mass[i]/ bufferSet->density[i];
	Vector3d v_i = bufferSet->vel[i];
	//a_d
	a_i += - zeta * v_i;
	
	
	//we will compute the 3 components at once
	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);

	//itrate on the fluid particles, also only take into account the particle that are not taged for removal
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
		if (bufferSet->neighborsDataSet->cell_id[j] != TAG_REMOVAL) {
			if (i != j) {
				Vector3d x_ij = p_i - bufferSet->pos[j];

				//a_b
				a_i += coef_ab * KERNEL_GRAD_W(data,x_ij) / bufferSet->density[j];
				
				RealCuda r = x_ij.norm();
				x_ij /= r;

				//a_rf
				//note at the point x_ij is normalized so it correspond to n_ij in the formula
				//*
				if (r<delta_s && r > r_limit) {
					a_i += x_ij * k_r * ((3 * c * c) / (r * r * r * r) - (2 * c) / (r * r * r));
				}
				else {
					a_i += x_ij * a_rf_r_limit;
				}
				//*/
			}
		}
	);

	//we need to remve the mass since the boundaries have avarible mass
	coef_ab /= (bufferSet->mass[i]*data.density0);

	//and ont he boundarie
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		Vector3d x_ij = p_i - data.boundaries_data_cuda->pos[j];

		//a_b
		a_i += coef_ab * KERNEL_GRAD_W(data,x_ij) * data.boundaries_data_cuda->mass[j];

		RealCuda r = x_ij.norm();
		x_ij /= r;

		//a_rf
		//note at the point x_ij is normalized so it correspond to n_ij in the formula
		//*
		if (r<delta_s && r > r_limit) {
			a_i += x_ij * k_r * ((3 * c * c) / (r * r * r * r) - (2 * c) / (r * r * r));
		}
		else {
			a_i += x_ij * a_rf_r_limit;
		}
		//*/
	);


	//we cannot drectly modify v_i here since we do not have delta_t
	bufferSet->acc[i] = a_i;

	//do it at the end with some luck the thread ill be desynched enougth for the atomic to not matter
	RealCuda v_i_sq_norm = v_i.squaredNorm();
	if (v_i_sq_norm > coef_to_compare_v_sq_to) {
		atomicToMax(max_v_norm_sq, v_i_sq_norm);
	}
}

template<bool compute_active_only>
__global__ void advance_in_time_particleSet_kernel(SPH::UnifiedParticleSet* particleSet, RealCuda dt) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	//do not do useless computation on particles that have already been taged for removal
	if (compute_active_only) {
		if (particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) {
			return;
		}
	}
	else {
		if (particleSet->neighborsDataSet->cell_id[i] == TAG_REMOVAL) {
			return;
		}
	}

	particleSet->vel[i] += particleSet->acc[i] * dt;
	particleSet->pos[i] += particleSet->vel[i] * dt;


}


void DynamicWindowInterface::initializeFluidToSurface(SPH::DFSPHCData& data) {

	//ok so I'll use the same method as for the dynamic boundary but to initialize a fluid
	//although the buffers wont contain the same data
	//I'll code it outside of the class for now Since I gues it will be another class
	//although it might now be but it will do for now
	///TODO: For this process I only use one unified particle set, as surch, I could just work inside the actual fluid buffer

	SPH::UnifiedParticleSet* backgroundFluidBufferSet;

	

	//surface descibing the simulation space and the fluid space
	//this most likely need ot be a mesh in the end or at least a union of surfaces
	BufferFluidSurface S_simulation;
	BufferFluidSurface S_fluid;

	//harcode for now but it would need to be given as parameters...
	S_simulation.setCuboid(Vector3d(0,2.5,0), Vector3d(0.5,2.5,0.5));
	S_fluid.setCuboid(Vector3d(0,1.5,0), Vector3d(0.5,1.5,0.5));


	int* outInt = NULL;
	cudaMallocManaged(&(outInt), sizeof(int));
	RealCuda* outRealCuda = NULL;
	cudaMallocManaged(&(outRealCuda), sizeof(RealCuda));


	
	//First I have to load a new background buffer file
	Vector3d min_fluid_buffer; 
	Vector3d max_fluid_buffer;
	SPH::UnifiedParticleSet* dummy = NULL;
	backgroundFluidBufferSet = new SPH::UnifiedParticleSet();
	backgroundFluidBufferSet->load_from_file(data.fluid_files_folder + "background_buffer_file.txt", false, &min_fluid_buffer, &max_fluid_buffer, false);
	allocate_and_copy_UnifiedParticleSet_vector_cuda(&dummy, backgroundFluidBufferSet, 1);




	//I'll center the loaded dataset (for now hozontally but in the end I'll also center it vertically)
	Vector3d displacement = max_fluid_buffer + min_fluid_buffer;
	displacement /= 2;
	//on y I just put the fluid slightly below the simulation space to dodge the special distribution around the borders
	displacement.y = -min_fluid_buffer.y -0.2;
	std::cout << "background buffer displacement: " << displacement.toString() << std::endl;
	{
		int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
		apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->pos, displacement, backgroundFluidBufferSet->numParticles);
		gpuErrchk(cudaDeviceSynchronize());
	}

	//now I need to apply a restriction on the domain to limit it to the current simulated domain
	//so tag the particles
	*outInt = 0;
	{
		int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
		surface_restrict_particleset_kernel<0, true> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_simulation, -(data.particleRadius/4), outInt);
		gpuErrchk(cudaDeviceSynchronize());
	}
	//and remove the particles
	remove_tagged_particles(backgroundFluidBufferSet, *outInt);

	

	//and we can finish the initialization
	//it's mostly to have the particles sorted here just for better spacial coherence
	backgroundFluidBufferSet->initNeighborsSearchData(data, true);
	backgroundFluidBufferSet->resetColor();

	
	//now we need to generate the fluid buffer
	*outInt = 0;
	{
		int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
		surface_restrict_particleset_kernel<0, true> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_fluid, 0, outInt);
		gpuErrchk(cudaDeviceSynchronize());
	}
	int count_outside_buffer = *outInt;


	//sort the buffer
	cub::DeviceRadixSort::SortPairs(backgroundFluidBufferSet->neighborsDataSet->d_temp_storage_pair_sort, backgroundFluidBufferSet->neighborsDataSet->temp_storage_bytes_pair_sort,
		backgroundFluidBufferSet->neighborsDataSet->cell_id, backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted,
		backgroundFluidBufferSet->neighborsDataSet->p_id, backgroundFluidBufferSet->neighborsDataSet->p_id_sorted, backgroundFluidBufferSet->numParticles);
	gpuErrchk(cudaDeviceSynchronize());

	cuda_sortData(*backgroundFluidBufferSet, backgroundFluidBufferSet->neighborsDataSet->p_id_sorted);
	gpuErrchk(cudaDeviceSynchronize());



	{
		std::cout << "init end check values" << std::endl;
		Vector3d min, max;
		get_UnifiedParticleSet_min_max_naive_cuda(*(backgroundFluidBufferSet), min, max);
		std::cout << "buffer informations: count particles (potential fluid)" << backgroundFluidBufferSet->numParticles << "  (" << backgroundFluidBufferSet->numParticles-count_outside_buffer << ") ";
		std::cout << " min/max " << min.toString() << " // " << max.toString() << std::endl;
	}
	

	
	////!!! WARNING from now on we must NEVER resort the particles or make sure the potential fluid particles stays first in the buffer

	//reinitialise the neighbor structure (might be able to delete it if I never use it
	//though I will most likely use it
	backgroundFluidBufferSet->initNeighborsSearchData(data, false);

	//OK the the initialiation is done and now I can start removing the particles causing surpressions
	//in the particular case of the initialization 
	//the only place where there can be any surpression is at the boundary
	//there are two ways to do it 
	//1 iterate on the boundaries particles and tag the fluid particles that are nearby
	//2 use the surface
	//The problem with 2 is that for complex simulation domains that are an union of surface 
	//OK there is a way simply check the distance to the surface if it is far inside any of the surfaces in the union then it is not realy on the surface
	//now as for wich one is the better one ... most likely the first one is better (although slower) in particular if I have
	//an object that has been generated from a complex mesh because I'm gessing the distance to a mesh might be slower that the distance to boundaries particles
	
	backgroundFluidBufferSet->resetColor();

	//a backup for some tests
	//*
	int background_numParticles = backgroundFluidBufferSet->numParticles;
	Vector3d* background_pos_backup = NULL;
	cudaMallocManaged(&(background_pos_backup), backgroundFluidBufferSet->numParticles * sizeof(Vector3d));
	gpuErrchk(cudaMemcpy(background_pos_backup, backgroundFluidBufferSet->pos, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
	//*/


	//first reset the cell_id buffer because I'll use it for tagging
	set_buffer_to_value<unsigned int>(backgroundFluidBufferSet->neighborsDataSet->cell_id, 0, backgroundFluidBufferSet->numParticles);
	set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->density, 0, backgroundFluidBufferSet->numParticles);
	set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->kappa, 0, backgroundFluidBufferSet->numParticles);
	set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->kappaV, 0, backgroundFluidBufferSet->numParticles);


	int count_potential_fluid = backgroundFluidBufferSet->numParticles -count_outside_buffer;

	//*
	//then do a preliminary tag to identify the particles that are close to the boundaries
	{
		int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
		tag_neighborhood_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, data.getKernelRadius(), count_potential_fluid);
		gpuErrchk(cudaDeviceSynchronize());
	}

	if (false) {
		for (int i = 0; i < backgroundFluidBufferSet->numParticles; i++) {
				backgroundFluidBufferSet->kappaV[i] = backgroundFluidBufferSet->neighborsDataSet->cell_id[i];
			
		}
	}

	//now we can use the iterative process to remove particles that have a density to high
	//no need to lighten the buffers for now since I only use one
	int total_to_remove = 0;
	for (int i = 0; i < 12; i++) {
		RealCuda limit_density = 1500-25*i;
		*outInt = 0;
		{
			int numBlocks = calculateNumBlocks(count_potential_fluid);
			evaluate_and_tag_high_density_from_buffer_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, limit_density, count_potential_fluid);
			gpuErrchk(cudaDeviceSynchronize());
		}
		total_to_remove += *outInt;

		std::cout << "initializeFluidToSurface: fitting fluid, iter: " << i << " nb removed (this iter): " << total_to_remove << " ("<<*outInt<<") " << std::endl;
		{
			RealCuda min_density = 10000;
			RealCuda max_density = 0;
			RealCuda avg_density = 0;
			int count = 0;
			for (int j = 0; j < count_potential_fluid; ++j) {
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE) {
					avg_density += backgroundFluidBufferSet->density[j];
					min_density = std::fminf(min_density, backgroundFluidBufferSet->density[j]);
					max_density = std::fmaxf(max_density, backgroundFluidBufferSet->density[j]);
					count++;
				}
			}
			avg_density /= count;
			std::cout << "                  avg/min/ max density this iter : " << avg_density << " / " << min_density << " / " << max_density << std::endl;
		}
	}
	//*/

	//a test of the particle shifting (that does not use the concentration as a differencial but directly compute the concentration gradiant)
	if (false){
		set_buffer_to_value<Vector3d>(background_pos_backup, Vector3d(0, 0, 0), background_numParticles);
		/*
		for (int i = 0; i < backgroundFluidBufferSet->numParticles; i++) {
			if (backgroundFluidBufferSet->neighborsDataSet->cell_id[i] == 0) {
				backgroundFluidBufferSet->neighborsDataSet->cell_id[i] = 1;
			}
		}//*/

		{
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			evaluate_and_tag_high_density_from_buffer_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, 4000, backgroundFluidBufferSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());
		}

		{
			RealCuda min_density = 10000;
			RealCuda max_density = 0;
			for (int j = 0; j < count_potential_fluid; ++j) {
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE) {
					min_density = std::fminf(min_density, backgroundFluidBufferSet->density[j]);
					max_density = std::fmaxf(max_density, backgroundFluidBufferSet->density[j]);
				}

			}
			std::cout << "min/ max density preshift : " << min_density << "  " << max_density << std::endl;
		}


	
		{
			int numBlocks = calculateNumBlocks(count_potential_fluid);
			particle_shift_test_kernel << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, background_pos_backup);
			gpuErrchk(cudaDeviceSynchronize());
		}

		{
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			evaluate_and_tag_high_density_from_buffer_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, 4000, backgroundFluidBufferSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());
		}

		{
			RealCuda min_density = 10000;
			RealCuda max_density = 0;
			for (int j = 0; j < count_potential_fluid; ++j) {
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE) {
					min_density = std::fminf(min_density, backgroundFluidBufferSet->density[j]);
					max_density = std::fmaxf(max_density, backgroundFluidBufferSet->density[j]);
				}

			}
			std::cout << "min/ max density postshift : " << min_density << "  " << max_density << std::endl;
		}


	}


	//ok let's try with a particle packing algorithm
	//this algo come from :
	//An improved particle packing algorithm for complexgeometries
	if (false) {
		//so first initialize the density for all particles since we are gonna need it
		//also set a density limit way high to be sure no aditional particles get tagged
		{
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			evaluate_and_tag_high_density_from_buffer_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, 4000, backgroundFluidBufferSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());
		}
		{
			RealCuda min_density = 10000;
			RealCuda max_density = 0;
			RealCuda avg_density = 0;
			int count = 0;
			for (int j = 0; j < count_potential_fluid; ++j) {
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE) {
					avg_density += backgroundFluidBufferSet->density[j];
					min_density = std::fminf(min_density, backgroundFluidBufferSet->density[j]);
					max_density = std::fmaxf(max_density, backgroundFluidBufferSet->density[j]);
					count++;
				}
			}
			avg_density /= count;
			std::cout << "avg/min/ max density preshift : " << avg_density << " / " << min_density << " / " << max_density << std::endl;
		}



		//prepare all cnstants (in the end I'll move them
		//params
		RealCuda delta_s = data.particleRadius * 2;
		RealCuda p_b = 25000 *delta_s;
		RealCuda k_r = 150*delta_s*delta_s;
		RealCuda zeta = 2*(SQRT_MACRO_CUDA(delta_s)+1)/delta_s;
		
		RealCuda dt_pb = 0.1 * data.getKernelRadius() / SQRT_MACRO_CUDA(p_b);
		RealCuda dt_zeta_first = SQRT_MACRO_CUDA(0.1 * data.getKernelRadius() / zeta);
		RealCuda coef_to_compare_v_sq_to = (dt_zeta_first * dt_zeta_first) / (dt_pb * dt_pb);
		coef_to_compare_v_sq_to *= coef_to_compare_v_sq_to;
		
		RealCuda c = delta_s * 2.0 / 3.0;
		RealCuda r_limit = delta_s / 2;

		//ok so this is pure bullshit
		//I add that factor to make my curve fit with the one the guy fucking drawn in his paper (maybe it will help getting to the stable solution)
		k_r *= 0.03;
		//and another factor to normalize the force on the same scale as a_b
		k_r /= 700;


		std::cout << "parameters values p_b/k_r/zeta: " << p_b << "  " << k_r << "  " << zeta << std::endl;


		//I'll itegrate this cofficient inside
		k_r *= 12;

		//this is the parenthesis for the case where r it set to the limit
		RealCuda a_rf_r_limit = k_r*((3 * c * c) / (r_limit * r_limit * r_limit * r_limit) - (2 * c) / (r_limit * r_limit * r_limit));

		std::cout << "arfrlimit: " << a_rf_r_limit <<"  "<< a_rf_r_limit/k_r << std::endl;

		//and now we can compute the acceleration
		set_buffer_to_value<Vector3d>(backgroundFluidBufferSet->vel, Vector3d(0, 0, 0), backgroundFluidBufferSet->numParticles);

		for (int i = 0; i < 200; i++) {
			set_buffer_to_value<Vector3d>(backgroundFluidBufferSet->acc, Vector3d(0, 0, 0), backgroundFluidBufferSet->numParticles);
			
			*outRealCuda = -1;
			{
				int numBlocks = calculateNumBlocks(count_potential_fluid);
				particle_packing_negi_2019_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr,count_potential_fluid ,
					delta_s,p_b,k_r,zeta,coef_to_compare_v_sq_to,
					c,r_limit,a_rf_r_limit,outRealCuda);
				gpuErrchk(cudaDeviceSynchronize());
			}
			
			RealCuda dt = *outRealCuda;
			if (dt > 0) {
				dt = SQRT_MACRO_CUDA(SQRT_MACRO_CUDA(dt)) * dt_zeta_first;
			}
			else {
				dt = dt_pb;
			}

			//std::cout << "test computations dt: " << *outRealCuda << "  " << coef_to_compare_v_sq_to << "  " << dt_pb << "  " << dt_zeta_first << std::endl;

			{
				int numBlocks = calculateNumBlocks(count_potential_fluid);
				advance_in_time_particleSet_kernel<true> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, dt);
				gpuErrchk(cudaDeviceSynchronize());
			}

			//to reevaluate the density I need to rebuild the neighborhood
			//though this would override the tagging I'm using
			//so I nee to backup the tagging and reload it after
			unsigned int* tag_array;
			cudaMallocManaged(&(tag_array), backgroundFluidBufferSet->numParticles * sizeof(unsigned int));
			gpuErrchk(cudaMemcpy(tag_array, backgroundFluidBufferSet->neighborsDataSet->cell_id, backgroundFluidBufferSet->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	
			backgroundFluidBufferSet->initNeighborsSearchData(data, false);
		
			gpuErrchk(cudaMemcpy(backgroundFluidBufferSet->neighborsDataSet->cell_id, tag_array, backgroundFluidBufferSet->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
			{
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				evaluate_and_tag_high_density_from_buffer_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, 4000, backgroundFluidBufferSet->numParticles);
				gpuErrchk(cudaDeviceSynchronize());
			}
			
			{
				RealCuda min_density = 10000;
				RealCuda max_density = 0;
				RealCuda avg_density = 0;
				Vector3d max_displacement(0);
				Vector3d min_displacement(10000000);
				Vector3d avg_displacement(0);

				for (int j = 0; j < count_potential_fluid; ++j) {
					if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE) 
					{
						avg_density += backgroundFluidBufferSet->density[j];
						min_density = std::fminf(min_density, backgroundFluidBufferSet->density[j]);
						max_density = std::fmaxf(max_density, backgroundFluidBufferSet->density[j]);
						avg_displacement += (backgroundFluidBufferSet->acc[j]).abs();//dt * dt *
						min_displacement.toMin(backgroundFluidBufferSet->acc[j]);//dt * dt *
						max_displacement.toMax(backgroundFluidBufferSet->acc[j]);//dt * dt *
					}

				}
				avg_density /= count_potential_fluid;
				std::cout << "iter: " << i << "  dt: " << dt << "  avg/min/ max density ?? avg/max_displacement : " << min_density << "  " << max_density << std::endl;
				std::cout << avg_displacement.toString() << " // " << min_displacement.toString() << " // " << max_displacement.toString() << std::endl;
			}

		}


	}

	if (false) {
		for (int i = 0; i < backgroundFluidBufferSet->numParticles; i++) {
			backgroundFluidBufferSet->kappa[i] = backgroundFluidBufferSet->density[i];
		}

	}
	if (false) {
		std::cout << "here" << std::endl;
		std::ofstream myfile("temp.csv", std::ofstream::trunc);
		if (myfile.is_open())
		{
			for (int i = 0; i < count_potential_fluid; i++) {
				if(backgroundFluidBufferSet->neighborsDataSet->cell_id[i]>0)
				myfile << backgroundFluidBufferSet->neighborsDataSet->cell_id[i] << "  " << backgroundFluidBufferSet->density[i] << "  " << backgroundFluidBufferSet->kappa[i] << "  " <<
					backgroundFluidBufferSet->kappaV[i] << " "  <<background_pos_backup[i].toString()<<std::endl;
			}
			myfile.close();
		}

			
		

	}
	
	std::vector<int> ids_to_remove;
	/*
	for (int i = 0; i < backgroundFluidBufferSet->numParticles; i++) {
		if (backgroundFluidBufferSet->neighborsDataSet->cell_id[i] == TAG_REMOVAL) {
			ids_to_remove.push_back(i);
			backgroundFluidBufferSet->mass[i] += 1;
		}
	}
	if (true) {
		for (int i = 0; i < ids_to_remove.size(); i++) {
			Vector3d* pos_temp = new Vector3d[backgroundFluidBufferSet->numParticles];
			read_UnifiedParticleSet_cuda(*backgroundFluidBufferSet, pos_temp, NULL, NULL);
			int id = ids_to_remove[i];
			if (backgroundFluidBufferSet->mass[id] < 1) {
				std::cout << "immediate test: " << i << "   " << background_pos_backup[id].toString() << "  // " << pos_temp[id].toString() << " ?? " << backgroundFluidBufferSet->mass[id] << std::endl;
			}
		}
		std::cout << "checked all" << std::endl;
	}
	//*/


	//now that we are done with the computations we have two things to do
	//first remove any particles that were not candidate for fluid particles (no problem juste remove en end particles)
	int new_num_particles = count_potential_fluid;
	backgroundFluidBufferSet->updateActiveParticleNumber(new_num_particles);

	if (false) {
		//I want to verify if all the particles that were tagged are still here
		//there are two way to know, the density and the cell_id
		//normally they should fit each other so let's check that everything is as expected
		int count_density_trigger = 0;
		int count_tag_trigger = 0;
		int count_id_trigger = 0;
		for (int j = 0; j < backgroundFluidBufferSet->numParticles; j++) {
			int c = 0;
			if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_REMOVAL) {
				count_tag_trigger++;
				c++;
			}

			if (backgroundFluidBufferSet->density[j] > 1050) {
				count_density_trigger++;
				c++;
			}

			for (int i = 0; i < ids_to_remove.size(); i++) {
				if (ids_to_remove[i] == j) {
					count_id_trigger++;
					c++;
				}
			}

			if (c != 0 && c != 3) {
				std::cout << "fucking hell the density, tag and stored ids do not fit " << std::endl;
			}
			
		}

		std::cout << "count triggers density tag ids: " <<count_density_trigger<<"  " <<count_tag_trigger<<"  " <<count_id_trigger<<"  " << std::endl;

	}
	if (false) {
		for (int i = 0; i < ids_to_remove.size(); i++) {
			Vector3d* pos_temp = new Vector3d[backgroundFluidBufferSet->numParticles];
			read_UnifiedParticleSet_cuda(*backgroundFluidBufferSet, pos_temp, NULL, NULL);
			int id = ids_to_remove[i];
			if (backgroundFluidBufferSet->mass[id] < 1) {
				std::cout << "immediate test: " << i << "   " << background_pos_backup[id].toString() << "  // " << pos_temp[id].toString() << " ?? " << backgroundFluidBufferSet->mass[id] << std::endl;
			}
		}
		std::cout << "checked all" << std::endl;
	}

	if (false) {
		for (int i = 0; i < ids_to_remove.size(); i++) {
			Vector3d* pos_temp = new Vector3d[backgroundFluidBufferSet->numParticles];
			read_UnifiedParticleSet_cuda(*backgroundFluidBufferSet, pos_temp, NULL, NULL);
			int id = ids_to_remove[i];
			if (backgroundFluidBufferSet->mass[id] < 1) {
				std::cout << "immediate test: " << i << "   " << background_pos_backup[id].toString() << "  // " << pos_temp[id].toString() << " ?? " << backgroundFluidBufferSet->mass[id] << std::endl;
			}
		}
		std::cout << "checked all" << std::endl;
	}


	//and now remove the partifcles that were tagged for the fitting
	remove_tagged_particles(backgroundFluidBufferSet, total_to_remove);

	if (false) {
		//i'll check if I actually removed the correct particles
		Vector3d* pos_temp = new Vector3d[backgroundFluidBufferSet->numParticles];
		read_UnifiedParticleSet_cuda(*backgroundFluidBufferSet, pos_temp, NULL, NULL);
		for (int i = 0; i < ids_to_remove.size(); i++) {
			Vector3d p_i = background_pos_backup[ids_to_remove[i]];
			for (int j = 0; j < backgroundFluidBufferSet->numParticles; j++) {
				if ((p_i - pos_temp[j]).norm() < data.particleRadius / 10) {
					std::cout << "ok huge fail: " << i<< "   "<< p_i.toString()<<"  // "<< pos_temp[j].toString()<< " ?? " << backgroundFluidBufferSet->mass[j] <<std::endl;
				}
			}
		}
		std::cout << "checked all" << std::endl;

	}


	if (false) {
		for (int j = 0; j < backgroundFluidBufferSet->numParticles; j++) {
			if(backgroundFluidBufferSet->mass[j] > 1){
				std::cout << "rhaaaaaaaaaaaaaaaa" << std::endl;
			}
		}
	}

	
	if (false) {
		//a test to see if I did some kind of fail
		backgroundFluidBufferSet->initNeighborsSearchData(data, false);
		backgroundFluidBufferSet->resetColor();
		
		set_buffer_to_value<unsigned int>(backgroundFluidBufferSet->neighborsDataSet->cell_id, 0, backgroundFluidBufferSet->numParticles);
		set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->density, 0, backgroundFluidBufferSet->numParticles);
		
		{
			int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
			tag_neighborhood_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, data.getKernelRadius(), backgroundFluidBufferSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());
		}


		RealCuda limit_density = 1050;
		*outInt = 0;
		{
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			evaluate_and_tag_high_density_from_buffer_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, limit_density, backgroundFluidBufferSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());
		}

		std::cout << "initializeFluidToSurface: fitting fluid, iter: " << backgroundFluidBufferSet->numParticles << "(this iter): (" << *outInt << ") " << std::endl;


		Vector3d* pos_temp = new Vector3d[backgroundFluidBufferSet->numParticles];
		read_UnifiedParticleSet_cuda(*backgroundFluidBufferSet, pos_temp, NULL, NULL);
		for (int j = 0; j < backgroundFluidBufferSet->numParticles; j++) {
			if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j]==TAG_REMOVAL) {
				std::cout << "verifiaction chack: " << j << "   " << backgroundFluidBufferSet->density[j] << "    " << backgroundFluidBufferSet->kappa[j] << "    " << 
					backgroundFluidBufferSet->kappaV[j] << "  //  "<<pos_temp[j].toString()<<std::endl;
			}
		}

	}


	//Now we have our fluid particles and we can just put them in the actual fluid buffer
	data.fluid_data->updateActiveParticleNumber(backgroundFluidBufferSet->numParticles);

	gpuErrchk(cudaMemcpy(data.fluid_data->mass, backgroundFluidBufferSet->mass, backgroundFluidBufferSet->numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(data.fluid_data->pos, backgroundFluidBufferSet->pos, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(data.fluid_data->vel, backgroundFluidBufferSet->vel, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(data.fluid_data->color, backgroundFluidBufferSet->color, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
	//data.fluid_data->resetColor();

	set_buffer_to_value<Vector3d>(data.fluid_data->vel, Vector3d(0, 0, 0), data.fluid_data->numParticles);
	set_buffer_to_value<RealCuda>(data.fluid_data->kappa, 0, data.fluid_data->numParticles);
	set_buffer_to_value<RealCuda>(data.fluid_data->kappaV, 0, data.fluid_data->numParticles);
	
	//here I most likely need to run a particle shift algorithm
	//However since all existing particles shift algorithms need a special boundary model to work I cannot currently use one



	if(false){
		Vector3d* pos= new Vector3d[backgroundFluidBufferSet->numParticles];
		read_UnifiedParticleSet_cuda(*backgroundFluidBufferSet, pos, NULL, NULL, NULL);

		std::ofstream myfile("temp.csv", std::ofstream::trunc);
		if (myfile.is_open())
		{
			for (int i = 0; i < backgroundFluidBufferSet->numParticles; i++) {
			
				myfile << pos[i].toString() << std::endl;
			}
			myfile.close();
		}

	}

	//and I'll just initialize the neighor struture to sort the particles
	data.fluid_data->initNeighborsSearchData(data, true);


	//I want to try the particle packing thingy
	//So I'll need a special buffer containing all but the particles that have been added to the fluid and the particles that had a too high density
	//although I still want the particles that hadd a too hig density but are in the background and not in the fluid
	//so I reload the background with all its particles
	//backgroundFluidBufferSet->updateActiveParticleNumber(background_numParticles);
	//backgroundFluidBufferSet->initNeighborsSearchData(data, true);
	//since the particles are sorted, I want all the particles after a given point and they will be the particles outisde of the fluide
	//gpuErrchk(cudaMemcpy(backgroundFluidBufferSet->pos, (background_pos_backup+ count_potential_fluid), (backgroundFluidBufferSet->numParticles+count_potential_fluid) * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

	//now I can try to use the particle packing to improve the distribution




	//just a test to see if there is any problem in the end
	/*
	if(false){
		backgroundFluidBufferSet->updateActiveParticleNumber(background_numParticles);
		backgroundFluidBufferSet->initNeighborsSearchData(data, true);
		gpuErrchk(cudaMemcpy(backgroundFluidBufferSet->pos, background_pos_backup, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		{
			int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
			evaluate_density_from_background_kernel << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr);
			gpuErrchk(cudaDeviceSynchronize());
		}

		 {
			std::ofstream myfile("temp.csv", std::ofstream::trunc);
			if (myfile.is_open())
			{
				for (int i = 0; i < data.fluid_data->numParticles; i++) {

					myfile << data.fluid_data->density[i] << std::endl;
				}
				myfile.close();
			}
		}
	}
	//*/




}

