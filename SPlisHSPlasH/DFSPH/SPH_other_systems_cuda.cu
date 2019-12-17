#include "SPH_other_systems_cuda.h"
#include "DFSPH_core_cuda.h"

#include <stdio.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <sstream>

#include "DFSPH_define_cuda.h"
#include "DFSPH_macro_cuda.h"
#include "DFSPH_static_variables_structure_cuda.h"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DFSPH_c_arrays_structure.h"
#include "cub.cuh"



#include <curand.h>
#include <curand_kernel.h>

#include "basic_kernels_cuda.cuh"


namespace OtherSystemsCuda
{
	__global__ void init_buffer_kernel(Vector3d* buff, unsigned int size, Vector3d val) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= size) { return; }

		buff[i] = val;
	}
}

void read_last_error_cuda(std::string msg) {
	std::cout << msg << cudaGetErrorString(cudaGetLastError()) <<"  "<< std::endl;
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
	unsigned int i = BLOCKSIZE / 2;
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

void get_UnifiedParticleSet_min_max_naive_cuda(SPH::UnifiedParticleSet& particleSet, Vector3d& min, Vector3d& max) {
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
	int count_values_for_median = 3;
	//RealCuda max_height[5] = { -2, -2, -2, -2, -2 };
	RealCuda max_height[3] = { -2, -2, -2 };
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
						max_height[k] = max_height[k + 1];
					}
					max_height[is_superior] = cur_height;
				}
			}
			if (count_actual_values>(count_values_for_median - 1)) {
				break;
			}
		}
	}

	//and we keep the median value only if there are enougth particles in the column (so that the result is relatively correct)
	column_max_height[i] = (count_actual_values>(count_values_for_median - 1)) ? max_height[(count_values_for_median - 1) / 2] : -2;

}



RealCuda find_fluid_height_cuda(SPH::DFSPHCData& data) {
	SPH::UnifiedParticleSet* particleSet = data.fluid_data;


	//we will need the neighbors data to know where the particles are
	particleSet->initNeighborsSearchData(data, false);



	//so first i need to kow the fluid height
	//the main problem is that I don't want to consider splash particles
	//so I need a special kernel for that
	//first I need the highest particle for each cell
	RealCuda* column_max_height = SVS_CU::get()->column_max_height;
	{
		int numBlocks = calculateNumBlocks(CELL_ROW_LENGTH*CELL_ROW_LENGTH);
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


__global__ void tag_particles_above_limit_hight_kernel(SPH::UnifiedParticleSet* particleSet, RealCuda target_height, int* count_flagged_particles) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	//put the particles that will be removed at the end
	if (particleSet->pos[i].y > target_height) {
		particleSet->neighborsDataSet->cell_id[i] = 30000000;
		atomicAdd(count_flagged_particles, 1);
	}
}


__global__ void get_min_max_pos_kernel(SPH::UnifiedParticleSet* particleSet, Vector3d* min_o, Vector3d *max_o, RealCuda particle_radius) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= 1) { return; }

	//the problem I have is that there wont be a particle in the exact corner
	//I I'll iter on some particles to be sure to reach smth near the corner
	Vector3d min = particleSet->pos[0];
	Vector3d max = particleSet->pos[particleSet->numParticles - 1];

	for (int k = 0; k < 10; ++k) {
		Vector3d p_min = particleSet->pos[k];
		Vector3d p_max = particleSet->pos[particleSet->numParticles - (1 + k)];

		if (min.x > p_min.x) { min.x = p_min.x; }
		if (min.y > p_min.y) { min.y = p_min.y; }
		if (min.z > p_min.z) { min.z = p_min.z; }

		if (max.x < p_max.x) { max.x = p_max.x; }
		if (max.y < p_max.y) { max.y = p_max.y; }
		if (max.z < p_max.z) { max.z = p_max.z; }
	}

	min += 2 * particle_radius;
	max -= 2 * particle_radius;

	*min_o = min;
	*max_o = max;
}


__global__ void find_column_max_height_kernel(SPH::UnifiedParticleSet* particleSet, RealCuda* column_max_height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= CELL_ROW_LENGTH*CELL_ROW_LENGTH) { return; }

	int z = i / CELL_ROW_LENGTH;
	int x = i - z*CELL_ROW_LENGTH;

	RealCuda max_height = -2;

	for (int y = CELL_ROW_LENGTH - 1; y >= 0; --y) {
		int cell_id = COMPUTE_CELL_INDEX(x, y, z);
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
	max_count_width += 1;



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
		effective_id = atomicAdd(count_created_particles, 1);
	}
	else {
		effective_id = id;
	}


	//read the actual height
	pos_temp = pos_temp / data.getKernelRadius() + data.gridOffset;
	pos_temp.toFloor();
	int column_id = pos_temp.x + pos_temp.z*CELL_ROW_LENGTH;
	pos_f.y = column_max_height[column_id] + p_distance + p_distance / 4.0;

	pos_f += pos_local;

	int global_id = effective_id + particleSet->numParticles;
	particleSet->mass[global_id] = particleSet->mass[0];
	particleSet->pos[global_id] = pos_f;
	particleSet->vel[global_id] = Vector3d(0, 0, 0);
	particleSet->kappa[global_id] = 0;
	particleSet->kappaV[global_id] = 0;
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
		int numBlocks = calculateNumBlocks(numParticles);

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
			int numBlocks = calculateNumBlocks(CELL_ROW_LENGTH*CELL_ROW_LENGTH);
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
		max_count_width.y = (target_height - global_height) / (p_distance);
		max_count_width.toFloor();
		//idk why but with that computation it's missing one particle so I'll add it
		max_count_width += 1;


		int count_new_particles = max_count_width.x*max_count_width.y*max_count_width.z;

#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
		std::cout << "num particles to be added: " << count_new_particles << std::endl;
#endif

		if (count_new_particles == 0) {
			throw("asked creating 0 particles, I can just skip it and return but for now it'll stop the program because it should not happen");
		}



		//if we need more than the max number of particles then we have to reallocate everything
		if ((particleSet->numParticles + count_new_particles) > particleSet->numParticlesMax) {
			change_fluid_max_particle_number(data, (particleSet->numParticles + count_new_particles)*1.5);
		}



		int numBlocks = calculateNumBlocks(count_new_particles);
		data.destructor_activated = false;
		Vector3d border_range = width / 3;
		border_range.y = 0;

		std::cout << "border_range: " << border_range.x << " " << border_range.y << " " << border_range.z << std::endl;

		int* count_created_particles = SVS_CU::get()->count_created_particles;
		*count_created_particles = 0;
		//and place the particles in the simulation
		place_additional_particles_right_above_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, column_max_height,
			count_new_particles, border_range,
			(border_range.squaredNorm()>0) ? count_created_particles : NULL);


		gpuErrchk(cudaDeviceSynchronize());
		data.destructor_activated = true;


		//and now you can update the number of particles
		int added_particles = ((border_range.squaredNorm()>0) ? (*count_created_particles) : count_new_particles);
		int new_num_particles = particleSet->numParticles + added_particles;
		particleSet->updateActiveParticleNumber(new_num_particles);
#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
		std::cout << "new number of particles: " << particleSet->numParticles << "with num added particles: " << added_particles << std::endl;
#endif
	}

#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
	std::cout << "end fluid level control" << std::endl;
#endif

}





Vector3d get_simulation_center_cuda(SPH::DFSPHCData& data) {
	//get the min and max
	get_min_max_pos_kernel << <1, 1 >> > (data.boundaries_data->gpu_ptr, data.bmin, data.bmax, data.particleRadius);
	gpuErrchk(cudaDeviceSynchronize());

	//std::cout<<"get_simulation_center_cuda min max: "<<
	//           data.bmin->x<<"  "<<data.bmin->z<<"  "<<data.bmax->x<<"  "<<data.bmax->z<<std::endl;

	//and computethe center
	return ((*data.bmax) + (*data.bmin)) / 2;
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
	source_id = (source_id / kernel_radius) + gridOffset;
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

	if (pos == (source_id + movement)) {
		//I'll also move the paticles away
		particleSet->pos[i].y += 2.0f;
		particleSet->neighborsDataSet->cell_id[i] = 25000000;
		atomicAdd(count_moved_particles, 1);

	}
	else if (pos == (target_id - movement)) {
		int id = atomicAdd(count_possible_particles, 1);
		particleSet->neighborsDataSet->p_id_sorted[id] = i;
	}
	else if (pos == target_id || pos == source_id) {
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


__global__ void translate_borderline_particles_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, RealCuda* column_max_height,
	int* count_moved_particles,
	int* moved_particles_min, int* moved_particles_max, int* count_invalid_position,
	Vector3d movement, int count_possible_pos, int count_remaining_pos,
	RealCuda start_height_min, RealCuda start_height_max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	RealCuda affected_distance_sq = data.particleRadius*1.5;
	affected_distance_sq *= affected_distance_sq;

	RealCuda precise_affected_distance_sq = data.particleRadius * 2;// 1.5;
	precise_affected_distance_sq *= precise_affected_distance_sq;



	//compute tsome constants
	Vector3d min = *data.bmin;
	Vector3d max = *data.bmax;
	RealCuda p_distance = data.particleRadius * 2;
	Vector3d plane_unit = movement.abs() / movement.norm();
	bool positive_motion = plane_unit.dot(movement)>0;
	Vector3d plane_unit_perp = (Vector3d(1, 0, 1) - plane_unit);
	//I need to know the width I have
	Vector3d width = (max)-(min);
	//and only kee the component oriented perpendicular with the plane
	int max_count_width = width.dot(plane_unit_perp) / p_distance;
	//idk why but with that computation it's missing one particle so I'll add it
	max_count_width++;

	int max_row = (width.dot(plane_unit) / 5) / p_distance;


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
	bool remaining_particle = particleSet->neighborsDataSet->cell_id[i] == 25000000;

	//so I know I onlyhave 2 damp planes the first one being the one near the min
	for (int k = 0; k < 2; ++k) {

		Vector3d plane = data.damp_planes[k];
		if (((particleSet->pos[i] * plane_unit - plane).squaredNorm() < affected_distance_sq) || remaining_particle) {
			//let's try to estimate the density to see if there are actual surpression
			bool distance_too_short = false;
			if (remaining_particle) {
				distance_too_short = true;
			}
			else {

				if (k == 0) {
					//we can do a simple distance check in essence

					Vector3d cur_particle_pos = particleSet->pos[i];


					Vector3i cell_pos = (particleSet->pos[i] / data.getKernelRadius()).toFloor() + data.gridOffset;
					cell_pos += Vector3i(0, -1, 0);
					//ok since I want to explore the bottom cell firts I need to move in the plane
					cell_pos -= plane_unit_perp;

					//potential offset
					Vector3d particle_offset = Vector3d(0, 0, 0);
					//*



					//we skipp some cases to only move the particles that are on one side
					if (positive_motion) {
						//for positive motion the lower plane is on the source
						if (plane_unit.dot(cur_particle_pos) <= plane_unit.dot(data.damp_planes[0])) {
							continue;
							cell_pos += plane_unit * 1;//since the particle lower than that have already been moved in the direction once
						}
						else {
							cell_pos -= plane_unit * 2;
						}
					}
					else {
						//if the motion is negative then the lower plane is the target
						if (plane_unit.dot(cur_particle_pos) <= plane_unit.dot(data.damp_planes[0])) {
							//the cell that need to be explored are on row away from us
							cell_pos += plane_unit;
						}
						else {
							continue;
							//we need to move the particle we are checking toward on rows in the direction of the movement
							particle_offset = plane_unit*data.getKernelRadius()*-1;
						}
					}
					//*/

					//I only need to check if the other side of the jonction border is too close, no need to check the same side since
					//it was part of a fluid at rest
					for (int k = 0; k<3; ++k) {//that's y
						for (int l = 0; l<3; ++l) {//that's the coordinate in the plane

							Vector3i cur_cell_pos = cell_pos + plane_unit_perp*l;
							int cur_cell_id = COMPUTE_CELL_INDEX(cur_cell_pos.x, cur_cell_pos.y + k, cur_cell_pos.z);
							UnifiedParticleSet* body = data.fluid_data_cuda;
							NeighborsSearchDataSet* neighborsDataSet = body->neighborsDataSet;
							unsigned int end = neighborsDataSet->cell_start_end[cur_cell_id + 1];
							for (unsigned int cur_particle = neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {
								unsigned int j = neighborsDataSet->p_id_sorted[cur_particle];
								if ((cur_particle_pos - (body->pos[j] + particle_offset)).squaredNorm() < precise_affected_distance_sq) {
									distance_too_short = true;
									break;
								}
							}
						}
						if (distance_too_short) { break; }
					}

				}
				else {
					Vector3d cur_particle_pos = particleSet->pos[i];


					Vector3i cell_pos = (particleSet->pos[i] / data.getKernelRadius()).toFloor() + data.gridOffset;
					cell_pos += Vector3i(0, -1, 0);
					//ok since I want to explore the bottom cell firts I need to move in the plane
					cell_pos -= plane_unit_perp;

					//on the target side the cell of the right side are a copy of the left side !
					// so we have to check the row agaisnt itself
					//but we will have to translate the particles depending on the side we are on
					Vector3d particle_offset = Vector3d(0, 0, 0);


					if (positive_motion) {
						if (plane_unit.dot(cur_particle_pos) > plane_unit.dot(data.damp_planes[1])) {
							//the cell that need to be explored are on row away from us
							cell_pos -= plane_unit;
						}
						else {
							continue;
							//we need to move the particle we are checking toward on rows in the direction of the movement
							particle_offset = plane_unit*data.getKernelRadius();
						}
					}
					else {
						if (plane_unit.dot(cur_particle_pos) > plane_unit.dot(data.damp_planes[1])) {
							continue;
							cell_pos -= plane_unit * 1;//since the particle lower than that have already been moved in the direction once
						}
						else {
							cell_pos += plane_unit * 2;
						}

					}


					//I only need to check if the other side of the jonction border is too close, no need to check the same side since
					//it was part of a fluid at rest
					for (int k = 0; k<3; ++k) {//that's y
						for (int l = 0; l<3; ++l) {//that's the coordinate in the plane

							Vector3i cur_cell_pos = cell_pos + plane_unit_perp*l;
							int cur_cell_id = COMPUTE_CELL_INDEX(cur_cell_pos.x, cur_cell_pos.y + k, cur_cell_pos.z);
							UnifiedParticleSet* body = data.fluid_data_cuda;
							NeighborsSearchDataSet* neighborsDataSet = body->neighborsDataSet;
							unsigned int end = neighborsDataSet->cell_start_end[cur_cell_id + 1];
							for (unsigned int cur_particle = neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {
								unsigned int j = neighborsDataSet->p_id_sorted[cur_particle];
								if ((cur_particle_pos - (body->pos[j] + particle_offset)).squaredNorm() < precise_affected_distance_sq) {
									distance_too_short = true;
									break;
								}
							}
							if (distance_too_short) { break; }
						}
					}

				}
			}


			if (!distance_too_short) {
				//that mean this particle is not too close for another and there is no need to handle it
				continue;
			}
			else {
				//for testing purposes
				//particleSet->pos[i].y+=2.0f;
				//return;
			}


			//get a unique id to compute the position
			//int id = atomicAdd((k==0)? moved_particles_back : moved_particles_front, 1);
			int id = atomicAdd(count_moved_particles, 1);


			//before trying to position it above the surface, if we sill have palce in front of the fluid we will put it there
			if (count_remaining_pos>0) {
				id -= count_remaining_pos;
				if (id<0) {
					id += count_possible_pos;
					//so so it means we have to put that paricle in the new layer
					int ref_particle_id = particleSet->neighborsDataSet->p_id_sorted[id];
					particleSet->pos[i] = particleSet->pos[ref_particle_id] + movement*data.getKernelRadius();
					particleSet->vel[i] = particleSet->vel[ref_particle_id];
					particleSet->kappa[i] = particleSet->kappa[ref_particle_id];
					particleSet->kappaV[i] = particleSet->kappaV[ref_particle_id];

					particleSet->neighborsDataSet->cell_id[i] = 0;

					//andwe have to reexecute the loop since that new pos may be next to a border
					k = -1;
					continue;
				}
			}




			//*
			//we place one third of the particles in the front the rest is placed in the back
			bool near_min = (positive_motion) ? true : false;
			if ((id % 10) == 0) {
				near_min = !near_min;
			}

			//we repeat until the particle is placed
			for (;;) {

				if (near_min) {
					id = atomicAdd(moved_particles_min, 1);
				}
				else {
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
					pos_local += (Vector3d(1, 0, 1)*(p_distance / 2.0f))*((near_min) ? 1 : -1);
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
				if (near_min) {
					//0 is the min plane
					if (plane_unit.dot(pos_temp) < plane_unit.dot(data.damp_planes[0])) {
						pos_temp -= (movement*data.getKernelRadius());
					}
				}
				else {
					//1 is the max plane
					if (plane_unit.dot(pos_temp) > plane_unit.dot(data.damp_planes[1])) {
						pos_temp -= (movement*data.getKernelRadius());
					}
				}

				pos_temp = pos_temp / data.getKernelRadius() + data.gridOffset;
				pos_temp.toFloor();

				//read the actual height
				int column_id = pos_temp.x + pos_temp.z*CELL_ROW_LENGTH;

				RealCuda start_height = ((near_min) ? start_height_min : start_height_max) + p_distance + p_distance / 4.0;
				RealCuda min_height = column_max_height[column_id] + p_distance;

				if ((start_height + pos_local.y)<min_height) {
					//this means we can't place a particle here so I need to get another
					atomicAdd(count_invalid_position, 1);
					continue;
				}

				pos_f.y = start_height;
				pos_f += pos_local;


				particleSet->pos[i] = pos_f;
				particleSet->vel[i] = Vector3d(0, 0, 0);
				particleSet->kappa[i] = 0;
				particleSet->kappaV[i] = 0;

				//if he particle was moved we are done
				return;

			}
		}
	}
}


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
		int numBlocks = calculateNumBlocks(numParticles);

		//move the particles
		apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (particleSet->pos, mov_pos, numParticles);
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
			int numBlocks = calculateNumBlocks(CELL_ROW_LENGTH*CELL_ROW_LENGTH);
			find_column_max_height_kernel << <numBlocks, BLOCKSIZE >> > (particleSet->gpu_ptr, column_max_height);
			gpuErrchk(cudaDeviceSynchronize());
		}




		//for the fluid I don't want to "move"the fluid, I have to rmv some particles and
		//add others to change the simulation area of the fluid
		//the particles that I'll remove are the ones in the second layer when a linear index is used
		//to find the second layer just take the first particle and you add 1to the cell id on the desired direction
		unsigned int numParticles = particleSet->numParticles;
		int numBlocks = calculateNumBlocks(numParticles);

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
		std::cout << "count particle delta: (moved particles, possible particles)" << *count_rmv_particles << "  " << *count_possible_particles << std::endl;
#endif
		std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

		//compute the positions of the 2 planes where there is a junction
		//the first of the two planes need to be the source one
		//calc the postion of the jonction planes
		//we updata the min max so that it now considers the new borders
		(*(data.bmin)) += mov_pos;
		(*(data.bmax)) += mov_pos;
		gpuErrchk(cudaDeviceSynchronize());
#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
		std::cout << "test min_max_2: " << data.bmin->x << " " << data.bmin->y << " " << data.bmin->z << " " << data.bmax->x << " " << data.bmax->y << " " << data.bmax->z << std::endl;
#endif

		//min plane
		RealCuda min_plane_precision = data.particleRadius / 1000;
		Vector3d plane = (*data.bmin)*mov_axis;
		plane /= data.getKernelRadius();
		plane.toFloor();
		plane += (movement.abs() == movement) ? movement : (movement.abs() * 2);
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
		plane -= (movement.abs() == movement) ? movement : 0;
		plane *= data.getKernelRadius();
		//we need to prevent going to close to 0,0,0
		if (plane.norm() < min_plane_precision) {
			plane = mov_axis*min_plane_precision;
		}
		data.damp_planes[data.damp_planes_count++] = plane;

		//always save the source
		if (movement.abs() == movement) {
			plane = data.damp_planes[data.damp_planes_count - 2];
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
		data.damp_borders = false;
		if (data.damp_borders) {
			data.damp_borders_steps_count = 10;
			add_border_to_damp_planes_cuda(data);
		}


		//what I what here is the minimum height in the area where the article will be placed near the surface
		//for both the min side and the max side
		int min_mov_dir = ((*data.bmin / data.getKernelRadius() + data.gridOffset)*mov_axis).toFloor().norm();
		int max_mov_dir = ((*data.bmax / data.getKernelRadius() + data.gridOffset)*mov_axis).toFloor().norm();

		Vector3d width = (*data.bmax) - (*data.bmin);
		int placement_row = (width.dot(mov_axis) / 5) / data.getKernelRadius();

		RealCuda height_near_min = 100000;
		RealCuda height_near_max = 100000;

		for (int j = 0; j<CELL_ROW_LENGTH; ++j) {
			for (int i = 0; i<CELL_ROW_LENGTH; ++i) {
				int column_id = i + j*CELL_ROW_LENGTH;
				if (column_max_height[column_id] <= 0) {
					continue;
				}


				int id_mov_dir = Vector3d(i, 0, j).dot(mov_axis);


				if (abs(id_mov_dir - min_mov_dir)<placement_row) {
					height_near_min = MIN_MACRO(height_near_min, column_max_height[column_id]);
				}

				if (abs(id_mov_dir - max_mov_dir)<placement_row) {
					height_near_max = MIN_MACRO(height_near_max, column_max_height[column_id]);
				}
			}
		}
#ifdef SHOW_MESSAGES_IN_CUDA_FUNCTIONS
		std::cout << "check start positon (front back): " << height_near_max << "   " << height_near_min << std::endl;
#endif





		//transate the particles that are too close to the jonction planes
		int count_possible_pos = *count_possible_particles;
		int count_remaining_pos = MAX_MACRO(count_possible_pos - (*count_rmv_particles), 0);

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
		std::cout << "number of particles displaced: " << *count_moved_particles - *count_invalid_position << "  with " <<
			*count_rmv_particles + *count_possible_particles << " at the surface and " <<
			*count_invalid_position << " rerolled positions" << std::endl;

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		float time = std::chrono::duration_cast<std::chrono::nanoseconds> (tp1 - start).count() / 1000000.0f;
		float time_1 = std::chrono::duration_cast<std::chrono::nanoseconds> (tp2 - tp1).count() / 1000000.0f;
		float time_2 = std::chrono::duration_cast<std::chrono::nanoseconds> (end - tp2).count() / 1000000.0f;
		std::cout << "time to move fluid simu: " << time + time_1 + time_2 << " ms  (" << time << "  " << time_1 << "  " << time_2 << ")" << std::endl;
#endif


		//add the wave canceler
		//do nnot use it it do not works properly
		data.cancel_wave = false;
		if (data.cancel_wave) {
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
			data.cancel_wave_lowest_point = global_height / 2.0;

			//and now fix the 2plane where the wave needs to be stoped
			data.cancel_wave_planes[0] = (*data.bmin)*mov_axis + mov_axis*placement_row*data.getKernelRadius();
			data.cancel_wave_planes[1] = (*data.bmax)*mov_axis - mov_axis*placement_row*data.getKernelRadius();
		}

	}

	//we can now update the offset on the grid
	data.gridOffset -= movement;
	data.dynamicWindowTotalDisplacement += mov_pos;

	//and we need ot updatethe neighbor structure for the static particles
	//I'll take the easy way and just rerun the neighbor computation
	//there shoudl eb a faster way but it will be enougth for now
	data.boundaries_data->initNeighborsSearchData(data, false);
}


void add_border_to_damp_planes_cuda(SPH::DFSPHCData& data) {

	get_min_max_pos_kernel << <1, 1 >> > (data.boundaries_data->gpu_ptr, data.bmin, data.bmax, data.particleRadius);
	gpuErrchk(cudaDeviceSynchronize());


	RealCuda min_plane_precision = data.particleRadius / 1000;
	data.damp_planes[data.damp_planes_count++] = Vector3d((abs(data.bmin->x) > min_plane_precision) ? data.bmin->x : min_plane_precision, 0, 0);
	data.damp_planes[data.damp_planes_count++] = Vector3d((abs(data.bmax->x) > min_plane_precision) ? data.bmax->x : min_plane_precision, 0, 0);
	data.damp_planes[data.damp_planes_count++] = Vector3d(0, 0, (abs(data.bmin->z) > min_plane_precision) ? data.bmin->z : min_plane_precision);
	data.damp_planes[data.damp_planes_count++] = Vector3d(0, 0, (abs(data.bmax->z) > min_plane_precision) ? data.bmax->z : min_plane_precision);
	data.damp_planes[data.damp_planes_count++] = Vector3d(0, (abs(data.bmin->y) > min_plane_precision) ? data.bmin->y : min_plane_precision, 0);

}

#define GAP_PLANE_POS 5


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

__global__ void DFSPH_reset_fluid_boundaries_remove_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, int* countRmv) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	//*
	if (particleSet->pos[i].x <= (-2.0 + GAP_PLANE_POS * data.getKernelRadius())) {
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

template<int nbr_layers>
__global__ void DFSPH_evaluate_density_field_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
	SPH::UnifiedParticleSet* bufferSet, Vector3d min, Vector3d max,
	Vector3i vec_count_samples, RealCuda* samples, RealCuda* samples_after_buffer, int count_samples, RealCuda sampling_spacing) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count_samples) { return; }

	int i_local = i / nbr_layers;
	Vector3d sampling_point;
	sampling_point.x = 0;
	sampling_point.y = i_local / vec_count_samples.z;
	sampling_point.z = i_local - (sampling_point.y)* vec_count_samples.z;
	sampling_point *= sampling_spacing;

	//add the gap plane position
	RealCuda plane_pos= (-2.0 + GAP_PLANE_POS * data.getKernelRadius()) + sampling_spacing;
	sampling_point.x = plane_pos;
	for (int k = 0; k < nbr_layers; ++k) {
		if (i < ((count_samples * k) / nbr_layers)) {sampling_point.x-=sampling_spacing;}
	}
	
	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(sampling_point, data.getKernelRadius(), Vector3d(0,0,0));


	RealCuda density = 0;
	RealCuda density_after_buffer = 0;

	//*
	//compute the fluid contribution
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
		RealCuda density_delta = fluidSet->mass[j] * KERNEL_W(data, sampling_point - fluidSet->pos[j]);
		if (fluidSet->pos[j].x > plane_pos) {
			density_after_buffer += density_delta;
		}
		density += density_delta;
	);

	return;
	//compute the buffer contribution
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
		RealCuda density_delta = bufferSet->mass[j] * KERNEL_W(data, sampling_point - bufferSet->pos[j]);
		density_after_buffer += density_delta;
	);


	samples[i] = density;
	samples_after_buffer[i] = density_after_buffer;
	
}



void handle_fluid_boundries_cuda(SPH::DFSPHCData& data, bool loading) {
	SPH::UnifiedParticleSet* particleSet = data.fluid_data;

	//the structture containing the fluid buffer
	static SPH::UnifiedParticleSet* fluidBufferSet = NULL;
	static Vector3d* pos_base=NULL;
	static int numParticles_base=0;

	if (!loading) {
		if (fluidBufferSet) {
			std::string msg = "handle_fluid_boundries_cuda: trying to change the boundary buffer size";
			std::cout << msg << std::endl;
			throw(msg);
		}

		//load the fluid buffer
		SPH::UnifiedParticleSet* dummy = NULL;
		fluidBufferSet = new SPH::UnifiedParticleSet(particleSet);
		allocate_and_copy_UnifiedParticleSet_vector_cuda(&dummy, fluidBufferSet, 1);
		fluidBufferSet->resetColor();

		numParticles_base = fluidBufferSet->numParticles;
		cudaMallocManaged(&(pos_base), sizeof(Vector3d) * numParticles_base);
		gpuErrchk(cudaMemcpy(pos_base, fluidBufferSet->pos, numParticles_base * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

		return;
	}

	if (!fluidBufferSet) {
		std::string msg = "handle_fluid_boundries_cuda: you have to load a buffer first";
		std::cout << msg << std::endl;
		throw(msg);
	}

	//now when loading the boundary buffer i need first to rmv the fluid particles that are on the buffer then load the buffer
	particleSet->resetColor();
	fluidBufferSet->resetColor();
	

	//get the min for further calculations
	Vector3d min, max;
	get_UnifiedParticleSet_min_max_naive_cuda(*particleSet, min, max);

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
			cudaMallocManaged(&(ids_near_plane_fluid), sizeof(int)*ids_count_size);
			cudaMallocManaged(&(ids_near_plane_buffer), sizeof(int)* ids_count_size);
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
		gpuErrchk(cudaMemcpy(fluidBufferSet->pos, pos_base, numParticles_base * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

		//first let's do a test
		//let's compute the density on the transition plane at regular interval in the fluid 
		//then compare with the values obtained when I add the buffer back in the simulation
		{
			//first define the structure that will hold the density values
#define NBR_LAYERS 5
			RealCuda sampling_spacing = data.particleRadius;
			Vector3i vec_count_samples(0, (max.y - min.y) / sampling_spacing + 1, (max.z - min.z) / sampling_spacing + 1);
			int count_samples = vec_count_samples.y * vec_count_samples.z;
			count_samples *= NBR_LAYERS;

			static RealCuda* density_field = NULL;
			static RealCuda* density_field_after_buffer = NULL;
			static int density_field_size = 0;

			if (count_samples > density_field_size) {
				CUDA_FREE_PTR(density_field);

				density_field_size = count_samples * 1.5;
				cudaMallocManaged(&(density_field), density_field_size * sizeof(RealCuda));
				cudaMallocManaged(&(density_field_after_buffer), density_field_size * sizeof(RealCuda));

			}

			//re-init the neighbor structure to be able to compute the density
			particleSet->initNeighborsSearchData(data, false);

			{
				int numBlocks = calculateNumBlocks(count_samples);
				DFSPH_evaluate_density_field_kernel<NBR_LAYERS> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->gpu_ptr,
					min, max, vec_count_samples, density_field, density_field_after_buffer, count_samples, sampling_spacing);
				gpuErrchk(cudaDeviceSynchronize());
			}

#undef NBR_LAYERS
		}




		//a first version that is a simule mouvement of the particle that are inside the buffer so that 
		//they are slightlymore spread out to compensate the gap
		//the problem is that it results in a too low density for the buffer which iduce motion in the simulation
		if (false){

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
			DFSPH_init_buffer_velocity_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->pos, fluidBufferSet->vel, fluidBufferSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());
		}

		//now we can remove the partices from the simulation
		{

			*countRmv = 0;
			int numBlocks = calculateNumBlocks(particleSet->numParticles);
			DFSPH_reset_fluid_boundaries_remove_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, countRmv);
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
	}
}