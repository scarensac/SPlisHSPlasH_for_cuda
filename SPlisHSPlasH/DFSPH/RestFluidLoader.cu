#include "RestFluidLoader.h"
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
	class RestFLuidLoader {
	public:
		bool _isInitialized; //mean we read data from disk
		bool _isDataTagged; //mean the data is now tagged to fit the desired conditions

		int* outInt;
		RealCuda* outRealCuda;


		SPH::UnifiedParticleSet* backgroundFluidBufferSet;
		int count_potential_fluid;

		RestFLuidLoader() {
			_isInitialized = false;
			_isDataTagged = false;

			cudaMallocManaged(&(outInt), sizeof(int));
			cudaMallocManaged(&(outRealCuda), sizeof(RealCuda));
			
			backgroundFluidBufferSet = NULL;
			count_potential_fluid = 0;
		}

		~RestFLuidLoader() {

		}

		static RestFLuidLoader& getStructure() {
			static RestFLuidLoader rfl;
			return rfl;
		}


		////!!! WARNING after this function is executed we must NEVER sort the particle data in the backgroundBuffer
		void init(DFSPHCData& data);

		bool isInitialized() { return _isInitialized; }

		bool isDataTagged() { return _isDataTagged; }

		//ok here I'll test a system to initialize a volume of fluid from
		//a large wolume of fluid (IE a technique to iinit the fluid at rest)
		void tagDataToSurface(SPH::DFSPHCData& data);

		//ok here I'll test a system to initialize a volume of fluid from
		//a large wolume of fluid (IE a technique to iinit the fluid at rest)
		void loadDataToSimulation(SPH::DFSPHCData& data);
	

		//so this is a function that will be used to move around the particles in the fluid to 
		//improove the stability of the fluid when the first time step is ran
		//Warning this function will erase the current fluid data no mather what
		void stabilizeFluid(SPH::DFSPHCData& data, RestFLuidLoaderInterface::StabilizationParameters& params);
	};
}


void RestFLuidLoaderInterface::init(DFSPHCData& data) {
	RestFLuidLoader::getStructure().init(data);
}

bool RestFLuidLoaderInterface::isInitialized() {
	return RestFLuidLoader::getStructure().isInitialized();
}


void RestFLuidLoaderInterface::initializeFluidToSurface(SPH::DFSPHCData& data) {
	if (!isInitialized()) {
		init(data);
	}
	RestFLuidLoader::getStructure().tagDataToSurface(data);
	RestFLuidLoader::getStructure().loadDataToSimulation(data);
}

void RestFLuidLoaderInterface::stabilizeFluid(SPH::DFSPHCData& data, RestFLuidLoaderInterface::StabilizationParameters& params) {
	RestFLuidLoader::getStructure().stabilizeFluid(data, params);
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
	else if (restrictionType == 1) {
		if (S.distanceToSurfaceSigned(particleSet->pos[i]) > (-offset)) {
			particleSet->neighborsDataSet->cell_id[i] += particleSet->numParticles;
			atomicAdd(countRmv, 1);
		}

	}
	else if (restrictionType == 2) {
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
	radius_sq = limit_distance * limit_distance;

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
				Vector3d delta_disp = bufferSet->getMass(j) / bufferSet->density[j] * KERNEL_GRAD_W(data, p_i - bufferSet->pos[j]);
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

	bufferSet->pos[i] -= total_shifting * 0.001;
}

//related paper: An improved particle packing algorithm for complexgeometries
template<bool compute_active_only>
__global__ void particle_packing_negi_2019_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int count_potential_fluid, RealCuda delta_s, RealCuda p_b,
	RealCuda k_r, RealCuda zeta, RealCuda coef_to_compare_v_sq_to, RealCuda c, RealCuda r_limit, RealCuda a_rf_r_limit, RealCuda* max_v_norm_sq) {
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
	RealCuda coef_ab = -p_b * bufferSet->mass[i] / bufferSet->density[i];
	Vector3d v_i = bufferSet->vel[i];
	//a_d
	a_i += -zeta * v_i;


	//we will compute the 3 components at once
	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);

	//itrate on the fluid particles, also only take into account the particle that are not taged for removal
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
		if (bufferSet->neighborsDataSet->cell_id[j] != TAG_REMOVAL) {
			if (i != j) {
				Vector3d x_ij = p_i - bufferSet->pos[j];

				//a_b
				a_i += coef_ab * KERNEL_GRAD_W(data, x_ij) / bufferSet->density[j];

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
	coef_ab /= (bufferSet->mass[i] * data.density0);

	//and ont he boundarie
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		Vector3d x_ij = p_i - data.boundaries_data_cuda->pos[j];

	//a_b
	a_i += coef_ab * KERNEL_GRAD_W(data, x_ij) * data.boundaries_data_cuda->mass[j];

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





void RestFLuidLoader::init(DFSPHCData& data) {
	//Essencially this function will load the background buffer and initialize it to the desired simulation domain

	//surface descibing the simulation space and the fluid space
	//this most likely need ot be a mesh in the end or at least a union of surfaces
	BufferFluidSurface S_simulation;
	BufferFluidSurface S_fluid;

	//harcode for now but it would need to be given as parameters...
	S_simulation.setCuboid(Vector3d(0, 2.5, 0), Vector3d(0.5, 2.5, 0.5));
	S_fluid.setCuboid(Vector3d(0, 1.5, 0), Vector3d(0.5, 1.5, 0.5));

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
	displacement.y = -min_fluid_buffer.y - 0.2;
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
		surface_restrict_particleset_kernel<0, true> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_simulation, -(data.particleRadius / 4), outInt);
		gpuErrchk(cudaDeviceSynchronize());
	}
	int count_to_rmv = *outInt;

	//and remove the particles
	remove_tagged_particles(backgroundFluidBufferSet, count_to_rmv);



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
	count_potential_fluid = backgroundFluidBufferSet->numParticles - count_outside_buffer;


	//sort the buffer
	cub::DeviceRadixSort::SortPairs(backgroundFluidBufferSet->neighborsDataSet->d_temp_storage_pair_sort, backgroundFluidBufferSet->neighborsDataSet->temp_storage_bytes_pair_sort,
		backgroundFluidBufferSet->neighborsDataSet->cell_id, backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted,
		backgroundFluidBufferSet->neighborsDataSet->p_id, backgroundFluidBufferSet->neighborsDataSet->p_id_sorted, backgroundFluidBufferSet->numParticles);
	gpuErrchk(cudaDeviceSynchronize());

	cuda_sortData(*backgroundFluidBufferSet, backgroundFluidBufferSet->neighborsDataSet->p_id_sorted);
	gpuErrchk(cudaDeviceSynchronize());

	//that buffer is used for tagging in the future so set it to zero now just to be sure
	set_buffer_to_value<unsigned int>(backgroundFluidBufferSet->neighborsDataSet->cell_id, 0, backgroundFluidBufferSet->numParticles);


	{
		std::cout << "init end check values" << std::endl;
		Vector3d min, max;
		get_UnifiedParticleSet_min_max_naive_cuda(*(backgroundFluidBufferSet), min, max);
		std::cout << "buffer informations: count particles (potential fluid)" << backgroundFluidBufferSet->numParticles << "  (" << backgroundFluidBufferSet->numParticles - count_outside_buffer << ") ";
		std::cout << " min/max " << min.toString() << " // " << max.toString() << std::endl;
	}

}


void RestFLuidLoader::tagDataToSurface(SPH::DFSPHCData& data) {
	if (!isInitialized()) {
		init(data);
	}


	//ok so I'll use the same method as for the dynamic boundary but to initialize a fluid
	//although the buffers wont contain the same data
	//I'll code it outside of the class for now Since I gues it will be another class
	//although it might now be but it will do for now
	///TODO: For this process I only use one unified particle set, as surch, I could just work inside the actual fluid buffer



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
		RealCuda limit_density = 1500 - 25 * i;
		*outInt = 0;
		{
			int numBlocks = calculateNumBlocks(count_potential_fluid);
			evaluate_and_tag_high_density_from_buffer_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, limit_density, count_potential_fluid);
			gpuErrchk(cudaDeviceSynchronize());
		}
		total_to_remove += *outInt;

		std::cout << "initializeFluidToSurface: fitting fluid, iter: " << i << " nb removed (this iter): " << total_to_remove << " (" << *outInt << ") " << std::endl;
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

	_isDataTagged = true;
	// THIS FUNCTION MUST END THERE (or at least after that there should only be debug functions

	//a test of the particle shifting (that does not use the concentration as a differencial but directly compute the concentration gradiant)
	if (false) {
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
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[i] > 0)
					myfile << backgroundFluidBufferSet->neighborsDataSet->cell_id[i] << "  " << backgroundFluidBufferSet->density[i] << "  " << backgroundFluidBufferSet->kappa[i] << "  " <<
					backgroundFluidBufferSet->kappaV[i] << " " << background_pos_backup[i].toString() << std::endl;
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

		std::cout << "count triggers density tag ids: " << count_density_trigger << "  " << count_tag_trigger << "  " << count_id_trigger << "  " << std::endl;

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
					std::cout << "ok huge fail: " << i << "   " << p_i.toString() << "  // " << pos_temp[j].toString() << " ?? " << backgroundFluidBufferSet->mass[j] << std::endl;
				}
			}
		}
		std::cout << "checked all" << std::endl;

	}


	if (false) {
		for (int j = 0; j < backgroundFluidBufferSet->numParticles; j++) {
			if (backgroundFluidBufferSet->mass[j] > 1) {
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
			if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_REMOVAL) {
				std::cout << "verifiaction chack: " << j << "   " << backgroundFluidBufferSet->density[j] << "    " << backgroundFluidBufferSet->kappa[j] << "    " <<
					backgroundFluidBufferSet->kappaV[j] << "  //  " << pos_temp[j].toString() << std::endl;
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



	if (false) {
		Vector3d* pos = new Vector3d[backgroundFluidBufferSet->numParticles];
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


void RestFLuidLoader::loadDataToSimulation(SPH::DFSPHCData& data) {
	if (!isInitialized()) {
		std::cout << "RestFLuidLoader::loadDataToSimulation Loading impossible data was not initialized" << std::endl;
		return;
	}

	if (!isDataTagged()) {
		std::cout << "!!!!!!!!!!! RestFLuidLoader::loadDataToSimulation you are loading untagged data !!!!!!!!!!!" << std::endl;
		return;
	}

	//just copy all the values to the fluid
	data.fluid_data->updateActiveParticleNumber(backgroundFluidBufferSet->numParticles);

	gpuErrchk(cudaMemcpy(data.fluid_data->mass, backgroundFluidBufferSet->mass, backgroundFluidBufferSet->numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(data.fluid_data->pos, backgroundFluidBufferSet->pos, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(data.fluid_data->vel, backgroundFluidBufferSet->vel, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(data.fluid_data->color, backgroundFluidBufferSet->color, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
	//data.fluid_data->resetColor();

	//clearing the warmstart values is necessary
	set_buffer_to_value<RealCuda>(data.fluid_data->kappa, 0, data.fluid_data->numParticles);
	set_buffer_to_value<RealCuda>(data.fluid_data->kappaV, 0, data.fluid_data->numParticles);

	//I'll also clear the velocities for now since I'll load fluid at rest
	//this will have to be removed at some point in the future
	set_buffer_to_value<Vector3d>(data.fluid_data->vel, Vector3d(0, 0, 0), data.fluid_data->numParticles);

}


void RestFLuidLoader::stabilizeFluid(SPH::DFSPHCData& data, RestFLuidLoaderInterface::StabilizationParameters& params) {

	if (params.method == 0) {
		//so the first method will be to actually simulate the fluid while potentially restricting it
		//this need to be fully manipulable to potentially only activate part of the simulation process
		//as for simulation only part of the simulation domain it may be hard but by sorting the data in the right order it may be feasible with the current implementation though the cache hit rate will go down hard
		//worst case I'll have to copy all the functions 
		//a relatively easy way to add a particle restriction to the current implementation would be to use a macro that I set to nothing when  don't consider tagging and set to a return when I do
		//but I'll think more about that later (you can also remove from the simulation all particle that are so far from the usefull ones that they wont have any impact 

		//first the data to the simulation
		loadDataToSimulation(data);

		//pretty much all that will have to be added to the params and this will be replace by a block reading the parameters
		bool useDivergenceSolver = params.useDivergenceSolver;
		bool useDensitySolver = params.useDensitySolver;
		bool useExternalForces = params.useExternalForces;
		RealCuda maxErrorV = params.maxErrorV;
		RealCuda maxIterV = params.maxIterV;
		RealCuda maxErrorD = params.maxErrorD;
		RealCuda maxIterD = params.maxIterD;
		RealCuda timeStep = params.timeStep;

		//for damping and clamping
		bool preUpdateVelocityClamping = params.preUpdateVelocityClamping;
		RealCuda preUpdateVelocityClamping_val = params.preUpdateVelocityClamping_val;
		bool preUpdateVelocityDamping = params.preUpdateVelocityDamping;
		RealCuda preUpdateVelocityDamping_val = params.preUpdateVelocityDamping_val;
		bool postUpdateVelocityClamping = params.postUpdateVelocityClamping;
		RealCuda postUpdateVelocityClamping_val = params.postUpdateVelocityClamping_val;
		bool postUpdateVelocityDamping = params.postUpdateVelocityDamping;
		RealCuda postUpdateVelocityDamping_val = params.postUpdateVelocityDamping_val;

		//for the particle checking necessary to reject overly bad simulations
		//though need to be deactivated for best timmings
		bool runCheckParticlesPostion = params.runCheckParticlesPostion;
		bool interuptOnLostParticle = params.interuptOnLostParticle;

		int iterV = 0;
		int iterD = 0;

		//all this process will be done with a constant timestep so I'll do that to make sure there is no initialization problem
		RealCuda old_timeStep = data.get_current_timestep();
		data.updateTimeStep(timeStep);
		data.updateTimeStep(timeStep);

		for (int iter = 0; iter < params.max_iter; iter++) {
			//neighborsearch 
			cuda_neighborsSearch(data, false);

			//divergence
			if (useDivergenceSolver)
			{
				iterV = cuda_divergenceSolve(data, maxIterV, maxErrorV);
			}
			else {
				//even if I don't use the warm start I'll still need that since it compute the density and everything
				//technically it even compute too much...
				cuda_divergence_warmstart_init(data);
			}

			//external forces
			if (useExternalForces) {
				cuda_externalForces(data);
				cuda_update_vel(data);
			}

			//density
			if (useDensitySolver) {
				iterD = cuda_pressureSolve(data, maxIterD, maxErrorD);
			}

			if (preUpdateVelocityClamping) {
				clamp_buffer_to_value<Vector3d, 4>(data.fluid_data->vel, Vector3d(preUpdateVelocityClamping_val), data.fluid_data->numParticles);
			}

			if (preUpdateVelocityDamping) {
				apply_factor_to_buffer(data.fluid_data->vel, Vector3d(preUpdateVelocityDamping_val), data.fluid_data->numParticles);
			}


			cuda_update_pos(data);


			if (postUpdateVelocityClamping) {
				clamp_buffer_to_value<Vector3d, 4>(data.fluid_data->vel, Vector3d(postUpdateVelocityClamping_val), data.fluid_data->numParticles);
			}

			if (postUpdateVelocityDamping) {
				apply_factor_to_buffer(data.fluid_data->vel, Vector3d(postUpdateVelocityDamping_val), data.fluid_data->numParticles);
			}


			//this will have to be commented by the end because it is waiting computation time if  the fluid is stable
			data.checkParticlesPositions(2);
		}

		//I need to clear the warmstart and velocity buffer
		set_buffer_to_value<RealCuda>(data.fluid_data->kappa,0,data.fluid_data->numParticles);
		set_buffer_to_value<RealCuda>(data.fluid_data->kappaV,0,data.fluid_data->numParticles);
		set_buffer_to_value<Vector3d>(data.fluid_data->vel,Vector3d(0),data.fluid_data->numParticles);

		//set the timestep back to the previous one
		data.updateTimeStep(old_timeStep);
		data.updateTimeStep(old_timeStep);


	}else if (params.method == 1) {
		//ok let's try with a particle packing algorithm
		//this algo come from :
		//An improved particle packing algorithm for complexgeometries
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
		RealCuda p_b = 25000 * delta_s;
		RealCuda k_r = 150 * delta_s * delta_s;
		RealCuda zeta = 2 * (SQRT_MACRO_CUDA(delta_s) + 1) / delta_s;

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
		RealCuda a_rf_r_limit = k_r * ((3 * c * c) / (r_limit * r_limit * r_limit * r_limit) - (2 * c) / (r_limit * r_limit * r_limit));

		std::cout << "arfrlimit: " << a_rf_r_limit << "  " << a_rf_r_limit / k_r << std::endl;

		//and now we can compute the acceleration
		set_buffer_to_value<Vector3d>(backgroundFluidBufferSet->vel, Vector3d(0, 0, 0), backgroundFluidBufferSet->numParticles);

		for (int i = 0; i < 200; i++) {
			set_buffer_to_value<Vector3d>(backgroundFluidBufferSet->acc, Vector3d(0, 0, 0), backgroundFluidBufferSet->numParticles);

			*outRealCuda = -1;
			{
				int numBlocks = calculateNumBlocks(count_potential_fluid);
				particle_packing_negi_2019_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, count_potential_fluid,
					delta_s, p_b, k_r, zeta, coef_to_compare_v_sq_to,
					c, r_limit, a_rf_r_limit, outRealCuda);
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
	else {
		std::cout << " RestFLuidLoader::stabilizeFluid no stabilization method selected" << std::endl;
	}

	   
	//for the evaluation
	if (params.evaluateStabilization) {
		//I can see 2 ways to evaluate the result
		//1: you check the density though sadly the density near the surface will cause evaluation problems
		//2: runa normal simulation step and check the velocities. Probably better since It will show you how to particle should move
		//		you have 2 main ways of doing that evaluation: check the max, check the avg at the border
		//btw you could do it on multiples simulation steps but since I want smth that is perfecty stable immediatly i'll just evaluate on one for now
		//though it might not be smart mayb there will be a curretn that slowly accumulate
		//SO maybe I'll code smth that incorporate time in the future


		RealCuda old_timeStep = data.get_current_timestep();
		data.updateTimeStep(params.timeStepEval);
		data.updateTimeStep(params.timeStepEval);


		//for now I'll use the solution of checking the max
		cuda_neighborsSearch(data, false);

		cuda_divergenceSolve(data, params.maxIterVEval, params.maxErrorVEval);

		cuda_externalForces(data);
		cuda_update_vel(data);

		cuda_pressureSolve(data, params.maxIterDEval, params.maxErrorDEval);

		cuda_update_pos(data);

		//set the timestep back to the previous one
		data.updateTimeStep(old_timeStep);
		data.updateTimeStep(old_timeStep);

		//read data to CPU
		static Vector3d* vel = NULL;
		int size = 0;
		if (data.fluid_data->numParticles > size) {
			if (vel != NULL) {
				delete[] vel;
			}
			vel = new Vector3d[data.fluid_data->numParticlesMax];
			size = data.fluid_data->numParticlesMax;

		}
		read_UnifiedParticleSet_cuda(*(data.fluid_data), NULL, vel, NULL);

		//read the actual evaluation
		RealCuda stabilzationEvaluation = -1;

		for (int i = 0; i < data.fluid_data->numParticles; ++i) {
			stabilzationEvaluation = MAX_MACRO_CUDA(stabilzationEvaluation, vel[i].squaredNorm());
		}
		SQRT_MACRO_CUDA(stabilzationEvaluation);

		params.stabilzationEvaluation = stabilzationEvaluation;
	}

}