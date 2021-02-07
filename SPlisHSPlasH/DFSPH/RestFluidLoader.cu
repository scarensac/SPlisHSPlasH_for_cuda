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

		unsigned int* tag_array_with_existing_fluid;
		unsigned int tag_array_with_existing_fluid_size;

		//if this is equals to 0 after tagging this mean the tagging phase extracted them
		int count_high_density_tagged_in_potential;
		int count_high_density_tagged_in_air;

		//a variable that tell me if I already have computed the active and active neghbor tagging
		bool _hasFullTaggingSaved;

		RestFLuidLoader() {
			_isInitialized = false;
			_isDataTagged = false;
			_hasFullTaggingSaved = false;

			cudaMallocManaged(&(outInt), sizeof(int));
			cudaMallocManaged(&(outRealCuda), sizeof(RealCuda));
			
			backgroundFluidBufferSet = NULL;
			count_potential_fluid = 0;
			count_high_density_tagged_in_potential = 0;
			count_high_density_tagged_in_air = 0;

			tag_array_with_existing_fluid=NULL;
			tag_array_with_existing_fluid_size=0;
		}

		~RestFLuidLoader() {

		}

		static RestFLuidLoader& getStructure() {
			static RestFLuidLoader rfl;
			return rfl;
		}

		void clear();

		////!!! WARNING after this function is executed we must NEVER sort the particle data in the backgroundBuffer
		//an explanation for the air particle range
		//it is used to limit the amount of air particles that are kept 
		//-1: no restriction; 0: no air(don't use that...), 1: air that are neighbors to fluid, 2: air that are neigbors to air particles that are neighbors to fluid
		void init(DFSPHCData& data, RestFLuidLoaderInterface::InitParameters& params);

		bool isInitialized() { return _isInitialized; }

		bool isDataTagged() { return _isDataTagged; }

		bool hasFullTaggingSaved() { return _hasFullTaggingSaved; }

		//ok here I'll test a system to initialize a volume of fluid from
		//a large wolume of fluid (IE a technique to iinit the fluid at rest)
		void tagDataToSurface(SPH::DFSPHCData& data, RestFLuidLoaderInterface::TaggingParameters& params);

		//ok here I'll test a system to initialize a volume of fluid from
		//a large wolume of fluid (IE a technique to iinit the fluid at rest)
		//return the number of FLUID particles (it may be less than the number of loaded particle since there are air particles)
		int loadDataToSimulation(SPH::DFSPHCData& data, RestFLuidLoaderInterface::LoadingParameters& params);

		//so this is a function that will be used to move around the particles in the fluid to 
		//improove the stability of the fluid when the first time step is ran
		//Warning this function will erase the current fluid data no mather what
		void stabilizeFluid(SPH::DFSPHCData& data, RestFLuidLoaderInterface::StabilizationParameters& params);
	};
}

void RestFLuidLoaderInterface::clear() {
	RestFLuidLoader::getStructure().clear();
}

void RestFLuidLoaderInterface::init(DFSPHCData& data, InitParameters& params) {
	RestFLuidLoader::getStructure().init(data, params);
}

bool RestFLuidLoaderInterface::isInitialized() {
	return RestFLuidLoader::getStructure().isInitialized();
}


void RestFLuidLoaderInterface::initializeFluidToSurface(SPH::DFSPHCData& data, bool center_loaded_fluid, TaggingParameters& params, 
	LoadingParameters& params_loading) {
	std::vector<std::string> timing_names{ "init","tag","load" };
	SPH::SegmentedTiming timings("RestFLuidLoaderInterface::initializeFluidToSurface", timing_names, true);
	timings.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)

	if (!isInitialized()) {
		InitParameters initParams;
		initParams.keep_existing_fluid = params.keep_existing_fluid;
		initParams.center_loaded_fluid = center_loaded_fluid;
		initParams.air_particles_restriction = 1;
		init(data,initParams);
	}


	timings.time_next_point();//time p1
	
	RestFLuidLoader::getStructure().tagDataToSurface(data,params);
	
	timings.time_next_point();//time p2
	
	if (params_loading.load_fluid) {
		int count_fluid = RestFLuidLoader::getStructure().loadDataToSimulation(data, params_loading);
	}
	
	timings.time_next_point();//time p3
	timings.end_step();//end point of the current step (if measuring avgs you need to call it at every end of the loop)
	timings.recap_timings();//writte timming to cout

	params.time_total = timings.getTimmingAvg(1);

	//the idea is that I'll get the min max density here
	//since they are already computed it should be fine since this is outside of the timmings
	if (params.output_density_information) {
		RealCuda min_density = 10000;
		RealCuda max_density = 0;
		RealCuda avg_density = 0;
		UnifiedParticleSet* particleSet = RestFLuidLoader::getStructure().backgroundFluidBufferSet;
		int count = 0;
		for (int j = 0; j < RestFLuidLoader::getStructure().count_potential_fluid; ++j) {
			if (particleSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE) {
				avg_density += particleSet->density[j];
				min_density = std::fminf(min_density, particleSet->density[j]);
				max_density = std::fmaxf(max_density, particleSet->density[j]);
				count++;
			}
		}
		avg_density /= count;

		//secodn pass for the stdev
		RealCuda stdev_density = 0;
		for (int j = 0; j < RestFLuidLoader::getStructure().count_potential_fluid; ++j) {
			if (particleSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE) {
				RealCuda delta = particleSet->density[j] - avg_density;
				delta *= delta;
				stdev_density += delta;
			}
		}
		stdev_density /= count;
		stdev_density = SQRT_MACRO(stdev_density);

		params.min_density_o = min_density;
		params.max_density_o = max_density;
		params.avg_density_o = avg_density;
		params.stdev_density_o = stdev_density;
	}
	
}

void RestFLuidLoaderInterface::stabilizeFluid(SPH::DFSPHCData& data, RestFLuidLoaderInterface::StabilizationParameters& params) {
	RestFLuidLoader::getStructure().stabilizeFluid(data, params);
}


void RestFLuidLoader::clear() {
	_isInitialized = false;
	_isDataTagged = false;
	_hasFullTaggingSaved = false;

	/*
	if (backgroundFluidBufferSet != NULL) {
		backgroundFluidBufferSet->clear();
		//delete backgroundFluidBufferSet;
		backgroundFluidBufferSet = NULL;
	}
	//*/

	if (tag_array_with_existing_fluid != NULL) {
		CUDA_FREE_PTR(tag_array_with_existing_fluid);
	}
	tag_array_with_existing_fluid_size = 0;
	//*/
}

//this will tag certain particles depending on the required restriction type
//0==> outside
//1==> inside
//2==> away from the surface
template<int restrictionType, bool override_existing_tagging>
__global__ void surface_restrict_particleset_kernel(SPH::UnifiedParticleSet* particleSet, BufferFluidSurface S, RealCuda offset, 
	int* countTagged) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (override_existing_tagging) {
		particleSet->neighborsDataSet->cell_id[i] = i;
	}
	if (restrictionType == 0) {
		if (S.distanceToSurfaceSigned(particleSet->pos[i]) < (-offset)) {
			particleSet->neighborsDataSet->cell_id[i] += particleSet->numParticles;
			atomicAdd(countTagged, 1);
		}
	}
	else if (restrictionType == 1) {
		if (S.distanceToSurfaceSigned(particleSet->pos[i]) > (-offset)) {
			particleSet->neighborsDataSet->cell_id[i] += particleSet->numParticles;
			atomicAdd(countTagged, 1);
		}

	}
	else if (restrictionType == 2) {
		if (S.distanceToSurface(particleSet->pos[i]) > (offset)) {
			particleSet->neighborsDataSet->cell_id[i] += particleSet->numParticles;
			atomicAdd(countTagged, 1);
		}
	}
	else {
		asm("trap;");
	}

}

template<bool retagSameTag>
__global__ void tag_outside_of_surface_kernel(SPH::UnifiedParticleSet* particleSet, BufferFluidSurface S,
	int* countTagged, unsigned int tag) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (!retagSameTag) {
		if (particleSet->neighborsDataSet->cell_id[i] == tag) {
			return;
		}
	}

	if (!S.isinside(particleSet->pos[i]) ) {
		particleSet->neighborsDataSet->cell_id[i] = tag;
		atomicAdd(countTagged, 1);
	}
}



//so this is the offset version for using when using a surface aggregation
//however there are restriction on its use
//It can only be used with a an intersection aggregation and the offset need to be tworsd the inside of the aggregation
//this restriction is due to the fact that I only know how to define a distance for that specific case
template<bool override_existing_tagging>
__global__ void surface_restrict_particleset_kernel(SPH::UnifiedParticleSet* particleSet, SurfaceAggregation S, RealCuda offset, int* countRmv) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (override_existing_tagging) {
		particleSet->neighborsDataSet->cell_id[i] = i;
	}
	RealCuda dist = 0;
	dist=S.distanceSigned<true, false>(particleSet->pos[i]);
	
	if (dist <= offset) {
		particleSet->neighborsDataSet->cell_id[i] += particleSet->numParticles;
		atomicAdd(countRmv, 1);
	}
}

//same as the one that does it for only 1 surface aggreg but this one does both addreg at the same time 
//the goal is to also remove the air particle that are too far from the fluid so they don't fuck with the neighbor search
//for the simulation domain I cna taka an aggregation but for the fluid I need a single surface because I'll have
//to compute a distance outside, the only other way ould be to restrict the aggreg for the fluid to unions
// anyway for now I only need the fluid suface to define the free surface of the fluid
//so no need to go all gung ho 
//just a side note:: don't fuck with the function... 
//		give it a positive offset for the simulation space
//		and give it a negative offset for the fluid as expected it would make no sense otherwise
__global__ void surface_restrict_particleset_kernel(SPH::UnifiedParticleSet* particleSet, SurfaceAggregation S_simu_aggr, RealCuda offset_simu, 
	BufferFluidSurface S_fluid, RealCuda offset_fluid, int* countRmv) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	particleSet->neighborsDataSet->cell_id[i] = i;
	//particleSet->densityAdv[i] = 0.0f;
	
	//handle the simulation space surface
	RealCuda dist = 0;
	dist = S_simu_aggr.distanceSigned<true, false>(particleSet->pos[i]);

	if (dist <= offset_simu) {
		particleSet->neighborsDataSet->cell_id[i] += particleSet->numParticles;
		atomicAdd(countRmv, 1);
		return;
	}

	//and now restrict it even more to only keep some of the air particles
	//btw I supose a negative offset because it make no sense to give a positive one
	//and remmeber the distance outside is negative
	dist = S_fluid.distanceToSurfaceSigned(particleSet->pos[i]);
	//particleSet->densityAdv[i] = dist;
	if (dist <= offset_fluid) {
		particleSet->neighborsDataSet->cell_id[i] += particleSet->numParticles;
		atomicAdd(countRmv, 1);
		return;
	}
}



//this will tag certain particles depending on the required restriction type
//0==> outside
//1==> inside
template<int restrictionType, bool override_existing_tagging>
__global__ void surface_restrict_particleset_kernel(SPH::UnifiedParticleSet* particleSet, SurfaceAggregation S, int* countRmv) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (override_existing_tagging) {
		particleSet->neighborsDataSet->cell_id[i] = i;
	}
	if (restrictionType == 0) {
		if (!S.isinside(particleSet->pos[i])) {
			particleSet->neighborsDataSet->cell_id[i] += particleSet->numParticles;
			atomicAdd(countRmv, 1);
		}
	}
	else if (restrictionType == 1) {
		if (S.isinside(particleSet->pos[i])) {
			particleSet->neighborsDataSet->cell_id[i] += particleSet->numParticles;
			atomicAdd(countRmv, 1);
		}

	}
	else {
		asm("trap;");
	}

}


__global__ void tag_close_to_boundaires_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, RealCuda min_dist, int* countRmv) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }



	ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, particleSet, i);
	
	//adapt the radius_sq to our distance
	radius_sq = min_dist * min_dist;

	//other
	ITER_NEIGHBORS_FROM_STRUCTURE(data.boundaries_data_cuda[0].neighborsDataSet, data.boundaries_data_cuda[0].pos,
		{
			//if anything go there then we tag it and end the process
			particleSet->neighborsDataSet->cell_id[i] = TAG_REMOVAL;
				
			if (countRmv != NULL) {
				atomicAdd(countRmv, 1);
			}
	
			return;
		}
	);

	///TODO warning if the solids add the code for the solids, though for the solids you will have to initialize them
	/*
	if (data.numDynamicBodies > 0) {
		for (int id_body = 0; id_body < data.numDynamicBodies; ++id_body) {
			ITER_NEIGHBORS_FROM_STRUCTURE(data.vector_dynamic_bodies_data_cuda[id_body].neighborsDataSet, data.vector_dynamic_bodies_data_cuda[id_body].pos,
				{ 
				

				//printf("wrote a neighbor %i\n", nb_neighbors_boundary+nb_neighbors_fluid+nb_neighbors_dynamic_objects);
				});
		}
	}
	//*/

}

//to do it I use the normal neighbor search process, altough it mean on important thing
//the limit_distance MUST be smaller or equals to the kernel radius
//btw no need to care about synchronization between threads
//at worst particles will be tagged multiples times which only waste computation time
template<bool tag_candidate_only, bool tag_untagged_only>
__global__ void tag_neighborhood_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, SPH::UnifiedParticleSet* particleSetToTag, 
	RealCuda limit_distance, int count_candidates, unsigned int tag_o=TAG_ACTIVE) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(particleSet->pos[i], data.getKernelRadius(), data.gridOffset);


	//override the kernel distance
	radius_sq = limit_distance * limit_distance;
	
	if (limit_distance < (data.getKernelRadius()*1.01)) {
		if (tag_candidate_only) {
			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(particleSetToTag->neighborsDataSet, particleSetToTag->pos,
				if (j < count_candidates) {
					if (tag_untagged_only) {
						if (particleSetToTag->neighborsDataSet->cell_id[j] == TAG_UNTAGGED) {
							particleSetToTag->neighborsDataSet->cell_id[j] = tag_o;
							//particleSetToTag->color[j] = Vector3d(1, 0, 1);
						}
					}
					else {
						particleSetToTag->neighborsDataSet->cell_id[j] = tag_o;
					}
				}
			);
		}
		else {
			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(particleSetToTag->neighborsDataSet, particleSetToTag->pos,
				if (tag_untagged_only) {
					if (particleSetToTag->neighborsDataSet->cell_id[j] == TAG_UNTAGGED) {
						particleSetToTag->neighborsDataSet->cell_id[j] = tag_o;
					}
				}
				else {
					particleSetToTag->neighborsDataSet->cell_id[j] = tag_o;
				}
			);

		}
	}
	else {
		//so if the tagging distance is larger than the kernel radius using the acceleration structure gets harder
		//since I have to explore further cells than nrmaly
		//meaning I cannot use my normal macros
		if (tag_candidate_only){
			NeighborsSearchDataSet* neighborsDataSet=particleSetToTag->neighborsDataSet;
			Vector3d* positions= particleSetToTag->pos;
			int range_increase = limit_distance / data.getKernelRadius();
			for (int k = -1 - range_increase; k < (2 + range_increase); ++k) {
				if ((y + k < 0) || (y + k >= CELL_ROW_LENGTH)) {
					continue;
				}
				for (int m = -1 - range_increase; m < (2 + range_increase); ++m) {
					if ((z + m < 0) || (z + m >= CELL_ROW_LENGTH)) {
						continue;
					}
					for (int n = -1 - range_increase; n < (2 + range_increase); ++n) {
						if ((x + n < 0) || (x + n >= CELL_ROW_LENGTH)) {
							continue;
						}
						unsigned int cur_cell_id = COMPUTE_CELL_INDEX(x + n, y + k, z + m); 
						unsigned int end = neighborsDataSet->cell_start_end[cur_cell_id + 1]; 
						for (unsigned int cur_particle = neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {
							unsigned int j = neighborsDataSet->p_id_sorted[cur_particle]; 
							if ((pos - positions[j]).squaredNorm() < radius_sq) {
								if (j < count_candidates) {
									if (tag_untagged_only) {
										if (particleSetToTag->neighborsDataSet->cell_id[j] == TAG_UNTAGGED) {
											particleSetToTag->neighborsDataSet->cell_id[j] = tag_o;
										}
									}
									else {
										particleSetToTag->neighborsDataSet->cell_id[j] = tag_o;
									}
								}
							}
						}
					}
				}
			}
		}
		else {
			NeighborsSearchDataSet* neighborsDataSet = particleSetToTag->neighborsDataSet;
			Vector3d* positions = particleSetToTag->pos;
			int range_increase = limit_distance / data.getKernelRadius();
			for (int k = -1 - range_increase; k < (2 + range_increase); ++k) {
				if ((y + k < 0) || (y + k >= CELL_ROW_LENGTH)) {
					continue;
				}
				for (int m = -1 - range_increase; m < (2 + range_increase); ++m) {
					if ((z + m < 0) || (z + m >= CELL_ROW_LENGTH)) {
						continue;
					}
					for (int n = -1 - range_increase; n < (2 + range_increase); ++n) {
						if ((x + n < 0) || (x + n >= CELL_ROW_LENGTH)) {
							continue;
						}
						unsigned int cur_cell_id = COMPUTE_CELL_INDEX(x + n, y + k, z + m);
						unsigned int end = neighborsDataSet->cell_start_end[cur_cell_id + 1];
						for (unsigned int cur_particle = neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {
							unsigned int j = neighborsDataSet->p_id_sorted[cur_particle];
							if ((pos - positions[j]).squaredNorm() < radius_sq) {
								if (tag_untagged_only) {
									if (particleSetToTag->neighborsDataSet->cell_id[j] == TAG_UNTAGGED) {
										particleSetToTag->neighborsDataSet->cell_id[j] = tag_o;
									}
								}
								else {
									particleSetToTag->neighborsDataSet->cell_id[j] = tag_o;
								}
							}
						}
					}
				}
			}
		}
	}
}

//I'll allow 4 types of tagging:
//0: higher than;; 1: lower than;; 2: higher than dist from rest;; 3: lower than dist from rest
template<bool tag_candidate_only, bool tag_target_only, int comparison_type>
__global__ void tag_densities_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, RealCuda density_limit, int count_candidates, 
	int tag_target, int tag_output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (tag_candidate_only) {
		if (i>= count_candidates) {
			return;
		}
	}

	if (tag_target_only) {
		if (particleSet->neighborsDataSet->cell_id[i] != tag_target) {
			return;
		}
	}

	if (comparison_type == 0) {
		if (particleSet->density[i] > density_limit) {
			particleSet->neighborsDataSet->cell_id[i] = tag_output;
		}
	}
	else if(comparison_type == 1) {
		if (particleSet->density[i] < density_limit) {
			particleSet->neighborsDataSet->cell_id[i] = tag_output;
		}
	}
	else if(comparison_type == 0) {
		RealCuda v = abs(data.density0 - particleSet->density[i]);
		if (v > density_limit) {
			particleSet->neighborsDataSet->cell_id[i] = tag_output;
		}
	}
	else if(comparison_type == 0) {
		RealCuda v = abs(data.density0 - particleSet->density[i]);
		if (v < density_limit) {
			particleSet->neighborsDataSet->cell_id[i] = tag_output;
		}
	}
}


template<bool compute_active_only, bool tag_active_only, bool tag_as_candidate, bool use_stored_neighbors>
__global__ void evaluate_and_tag_high_density_from_buffer_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int* countRmv, 
	RealCuda limit_density, int count_affected, int* count_stored_density, SPH::UnifiedParticleSet* existingFluidSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count_affected) { return; }

	//do not do useless computation on particles that have already been taged for removal
	if (compute_active_only) {
		if (bufferSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) {
			return;
		}
	}

	if (bufferSet->neighborsDataSet->cell_id[i] == TAG_AIR) {
		bufferSet->color[i] = Vector3d(0, 0, 1);
	}

	Vector3d p_i = bufferSet->pos[i];

	RealCuda density = bufferSet->getMass(i) * data.W_zero;
	RealCuda density_fluid = 0;
	RealCuda density_boundaries = 0;

	if (existingFluidSet != NULL) {
		//for existing fluid we cannot use stored neighbors
		//the reason is that since I'm having neightbors form particle set that are at least partialy superposed
		//it might happen that I go above the maximum number of neighbors allowed 
		//and I'm not augmenting to double the number taking double the memory just to optimize this algorithm
		ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, bufferSet, i);
		ITER_NEIGHBORS_FROM_STRUCTURE(existingFluidSet->neighborsDataSet, existingFluidSet->pos,
			{
				RealCuda density_delta = existingFluidSet->getMass(j) * KERNEL_W(data, p_i - existingFluidSet->pos[j]);
				density += density_delta;
			});
	}

	if (use_stored_neighbors) {
		ITER_NEIGHBORS_INIT(m_data, bufferSet, i);
		SPH::UnifiedParticleSet* otherSet;

		//first neighbors from same set
		otherSet = bufferSet;
		ITER_NEIGHBORS_FROM_STORAGE(data, bufferSet, i, 0,
			if (bufferSet->neighborsDataSet->cell_id[neighborIndex] != TAG_REMOVAL) {
				RealCuda density_delta = otherSet->getMass(neighborIndex) * KERNEL_W(data, p_i - otherSet->pos[neighborIndex]);
				density += density_delta;
				density_fluid += density_delta;
			}
		);

		//then boundaires
		otherSet = data.boundaries_data_cuda;
		ITER_NEIGHBORS_FROM_STORAGE(data, bufferSet, i, 1,
			{
				RealCuda density_delta = otherSet->getMass(neighborIndex) * KERNEL_W(data, p_i - otherSet->pos[neighborIndex]);
				density += density_delta;
				density_boundaries += density_delta;
			}
		);

		//and finaly solids (which I'll count as boundaries)
		//*
		ITER_NEIGHBORS_FROM_STORAGE(data, bufferSet, i, 2,
			{
				int dummy = neighborIndex;
				READ_DYNAMIC_BODIES_PARTICLES_INDEX(dummy, bodyIndex, neighborIndex);
				const SPH::UnifiedParticleSet & body = data.vector_dynamic_bodies_data_cuda[bodyIndex]; 

				RealCuda density_delta = body.getMass(neighborIndex) * KERNEL_W(data, p_i - body.pos[neighborIndex]);
				density += density_delta;
				density_boundaries += density_delta;
			}
		);
		//*/


	}
	else {
		ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);

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

		//*
		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
			RealCuda density_delta = data.boundaries_data_cuda->getMass(j) * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
		density += density_delta;
		density_boundaries += density_delta;
		);
		//*/
	}

	if ((!tag_active_only) || (bufferSet->neighborsDataSet->cell_id[i] == TAG_ACTIVE)) {
		if (density > limit_density) {
			atomicAdd(countRmv, 1);
			bufferSet->neighborsDataSet->cell_id[i] = ((tag_as_candidate)?TAG_REMOVAL_CANDIDATE:TAG_REMOVAL);
			//bufferSet->color[i] = Vector3d(0, 1, 0);
		}

		if (count_stored_density != NULL) {
			int id_stored=atomicAdd(count_stored_density, 1);
			bufferSet->densityAdv[id_stored] = density;
		}
	}


	//only for debug it's useless in the final execution
	bufferSet->density[i] = density;
	/*
	if (abs(1000 - density) > 3) {

		RealCuda k = MAX_MACRO_CUDA(MIN_MACRO_CUDA((1000 - density + 25) / 50, 1), 0);
		bufferSet->color[i] = Vector3d(1-k, k, 0);
		
	}
	//*/
	//*
	if (density>1005) {
		bufferSet->color[i] = Vector3d(1, 0, 0);
	}
	if (density < 990) {
		bufferSet->color[i] = Vector3d(0, 1, 0);
	}
	/*
	if (density < 850) {
		bufferSet->color[i] = Vector3d(0, 0, 1);
	}
	//*/
	//bufferSet->kappa[i] = density_fluid;
	//bufferSet->kappaV[i] = density_boundaries;



}


//this function requires having the neighbors stored
template<bool output_densities, bool compute_on_neighbors>
__global__ void particle_selection_rule_1_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, 
	RealCuda limit_density, int count_potential_fluid, RealCuda* sum_densities, int* count_summed, RealCuda* max_density,
	int iterId, SPH::UnifiedParticleSet* existingFluidSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count_potential_fluid) { return; }

	if (compute_on_neighbors) {
		if ((bufferSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) &&
			(bufferSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE_NEIGHBORS)) {
			return;
		}
	}
	else {
		if ((bufferSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE)) {
			return;
		}
	}
	
	Vector3d p_i = bufferSet->pos[i];
	RealCuda density = bufferSet->getMass(i) * data.W_zero;
	RealCuda densityConstant = 0;
	RealCuda min_dist = 100000;
	RealCuda factor = 1;

	//handle existing fluid
	if (iterId == 0)
	{
		if (existingFluidSet != NULL) {
			//for existing fluid we cannot use stored neighbors
			//the reason is that since I'm having neightbors form particle set that are at least partialy superposed
			//it might happen that I go above the maximum number of neighbors allowed 
			//and I'm not augmenting to double the number taking double the memory just to optimize this algorithm
			int count_neigbors_existing_fluid = 0;
			//RealCuda min_dist = 1000;
			ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, bufferSet, i);
			ITER_NEIGHBORS_FROM_STRUCTURE(existingFluidSet->neighborsDataSet, existingFluidSet->pos,
				{
					RealCuda density_delta = existingFluidSet->getMass(j) * KERNEL_W(data, p_i - existingFluidSet->pos[j]);
					densityConstant += density_delta;
					//count_neigbors_existing_fluid++;
					//min_dist = MIN_MACRO_CUDA(min_dist, ((p_i - existingFluidSet->pos[j]).norm()));
				});
			//printf("density constant from fluid and neighbors count: %i // %f %i // %f\n", i, densityConstant, 
			//	count_neigbors_existing_fluid, min_dist);
			bufferSet->kappa[i] = densityConstant;
			//bufferSet->kappaV[i] = min_dist;
		}
	}

	{
		ITER_NEIGHBORS_INIT(m_data, bufferSet, i);
		SPH::UnifiedParticleSet* otherSet;

		//first neighbors from same set
		otherSet = bufferSet;
		ITER_NEIGHBORS_FROM_STORAGE(data, bufferSet, i, 0,
			if (bufferSet->neighborsDataSet->cell_id[neighborIndex] != TAG_REMOVAL) {
				RealCuda density_delta = otherSet->getMass(neighborIndex) * KERNEL_W(data, p_i - otherSet->pos[neighborIndex]);
				density += density_delta;

				RealCuda dist = (p_i - otherSet->pos[neighborIndex]).norm();
				if (min_dist > dist) {
					min_dist = dist;
				}
			}
		);

		//for boundaries and solids since they do not move it is only needed to compute it once at the start
		if (iterId == 0) 
		{
			//then boundaires
			otherSet = data.boundaries_data_cuda;
			ITER_NEIGHBORS_FROM_STORAGE(data, bufferSet, i, 1,
				{
					RealCuda density_delta = otherSet->getMass(neighborIndex) * KERNEL_W(data, p_i - otherSet->pos[neighborIndex]);
					densityConstant += density_delta;

					RealCuda dist = (p_i - otherSet->pos[neighborIndex]).norm();
					if (min_dist > dist) {
						min_dist = dist;
						factor = -1;
					}

				}
			);

			//and finaly solids (which I'll count as boundaries)
			//*
			ITER_NEIGHBORS_FROM_STORAGE(data, bufferSet, i, 2,
				{
					int dummy = neighborIndex;
					READ_DYNAMIC_BODIES_PARTICLES_INDEX(dummy, bodyIndex, neighborIndex);
					const SPH::UnifiedParticleSet & body = data.vector_dynamic_bodies_data_cuda[bodyIndex];

					RealCuda density_delta = body.getMass(neighborIndex) * KERNEL_W(data, p_i - body.pos[neighborIndex]);
					densityConstant += density_delta;
				}
			);

			//bufferSet->kappaV[i] = densityConstant - bufferSet->kappa[i];
			//printf("density  %i // %f %f %f\n", i, density ,densityConstant, temp_calc);
		}
		//*/

	}

	//I can save the density that I'm sure doesn't change between iterations
	//meaning the density from solids, boundaries and existing fluids
	
	if (iterId == 0) {

		bufferSet->densityAdv[i] = densityConstant;
	}
	else {
		/*
		if (densityConstant != bufferSet->densityAdv[i]) {
			printf("density constant is not the same: %i // %f!=%f\n", i, densityConstant, bufferSet->densityAdv[i]);
		}
		//*/
		
		densityConstant = bufferSet->densityAdv[i];
	}

	//add the constant
	density += densityConstant;
	

	//*
	if ((bufferSet->neighborsDataSet->cell_id[i] == TAG_ACTIVE)) {
		//printf("dist/density: %f %f\n", factor*min_dist/data.particleRadius,density);
		//ok if the sum of existing fluid and boundaries is too high I can directly delete the particle
		//*
		if (densityConstant > (data.density0*0.70)) {
			//we can directly remove them without considering them as candidate
			//since their condition is on a constant value
			bufferSet->neighborsDataSet->cell_id[i] = TAG_REMOVAL;
		}
		else
		//*/
		{
			if (density > limit_density) {
				bufferSet->neighborsDataSet->cell_id[i] =  TAG_REMOVAL_CANDIDATE;
				//bufferSet->color[i] = Vector3d(0, 1, 0);
			}
			if (max_density != NULL) {
				atomicToMax(max_density, density);
			}
			//*
			//doing it with atomic is as fast/faster than cub since only a few thread need to add their values
			//also when chaining atomic it seems that only the first one has a cost
			atomicAdd(sum_densities, density);
			atomicAdd(count_summed, 1);
			//*/
		}
		
	}
	//*/

	

	//set the density for the other rules application
	if (output_densities) {
		bufferSet->density[i] = density;
	}
}


__global__ void compute_density_and_extract_large_contribution_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet,
	RealCuda limit_density, SPH::UnifiedParticleSet* existingFluidSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }


	Vector3d p_i = bufferSet->pos[i];
	RealCuda densityConstant = 0;

	{
		ITER_NEIGHBORS_INIT(data, bufferSet, i);
		SPH::UnifiedParticleSet* otherSet;

		//I need to skip the fluid particles from the buffer since I only whan the constant contribution
		if (false) {
			//first neighbors from same set
			otherSet = bufferSet;
			ITER_NEIGHBORS_FROM_STORAGE(data, bufferSet, i, 0,
				{
				}
			);
		}
		else {
			ADVANCE_END_PTR(end_ptr, bufferSet->getNumberOfNeighbourgs(i, 0));
			neighbors_ptr = end_ptr;
		}

		//for boundaries and solids since they do not move it is only needed to compute it once at the start
		//then boundaires
		otherSet = data.boundaries_data_cuda;
		ITER_NEIGHBORS_FROM_STORAGE(data, bufferSet, i, 1,
			{
				RealCuda density_delta = otherSet->getMass(neighborIndex) * KERNEL_W(data, p_i - otherSet->pos[neighborIndex]);
				densityConstant += density_delta;
			}
		);

		//and finaly solids (which I'll count as boundaries)
		//*
		ITER_NEIGHBORS_FROM_STORAGE(data, bufferSet, i, 2,
			{
				int dummy = neighborIndex;
				READ_DYNAMIC_BODIES_PARTICLES_INDEX(dummy, bodyIndex, neighborIndex);
				const SPH::UnifiedParticleSet & body = data.vector_dynamic_bodies_data_cuda[bodyIndex];

				RealCuda density_delta = body.getMass(neighborIndex) * KERNEL_W(data, p_i - body.pos[neighborIndex]);
				densityConstant += density_delta;
			}
		);

		//bufferSet->kappaV[i] = densityConstant - bufferSet->kappa[i];
		//printf("density  %i // %f %f %f\n", i, density ,densityConstant, temp_calc);

	//*/

	}

	if (existingFluidSet != NULL) {
		//for existing fluid we cannot use stored neighbors
		//the reason is that since I'm having neightbors form particle set that are at least partialy superposed
		//it might happen that I go above the maximum number of neighbors allowed 
		//and I'm not augmenting to double the number taking double the memory just to optimize this algorithm
		//RealCuda min_dist = 1000;
		ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, bufferSet, i);
		ITER_NEIGHBORS_FROM_STRUCTURE(existingFluidSet->neighborsDataSet, existingFluidSet->pos,
			{
				RealCuda density_delta = existingFluidSet->getMass(j) * KERNEL_W(data, p_i - existingFluidSet->pos[j]);
				densityConstant += density_delta;
			});
	}




	if (densityConstant > (limit_density)) {
		//we can directly remove them without considering them as candidate
		//since their condition is on a constant value
		bufferSet->neighborsDataSet->cell_id[i] = TAG_REMOVAL;
	}
	else {
		//I can save the density that I'm sure doesn't change between iterations
		//meaning the density from solids, boundaries and existing fluids
		bufferSet->densityAdv[i] = densityConstant;
	}

}





__global__ void untag_candidate_below_limit_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, 
	RealCuda limit_density, int count_potential_fluid) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count_potential_fluid) { return; }

	if ((bufferSet->neighborsDataSet->cell_id[i] != TAG_REMOVAL_CANDIDATE)) {
		return;
	}

	if (bufferSet->density[i] < limit_density) {
		bufferSet->neighborsDataSet->cell_id[i] = TAG_ACTIVE;
	}

}

template <bool run_multithread>
__global__ void save_usefull_candidates_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int* countRmv, RealCuda limit_density, int count_potential_fluid) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (run_multithread) {
		if (id >= count_potential_fluid) { return; }

		int i = id;

		if (bufferSet->neighborsDataSet->cell_id[i] != TAG_REMOVAL_CANDIDATE) {
			return;
		}

		//if by removing that particel I make anther particle go below 900 density prevent removing that particle
		bool need_saving = false;

		//and remove its contribution to neighbors candidates
		Vector3d p_i = bufferSet->pos[i];
		ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);
		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
			if (!need_saving) {
				if (j < count_potential_fluid) {
					if (bufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE || bufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE_NEIGHBORS) {

						if (i != j) {
							//check if OUR impact on the neighbor is important
							RealCuda density_delta = bufferSet->getMass(i) * KERNEL_W(data, p_i - bufferSet->pos[j]);
							if (density_delta > 10) {
								if ((bufferSet->density[j] - density_delta) < limit_density) {
									//printf("%f   ==> %i  %f  %f\n", bufferSet->density[i], j,bufferSet->density[j], density_delta);
									need_saving = true;
									break;
								}
							}
						}
					}
				}
			}
			else {
				break;
			}
		);

		if (need_saving) {
			bufferSet->neighborsDataSet->cell_id[i] = TAG_SAVE;
			atomicAdd(countRmv, 1);
		}


		


	}
	else {
		if (id >= 1) { return; }
		for (int i = 0; i < count_potential_fluid; ++i) {

			if (bufferSet->neighborsDataSet->cell_id[i] != TAG_REMOVAL_CANDIDATE) {
				continue;
			}

			//if by removing that particel I make anther particle go below 900 density prevent removing that particle
			bool need_saving = false;

			//and remove its contribution to neighbors candidates
			Vector3d p_i = bufferSet->pos[i];
			ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);
			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
				if (!need_saving) {
					if (j < count_potential_fluid) {
						if (bufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE || bufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE_NEIGHBORS) {

							if (i != j) {
								RealCuda density_delta = bufferSet->getMass(j) * KERNEL_W(data, p_i - bufferSet->pos[j]);
								if (density_delta > 10) {
									if ((bufferSet->density[j] - density_delta) < limit_density) {
										//printf("%f   ==> %i  %f  %f\n", bufferSet->density[i], j,bufferSet->density[j], density_delta);
										need_saving = true;
										break;
									}
								}
							}
						}
					}
				}
				else {
					break;
				}
			);

			if (need_saving) {
				bufferSet->neighborsDataSet->cell_id[i] = TAG_ACTIVE;
				(*countRmv)++;
			}
			else {
				//here i should remove the contribution from the neighbors particles,
				//but I'm not sure if it is that necessary
			}
		}

	}

}

//this version will work the opposite way
//if will check the particles with a low enougth density 
//and save the neighbors that are required to prevent them from going below the min limit for the density
__global__ void save_usefull_candidates_kernel_v2(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int* countRmv, RealCuda limit_density, int count_potential_fluid) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count_potential_fluid) { return; }

	if (bufferSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE &&
		bufferSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE_NEIGHBORS &&
		bufferSet->neighborsDataSet->cell_id[i] != TAG_SAVE) {
		return;
	}


	//and remove its contribution to neighbors candidates
	//I'll set a special tag
	Vector3d p_i = bufferSet->pos[i];
	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,		
			if (j < count_potential_fluid) {
				if (bufferSet->neighborsDataSet->cell_id[j] == TAG_REMOVAL_CANDIDATE) {
					if (i != j) {
						//check if wee need the impact of that neighbors
						RealCuda density_delta = bufferSet->getMass(j) * KERNEL_W(data, p_i - bufferSet->pos[j]);
						if (density_delta > 1) {
							if ((bufferSet->density[i] - density_delta) < limit_density) {
								bufferSet->neighborsDataSet->cell_id[j] = TAG_SAVE;
							}
						}
					}
				}
			}
		
	);

	
}


//the goal is be able to keep some particle swhen 2 haigh density particels are paired together
__global__ void verify_candidate_tagging_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int* countRmv, RealCuda limit_density, int count_potential_fluid) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= 1) { return; }

	
	for(int i = 0;i<count_potential_fluid;++i){

		if (bufferSet->neighborsDataSet->cell_id[i] != TAG_REMOVAL_CANDIDATE) {
			continue;
		}

		//check if the density is indeed too high
		if (bufferSet->density[i] > limit_density) {
			bufferSet->neighborsDataSet->cell_id[i] = TAG_REMOVAL;


			//and remove its contribution to neighbors candidates
			Vector3d p_i = bufferSet->pos[i];
			ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);
			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
				if (bufferSet->neighborsDataSet->cell_id[j] == TAG_REMOVAL_CANDIDATE) {
					if (i != j) {
						RealCuda density_delta = bufferSet->getMass(j) * KERNEL_W(data, p_i - bufferSet->pos[j]);
						bufferSet->density[j] -= density_delta;
					}
				}
			);
		}
		else {
			bufferSet->neighborsDataSet->cell_id[i] = TAG_ACTIVE;
			(*countRmv)++;
		}

	}


}


//the goal is be able to keep some particle swhen 2 haigh density particels are paired together
//this is the multithreaded version
//my logic is if we have two candidates that are close to each other we will save the one with the lowest density
//since it is multithreaded it may do some strange things sometimes, but since it is inside a loop it should be fine
template<bool confirm_remaining_candidate>
__global__ void verify_candidate_tagging_multithread_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int* countSaved,
	RealCuda limit_density, RealCuda density_delta_threshold, int count_potential_fluid) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count_potential_fluid) { return; }

	if (bufferSet->neighborsDataSet->cell_id[i] != TAG_REMOVAL_CANDIDATE) {
		return;
	}

	Vector3d p_i = bufferSet->pos[i];
	Vector3d den_i = bufferSet->density[i];

	ITER_NEIGHBORS_INIT(m_data, bufferSet, i);
	SPH::UnifiedParticleSet* otherSet;

	//first neighbors from same set
	ITER_NEIGHBORS_FROM_STORAGE(data, bufferSet, i, 0,
		//only do smth when the neighbor is also a candidate
		if (bufferSet->neighborsDataSet->cell_id[neighborIndex] == TAG_REMOVAL_CANDIDATE) {
			//only act if I'm at a lower density than that neighbor
			if (den_i > bufferSet->density[neighborIndex]) {
				continue;
			}

			//check if the impact between the particles is high enougth
			RealCuda density_delta = bufferSet->getMass(neighborIndex) * KERNEL_W(data, p_i - bufferSet->pos[neighborIndex]);
			if (density_delta > density_delta_threshold) {
				if ((den_i - density_delta) < limit_density) {
					bufferSet->neighborsDataSet->cell_id[i] = TAG_ACTIVE;
					atomicAdd(countSaved, 1);
					return;
				}
			}
		}
	);

	if (confirm_remaining_candidate) {
		bufferSet->neighborsDataSet->cell_id[i] = TAG_REMOVAL;
	}
}


__global__ void confirm_candidates_kernel( SPH::UnifiedParticleSet* bufferSet, int count_potential_fluid, 
	int* countConfirm, RealCuda* sumDensityConfirm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count_potential_fluid) { return; }

	//do not do useless computation on particles that have already been taged for removal
	if (bufferSet->neighborsDataSet->cell_id[i] == TAG_REMOVAL_CANDIDATE) {
		bufferSet->neighborsDataSet->cell_id[i] = TAG_REMOVAL;
		if (countConfirm != NULL) {
			atomicAdd(countConfirm, 1);
		}

		if (sumDensityConfirm != NULL) {
			atomicAdd(sumDensityConfirm, bufferSet->density[i]);
		}
		
	}
}


//this kernel uses the suposition that the air particles are store at the end of the buffer
__global__ void tag_high_density_in_air_kernel(SPH::UnifiedParticleSet* bufferSet, int count_potential_fluid, int density_limit) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	i += count_potential_fluid;
	if (i >= bufferSet->numParticles) { return; }

	//do not do useless computation on particles that have already been taged for removal
	if (bufferSet->density[i] >density_limit) {
		bufferSet->neighborsDataSet->cell_id[i] = TAG_REMOVAL;
	}
}


//the goal is be able to keep some particle swhen 2 haigh density particels are paired together
__global__ void save_particles_tagged_for_removal_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int* countRmv, RealCuda limit_density, int count_potential_fluid) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= 1) { return; }


	for (int i = 0; i < count_potential_fluid; ++i) {

		if (bufferSet->neighborsDataSet->cell_id[i] != TAG_REMOVAL) {
			continue;
		}

		//compute the density and it it is low enougth reactive the particle
		RealCuda density = bufferSet->getMass(i) * data.W_zero;
		

		//and remove its contribution to neighbors candidates
		Vector3d p_i = bufferSet->pos[i];
		ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);
		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
			if (bufferSet->neighborsDataSet->cell_id[j] != TAG_REMOVAL) {
				if (i != j) {
					RealCuda density_delta = bufferSet->getMass(j) * KERNEL_W(data, p_i - bufferSet->pos[j]);
					density += density_delta;
				}
			}
		);

		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
			RealCuda density_delta = data.boundaries_data_cuda->getMass(j) * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
			density += density_delta;
		);
		//*/

		if (density < limit_density) {
			atomicAdd(countRmv, 1);

			bufferSet->neighborsDataSet->cell_id[i] = TAG_ACTIVE;
			//bufferSet->color[i] = Vector3d(0, 1, 0);
		}
	}
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


template<bool compute_active_only, bool apply_acceleration>
__global__ void advance_in_time_particleSet_kernel(SPH::UnifiedParticleSet* particleSet, RealCuda dt, RealCuda velDamping=-1, bool dampBeforeAcc=true) {
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

	if ((velDamping >= 0)&&(dampBeforeAcc)) {
		particleSet->vel[i] *= velDamping;
	}

	if (apply_acceleration) {
		particleSet->vel[i] += particleSet->acc[i] * dt;
	}

	if ((velDamping >= 0) && (!dampBeforeAcc)) {
		particleSet->vel[i] *= velDamping;
	}

	particleSet->pos[i] += particleSet->vel[i] * dt;


}

__global__ void test_kernel(BufferFluidSurface S) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= 1) { return; }
	{

		printf("test nbr 1\n");

		bool test = S.isinside(Vector3d(0, 0.5, 0));
		if (test) {
			printf("inside the surface\n");
		}
		else {
			printf("not inside the surface\n");
		}
	}

	{

		printf("test nbr 2\n");

		bool test = S.isinside(Vector3d(0, -4, 0));
		if (test) {
			printf("inside the surface\n");
		}
		else {
			printf("not inside the surface\n");
		}
	}


}



void RestFLuidLoader::init(DFSPHCData& data, RestFLuidLoaderInterface::InitParameters& params) {
	bool center_loaded_fluid=params.center_loaded_fluid; 
	int air_particles_restriction = params.air_particles_restriction;
	bool keep_existing_fluid = params.keep_existing_fluid;

	//clear anything that was loaded before
	if (params.clear_data) {
		clear();
	}

	_isInitialized = false;
	_isDataTagged = false;
	//Essencially this function will load the background buffer and initialize it to the desired simulation domain


	//surface descibing the simulation space and the fluid space
	//this most likely need ot be a mesh in the end or at least a union of surfaces
	//I'll take into consideration that S_simulation will be applied always  before S_fluid to lighten the computation
	//by not adding the surface defining the simultion volume it should lighten the computation quite a bit
	SurfaceAggregation S_simulation_aggr;
	SurfaceAggregation S_fluid_aggr;

	std::vector<std::string> timing_names{ "mesh load","void","load", "center background","restict to simulation","restrict distance to boundary particles","tag to fluid"," sort by tagging" };
	SPH::SegmentedTiming timings("fluid loader init", timing_names, true);
	timings.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)


	BufferFluidSurface S_simulation;
	BufferFluidSurface S_fluid;
	int simulation_config = params.simulation_config;


	if (simulation_config == 0) {
		S_simulation.setCuboid(Vector3d(0, 2.5, 0), Vector3d(0.5, 2.5, 0.5));
		//S_fluid.setCuboid(Vector3d(0, 1, 0), Vector3d(0.5, 1, 0.5));
		//S_fluid.setCuboid(Vector3d(0, 1, 0), Vector3d(0.5, 1.04, 0.5));
		//S_fluid.setCuboid(Vector3d(0, 0.95, 0), Vector3d(0.5, 1, 0.5));
		//S_simulation.setCuboid(Vector3d(0, 1, 0), Vector3d(0.5, 1, 0.5));
		//S_fluid.move(Vector3d(0, 1, 0));
		S_fluid.setPlane(Vector3d(0, 2.0, 0), Vector3d(0, -1, 0));
	}
	else if(simulation_config == 1) {
		//std::string simulation_area_file_name = data.fluid_files_folder + "../models/complex_pyramid_scale_0_8.obj";
		std::string simulation_area_file_name = data.fluid_files_folder + "../models/complex_pyramid.obj";
		S_simulation.setMesh(simulation_area_file_name);
		//std::string fluid_area_file_name = data.fluid_files_folder + "../models/complex_pyramid_scale075_cut_1_6m.obj";
		//S_fluid.setMesh(fluid_area_file_name);
		S_fluid.setPlane(Vector3d(0, 3.5, 0), Vector3d(0, -1, 0));
	}
	else if(simulation_config == 2) {
		//box plus sphere
		S_simulation.setCuboid(Vector3d(0, 2.5, 0), Vector3d(0.5, 2.5, 0.5));
		S_fluid.setCuboid(Vector3d(0, 1, 0), Vector3d(0.5, 1, 0.5));

		//add a sphere
		BufferFluidSurface S_sphere; 
		std::string sphere_file_name = data.fluid_files_folder + "../models/spherev2.obj";
		S_sphere.setMesh(sphere_file_name);
		S_sphere.setReversedSurface(true);
		S_simulation_aggr.addSurface(S_sphere);
	}
	else if(simulation_config == 3) {
		//box plus sphere
		S_simulation.setCuboid(Vector3d(0, 2.5, 0), Vector3d(1, 2.5, 1));
		S_fluid.setCuboid(Vector3d(0, 1, 0), Vector3d(1, 1, 1));

		//add a sphere
		BufferFluidSurface S_sphere;
		std::string sphere_file_name = data.fluid_files_folder + "../models/sphere_d1m.obj";
		S_sphere.setMesh(sphere_file_name);
		S_sphere.setReversedSurface(true);
		S_sphere.move(Vector3d(0, 2, 0));

		S_simulation_aggr.addSurface(S_sphere);
		S_fluid_aggr.addSurface(S_sphere);

		S_simulation_aggr.setIsUnion(false);
		S_fluid_aggr.setIsUnion(false);

	}
	else if (simulation_config == 4) {
		//box plus box
		S_simulation.setCuboid(Vector3d(0, 2.5, 0), Vector3d(1.75, 2.5, 1.75));
		S_fluid.setCuboid(Vector3d(0, 1.5, 0), Vector3d(1.75, 1.5, 1.75));

		//add a rotated box
		BufferFluidSurface S_obj;
		std::string obj_file_name = data.fluid_files_folder + "../models/2mBox_rotated45.obj";
		S_obj.setMesh(obj_file_name);
		S_obj.setReversedSurface(true);
		S_obj.move(Vector3d(0, 3, 0));

		S_simulation_aggr.addSurface(S_obj);
		S_fluid_aggr.addSurface(S_obj);

		S_simulation_aggr.setIsUnion(false);
		S_fluid_aggr.setIsUnion(false);
	}
	else if (simulation_config == 5) {

		S_simulation.setCuboid(Vector3d(0, 2.5, 0), Vector3d(1, 2.5, 1));
		S_fluid.setCuboid(Vector3d(0, 1, 0), Vector3d(1, 1, 1));
		//S_fluid.setPlane(Vector3d(0, 2, 0), Vector3d(0, -1, 0));

	}
	else if (simulation_config == 6) {
		//2m fluid box
		S_simulation.setCuboid(Vector3d(0, 2.5, 0), Vector3d(1, 2.5, 1));
		//S_fluid.setCuboid(Vector3d(0, 1, 0), Vector3d(1, 1, 1));
		S_fluid.setPlane(Vector3d(0, 2, 0), Vector3d(0, -1, 0));

	}
	else if (simulation_config == 7) {
		//2.4m fluid box 
		//the .4 is necessary to obtain a similar number of active particlesas the config 8
		S_simulation.setCuboid(Vector3d(0, 2.5, 0), Vector3d(1.15, 2.5, 1.15));
		//S_fluid.setCuboid(Vector3d(0, 1, 0), Vector3d(2, 2, 2));
		S_fluid.setPlane(Vector3d(0, 2.00, 0), Vector3d(0, -1, 0));

	}
	else if (simulation_config == 8) {
		//init a 2m fluid box within a 2,5m box with fluid near the border already at rest
		S_simulation.setCuboid(Vector3d(0, 2.5, 0), Vector3d(1.25, 2.5, 1.25));
		S_fluid.setCuboid(Vector3d(0, 1.5, 0), Vector3d(1, 1, 1));
		//S_fluid.setPlane(Vector3d(0, 2.5, 0), Vector3d(0, -1, 0));

	}
	else if (simulation_config == 9) {
		//2.5 m fluid box
		S_simulation.setCuboid(Vector3d(0, 2.5, 0), Vector3d(1.25, 2.5, 1.25));
		//S_fluid.setCuboid(Vector3d(0, 1.25, 0), Vector3d(2, 2, 2));
		S_fluid.setPlane(Vector3d(0, 2.54, 0), Vector3d(0, -1, 0));

	}
	else if (simulation_config == 10) {
		//2.5 m fluid box
		S_simulation.setCuboid(Vector3d(0, 2.5, 0), Vector3d(1.25, 2.5, 1.25));
		//S_fluid.setCuboid(Vector3d(0, 1.25, 0), Vector3d(2, 2, 2));
		S_fluid.setPlane(Vector3d(0, 2.54, 0), Vector3d(0, -1, 0));

	}
	else {
		exit(5986);
	}

	S_simulation_aggr.addSurface(S_simulation);
	S_fluid_aggr.addSurface(S_fluid);


	timings.time_next_point();//time 


	if (params.show_debug) {
		std::cout << "Simulation space: " << S_simulation_aggr.toString() << std::endl;
		std::cout << "Fluid space: " << S_fluid_aggr.toString() << std::endl;
	}

	//a test for the mesh surface
	if(false){
		test_kernel << <1, 1>> > (S_simulation);
		gpuErrchk(cudaDeviceSynchronize());
		exit(0);
	}

	timings.time_next_point();//time 

	//First I have to load a new background buffer file
	static Vector3d min_fluid_buffer;
	static Vector3d max_fluid_buffer;
	static Vector3d* background_file_positions = NULL;
	static RealCuda background_file_positions_size = 0;

	if (background_file_positions == NULL) {
		SPH::UnifiedParticleSet* dummy = NULL;
		backgroundFluidBufferSet = new SPH::UnifiedParticleSet();
		backgroundFluidBufferSet->load_from_file(data.fluid_files_folder + "background_buffer_file.txt", false, &min_fluid_buffer, &max_fluid_buffer, false);
		allocate_and_copy_UnifiedParticleSet_vector_cuda(&dummy, backgroundFluidBufferSet, 1);
	
		//backup the positions
		cudaMallocManaged(&(background_file_positions), backgroundFluidBufferSet->numParticles * sizeof(Vector3d));
		gpuErrchk(cudaMemcpy(background_file_positions, backgroundFluidBufferSet->pos, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		background_file_positions_size = backgroundFluidBufferSet->numParticles;
	}
	else {
		//backgroundFluidBufferSet = new SPH::UnifiedParticleSet();
		//backgroundFluidBufferSet->init(background_file_positions_size, true, true, false, true);
		backgroundFluidBufferSet->updateActiveParticleNumber(background_file_positions_size);
		gpuErrchk(cudaMemcpy(backgroundFluidBufferSet->pos, background_file_positions, background_file_positions_size * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		backgroundFluidBufferSet->resetColor();
	}


	timings.time_next_point();//time 

	//I'll center the loaded dataset (for now hozontally but in the end I'll also center it vertically)
	if (center_loaded_fluid||params.apply_additional_offset) {
		Vector3d displacement = Vector3d(0, 0, 0);

		if (center_loaded_fluid) {
			displacement = max_fluid_buffer + min_fluid_buffer;
			displacement /= 2;
			//on y I just put the fluid slightly below the simulation space to dodge the special distribution around the borders
			displacement.y = -min_fluid_buffer.y - 0.2;
			//displacement.y = 0;
			if (params.show_debug) {
				std::cout << "background buffer displacement (centering): " << displacement.toString() << std::endl;
			}
		}

		if (params.apply_additional_offset) {
			displacement += params.additional_offset;
			if (params.show_debug) {
				std::cout << "background buffer displacement (manual offsest): " << params.additional_offset.toString() << std::endl;
			}
		}

		if (center_loaded_fluid && params.apply_additional_offset) {
			if (params.show_debug) {
				std::cout << "background buffer displacement (total): " << displacement.toString() << std::endl;
			}
		}

		{
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->pos, displacement, backgroundFluidBufferSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());
		}
	}


	timings.time_next_point();//time 


	//now I need to apply a restriction on the domain to limit it to the current simulated domain
	//so tag the particles
	*outInt = 0;
	if (air_particles_restriction > 0) {
		if (params.show_debug) {
			std::cout << "with added air restriction" << std::endl;
		}


		RealCuda offset_simu_space = (data.particleRadius / 2.0);
		RealCuda air_particle_conservation_distance = -1.0f*air_particles_restriction * data.getKernelRadius();
		

		{
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			surface_restrict_particleset_kernel<< <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_simulation_aggr, offset_simu_space, 
				S_fluid, air_particle_conservation_distance, outInt);
			gpuErrchk(cudaDeviceSynchronize());
		}

		/*
		for (int i = 0; i < backgroundFluidBufferSet->numParticles; ++i) {
			if ((backgroundFluidBufferSet->densityAdv[i] < 0.0f)&& (backgroundFluidBufferSet->densityAdv[i] > air_particle_conservation_distance)) {
				std::cout << "check: " << i << "  " << backgroundFluidBufferSet->densityAdv[i] << "  " << backgroundFluidBufferSet->neighborsDataSet->cell_id[i] << std::endl;
			}
		}
		//*/
	}
	else {
		std::cout << "no added air restriction" << std::endl;
		bool allow_offset_range_simulation = true;
		if (allow_offset_range_simulation) {
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			//surface_restrict_particleset_kernel<0, true> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_simulation, -(data.particleRadius / 4), outInt);
			//surface_restrict_particleset_kernel<true> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_simulation_aggr, (data.particleRadius / 4.0), outInt);
			surface_restrict_particleset_kernel<true> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_simulation_aggr, (data.particleRadius/2.0), outInt);
			//surface_restrict_particleset_kernel<true> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_simulation_aggr, 0.000001, outInt);
			gpuErrchk(cudaDeviceSynchronize());
		}
		else {
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			surface_restrict_particleset_kernel<0, true> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_simulation_aggr, outInt);
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	int count_to_rmv = *outInt;

	//and remove the particles	
	remove_tagged_particles(backgroundFluidBufferSet, backgroundFluidBufferSet->neighborsDataSet->cell_id,
		backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted, count_to_rmv);

	if (params.show_debug) {
		std::cout << "Restricting to simulation area count remaining(count removed): " << backgroundFluidBufferSet->numParticles <<
			" (" << count_to_rmv << ")" << std::endl;
	}

	timings.time_next_point();//time 

	//let's add another goemetric condition on the distance to the boundary particles
	//btw since I only need the neighbor structure of the boundaries to do that it should be fine
	//ince the neighbor structure of the boundaries is build when loading the boundaries
	if(true){
		RealCuda cut_dist = 0.5 * data.particleRadius;
		*outInt = 0;
		int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
		tag_close_to_boundaires_kernel << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, cut_dist,outInt);
		gpuErrchk(cudaDeviceSynchronize());

		count_to_rmv = *outInt;

		//and remove the particles	
		remove_tagged_particles(backgroundFluidBufferSet, backgroundFluidBufferSet->neighborsDataSet->cell_id,
			backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted, count_to_rmv);

		std::cout << "Restricting boundary particle distance dist/remaining/removed: " <<cut_dist/data.particleRadius<<"  "<< backgroundFluidBufferSet->numParticles <<
			" " << count_to_rmv << std::endl;
	}


	timings.time_next_point();//time 


	//and we can finish the initialization
	//it's mostly to have the particles sorted here just for better spacial coherence
	backgroundFluidBufferSet->initNeighborsSearchData(data, true);
	backgroundFluidBufferSet->resetColor();


	//now we need to generate the fluid buffer
	*outInt = 0;
	bool allow_offset_range_fluid = false;
	if(allow_offset_range_fluid){
		int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
		surface_restrict_particleset_kernel<0, true> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_fluid, 0, outInt);
		gpuErrchk(cudaDeviceSynchronize());
	}
	else {
		int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
		surface_restrict_particleset_kernel<0, true> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_fluid_aggr, outInt);
		gpuErrchk(cudaDeviceSynchronize());
	}
	int count_outside_buffer = *outInt;
	count_potential_fluid = backgroundFluidBufferSet->numParticles - count_outside_buffer;

	timings.time_next_point();//time 

	if (params.show_debug) {
		std::cout << "Restricting to fluid area count remaining(count removed): " << count_potential_fluid << " (" << count_outside_buffer << ")" << std::endl;
	}
	//sort the buffer
	cub::DeviceRadixSort::SortPairs(backgroundFluidBufferSet->neighborsDataSet->d_temp_storage_pair_sort, backgroundFluidBufferSet->neighborsDataSet->temp_storage_bytes_pair_sort,
		backgroundFluidBufferSet->neighborsDataSet->cell_id, backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted,
		backgroundFluidBufferSet->neighborsDataSet->p_id, backgroundFluidBufferSet->neighborsDataSet->p_id_sorted, backgroundFluidBufferSet->numParticles);
	gpuErrchk(cudaDeviceSynchronize());

	cuda_sortData(*backgroundFluidBufferSet, backgroundFluidBufferSet->neighborsDataSet->p_id_sorted);
	gpuErrchk(cudaDeviceSynchronize());

	//that buffer is used for tagging in the future so set it to untagged now just to be sure
	set_buffer_to_value<unsigned int>(backgroundFluidBufferSet->neighborsDataSet->cell_id, TAG_UNTAGGED, backgroundFluidBufferSet->numParticles);

	_isInitialized = true;

	timings.time_next_point();//time 
	timings.end_step();//end point of the current step (if measuring avgs you need to call it at every end of the loop)
	timings.recap_timings();//writte timming to cout

	//check the min max using existing functions
	if(false){
		std::cout << "init end check values" << std::endl;
		Vector3d min, max;
		get_UnifiedParticleSet_min_max_naive_cuda(*(backgroundFluidBufferSet), min, max);
		std::cout << "buffer informations: count particles (potential fluid)" << backgroundFluidBufferSet->numParticles << "  (" << count_potential_fluid << ") ";
		std::cout << " min/max " << min.toString() << " // " << max.toString() << std::endl;
	}

	if (false && params.keep_existing_fluid) {
		Vector3d min, max;
		get_UnifiedParticleSet_min_max_naive_cuda(*(data.fluid_data), min, max);
		std::cout << "existing fluid informations: count particles " << data.fluid_data->numParticles << " ";
		std::cout << " min/max " << min.toString() << " // " << max.toString() << std::endl;
	}

	if (false) {
		Vector3d min, max;
		get_UnifiedParticleSet_min_max_naive_cuda(*(data.boundaries_data), min, max);
		std::cout << "existing fluid informations: count particles " << data.boundaries_data->numParticles << " ";
		std::cout << " min/max " << min.toString() << " // " << max.toString() << std::endl;
	}

	if (false) {
		//check the min max of flud particles
		//I dn't want to bother with a kernel so I'll do it by copiing the position info to cpu
		Vector3d* pos_temp = new Vector3d[backgroundFluidBufferSet->numParticles];

		std::cout << "create temp succesful" << std::endl;
		read_UnifiedParticleSet_cuda(*backgroundFluidBufferSet, pos_temp, NULL, NULL, NULL);

		std::cout << "read data succesful"<< std::endl;
		Vector3d min = pos_temp[0];
		Vector3d max = pos_temp[0];
		for (int i = 0; i < (count_potential_fluid); ++i) {
			min.toMin(pos_temp[i]);
			max.toMax(pos_temp[i]);
		}
		std::cout << "fluid min/max " << min.toString() << " // " << max.toString() << std::endl;
	}

	S_simulation_aggr.prepareForDestruction();
	S_fluid_aggr.prepareForDestruction();
	S_simulation.prepareForDestruction();
	S_fluid.prepareForDestruction();

	gpuErrchk(read_last_error_cuda("check cuda error end init ", params.show_debug));
	
}

template<bool tag_untagged_only, bool use_neighbors_storage>
__global__ void tag_neighbors_of_tagged_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, int tag_i, int tag_o) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (particleSet->neighborsDataSet->cell_id[i] == tag_i) {
		Vector3d p_i = particleSet->pos[i];


		if (use_neighbors_storage) {
			ITER_NEIGHBORS_INIT_FROM_STORAGE(data, particleSet, i);

			ITER_NEIGHBORS_SELF_FROM_STORAGE(data, particleSet, i,
				if (i != neighborIndex) {
					if (particleSet->neighborsDataSet->cell_id[neighborIndex] == TAG_UNTAGGED) {
						particleSet->neighborsDataSet->cell_id[neighborIndex] = tag_o;
					}
				}
			);
		}
		else {
			ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);


			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(particleSet->neighborsDataSet, particleSet->pos,
				if (i != j) {
					if (particleSet->neighborsDataSet->cell_id[j] == TAG_UNTAGGED) {
						particleSet->neighborsDataSet->cell_id[j] = tag_o;
					}
				}
			);
		}

	}
}

template<bool tag_untagged_only, bool tag_candidate_only, bool tag_air_separate>
__global__ void tag_neighbors_of_tagged_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, 
	RealCuda tagging_distance, int tag_i, int tag_o, int count_candidate=-1, int tag_o_air= TAG_AIR) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	//particleSet->color[i] = Vector3d(-1, -1, -1);
	if (particleSet->neighborsDataSet->cell_id[i] == tag_i) {
		Vector3d p_i = particleSet->pos[i];
		//particleSet->color[i] = Vector3d(1, 0, 0);

		ITER_NEIGHBORS_INIT_CELL_COMPUTATION(particleSet->pos[i], data.getKernelRadius(), data.gridOffset);

		//override the kernel distance after that the grid cell computation is done
		radius_sq = tagging_distance * tagging_distance;
		
		NeighborsSearchDataSet* neighborsDataSet = particleSet->neighborsDataSet;
		Vector3d* positions = particleSet->pos;
		int range_increase = tagging_distance / data.getKernelRadius();
		/*
		if (i == 9400) {
			printf("range increase test : %i = %f / %f \n", 
				range_increase , tagging_distance , data.getKernelRadius());
		}
		//*/
		for (int k = -1 - range_increase; k < (2 + range_increase); ++k) {
			if ((y + k < 0) || (y + k >= CELL_ROW_LENGTH)) {
				continue;
			}
			for (int m = -1 - range_increase; m < (2 + range_increase); ++m) {
				if ((z + m < 0) || (z + m >= CELL_ROW_LENGTH)) {
					continue;
				}
				for (int n = -1 - range_increase; n < (2 + range_increase); ++n) {
					if ((x + n < 0) || (x + n >= CELL_ROW_LENGTH)) {
						continue;
					}
					unsigned int cur_cell_id = COMPUTE_CELL_INDEX(x + n, y + k, z + m);
					/*
					if (i == 9400) {
						static int count_pass = 0;
						count_pass++;
						printf("count pass : %i // %i : %i %i %i \n",count_pass, cur_cell_id,
							x + n, y + k, z + m);
					}
					//*/
					unsigned int end = neighborsDataSet->cell_start_end[cur_cell_id + 1];
					for (unsigned int cur_particle = neighborsDataSet->cell_start_end[cur_cell_id]; cur_particle < end; ++cur_particle) {
						unsigned int j = neighborsDataSet->p_id_sorted[cur_particle];
						if ((pos - positions[j]).squaredNorm() < radius_sq) {
							if ((!tag_candidate_only) || (j < count_candidate)) {
								if (tag_untagged_only) {
									if (particleSet->neighborsDataSet->cell_id[j] == TAG_UNTAGGED) {
										particleSet->neighborsDataSet->cell_id[j] = tag_o;

									}
								}
								else {
									if (tag_air_separate && (particleSet->neighborsDataSet->cell_id[j] == TAG_AIR ||
										particleSet->neighborsDataSet->cell_id[j] == tag_o_air)) {
										particleSet->neighborsDataSet->cell_id[j] = tag_o_air;
										//particleSet->color[j] = Vector3d(0, 1, 0);
										//printf("test tagging air neighbors\n");
									}else if (particleSet->neighborsDataSet->cell_id[j] != TAG_ACTIVE) {
										particleSet->neighborsDataSet->cell_id[j] = tag_o;
										//particleSet->color[j] = Vector3d(1, 1, 0);
										/*
										if (i == 9400) {
											printf("neighbor info : %i  \n", j);
										}
										//*/
									}
								}
							}
						}
					}
				}
			}
		}
		


	}
}




__global__ void count_particle_with_tag_kernel(SPH::UnifiedParticleSet* particleSet, int tag, int* count_tagged_other, int count_candidates=-1, int* count_tagged_candidates=NULL) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (particleSet->neighborsDataSet->cell_id[i] == tag) {
		if (i < count_candidates) {
			atomicAdd(count_tagged_candidates, 1);
		}
		else {
			atomicAdd(count_tagged_other, 1);
		}
	}
}


__global__ void convert_tag_kernel(SPH::UnifiedParticleSet* particleSet, int tag_i, int tag_o) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (particleSet->neighborsDataSet->cell_id[i] == tag_i) {
		particleSet->neighborsDataSet->cell_id[i] = tag_o;
	}
}

__global__ void convert_tag_kernel(unsigned int* tag_array, int size, int tag_i, int tag_o) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size) { return; }

	if (tag_array[i] == tag_i) {
		tag_array[i] = tag_o;
	}
}

void show_extensive_density_information(SPH::UnifiedParticleSet* particleSet, int count_fluid_particles) {

	{
		RealCuda min_density = 10000;
		RealCuda max_density = 0;
		RealCuda avg_density = 0;
		RealCuda min_density_all = 10000;
		RealCuda max_density_all = 0;
		RealCuda avg_density_all = 0;
		RealCuda min_density_neighbors = 10000;
		RealCuda max_density_neighbors = 0;
		RealCuda avg_density_neighbors = 0;
		RealCuda min_density_neighbors2 = 10000;
		RealCuda max_density_neighbors2 = 0;
		RealCuda avg_density_neighbors2 = 0;
		int count = 0;
		int count_all = 0;
		int count_neighbors = 0;
		int count_neighbors2 = 0;
		for (int j = 0; j < count_fluid_particles; ++j) {
			if (particleSet->neighborsDataSet->cell_id[j] != TAG_REMOVAL) {
				avg_density_all += particleSet->density[j];
				min_density_all = std::fminf(min_density_all, particleSet->density[j]);
				max_density_all = std::fmaxf(max_density_all, particleSet->density[j]);
				count_all++;
			}
			if (particleSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE) {
				avg_density += particleSet->density[j];
				min_density = std::fminf(min_density, particleSet->density[j]);
				max_density = std::fmaxf(max_density, particleSet->density[j]);
				count++;
			}
			if (particleSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE_NEIGHBORS) {
				avg_density_neighbors += particleSet->density[j];
				min_density_neighbors = std::fminf(min_density_neighbors, particleSet->density[j]);
				max_density_neighbors = std::fmaxf(max_density_neighbors, particleSet->density[j]);
				count_neighbors++;
			}
			if (particleSet->neighborsDataSet->cell_id[j] == TAG_1) {
				avg_density_neighbors2 += particleSet->density[j];
				min_density_neighbors2 = std::fminf(min_density_neighbors2, particleSet->density[j]);
				max_density_neighbors2 = std::fmaxf(max_density_neighbors2, particleSet->density[j]);
				count_neighbors2++;
			}
		}
		avg_density /= count;
		avg_density_all /= count_all;
		avg_density_neighbors /= count_neighbors;
		avg_density_neighbors2 /= count_neighbors2;
		std::cout << "                  count/avg/min/max density this iter : " << count << " / " << avg_density << " / " <<
			min_density << " / " << max_density << std::endl;
		std::cout << "                  count/avg/min/max         neighbors : " << count_neighbors << " / " << avg_density_neighbors << " / " <<
			min_density_neighbors << " / " << max_density_neighbors << std::endl;
		std::cout << "                  count/avg/min/max        neighbors2 : " << count_neighbors2 << " / " << avg_density_neighbors2 << " / " <<
			min_density_neighbors2 << " / " << max_density_neighbors2 << std::endl;
		std::cout << "                  count/avg/min/max               all : " << count_all << " / " << avg_density_all << " / " <<
			min_density_all << " / " << max_density_all << std::endl;
	}
}

//this function tags for removal any particle that is too close to the existing fluid
/*
__global__ void tag_close_to__kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, SPH::UnifiedParticleSet* existingFluidSet,
	RealCuda limit_distance_sq, int count_candidates) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (i > count_candidates) {
		return;
	}

	if (particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) {
		return;
	}

	Vector3d p_i = particleSet->pos[i];
	ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, particleSet, i);
	ITER_NEIGHBORS_FROM_STRUCTURE(existingFluidSet->neighborsDataSet, existingFluidSet->pos,
		{
			if ((p_i - existingFluidSet->pos[j]).squaredNorm() < limit_distance_sq) {
				particleSet->neighborsDataSet->cell_id[i] != TAG_REMOVAL;
				return;
			}
		});
}
//*/


void RestFLuidLoader::tagDataToSurface(SPH::DFSPHCData& data, RestFLuidLoaderInterface::TaggingParameters& params) {
	if (!isInitialized()) {
		std::cout << "RestFLuidLoader::tagDataToSurface tagging impossible data was not initialized" << std::endl;
		return;
	}


	gpuErrchk(read_last_error_cuda("check error set up tagging start: ", params.show_debug));

	//reset the output values
	params.count_iter = 0;



	//ok so I'll use the same method as for the dynamic boundary but to initialize a fluid
	//although the buffers wont contain the same data
	//I'll code it outside of the class for now Since I gues it will be another class
	//although it might now be but it will do for now
	///TODO: For this process I only use one unified particle set, as surch, I could just work inside the actual fluid buffer
	///TODO:  ^	NAH no need for that  
	bool show_debug = params.show_debug;
	bool send_result_to_file = false;

	//let's try to apply the selection on the air particles too (while only keeping one layer of air particles
	//it should lower the maximum densities observed and improve the stability of the stabilization step
	bool apply_selection_to_air = false;
	int old_count_potential_fluid = count_potential_fluid;
	if (apply_selection_to_air) {
		count_potential_fluid = backgroundFluidBufferSet->numParticles;
	}

	//*
	std::vector<std::string> timing_names{ "neighbor","void","tagging","void","constant density based extraction",
		"loop","count_tagged_for_removal","void","cleartag" };
	SPH::SegmentedTiming timings("tag data", timing_names, true);
	timings.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)


	//*/
	//reinitialise the neighbor structure (might be able to delete it if I never use it
	//though I will most likely use it
	//backgroundFluidBufferSet->initNeighborsSearchData(data, false);
	backgroundFluidBufferSet->initAndStoreNeighbors(data, false);

	if (params.keep_existing_fluid) {
		data.fluid_data->initNeighborsSearchData(data, false);
	}

	timings.time_next_point();//time 

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
	if (show_debug) {
		cudaMallocManaged(&(background_pos_backup), backgroundFluidBufferSet->numParticles * sizeof(Vector3d));
		gpuErrchk(cudaMemcpy(background_pos_backup, backgroundFluidBufferSet->pos, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
	}
	//*/

	//zero everything
	set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->density, 0, backgroundFluidBufferSet->numParticles);
	set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->densityAdv, 0, backgroundFluidBufferSet->numParticles);
	set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->kappa, 0, backgroundFluidBufferSet->numParticles);
	set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->kappaV, 0, backgroundFluidBufferSet->numParticles);


	timings.time_next_point();//time 


	gpuErrchk(read_last_error_cuda("check error before tagging early removal: ", params.show_debug));

	//*
	//then do a preliminary tag to identify the particles that are close to the boundaries
	bool active_particles_first = true;
	{
		//tag the air and the fluid with preliminary tags
		set_buffer_to_value<unsigned int>(backgroundFluidBufferSet->neighborsDataSet->cell_id, TAG_AIR, backgroundFluidBufferSet->numParticles);
		set_buffer_to_value<unsigned int>(backgroundFluidBufferSet->neighborsDataSet->cell_id, TAG_UNTAGGED, count_potential_fluid);

		///TODO reverse that so that I explore the fluid and check if they have boundries/solid neighbors
		//tag boundaries neigbors
		{
			int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
			tag_neighborhood_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data->gpu_ptr, backgroundFluidBufferSet->gpu_ptr,
				data.getKernelRadius(), count_potential_fluid);
			//the synch is done at the end since all those kernel can be executed at once
			//gpuErrchk(cudaDeviceSynchronize());
		}
		//tag dynamic objects neighbors
		for (int i = 0; i < data.numDynamicBodies; ++i) {
			int numBlocks = calculateNumBlocks(data.vector_dynamic_bodies_data[i].numParticles);
			tag_neighborhood_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.vector_dynamic_bodies_data[i].gpu_ptr, backgroundFluidBufferSet->gpu_ptr,
				data.getKernelRadius(), count_potential_fluid);
		}
		gpuErrchk(cudaDeviceSynchronize());

		//tag existing fluid neighbors
		if (params.keep_existing_fluid) {
			int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
			//here I can optimize the comutation by tagging for removal any particle that is extremely close to an existing fluid particle
			bool remove_close_to_fluid = true;
			if (remove_close_to_fluid) {
				RealCuda distance_to_exiting_limit = data.particleRadius*0.5;
				tag_neighborhood_kernel<false, false> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, backgroundFluidBufferSet->gpu_ptr,
					distance_to_exiting_limit, backgroundFluidBufferSet->numParticles, TAG_REMOVAL);
				gpuErrchk(cudaDeviceSynchronize());
			}
			
			//do the actual tag as actives
			tag_neighborhood_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, backgroundFluidBufferSet->gpu_ptr,
				data.getKernelRadius(), count_potential_fluid);
			gpuErrchk(cudaDeviceSynchronize());


			//if we have existing fluid we can directly eliminate any particle too close from it 

		}

		//and tag their neighbors if I need it
		bool tag_neigbors = true;
		if (tag_neigbors) {
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS);
			gpuErrchk(cudaDeviceSynchronize());

		}

		if (false) {//for some reason this time it is slower when doing that...
			//run the sort
			cub::DeviceRadixSort::SortPairs(data.fluid_data->neighborsDataSet->d_temp_storage_pair_sort, data.fluid_data->neighborsDataSet->temp_storage_bytes_pair_sort,
				data.fluid_data->neighborsDataSet->cell_id, data.fluid_data->neighborsDataSet->cell_id_sorted,
				data.fluid_data->neighborsDataSet->p_id, data.fluid_data->neighborsDataSet->p_id_sorted, data.fluid_data->numParticles);
			gpuErrchk(cudaDeviceSynchronize());

			cuda_sortData(*(data.fluid_data), data.fluid_data->neighborsDataSet->p_id_sorted);
			gpuErrchk(cudaDeviceSynchronize());

			//doing it forces us to rebuild the neighbors and the tag
			backgroundFluidBufferSet->initAndStoreNeighbors(data, false);

			//tag the air and the fluid with preliminary tags
			set_buffer_to_value<unsigned int>(backgroundFluidBufferSet->neighborsDataSet->cell_id, TAG_AIR, backgroundFluidBufferSet->numParticles);
			set_buffer_to_value<unsigned int>(backgroundFluidBufferSet->neighborsDataSet->cell_id, TAG_UNTAGGED, count_potential_fluid);


			//tag boundaries neigbors
			{
				int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
				tag_neighborhood_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data->gpu_ptr, backgroundFluidBufferSet->gpu_ptr,
					data.getKernelRadius(), count_potential_fluid);
				gpuErrchk(cudaDeviceSynchronize());
			}
			//tag dynamic objects neighbors
			for (int i = 0; i < data.numDynamicBodies; ++i) {
				int numBlocks = calculateNumBlocks(data.vector_dynamic_bodies_data[i].numParticles);
				tag_neighborhood_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.vector_dynamic_bodies_data[i].gpu_ptr, backgroundFluidBufferSet->gpu_ptr,
					data.getKernelRadius(), count_potential_fluid);
				gpuErrchk(cudaDeviceSynchronize());
			}

			//and tag their neighbors if I need it
			bool tag_neigbors = true;
			if (tag_neigbors) {
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS);
				gpuErrchk(cudaDeviceSynchronize());

			}
		}

		if (show_debug) {

			//for debug purposes check the numbers
			{
				int tag = TAG_ACTIVE;
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
			{
				int tag = TAG_ACTIVE_NEIGHBORS;
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
			{
				int tag = TAG_AIR;
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}

			{
				int tag = TAG_UNTAGGED;
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}

			{
				int tag = TAG_REMOVAL;
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
		}
	}


	timings.time_next_point();//time 

	//backup index (a debug test
	if (false) {
		for (int i = 0; i < backgroundFluidBufferSet->numParticles; i++) {
			backgroundFluidBufferSet->kappaV[i] = backgroundFluidBufferSet->neighborsDataSet->cell_id[i];

		}
	}

	//evaluate the density and show it (only debug)
	if(show_debug){
		{
			
			int numBlocks = calculateNumBlocks(count_potential_fluid);
			evaluate_and_tag_high_density_from_buffer_kernel<false, true, true, true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr,
				outInt, 4000, count_potential_fluid, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
			gpuErrchk(cudaDeviceSynchronize());
		}


		std::cout << "!!!!!!!!!!!!! before comp informations !!!!!!!!!!!!!!!" << std::endl;
		show_extensive_density_information(backgroundFluidBufferSet, count_potential_fluid);
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
	}
	//a test to compare the code using store neighbors and on fly neighbors
	if (false) {
		{
			int numBlocks = calculateNumBlocks(count_potential_fluid);
			evaluate_and_tag_high_density_from_buffer_kernel<false, true, true, false> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, 
				outInt, 4000, count_potential_fluid, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
			gpuErrchk(cudaDeviceSynchronize());
		}

		{
			RealCuda min_density = 10000;
			RealCuda max_density = 0;
			RealCuda avg_density = 0;
			RealCuda min_density_all = 10000;
			RealCuda max_density_all = 0;
			RealCuda avg_density_all = 0;
			RealCuda min_density_neighbors = 10000;
			RealCuda max_density_neighbors = 0;
			RealCuda avg_density_neighbors = 0;
			RealCuda min_density_neighbors2 = 10000;
			RealCuda max_density_neighbors2 = 0;
			RealCuda avg_density_neighbors2 = 0;
			int count = 0;
			int count_all = 0;
			int count_neighbors = 0;
			int count_neighbors2 = 0;
			for (int j = 0; j < count_potential_fluid; ++j) {
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] != TAG_REMOVAL) {
					avg_density_all += backgroundFluidBufferSet->density[j];
					min_density_all = std::fminf(min_density_all, backgroundFluidBufferSet->density[j]);
					max_density_all = std::fmaxf(max_density_all, backgroundFluidBufferSet->density[j]);
					count_all++;
				}
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE) {
					avg_density += backgroundFluidBufferSet->density[j];
					min_density = std::fminf(min_density, backgroundFluidBufferSet->density[j]);
					max_density = std::fmaxf(max_density, backgroundFluidBufferSet->density[j]);
					count++;
				}
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE_NEIGHBORS) {
					avg_density_neighbors += backgroundFluidBufferSet->density[j];
					min_density_neighbors = std::fminf(min_density_neighbors, backgroundFluidBufferSet->density[j]);
					max_density_neighbors = std::fmaxf(max_density_neighbors, backgroundFluidBufferSet->density[j]);
					count_neighbors++;
				}
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_1) {
					avg_density_neighbors2 += backgroundFluidBufferSet->density[j];
					min_density_neighbors2 = std::fminf(min_density_neighbors2, backgroundFluidBufferSet->density[j]);
					max_density_neighbors2 = std::fmaxf(max_density_neighbors2, backgroundFluidBufferSet->density[j]);
					count_neighbors2++;
				}
			}
			avg_density /= count;
			avg_density_all /= count_all;
			avg_density_neighbors /= count_neighbors;
			avg_density_neighbors2 /= count_neighbors2;
			std::cout << "!!!!!!!!!!!!! before comp informations !!!!!!!!!!!!!!!" << std::endl;
			std::cout << "                  count/avg/min/max density this iter : " << count << " / " << avg_density << " / " <<
				min_density << " / " << max_density << std::endl;
			std::cout << "                  count/avg/min/max         neighbors : " << count_neighbors << " / " << avg_density_neighbors << " / " <<
				min_density_neighbors << " / " << max_density_neighbors << std::endl;
			std::cout << "                  count/avg/min/max        neighbors2 : " << count_neighbors2 << " / " << avg_density_neighbors2 << " / " <<
				min_density_neighbors2 << " / " << max_density_neighbors2 << std::endl;
			std::cout << "                  count/avg/min/max               all : " << count_all << " / " << avg_density_all << " / " <<
				min_density_all << " / " << max_density_all << std::endl;
			std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		}

	}



	//now we can use the iterative process to remove particles that have a density to high
	//no need to lighten the buffers for now since I only use one
	RealCuda limit_density = 0;
	int total_to_remove = 0;

	std::vector<std::string> timing_names_loop{ "eval","compute_avg","change_step_size","save1","save2","confirm" };
	SPH::SegmentedTiming timings_loop("tag data loop", timing_names_loop, true);

	RealCuda density_start = params.density_start;
	RealCuda density_end = params.density_end;
	RealCuda step_density = params.step_density;
	limit_density = density_start;
	int i = 0;//a simple counter
	bool use_cub_for_avg = false;//ok for some reason using cub has consequences that augent the computation time... don't ask me
	int* outInt2 = SVS_CU::get()->count_rmv_particles;
	int count_active_ini = 0;
	if (use_cub_for_avg) 
	{
		{
			int tag = TAG_ACTIVE;
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			*(SVS_CU::get()->tagged_particles_count) = 0;
			count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
			gpuErrchk(cudaDeviceSynchronize());

			count_active_ini=*(SVS_CU::get()->tagged_particles_count);
		}
	}
	bool onepass_removal_counting = true;
	bool successful = false;
	const bool use_clean_version = true;
	const bool candidate_validation_separate = true;
	RealCuda* max_density_first_step= SVS_CU::get()->avg_density_err;
	*max_density_first_step = 0;
	//clear buffers
	set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->densityAdv, 0, backgroundFluidBufferSet->numParticles);
	set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->kappaV, 0, backgroundFluidBufferSet->numParticles);
	set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->kappa, 0, backgroundFluidBufferSet->numParticles);


	timings.time_next_point();//time 


	gpuErrchk(read_last_error_cuda("check error before constant density based early removal: ", params.show_debug));

	//let's compute the constant density contribution before anything
	if (use_clean_version) {
		
		int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
		compute_density_and_extract_large_contribution_kernel << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr,
			700, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
		
		gpuErrchk(cudaDeviceSynchronize());

		if (show_debug) {
			//for debug purposes check the numbers
			{
				int tag = TAG_ACTIVE;
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
			{
				int tag = TAG_ACTIVE_NEIGHBORS;
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
			{
				int tag = TAG_AIR;
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}

			{
				int tag = TAG_UNTAGGED;
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}

			{
				int tag = TAG_REMOVAL;
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
		}

		if (show_debug) {
			{

				int numBlocks = calculateNumBlocks(count_potential_fluid);
				evaluate_and_tag_high_density_from_buffer_kernel<false, true, true, true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr,
					outInt, 4000, count_potential_fluid, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
				gpuErrchk(cudaDeviceSynchronize());
			}


			std::cout << "!!!!!!! After constant density component based elimination informations !!!!!!!!!!" << std::endl;
			show_extensive_density_information(backgroundFluidBufferSet, count_potential_fluid);
			std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		}
	}

	timings.time_next_point();//time 


	gpuErrchk(read_last_error_cuda("check error before selection loop: ", params.show_debug));

	//there is a condition inside the loop to end it
	while (true) {
		 ++i;//a simple counter

		set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->kappa, 0, backgroundFluidBufferSet->numParticles);


		if (max_density_first_step != NULL) {
			*max_density_first_step = 0;
		}
		//set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->density, 0, backgroundFluidBufferSet->numParticles);

		timings_loop.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)
		
		limit_density -= step_density;
		if (limit_density < density_end) {
			limit_density = density_end-0.001;
		}
		//I will use the kappa buffer to compute the avg density of active particles
		RealCuda avg_density = 0;
		RealCuda sum_density_active = 0;
		RealCuda count_density_active = 0;
		int count_to_rmv_this_step = 0;
		if (use_clean_version) {
		
			*outRealCuda = 0;
			*outInt = 0;
			{
				int numBlocks = calculateNumBlocks(count_potential_fluid);
				particle_selection_rule_1_kernel<true, false> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr,
					limit_density, count_potential_fluid, outRealCuda, outInt,
					max_density_first_step, 5, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
				gpuErrchk(cudaDeviceSynchronize());
			}

	
			if (false) {
				if (i == 1) {
					Vector3d* pos = new Vector3d[backgroundFluidBufferSet->numParticles];
					read_UnifiedParticleSet_cuda(*backgroundFluidBufferSet, pos, NULL, NULL, NULL);

					std::ofstream myfile("temp5.csv", std::ofstream::trunc);
					if (myfile.is_open())
					{
						for (int j = 0; j < backgroundFluidBufferSet->numParticles; j++) {

							myfile << backgroundFluidBufferSet->neighborsDataSet->cell_id[j] << "  " << pos[j].toString() << "  " <<
								backgroundFluidBufferSet->density[j] << "  " <<
								backgroundFluidBufferSet->density[j] - backgroundFluidBufferSet->densityAdv[j] << "  " <<
								backgroundFluidBufferSet->densityAdv[j] << "  " << backgroundFluidBufferSet->kappa[j] << "  " <<
								backgroundFluidBufferSet->densityAdv[j] - backgroundFluidBufferSet->kappa[j] << "  " <<
								backgroundFluidBufferSet->kappaV[j] << "  " << std::endl;
						}
						myfile.close();
					}

					exit(0);
				}
			}

			if (false) {
				{
					// a test to compare the zone between the fluid-fluid and border-fluid
					UnifiedParticleSet* particleSet = backgroundFluidBufferSet;
					RealCuda min_density = 10000;
					RealCuda max_density = 0;
					RealCuda avg_density = 0;
					RealCuda min_density_all = 10000;
					RealCuda max_density_all = 0;
					RealCuda avg_density_all = 0;
					int count = 0;
					int count_all = 0;
					for (int j = 0; j < count_potential_fluid; ++j) {
						if (particleSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE) {
							if (particleSet->getNumberOfNeighbourgs(j, 1) > 0) {
								avg_density += particleSet->density[j];
								min_density = std::fminf(min_density, particleSet->density[j]);
								max_density = std::fmaxf(max_density, particleSet->density[j]);
								count++;
							}
							else {
								avg_density_all += particleSet->density[j];
								min_density_all = std::fminf(min_density_all, particleSet->density[j]);
								max_density_all = std::fmaxf(max_density_all, particleSet->density[j]);
								count_all++;
							}
						}
					}
					avg_density /= count;
					avg_density_all /= count_all;
					std::cout << "                  count/avg/min/max density boundary : " << count << " / " << avg_density << " / " <<
						min_density << " / " << max_density << std::endl;
					std::cout << "                  count/avg/min/max density fluid    : " << count_all << " / " << avg_density_all << " / " <<
						min_density_all << " / " << max_density_all << std::endl;
				}
			}


			//first we need to check if there is any particle left
			if ((*outInt) == 0) {
				//ok i'll do a gross suposition, if there is no candidate particle left for the selection
				//it very likely means that there is no particle remaining that would be added to the simulation
				//if I were to continue with the process
				_isDataTagged = false;
				return;
			}

			//exit(0);
			//avg_density = 1000;//(*outRealCuda) / (*outInt);

			sum_density_active = *outRealCuda;
			count_density_active = *outInt;
			avg_density = sum_density_active / count_density_active;
			//avg_density = (*outRealCuda) / (count_active_ini - total_to_remove);

			if (max_density_first_step != NULL) {
				if (show_debug) {
					std::cout << "max density: " << *max_density_first_step << std::endl;
				}
				if (limit_density > (*max_density_first_step)) {
					if (show_debug) {
						std::cout << "currently limit bellow max so changing max to fit max density (old/new): " << 
							limit_density<< " / "<< (*max_density_first_step) << std::endl;
						std::cout << std::endl;
						std::cout << "initializeFluidToSurface: fitting fluid, iter: " << i << "  skipped"<<std::endl;

					}

					//set a new limit
					//knowing this valu will be reduced byt the step
					limit_density = (*max_density_first_step);

					//yeah if we have the timer we need to end its step
					timings_loop.end_step(true);//time
					
					//deactivate that since for next stepsince it isimpossible for it to trigger for 2 steps
					max_density_first_step = NULL;



					//go to the next step
					continue;
				}
			}
			else {
				//reactivate the utation of the max in case there is a gap after some steps
				//Normaly this should NEVER happens exact when there is extremely few remaining particles
				//when the existing fluid is used this fucking case happens sometime due to some optimizations 
				//so I might as well use it as an advantage
				//and now with the step regulator it also happens when there is no existing fluid so i'll keep it activated
				//if (params.keep_existing_fluid) 
				{
					max_density_first_step = SVS_CU::get()->avg_density_err;
					*max_density_first_step = 0;
				}
				
			}

			/*
			RealCuda* sum_densities = SVS_CU::get()->avg_density_err;
			cub::DeviceReduce::Sum(backgroundFluidBufferSet->d_temp_storage, backgroundFluidBufferSet->temp_storage_bytes,
				backgroundFluidBufferSet->density, sum_densities, count_potential_fluid);
			gpuErrchk(cudaDeviceSynchronize());
			avg_density = (*sum_densities) / (count_active_ini-total_to_remove);
			//*/
			//


			//std::cout << "sum density comparison: " << *outRealCuda << "  " << *sum_densities << std::endl;
			//std::cout << "count active comparison: " << *outInt<< "  " << count_active_ini-total_to_remove << std::endl;
			timings_loop.time_next_point();//time
		}
		else {
			if (use_cub_for_avg) {
				set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->densityAdv, 0, *outInt2);
				*outInt2 = 0;
			}else{
				outInt2 = NULL;
			}
			*outInt = 0;
			{
				int numBlocks = calculateNumBlocks(count_potential_fluid);
				evaluate_and_tag_high_density_from_buffer_kernel<false, true, true, true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr,
					outInt, limit_density, count_potential_fluid, outInt2, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
				gpuErrchk(cudaDeviceSynchronize());
			}
			count_to_rmv_this_step = *outInt;

			if (!use_cub_for_avg) {
				outInt2 = SVS_CU::get()->count_rmv_particles;
			}

			timings_loop.time_next_point();//time

			//check the avg
			{
				int count = 0;

				if (use_cub_for_avg) {
					RealCuda* sum_densities = SVS_CU::get()->avg_density_err;
					cub::DeviceReduce::Sum(backgroundFluidBufferSet->d_temp_storage, backgroundFluidBufferSet->temp_storage_bytes,
						backgroundFluidBufferSet->densityAdv, sum_densities, *outInt2);

					gpuErrchk(cudaDeviceSynchronize());

					count = count_active_ini - total_to_remove;
					avg_density = *sum_densities / count;
					//std::cout << "check avg density avg/count: " << avg_density << "   " << count << std::endl;
				}
				else
				{
					avg_density = 0;
					count = 0;
					for (int j = 0; j < count_potential_fluid; ++j) {
						if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE ||
							backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_REMOVAL_CANDIDATE) {
							avg_density += backgroundFluidBufferSet->density[j];
							count++;
						}
					}
					avg_density /= count;
				}
				//std::cout << "check avg density avg/count: " << avg_density << "   " << count << std::endl;
			}
		}

		if (show_debug) {
			std::cout << "avg density before check end avg density (num particle contributing): " << avg_density <<
				"  ("<<count_density_active<<")"<<std::endl;
		}

		//end the process if avg reach target
		{
			if (((avg_density - density_end) < 0)) {
				if (show_debug) {
					std::cout << "Rest density reached at iter/limit_density/avg_density: " << i << "  " << limit_density << "  " << avg_density << std::endl;
				}

				//clear the candidate tagging
				for (int j = 0; j < count_potential_fluid; ++j) {
					if(backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_REMOVAL_CANDIDATE) {
						backgroundFluidBufferSet->neighborsDataSet->cell_id[j] = TAG_ACTIVE;
					}
				}

				//I have to end all the timers iteration
				timings_loop.time_next_point();//time
				timings_loop.time_next_point();//time
				timings_loop.time_next_point();//time
				timings_loop.time_next_point();//time
				timings_loop.time_next_point();//time
				timings_loop.end_step();

				successful = true;

				//and end the iteration process
				break;
			}
		}

		timings_loop.time_next_point();//time

		//if the current avg density is too close to the target with a step toolarge we need to reduce the step so we don't
		//actually skip the target density too much
		if (params.useStepSizeRegulator) {
			if(step_density>params.min_step_density){
				if (show_debug) {
					std::cout << "avg density before step modification: " << avg_density << std::endl;
				}
				RealCuda delta_to_target = avg_density - density_end;
				if (delta_to_target < (params.step_density / 2.0f)) {
					RealCuda old_step_density = step_density;
					step_density = static_cast<int>(delta_to_target)*2;
					step_density = MAX_MACRO_CUDA(step_density, params.min_step_density);
					limit_density += old_step_density - step_density;
					if (show_debug) {
						std::cout << "changing step size from         " << old_step_density << " to " << step_density << std::endl;
						std::cout << "resulting in limit density from " << limit_density+step_density-old_step_density << " to " << 
							limit_density << std::endl;
					}

					//and if the limit density has changed I need to untag part of the candidates
					//that are now above the new limit desity
					{
						int numBlocks = calculateNumBlocks(count_potential_fluid);
						untag_candidate_below_limit_kernel << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr,
							limit_density, count_potential_fluid);
						gpuErrchk(cudaDeviceSynchronize());
					}
				}
			}
		}

		timings_loop.time_next_point();//time

		
		//this is simple an empty line to make the output more lisible
		if (show_debug) {
			std::cout << std::endl;
		}

		//a variable that count te total number of saved particles
		int count_saved = 0;
		int count_saved_1 = 0;
		//let's try to not remove candidates that are too clase to low points
		bool invert_save = false;
		if (params.useRule2){
			*outInt = 0;
			if (active_particles_first) {
				throw("not sure I only use stored neighbors here so i can't simply use that boolean");
			}
			//idealy this would only be run by one thread, but it would take a massive amount of time
			//so I'll run it multithread with the risk of loosing some usefull particles
			bool use_multithread = false;
			RealCuda min_density = params.min_density;
			if (use_multithread) {
				int numBlocks = calculateNumBlocks(count_potential_fluid);
				if (invert_save) {
					save_usefull_candidates_kernel_v2 << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, min_density, count_potential_fluid);
				}else{
					save_usefull_candidates_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, min_density, count_potential_fluid);
				}
				if (show_debug)
				{
					convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, TAG_SAVE, TAG_ACTIVE);
				}
			}
			else {
				save_usefull_candidates_kernel<false> << <1, 1 >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, min_density, count_potential_fluid);
			}
			gpuErrchk(cudaDeviceSynchronize());
			count_saved_1 = *outInt;
			count_saved += count_saved_1;
		}


		timings_loop.time_next_point();//time
		int count_saved_2 = 0;
		//it is doing a good job for high density limit, bu as long as I ask for very low density limits it is useless
		//it also works pretty well when doing large juumps
		//essencially this system is to make sure we don't remove "packs" of particles at the same time which would cause aps in the fluid
		if(params.useRule3){
			*outInt = 0;
			bool use_multithread = true;
			if (use_multithread) {
				RealCuda density_delta_threshold = params.density_delta_threshold;
				int numBlocks = calculateNumBlocks(count_potential_fluid);
				if (onepass_removal_counting&&(!candidate_validation_separate)) {
					verify_candidate_tagging_multithread_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, 
						limit_density, density_delta_threshold, count_potential_fluid);
				}
				else {
					verify_candidate_tagging_multithread_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt,
						limit_density, density_delta_threshold, count_potential_fluid);
				}
			}
			else {
				//this can only be run by one threadint numBlocks = calculateNumBlocks(count_potential_fluid);
				verify_candidate_tagging_kernel << <1, 1>> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, limit_density, count_potential_fluid);
			}
			gpuErrchk(cudaDeviceSynchronize());
			count_saved_2 = *outInt;
			count_saved += count_saved_2;
		}
		int count_confirm = 0;

		timings_loop.time_next_point();//time
		
		if (onepass_removal_counting && candidate_validation_separate) {
			//*outInt = 0;
			//*outRealCuda = 0;
			{
				int numBlocks = calculateNumBlocks(count_potential_fluid);
				confirm_candidates_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, count_potential_fluid, 
					NULL, NULL);
				gpuErrchk(cudaDeviceSynchronize());
			}
			/*
			sum_density_active -= *outRealCuda;
			count_density_active -= *outInt;
			avg_density = sum_density_active / count_density_active;
			//*/
		}

		//convert the remaining cnadidates to actual removal
		if (!onepass_removal_counting) {
			{
				*outInt = 0;
				{
					int numBlocks = calculateNumBlocks(count_potential_fluid);
					confirm_candidates_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, count_potential_fluid, 
						outInt, NULL);
					gpuErrchk(cudaDeviceSynchronize());
				}
				count_confirm += *outInt;
			}
			if (use_clean_version) {
				count_to_rmv_this_step = count_confirm;
				total_to_remove += count_to_rmv_this_step;
			}
			else {
				throw("Verify that because i most likely don't wokrd like that anymomre, I added the count_confirm variable");
				if (invert_save) {
					count_saved_1= count_to_rmv_this_step - *outInt -count_saved_2;
					count_saved = count_saved_1+count_saved_2;
				}
				else {
					if (count_saved != (count_to_rmv_this_step - *outInt)) {
						std::cout << "LOLILOL there is a computation error in the count nbr expected: " << count_to_rmv_this_step - *outInt << std::endl;
					}
				}
				total_to_remove += count_to_rmv_this_step-count_saved;
			}
		}

		timings_loop.time_next_point();//time	
		timings_loop.end_step();//end point of the current step (if measuring avgs you need to call it at every end of the loop)



		if(show_debug){
			{
				int tag = TAG_ACTIVE;
				int numBlocks = calculateNumBlocks(count_potential_fluid);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}

			{
				int tag = TAG_REMOVAL;
				int numBlocks = calculateNumBlocks(count_potential_fluid);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());
				total_to_remove = *(SVS_CU::get()->tagged_particles_count);

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}


			std::cout << "initializeFluidToSurface: fitting fluid, iter: " << i << 
				"  target density: "<<limit_density<<"   nb rmv tot / step (cur candi/(save1,save2)): " << total_to_remove <<
				"   " << count_to_rmv_this_step - count_saved <<
				" (" << count_to_rmv_this_step << "  //  (" << count_saved_1 <<","<< count_saved_2 << ") " << std::endl;
			{
				{
					int numBlocks = calculateNumBlocks(count_potential_fluid);
					evaluate_and_tag_high_density_from_buffer_kernel<false, true, true, true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr,
						outInt, 4000, count_potential_fluid, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
					gpuErrchk(cudaDeviceSynchronize());
				}


				show_extensive_density_information(backgroundFluidBufferSet, count_potential_fluid);
			}

			//a check to know if the number of particles is right
			{
				std::cout << "suposed curretn number of particles: " << count_potential_fluid - total_to_remove << std::endl;
			}

			if (false) {
				//check the min max of flud particles
				//I dn't want to bother with a kernel so I'll do it by copiing the position info to cpu
				Vector3d* pos_temp = new Vector3d[backgroundFluidBufferSet->numParticles];

				read_UnifiedParticleSet_cuda(*backgroundFluidBufferSet, pos_temp, NULL, NULL, NULL);

				Vector3d min = pos_temp[0];
				Vector3d max = pos_temp[0];
				for (int j = 0; j < (count_potential_fluid); ++j) {
					if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] != TAG_REMOVAL) {
						min.toMin(pos_temp[j]);
						max.toMax(pos_temp[j]);
					}
				}
				std::cout << "fluid min/max " << min.toString() << " // " << max.toString() << std::endl;
			}

			
		}
	
		

	}


	gpuErrchk(read_last_error_cuda("check error after selection loop: ", params.show_debug));

	//save the count of iter as an output
	params.count_iter = i;

	timings.time_next_point();//time 
	timings_loop.recap_timings();//writte timming to cout

	//revert the affected particles to only the fluid particles
	if (apply_selection_to_air) {
		count_potential_fluid = old_count_potential_fluid;

		
		//okay let's now do a last force removalof anything too high that is remaining in the air
		{
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles-count_potential_fluid);
			tag_high_density_in_air_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, count_potential_fluid, 1050);
			gpuErrchk(cudaDeviceSynchronize());
		}

	
		//a verification check
		if (false) {
			if(false){
				set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->density, 0, backgroundFluidBufferSet->numParticles);
				{
					int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
					evaluate_and_tag_high_density_from_buffer_kernel<false, true, true, true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr,
						outInt, 4000, backgroundFluidBufferSet->numParticles, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
					gpuErrchk(cudaDeviceSynchronize());
				}
			}

			//*
			std::cout << "!!!!!!!!!!!!! after with air           !!!!!!!!!!!!!!!" << std::endl;
			show_extensive_density_information(backgroundFluidBufferSet, backgroundFluidBufferSet->numParticles);
			std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			//*/



			std::ofstream myfile("temp2.csv", std::ofstream::trunc);
			if (myfile.is_open())
			{
				for (int i_test = 0; i_test < backgroundFluidBufferSet->numParticles; ++i_test) {
					myfile << i_test << "   " << backgroundFluidBufferSet->density[i_test] << "  " <<
						backgroundFluidBufferSet->neighborsDataSet->cell_id[i_test] << "  " <<
						(i_test<count_potential_fluid)<<"  "<< backgroundFluidBufferSet->getNumberOfNeighbourgs(i_test, 0) << "  " <<
						backgroundFluidBufferSet->getNumberOfNeighbourgs(i_test, 1) << "  " << std::endl;

				}
			}
		}


	}

	if (false) {
		{
			Vector3d* pos = new Vector3d[backgroundFluidBufferSet->numParticles];
			read_UnifiedParticleSet_cuda(*backgroundFluidBufferSet, pos, NULL, NULL, NULL);

			std::ofstream myfile("temp5.csv", std::ofstream::trunc);
			if (myfile.is_open())
			{
				for (int j = 0; j < backgroundFluidBufferSet->numParticles; j++) {

					myfile << backgroundFluidBufferSet->neighborsDataSet->cell_id[j] << "  " << pos[j].toString() << "  " <<
						backgroundFluidBufferSet->density[j] << "  " <<
						backgroundFluidBufferSet->density[j] - backgroundFluidBufferSet->densityAdv[j] << "  " <<
						backgroundFluidBufferSet->densityAdv[j] << "  " << backgroundFluidBufferSet->kappa[j] << "  " <<
						backgroundFluidBufferSet->densityAdv[j] - backgroundFluidBufferSet->kappa[j] << "  " <<
						backgroundFluidBufferSet->kappaV[j] << "  " << std::endl;
				}
				myfile.close();
			}

			exit(0);
		}
	}

	//count the number of particles to remove at the end
	//if the air particles are part of the selection I'll use a two pass process for now 
	//if necessary create a specialized function to do it in one pass
	if(onepass_removal_counting){
		count_high_density_tagged_in_potential = 0;
		count_high_density_tagged_in_air = 0;

		int tag = TAG_REMOVAL; 
		int* count_tagged_candidates = (SVS_CU::get()->tagged_particles_count);
		int* count_tagged_other = SVS_CU::get()->count_created_particles;
		*count_tagged_candidates = 0;
		*count_tagged_other = 0;
		{
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, count_tagged_other,
				count_potential_fluid, count_tagged_candidates);
			gpuErrchk(cudaDeviceSynchronize());

		}
		total_to_remove = *count_tagged_candidates;
		count_high_density_tagged_in_potential = *count_tagged_candidates;
		count_high_density_tagged_in_air = *count_tagged_other;

	
	
		if (show_debug) {
			std::cout << "total number of particle to remove: " << count_high_density_tagged_in_potential <<
				"  "<< count_high_density_tagged_in_air << std::endl;
		}

		//test the count on cpu
		if (false) {
			int count_temp1 = 0;
			int count_temp2 = 0;
			for (int i_test = 0; i_test < backgroundFluidBufferSet->numParticles; ++i_test) {
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[i_test] == TAG_REMOVAL) {
					if (i_test < count_potential_fluid) {
						count_temp1++;
					}
					else {
						count_temp2++;
					}
				}
			}
			std::cout << "total count from cpu: " << count_temp1 <<"  "<<count_temp2<< std::endl;
		}

	}

	timings.time_next_point();//time 


	//if we had to use the last iter of the loop because we were never successful before it
	if (!successful) {
		//reevaluta to see if the last iter was enougth and if not just say it
		{
			int numBlocks = calculateNumBlocks(count_potential_fluid);
			evaluate_and_tag_high_density_from_buffer_kernel<false, true, true, true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr,
				outInt, 4000, count_potential_fluid, outInt2, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
			gpuErrchk(cudaDeviceSynchronize());
		}

		//and compute the avg
		RealCuda avg_density = 0;
		int count = 0;
		{
			for (int j = 0; j < count_potential_fluid; ++j) {
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE ||
					backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_REMOVAL_CANDIDATE) {
					avg_density += backgroundFluidBufferSet->density[j];
					count++;
				}
			}
			avg_density /= count;
		}
		//std::cout << "check avg density avg/count: " << avg_density << "   " << count << std::endl;

		if ((avg_density - data.density0) > 1) {
			std::cout << "Never reached the desired average density current/target: " << avg_density << "  " << data.density0 << std::endl;
		}
	}

	//let's do one last test by regoing through every particle tagged for removal and chcking if there arent some that I can get back
	//currently I save every particle that is essencial already so I don't need to care about this
	//It may even worsen the result
	if(false){
		*outInt = 0;
		{
			//this can only be run by one thread
			save_particles_tagged_for_removal_kernel << <1, 1 >> > (data, backgroundFluidBufferSet->gpu_ptr, outInt, limit_density, count_potential_fluid);
			gpuErrchk(cudaDeviceSynchronize());
		}
		int count_saved = *outInt;
		total_to_remove -= count_saved;

		std::cout << "trying to save particle after completion count_to_rmv/nbr_saved: " << total_to_remove << " / " << count_saved<< std::endl;

	}

	//*/

	//check the actual density values 
	//it may differ from the one seem at the last iteration since the iterations are multithreaded
	if (show_debug) {
		*outInt = 0;
		{
			int numBlocks = calculateNumBlocks(count_potential_fluid);
			evaluate_and_tag_high_density_from_buffer_kernel<true,false,false, true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, 
				outInt, 4000, count_potential_fluid, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
			gpuErrchk(cudaDeviceSynchronize());
		}
		total_to_remove += *outInt;

		{
			RealCuda min_density = 10000;
			RealCuda max_density = 0;
			RealCuda avg_density = 0;
			RealCuda min_density_all = 10000;
			RealCuda max_density_all = 0;
			RealCuda avg_density_all = 0;
			RealCuda min_density_neighbors = 10000;
			RealCuda max_density_neighbors = 0;
			RealCuda avg_density_neighbors = 0;
			RealCuda min_density_neighbors2 = 10000;
			RealCuda max_density_neighbors2 = 0;
			RealCuda avg_density_neighbors2 = 0;
			int count = 0;
			int count_all = 0;
			int count_neighbors = 0;
			int count_neighbors2 = 0;
			for (int j = 0; j < count_potential_fluid; ++j) {
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] != TAG_REMOVAL) {
					avg_density_all += backgroundFluidBufferSet->density[j];
					min_density_all = std::fminf(min_density_all, backgroundFluidBufferSet->density[j]);
					max_density_all = std::fmaxf(max_density_all, backgroundFluidBufferSet->density[j]);
					count_all++;
				}
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE) {
					avg_density += backgroundFluidBufferSet->density[j];
					min_density = std::fminf(min_density, backgroundFluidBufferSet->density[j]);
					max_density = std::fmaxf(max_density, backgroundFluidBufferSet->density[j]);
					count++;
				}
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE_NEIGHBORS) {
					avg_density_neighbors += backgroundFluidBufferSet->density[j];
					min_density_neighbors = std::fminf(min_density_neighbors, backgroundFluidBufferSet->density[j]);
					max_density_neighbors = std::fmaxf(max_density_neighbors, backgroundFluidBufferSet->density[j]);
					count_neighbors++;
				}
				if (backgroundFluidBufferSet->neighborsDataSet->cell_id[j] == TAG_1) {
					avg_density_neighbors2 += backgroundFluidBufferSet->density[j];
					min_density_neighbors2 = std::fminf(min_density_neighbors2, backgroundFluidBufferSet->density[j]);
					max_density_neighbors2 = std::fmaxf(max_density_neighbors2, backgroundFluidBufferSet->density[j]);
					count_neighbors2++;
				}
			}
			avg_density /= count;
			avg_density_all /= count_all;
			avg_density_neighbors /= count_neighbors;
			avg_density_neighbors2 /= count_neighbors2;
			std::cout << "!!!!!!!!!!!!! after end informations !!!!!!!!!!!!!!!!!" << std::endl;
			std::cout << "                  count/avg/min/max density this iter : " << count << " / " << avg_density << " / " <<
				min_density << " / " << max_density << std::endl;
			std::cout << "                  count/avg/min/max         neighbors : " << count_neighbors << " / " << avg_density_neighbors << " / " <<
				min_density_neighbors << " / " << max_density_neighbors << std::endl;
			std::cout << "                  count/avg/min/max        neighbors2 : " << count_neighbors2 << " / " << avg_density_neighbors2 << " / " <<
				min_density_neighbors2 << " / " << max_density_neighbors2 << std::endl;
			std::cout << "                  count/avg/min/max               all : " << count_all << " / " << avg_density_all << " / " <<
				min_density_all << " / " << max_density_all << std::endl;
			std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		}
	}
	
	if (send_result_to_file) {
		static bool first_time = true;
		if (first_time) {
			first_time = false;
			std::ofstream myfile("temp.csv", std::ofstream::trunc);
			if (myfile.is_open())
			{
				myfile << "r2 r3 delta avg min max" << std::endl;
			}
		}

		{
			*outInt = 0;
			{
				int numBlocks = calculateNumBlocks(count_potential_fluid);
				evaluate_and_tag_high_density_from_buffer_kernel<true, false, false, true> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, 
					outInt, 4000, count_potential_fluid, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
				gpuErrchk(cudaDeviceSynchronize());
			}
			total_to_remove += *outInt;

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
				
				std::ofstream myfile("temp.csv", std::ofstream::app);
				if (myfile.is_open())
				{
			
					myfile << params.useRule2 << "  " << params.useRule3 << "  " << params.step_density << "  " << avg_density << "  " <<
						min_density << "  " << max_density<< std::endl;
			
					myfile.close();
				}
			}
		}


	}

	timings.time_next_point();//time 

	//check the tags at the end
	if (show_debug) {
		//for debug purposes check the numbers
		{
			int tag = TAG_ACTIVE;
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			*(SVS_CU::get()->tagged_particles_count) = 0;
			count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
			gpuErrchk(cudaDeviceSynchronize());

			std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
		}
		{
			int tag = TAG_ACTIVE_NEIGHBORS;
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			*(SVS_CU::get()->tagged_particles_count) = 0;
			count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
			gpuErrchk(cudaDeviceSynchronize());

			std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
		}
		{
			int tag = TAG_AIR;
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			*(SVS_CU::get()->tagged_particles_count) = 0;
			count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
			gpuErrchk(cudaDeviceSynchronize());

			std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
		}

		{
			int tag = TAG_UNTAGGED;
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			*(SVS_CU::get()->tagged_particles_count) = 0;
			count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
			gpuErrchk(cudaDeviceSynchronize());

			std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
		}

		{
			int tag = TAG_REMOVAL;
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			*(SVS_CU::get()->tagged_particles_count) = 0;
			count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
			gpuErrchk(cudaDeviceSynchronize());

			std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
		}
		{
			int tag = TAG_REMOVAL_CANDIDATE;
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			*(SVS_CU::get()->tagged_particles_count) = 0;
			count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
			gpuErrchk(cudaDeviceSynchronize());

			std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
		}
	}


	//I need to remove all debug tag
	//for now just wipe all that is not the removal tag and the active tag
	//technically I could save quite some time by saving the tagging of the actve and active neighbors
	for (int i = 0; i < backgroundFluidBufferSet->numParticles; i++) {
		if (backgroundFluidBufferSet->neighborsDataSet->cell_id[i] != TAG_REMOVAL &&
			backgroundFluidBufferSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) {
			backgroundFluidBufferSet->neighborsDataSet->cell_id[i] = TAG_UNTAGGED;
		}
	}

	//sadly I don't think there is a way to do the following tagging here
	//so there is no real reason to do the particle extraction here
	//my main reason is that for the dynamic window the extract is stricly impossible here ...
	///TODO: currently the code doesn't expect this to be done here but for the removal to be done throught the loading function
	///			I'm not even sure using it here would not break, especially if the air particle are considered in the selectio process
	bool extract_particle_to_remove = false;
	bool keep_air_particles = true;
	//ok so here I'll remove the particlesthat have to be removed here, the goal is to be able to set
	//the final tags after that so I don't have to redo them ever
	if (extract_particle_to_remove) {
		if (keep_air_particles) {
			//here it is more complicated since I want to remove the tagged particles without 
			//breaking the order of the particles

			//I know that all air paticles and accepted fluid particles have cell_id< TAG_active which is < numPaticles
			//so the easiest ay to maintain the order is to add to each particle tag it's index
			remove_tagged_particles(backgroundFluidBufferSet, backgroundFluidBufferSet->neighborsDataSet->cell_id,
				backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted, count_high_density_tagged_in_potential, true);

			//update the number of particles that are fluid potential
			count_potential_fluid = count_potential_fluid - count_high_density_tagged_in_potential;

			//set it to 0 to indicate the other system that I have etracted them
			count_high_density_tagged_in_potential = 0;
		}
		else {
			//remove all that is not fluid
			backgroundFluidBufferSet->updateActiveParticleNumber(count_potential_fluid);

			//and now remove the partifcles that were tagged for the fitting
			remove_tagged_particles(data.fluid_data, backgroundFluidBufferSet->neighborsDataSet->cell_id,
				backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted, count_high_density_tagged_in_potential);

			//update the number of particles that are fluid potential
			count_potential_fluid = data.fluid_data->numParticles;


			//set it to 0 to indicate the other system that I have etracted them
			count_high_density_tagged_in_potential = 0;
		}
	}


	_isDataTagged = true;
	// THIS FUNCTION MUST END THERE (or at least after that there should only be debug functions


	timings.time_next_point();//time 
	timings.end_step();//end point of the current step (if measuring avgs you need to call it at every end of the loop)
	timings.recap_timings();//writte timming to cout

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
			evaluate_and_tag_high_density_from_buffer_kernel<false, false, false, false> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, 
				outInt, 4000, backgroundFluidBufferSet->numParticles, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
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
			evaluate_and_tag_high_density_from_buffer_kernel<false, false, false, false> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, 
				outInt, 4000, backgroundFluidBufferSet->numParticles, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
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

		set_buffer_to_value<unsigned int>(backgroundFluidBufferSet->neighborsDataSet->cell_id, TAG_UNTAGGED, backgroundFluidBufferSet->numParticles);
		set_buffer_to_value<RealCuda>(backgroundFluidBufferSet->density, 0, backgroundFluidBufferSet->numParticles);

		{
			int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
			tag_neighborhood_kernel<false, true> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, data.getKernelRadius(), backgroundFluidBufferSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());
		}


		RealCuda limit_density = 1050;
		*outInt = 0;
		{
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			evaluate_and_tag_high_density_from_buffer_kernel<true, false, false, false> << <numBlocks, BLOCKSIZE >> > (data, backgroundFluidBufferSet->gpu_ptr, 
				outInt, limit_density, backgroundFluidBufferSet->numParticles, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
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


int RestFLuidLoader::loadDataToSimulation(SPH::DFSPHCData& data, RestFLuidLoaderInterface::LoadingParameters& params) {
	if (!isInitialized()) {
		std::cout << "RestFLuidLoader::loadDataToSimulation Loading impossible data was not initialized" << std::endl;
		return -1;
	}

	if (!isDataTagged()) {
		std::cout << "!!!!!!!!!!! RestFLuidLoader::loadDataToSimulation you are loading untagged data !!!!!!!!!!!" << std::endl;
		return -1;
	}


	gpuErrchk(read_last_error_cuda("check error before loading: ", params.show_debug));

	std::vector<std::string> timing_names{ "copy","tagging"};
	SPH::SegmentedTiming timings("RestFLuidLoader::loadDataToSimulation", timing_names, true);
	timings.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)

	
	int nbr_fluid_particles;

	if (params.show_debug) {
		std::cout << "count to rmv in fluid/air " <<count_high_density_tagged_in_potential <<"  "<<
			count_high_density_tagged_in_air << std::endl;
	}

	if (params.keep_existing_fluid) {
		if (params.keep_air_particles) {
			throw("RestFLuidLoader::loadDataToSimulation keeping the air particles while keeping existing fluid is not currently allowed");
		}

	

		//just copy all the potential values to the fluid
		int count_existing_fluid_particles = data.fluid_data->numParticles;
		data.fluid_data->updateActiveParticleNumber(count_potential_fluid + count_existing_fluid_particles);

		gpuErrchk(cudaMemcpy(data.fluid_data->mass + count_existing_fluid_particles, backgroundFluidBufferSet->mass,
			count_potential_fluid * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(data.fluid_data->pos + count_existing_fluid_particles, backgroundFluidBufferSet->pos,
			count_potential_fluid * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(data.fluid_data->vel + count_existing_fluid_particles, backgroundFluidBufferSet->vel,
			count_potential_fluid * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(data.fluid_data->color + count_existing_fluid_particles, backgroundFluidBufferSet->color,
			count_potential_fluid * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		//data.fluid_data->resetColor();



		//the main problem is that I have ot extract the particles to rmv
		//however now I don't have a cell index for all existing particles so I have to first build it

		//first clear the index for exisitng particles
		//set_buffer_to_value<unsigned int>(data.fluid_data->neighborsDataSet->cell_id, 0, data.fluid_data->numParticles);
		gpuErrchk(cudaMemset(data.fluid_data->neighborsDataSet->cell_id, TAG_UNTAGGED, 
			data.fluid_data->numParticles * sizeof(unsigned int)));
		
		//now add the tags for the new particles
		gpuErrchk(cudaMemcpy(data.fluid_data->neighborsDataSet->cell_id + count_existing_fluid_particles, 
			backgroundFluidBufferSet->neighborsDataSet->cell_id,
			count_potential_fluid * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

		//and now remove the partifcles that were tagged for the fitting
		remove_tagged_particles(data.fluid_data, data.fluid_data->neighborsDataSet->cell_id,
			data.fluid_data->neighborsDataSet->cell_id_sorted, count_high_density_tagged_in_potential);


		//clearing the warmstart values is necessary
		set_buffer_to_value<RealCuda>(data.fluid_data->kappa, 0, data.fluid_data->numParticles);
		set_buffer_to_value<RealCuda>(data.fluid_data->kappaV, 0, data.fluid_data->numParticles);

		//I'll also clear the velocities for now since I'll load fluid at rest
		//this will have to be removed at some point in the future
		set_buffer_to_value<Vector3d>(data.fluid_data->vel, Vector3d(0, 0, 0), data.fluid_data->numParticles);

		nbr_fluid_particles = data.fluid_data->numParticles;

		timings.time_next_point();//time

		

		
		//ok so when keeping existing fluid i can't simply stock it in the background structure
		if (params.set_up_tagging) {
			if (params.show_debug) {
				std::cout << "RestFLuidLoader::loadDataToSimulation setting up the tagging for stabilization step" << std::endl;
			}

			//I have to allocate dedicated memory since I can't store it into the background arrays because ofthe preexisting particles
			if (tag_array_with_existing_fluid_size < data.fluid_data->numParticles) {
				CUDA_FREE_PTR(tag_array_with_existing_fluid);
			}

			if (tag_array_with_existing_fluid == NULL) {
				cudaMallocManaged(&(tag_array_with_existing_fluid), data.fluid_data->numParticlesMax * sizeof(unsigned int));
				tag_array_with_existing_fluid_size = data.fluid_data->numParticlesMax;
			}

			//so the main difficulty is that we have to also simulate the border with the fluid
			//although since the cell id sorted currently old the previously used tag we can reuse it
			//so first we recover the tagging to recover the active tagging
			gpuErrchk(cudaMemcpy(tag_array_with_existing_fluid,
				data.fluid_data->neighborsDataSet->cell_id_sorted,
				data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

			//first init the neighbor structure
			data.fluid_data->initNeighborsSearchData(data, false);

			//load the backuped tag
			gpuErrchk(cudaMemcpy(data.fluid_data->neighborsDataSet->cell_id,
				tag_array_with_existing_fluid,
				data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));


			
			//then set anything that is not the active untagged to be sure
			for (int i = 0; i < data.fluid_data->numParticles; i++) {
				if (data.fluid_data->neighborsDataSet->cell_id[i] != TAG_ACTIVE) {
					data.fluid_data->neighborsDataSet->cell_id[i] = TAG_UNTAGGED;
				}
			}

			//count hte number of tagged particles and back the tag array (at the same sime to parralel everything (comp and mem transfer)
			if (params.show_debug) {
				int tag = TAG_ACTIVE;
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				data.count_active = *(SVS_CU::get()->tagged_particles_count);
			}

			//do a wide tagging of their neighbors
			//by wide I mean you nee to do the tagging like if they had a slightly extended neighborhood
			//*
			if (true) {
				//data.fluid_data->resetColor();
				RealCuda tagging_distance = data.getKernelRadius() * params.neighbors_tagging_distance_coef;
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				tag_neighbors_of_tagged_kernel<false, false, false> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, tagging_distance,
					TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS);
				gpuErrchk(cudaDeviceSynchronize());
			
			}
			if (params.show_debug) {
				int tag = TAG_ACTIVE_NEIGHBORS;
				int* count_tagged_candidates = (SVS_CU::get()->tagged_particles_count);
				int* count_tagged_other = SVS_CU::get()->count_created_particles;
				*count_tagged_candidates = 0;
				*count_tagged_other = 0;
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, count_tagged_other, nbr_fluid_particles, count_tagged_candidates);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged (candidate/others): " << *count_tagged_candidates << "   " << *count_tagged_other << std::endl;
				data.count_active_neighbors = (*count_tagged_other) + (*count_tagged_candidates);
			}

		

			//sort de data following the tag so that the particles that interest us are stacked at the front
			if (false) {
				//run the sort
				cub::DeviceRadixSort::SortPairs(data.fluid_data->neighborsDataSet->d_temp_storage_pair_sort, data.fluid_data->neighborsDataSet->temp_storage_bytes_pair_sort,
					data.fluid_data->neighborsDataSet->cell_id, data.fluid_data->neighborsDataSet->cell_id_sorted,
					data.fluid_data->neighborsDataSet->p_id, data.fluid_data->neighborsDataSet->p_id_sorted, data.fluid_data->numParticles);
				gpuErrchk(cudaDeviceSynchronize());

				cuda_sortData(*(data.fluid_data), data.fluid_data->neighborsDataSet->p_id_sorted);
				gpuErrchk(cudaDeviceSynchronize());

				//and backup the tag
				//WARNING the reason why I don't store it in cell id is because cell id still have to maintain
				//			the storage of the particles id that must me removed if I ever call that function again 
				gpuErrchk(cudaMemcpy(tag_array_with_existing_fluid, data.fluid_data->neighborsDataSet->cell_id_sorted,
					data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

				if (false) {
					UnifiedParticleSet* tempSet = data.fluid_data;
					tempSet->initAndStoreNeighbors(data, false);
					cuda_divergence_warmstart_init(data);

					std::ofstream myfile("temp7.csv", std::ofstream::trunc);
					if (myfile.is_open())
					{
						for (int i_test = 0; i_test < tempSet->numParticles; ++i_test) {
							myfile << i_test << "   " << tempSet->density[i_test] << "  " <<
								tempSet->neighborsDataSet->cell_id[i_test] << "  " <<
								backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted[i_test] << "  " <<
								(i_test<nbr_fluid_particles) << "  " <<
								(i_test<count_potential_fluid) << "  " << tempSet->getNumberOfNeighbourgs(i_test, 0) << "  " <<
								tempSet->getNumberOfNeighbourgs(i_test, 1) << "  " <<
								tempSet->kappa[i_test] << "  " << tempSet->kappaV[i_test] * 2 << "  " << std::endl;

						}
					}
				}

			}
			else {
				//backup the tag
				gpuErrchk(cudaMemcpy(tag_array_with_existing_fluid, data.fluid_data->neighborsDataSet->cell_id,
					data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
			}

			//do an end count of the number of active since It practivcal to have it
			{
				int tag = TAG_ACTIVE;
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				data.count_active = *(SVS_CU::get()->tagged_particles_count);
			}


			//set a bool to indicate the fullowing system they don't have to recompute the tagging
			_hasFullTaggingSaved = true;
		}
		else {

			//set a bool to indicate the fullowing system they don't have to recompute the tagging
			_hasFullTaggingSaved = false;

			throw("RestFLuidLoader::loadDataToSimulation when keeping existing fluid it is required to pre-compute the tagging currently");
		}
	}
	else {
		if (params.keep_air_particles) {
			if (params.show_debug) {
				std::cout << "keeping air particles" << std::endl;
			}

			//note on that thorw, I don't know if there are modifications that are needed but you better check
			throw("this need to be modified to get the new standart where the tagging is read from the already existing tagging");


			//here it is more complicated since I want to remove the tagged particles without 
			//breaking the order of the particles

			//just copy all the values to the fluid (including air)
			data.fluid_data->updateActiveParticleNumber(backgroundFluidBufferSet->numParticles);

			gpuErrchk(cudaMemcpy(data.fluid_data->mass, backgroundFluidBufferSet->mass, backgroundFluidBufferSet->numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(data.fluid_data->pos, backgroundFluidBufferSet->pos, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(data.fluid_data->vel, backgroundFluidBufferSet->vel, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(data.fluid_data->color, backgroundFluidBufferSet->color, backgroundFluidBufferSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
			//data.fluid_data->resetColor();


			//I know that all air paticles and accepted fluid particles have cell_id< TAG_active which is < numPaticles
			//so the easiest ay to maintain the order is to add to each particle tag it's index
			gpuErrchk(cudaMemcpy(backgroundFluidBufferSet->neighborsDataSet->local_id,
				backgroundFluidBufferSet->neighborsDataSet->cell_id,
				backgroundFluidBufferSet->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

			//a test 
			gpuErrchk(cudaMemcpy(data.fluid_data->kappa, backgroundFluidBufferSet->density,
				backgroundFluidBufferSet->numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));


			remove_tagged_particles(data.fluid_data, backgroundFluidBufferSet->neighborsDataSet->local_id,
				backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted,
				count_high_density_tagged_in_potential + count_high_density_tagged_in_air, true);

			nbr_fluid_particles = count_potential_fluid - count_high_density_tagged_in_potential;

			gpuErrchk(cudaMemcpy(data.fluid_data->density, data.fluid_data->kappa,
				data.fluid_data->numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

			//a check to see if there is still some particles that should have been removed
			//if everything is working properly this test should be outputting files where you can get the same density for the corresponfding particles
			if (false) {
				//after the copy
				if (true) {
					UnifiedParticleSet* tempSet = data.fluid_data;

					tempSet->initAndStoreNeighbors(data, false);
					if (true) {
						set_buffer_to_value<RealCuda>(tempSet->density, 0, tempSet->numParticles);
						{
							int numBlocks = calculateNumBlocks(tempSet->numParticles);
							evaluate_and_tag_high_density_from_buffer_kernel<false, true, true, true> << <numBlocks, BLOCKSIZE >> > (data, tempSet->gpu_ptr,
								outInt, 4000, tempSet->numParticles, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
							gpuErrchk(cudaDeviceSynchronize());
						}
					}

					gpuErrchk(cudaMemcpy(tempSet->kappaV, tempSet->density,
						tempSet->numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));

					if (true) {
						cuda_divergence_warmstart_init(data);
					}

					//*
					if (params.show_debug) {
						std::cout << "!!!!!!!!!!!!! after with air           !!!!!!!!!!!!!!!" << std::endl;
						show_extensive_density_information(tempSet, tempSet->numParticles);
						std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
					}
					//*/



					std::ofstream myfile("temp.csv", std::ofstream::trunc);
					if (myfile.is_open())
					{
						for (int i_test = 0; i_test < tempSet->numParticles; ++i_test) {
							myfile << i_test << "   " << tempSet->density[i_test] << "  " <<
								tempSet->neighborsDataSet->cell_id[i_test] << "  " <<
								backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted[i_test] << "  " <<
								(i_test<nbr_fluid_particles) << "  " <<
								(i_test<count_potential_fluid) << "  " << tempSet->getNumberOfNeighbourgs(i_test, 0) << "  " <<
								tempSet->getNumberOfNeighbourgs(i_test, 1) << "  " <<
								tempSet->kappa[i_test] << "  " << tempSet->kappaV[i_test] * 2 << "  " << std::endl;

						}
					}
				}

				//before the copy
				if (true) {
					UnifiedParticleSet* tempSet = backgroundFluidBufferSet;

					if (true) {
						set_buffer_to_value<RealCuda>(tempSet->density, 0, tempSet->numParticles);
						{
							int numBlocks = calculateNumBlocks(tempSet->numParticles);
							evaluate_and_tag_high_density_from_buffer_kernel<false, true, true, true> << <numBlocks, BLOCKSIZE >> > (data, tempSet->gpu_ptr,
								outInt, 4000, tempSet->numParticles, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
							gpuErrchk(cudaDeviceSynchronize());
						}
					}

					//*
					std::cout << "!!!!!!!!!!!!! after with air           !!!!!!!!!!!!!!!" << std::endl;
					show_extensive_density_information(tempSet, tempSet->numParticles);
					std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
					//*/



					std::ofstream myfile("temp3.csv", std::ofstream::trunc);
					if (myfile.is_open())
					{
						for (int i_test = 0; i_test < tempSet->numParticles; ++i_test) {
							myfile << i_test << "   " << tempSet->density[i_test] << "  " <<
								tempSet->neighborsDataSet->cell_id[i_test] << "  " <<
								tempSet->neighborsDataSet->cell_id_sorted[i_test] << "  " <<
								(i_test<nbr_fluid_particles) << "  " <<
								(i_test<count_potential_fluid) << "  " << tempSet->getNumberOfNeighbourgs(i_test, 0) << "  " <<
								tempSet->getNumberOfNeighbourgs(i_test, 1) << "  " << std::endl;

						}
					}
				}
			}

		}
		else {

			if (params.show_debug) {
				//for debug purposes check the numbers
				{
					int tag = TAG_ACTIVE;
					int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
				{
					int tag = TAG_ACTIVE_NEIGHBORS;
					int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
				{
					int tag = TAG_AIR;
					int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}

				{
					int tag = TAG_UNTAGGED;
					int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}

				{
					int tag = TAG_REMOVAL;
					int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
				{
					int tag = TAG_REMOVAL_CANDIDATE;
					int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
			}



			//just copy all the potential values to the fluid
			data.fluid_data->updateActiveParticleNumber(count_potential_fluid);

			gpuErrchk(cudaMemcpy(data.fluid_data->mass, backgroundFluidBufferSet->mass, count_potential_fluid * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(data.fluid_data->pos, backgroundFluidBufferSet->pos, count_potential_fluid * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(data.fluid_data->vel, backgroundFluidBufferSet->vel, count_potential_fluid * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaMemcpy(data.fluid_data->color, backgroundFluidBufferSet->color, count_potential_fluid * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
			//data.fluid_data->resetColor();

			//we can remove all that is not fluid

			//and now remove the partifcles that were tagged for the fitting
			remove_tagged_particles(data.fluid_data, backgroundFluidBufferSet->neighborsDataSet->cell_id,
				backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted, count_high_density_tagged_in_potential,false,true);


			if (false){
				static bool first_time = true;
				//if (first_time) 
				{
					std::ofstream myfile("temp25.csv", std::ofstream::trunc);
					if (myfile.is_open())
					{
						
						for (int i = 0; i < count_potential_fluid- count_high_density_tagged_in_potential; ++i) {
							int new_id = data.fluid_data->neighborsDataSet->p_id_sorted[i];
							int old_pos_tag = backgroundFluidBufferSet->neighborsDataSet->cell_id[i];
							int new_pos_tag = backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted[new_id];
							
							myfile << i << "   " << new_id << "  " << old_pos_tag<<
								"   "<<new_pos_tag<<std::endl;


								
						}
					}

					myfile.close();
					
					first_time = false;
				}
			}



			//clearing the warmstart values is necessary
			set_buffer_to_value<RealCuda>(data.fluid_data->kappa, 0, data.fluid_data->numParticles);
			set_buffer_to_value<RealCuda>(data.fluid_data->kappaV, 0, data.fluid_data->numParticles);

			//I'll also clear the velocities for now since I'll load fluid at rest
			//this will have to be removed at some point in the future
			set_buffer_to_value<Vector3d>(data.fluid_data->vel, Vector3d(0, 0, 0), data.fluid_data->numParticles);

			nbr_fluid_particles = data.fluid_data->numParticles;
		}


		timings.time_next_point();//time

		if (params.set_up_tagging) {
			if (params.show_debug) {
				std::cout << "RestFLuidLoader::loadDataToSimulation setting up the tagging for stabilization step" << std::endl;
			}
			if (params.keep_air_particles) {
				//note on that thorw, I don't know if there are modifications that are needed but you better check
				throw("this need to be modified to get the new standart where the tagging is read from the already existing tagging");

			}

			
			//then we can init the neighbor structure
			data.fluid_data->initNeighborsSearchData(data, false);

			//and reload the tagging
			gpuErrchk(cudaMemcpy(data.fluid_data->neighborsDataSet->cell_id, backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted,
				data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

			

			//then set anything that is not the active untagged to be sure
			for (int i = 0; i < data.fluid_data->numParticles; i++) {
				if (data.fluid_data->neighborsDataSet->cell_id[i] != TAG_ACTIVE) {
					data.fluid_data->neighborsDataSet->cell_id[i] = TAG_UNTAGGED;
				}
			}

			//set an additional order of neighbors as active 
			//an order of neighbors is the neighbors of neighbors
			///TODO verify that this works properly...
			///thoughI'm not using it currently normaly
			int additional_neighbors_order_tagging = 0;
			for (int i = 0; i < (additional_neighbors_order_tagging); ++i) {
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);

				//tag the first order neighbors
				if (i == 0) {
					tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS);
					gpuErrchk(cudaDeviceSynchronize());
				}

				//then the second order
				tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE_NEIGHBORS, TAG_1);
				gpuErrchk(cudaDeviceSynchronize());


				//then cnvert the tags
				convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_ACTIVE_NEIGHBORS, TAG_ACTIVE);
				gpuErrchk(cudaDeviceSynchronize());
				if (i < (additional_neighbors_order_tagging - 1)) {
					convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_1, TAG_ACTIVE_NEIGHBORS);
				}
				else {
					convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_1, TAG_UNTAGGED);
				}
				gpuErrchk(cudaDeviceSynchronize());
			}

			//count hte number of tagged particles and back the tag array (at the same sime to parralel everything (comp and mem transfer)
			if (params.show_debug) {
				int tag = TAG_ACTIVE;
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				data.count_active = *(SVS_CU::get()->tagged_particles_count);
			}



			//do a wide tagging of their neighbors
			//by wide I mean you nee to do the tagging like if they had a slightly extended neighborhood
			//*
			if (true) {
				RealCuda tagging_distance = data.getKernelRadius() * params.neighbors_tagging_distance_coef;
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				tag_neighbors_of_tagged_kernel<false, false, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, tagging_distance,
					TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS, nbr_fluid_particles, TAG_AIR_ACTIVE_NEIGHBORS);
				gpuErrchk(cudaDeviceSynchronize());
			}
			if (params.show_debug) {
				int tag = TAG_ACTIVE_NEIGHBORS;
				int* count_tagged_candidates = (SVS_CU::get()->tagged_particles_count);
				int* count_tagged_other = SVS_CU::get()->count_created_particles;
				*count_tagged_candidates = 0;
				*count_tagged_other = 0;
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, count_tagged_other, nbr_fluid_particles, count_tagged_candidates);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged (candidate/others): " << *count_tagged_candidates << "   " << *count_tagged_other << std::endl;
				data.count_active_neighbors = (*count_tagged_other) + (*count_tagged_candidates);
			}
			//*/
			int count_neighbors_order_2 = 0;
			if (false) {
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE_NEIGHBORS, TAG_ACTIVE_NEIGHBORS_ORDER_2);
				gpuErrchk(cudaDeviceSynchronize());
			}
			if (params.show_debug) {
				int tag = TAG_ACTIVE_NEIGHBORS_ORDER_2;
				int* count_tagged_candidates = (SVS_CU::get()->tagged_particles_count);
				int* count_tagged_other = SVS_CU::get()->count_created_particles;
				*count_tagged_candidates = 0;
				*count_tagged_other = 0;
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, count_tagged_other, nbr_fluid_particles, count_tagged_candidates);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged (candidate/others): " << *count_tagged_candidates << "   " << *count_tagged_other << std::endl;
				count_neighbors_order_2 = (*count_tagged_other) + (*count_tagged_candidates);
			}

			//a print to disk to do some studies before sorting
			if (false) {
				gpuErrchk(cudaMemcpy(backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted, data.fluid_data->neighborsDataSet->cell_id,
					data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

				UnifiedParticleSet* tempSet = data.fluid_data;
				tempSet->initAndStoreNeighbors(data, false);
				cuda_divergence_warmstart_init(data);

				std::ofstream myfile("temp6.csv", std::ofstream::trunc);
				if (myfile.is_open())
				{
					for (int i_test = 0; i_test < tempSet->numParticles; ++i_test) {
						myfile << i_test << "   " << tempSet->density[i_test] << "  " <<
							tempSet->neighborsDataSet->cell_id[i_test] << "  " <<
							backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted[i_test] << "  " <<
							(i_test<nbr_fluid_particles) << "  " <<
							(i_test<count_potential_fluid) << "  " << tempSet->getNumberOfNeighbourgs(i_test, 0) << "  " <<
							tempSet->getNumberOfNeighbourgs(i_test, 1) << "  " <<
							tempSet->kappa[i_test] << "  " << tempSet->kappaV[i_test] * 2 << "  " << std::endl;

					}
				}

				gpuErrchk(cudaMemcpy(data.fluid_data->neighborsDataSet->cell_id, backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted,
					data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
			}


			//sort de data following the tag so that the particles that interest us are stacked at the front
			if (true) {
				//run the sort
				cub::DeviceRadixSort::SortPairs(data.fluid_data->neighborsDataSet->d_temp_storage_pair_sort, data.fluid_data->neighborsDataSet->temp_storage_bytes_pair_sort,
					data.fluid_data->neighborsDataSet->cell_id, data.fluid_data->neighborsDataSet->cell_id_sorted,
					data.fluid_data->neighborsDataSet->p_id, data.fluid_data->neighborsDataSet->p_id_sorted, data.fluid_data->numParticles);
				gpuErrchk(cudaDeviceSynchronize());

				cuda_sortData(*(data.fluid_data), data.fluid_data->neighborsDataSet->p_id_sorted);
				gpuErrchk(cudaDeviceSynchronize());

				//and backup the tag
				//WARNING the reason why I don't store it in cell id is because cell id still have to maintain
				//			the storage of the particles id that must me removed if I ever call that function again 
				gpuErrchk(cudaMemcpy(backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted, data.fluid_data->neighborsDataSet->cell_id_sorted,
					data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

				if (false) {
					UnifiedParticleSet* tempSet = data.fluid_data;
					tempSet->initAndStoreNeighbors(data, false);
					cuda_divergence_warmstart_init(data);

					std::ofstream myfile("temp7.csv", std::ofstream::trunc);
					if (myfile.is_open())
					{
						for (int i_test = 0; i_test < tempSet->numParticles; ++i_test) {
							myfile << i_test << "   " << tempSet->density[i_test] << "  " <<
								tempSet->neighborsDataSet->cell_id[i_test] << "  " <<
								backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted[i_test] << "  " <<
								(i_test<nbr_fluid_particles) << "  " <<
								(i_test<count_potential_fluid) << "  " << tempSet->getNumberOfNeighbourgs(i_test, 0) << "  " <<
								tempSet->getNumberOfNeighbourgs(i_test, 1) << "  " <<
								tempSet->kappa[i_test] << "  " << tempSet->kappaV[i_test] * 2 << "  " << std::endl;

						}
					}
				}

			}
			else {
				//and backup the tag
				gpuErrchk(cudaMemcpy(backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted, data.fluid_data->neighborsDataSet->cell_id,
					data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
			}

			//if I have kept the air particles then I have to tag the air particles that are neighbors so that they 
			//have their density computed
			if (params.keep_air_particles)
			{
				//since I have set a special tag for the air particles that are neighbors of active, I can simply trasnform the tag
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted,
					data.fluid_data->numParticles, TAG_AIR_ACTIVE_NEIGHBORS, TAG_ACTIVE_NEIGHBORS);
				gpuErrchk(cudaDeviceSynchronize());
			}

			//the goal here is to rmv the block of particles that are not used for the stabilization process
			if (false) {
				//let's do a retarded test
				//I'll actually remove any particle that is not acting in the stabilization process and see if it make any difference
				//this does not give better results
				data.true_particle_count = data.fluid_data->numParticles;
				data.fluid_data->updateActiveParticleNumber(data.count_active + data.count_active_neighbors + count_neighbors_order_2);
			}

			//do an end count of the number of active since It practivcal to have it
			{
				int tag = TAG_ACTIVE;
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				if (params.show_debug) {
					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
				data.count_active = *(SVS_CU::get()->tagged_particles_count);
			}



			//set a bool to indicate the fullowing system they don't have to recompute the tagging
			_hasFullTaggingSaved = true;
		}
		else {
			_hasFullTaggingSaved = false;
		}
	}
	



	timings.time_next_point();//time p3
	timings.end_step();//end point of the current step (if measuring avgs you need to call it at every end of the loop)
	timings.recap_timings();//writte timming to cout

	return nbr_fluid_particles;
}



struct ParticlePackingDebug
{
	int size;

	Vector3d* a_b_f;
	Vector3d* a_b_b;
	Vector3d* a_rf_f;
	Vector3d* a_rf_b;
	Vector3d* a_d;
	RealCuda* gamma_f;
	RealCuda* gamma_b;

	RealCuda a_i_avg;
	RealCuda a_b_avg;
	RealCuda a_b_f_avg;
	RealCuda a_b_b_avg;
	RealCuda a_rf_avg;
	RealCuda a_rf_f_avg;
	RealCuda a_rf_b_avg;
	RealCuda a_d_avg;

	RealCuda a_i_max;
	RealCuda a_b_max;
	RealCuda a_b_f_max;
	RealCuda a_b_b_max;
	RealCuda a_rf_max;
	RealCuda a_rf_f_max;
	RealCuda a_rf_b_max;
	RealCuda a_d_max;

	RealCuda gamma_min;
	RealCuda gamma_max;
	RealCuda gamma_avg;

	void alloc(int size_i) {
		size = size_i;

		cudaMallocManaged(&(a_b_f), size * sizeof(Vector3d));
		cudaMallocManaged(&(a_b_b), size * sizeof(Vector3d));
		cudaMallocManaged(&(a_rf_f), size * sizeof(Vector3d));
		cudaMallocManaged(&(a_rf_b), size * sizeof(Vector3d));
		cudaMallocManaged(&(a_d), size * sizeof(Vector3d));
		cudaMallocManaged(&(gamma_f), size * sizeof(RealCuda));
		cudaMallocManaged(&(gamma_b), size * sizeof(RealCuda));
		reset();
	}

	void reset() {
		set_buffer_to_value<Vector3d>(a_b_f, Vector3d(0, 0, 0), size);
		set_buffer_to_value<Vector3d>(a_b_b, Vector3d(0, 0, 0), size);
		set_buffer_to_value<Vector3d>(a_rf_f, Vector3d(0, 0, 0), size);
		set_buffer_to_value<Vector3d>(a_rf_b, Vector3d(0, 0, 0), size);
		set_buffer_to_value<Vector3d>(a_d, Vector3d(0, 0, 0), size);
		set_buffer_to_value<RealCuda>(gamma_f, 0, size);
		set_buffer_to_value<RealCuda>(gamma_b, 0, size);
	}



	void readAvgAndMax(unsigned int* tagArray) {

		a_i_avg = 0;
		a_b_avg = 0;
		a_b_f_avg = 0;
		a_b_b_avg = 0;
		a_rf_avg = 0;
		a_rf_f_avg = 0;
		a_rf_b_avg = 0;
		a_d_avg = 0;

		a_i_max = 0;
		a_b_max = 0;
		a_b_f_max = 0;
		a_b_b_max = 0;
		a_rf_max = 0;
		a_rf_f_max = 0;
		a_rf_b_max = 0;
		a_d_max = 0;

		gamma_avg = 0;
		gamma_min = 10000;
		gamma_max = 0;

		int count = 0;
		for (int i = 0; i < size; ++i) {
			if (tagArray[i] == TAG_ACTIVE) {
				Vector3d a_b = a_b_f[i] + a_b_b[i];
				Vector3d a_rf = a_rf_f[i] + a_rf_b[i];
				Vector3d a_i = a_b + a_rf;

				a_i_avg += a_i.norm();
				a_b_avg += a_b.norm();
				a_b_f_avg += a_b_f[i].norm();
				a_b_b_avg += a_b_b[i].norm();
				a_rf_avg += a_rf.norm();
				a_rf_f_avg += a_rf_f[i].norm();
				a_rf_b_avg += a_rf_b[i].norm();
				a_d_avg += a_d[i].norm();


				a_i_max = std::fmaxf(a_i_max, a_i.norm());
				a_b_max = std::fmaxf(a_b_max, a_b.norm());
				a_b_f_max = std::fmaxf(a_b_f_max, a_b_f[i].norm());
				a_b_b_max = std::fmaxf(a_b_b_max, a_b_b[i].norm());
				a_rf_max = std::fmaxf(a_rf_max, a_rf.norm());
				a_rf_f_max = std::fmaxf(a_rf_f_max, a_rf_f[i].norm());
				a_rf_b_max = std::fmaxf(a_rf_b_max, a_rf_b[i].norm());
				a_d_max = std::fmaxf(a_d_max, a_d[i].norm());

				RealCuda gamma_i= gamma_f[i] + gamma_b[i];
				gamma_avg += gamma_i;
				gamma_min = std::fminf(gamma_min, gamma_i);
				gamma_max = std::fmaxf(gamma_max, gamma_i);

				count++;
			}
		}
		a_i_avg /= count;
		a_b_avg /= count;
		a_b_f_avg /= count;
		a_b_b_avg /= count;
		a_rf_avg /= count;
		a_rf_f_avg /= count;
		a_rf_b_avg /= count; 
		a_d_avg /= count;
		gamma_avg /= count;
	}

	std::string avgAndMaxToString(bool show_ai = true, bool show_abf = true, bool show_ab = true, bool show_abb = true,
		bool show_arf = true, bool show_arff = true, bool show_arfb = true, bool show_ad = false, bool show_gamma = true) {
		std::ostringstream oss;

		if (show_ai) {
			oss << "a_i avg/max:      " << a_i_avg << "   " << a_i_max << std::endl;
		}
		if (show_ab) {
			oss << "a_b avg/max:      " << a_b_avg << "   " << a_b_max << std::endl;
		}
		if (show_abf) {
			oss << "a_b_f avg/max:      " << a_b_f_avg << "   " << a_b_f_max << std::endl;
		}
		if (show_abb) {
			oss << "a_b_b avg/max:      " << a_b_b_avg << "   " << a_b_b_max << std::endl;
		}
		if (show_arf) {
			oss << "a_rf avg/max:     " << a_rf_avg << "   " << a_rf_max << std::endl;
		}
		if (show_arff) {
			oss << "a_rf_f avg/max:     " << a_rf_f_avg << "   " << a_rf_f_max << std::endl;
		}
		if (show_arfb) {
			oss << "a_rf_b avg/max:     "<< a_rf_b_avg <<"   "<< a_rf_b_max <<std::endl;
		}
		if (show_ad) {
			oss << "a_d avg/max:        " << a_d_avg << "   " << a_d_max << std::endl;
		}
		if (show_gamma) {
			oss << "gamma avg/min/max:  " << gamma_avg << "   " <<gamma_min<< "   " <<gamma_max << std::endl;
		}

		return oss.str();
	}

	std::string particleInfoToString(int id, bool show_abf = true, bool show_ab = true, bool show_abb = true, 
		bool show_arf = true, bool show_arff = true, bool show_arfb = true, bool show_ad = true, bool show_gamma = true) {
		std::ostringstream oss;


		std::cout << "particle info: " << id << std::endl;
		if (show_ab) {
			oss << "	a_b :      " << (a_b_f[id]+ a_b_b[id]).toString() << std::endl;
		}
		if (show_abf) {
			oss << "	a_b_f :      " << a_b_f[id].toString() << std::endl;
		}
		if (show_abb) {
			oss << "	a_b_b :      " << a_b_b[id].toString() << std::endl;
		}
		if (show_arff) {
			oss << "	a_rf :     " << (a_rf_f[id]+ a_rf_b[id]).toString() << std::endl;
		}
		if (show_arff) {
			oss << "	a_rf_f :     " << a_rf_f[id].toString() << std::endl;
		}
		if (show_arfb) {
			oss << "	a_rf_b :     " << a_rf_b[id].toString() << std::endl;
		}
		if (show_ad) {
			oss << "	a_d :        " << a_d[id].toString() << std::endl;
		}
		if (show_gamma) {
			oss << "	gamma :      " << gamma_f[id]+ gamma_b[id] << std::endl;
		}

		return oss.str();
	}

};

__global__ void compute_gamma_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int size, RealCuda* gamma_o) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size) { return; }


	Vector3d p_i = bufferSet->pos[i];
	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);

	RealCuda gamma = data.W_zero * bufferSet->mass[i] / bufferSet->density[i];

	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
		if (i != j) {
			Vector3d x_ij = p_i - bufferSet->pos[j];
			gamma += KERNEL_W(data, x_ij) * bufferSet->mass[j] / bufferSet->density[j];			
		}
	);

	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		Vector3d x_ij = p_i - data.boundaries_data_cuda->pos[j];
		gamma += KERNEL_W(data, x_ij) * data.boundaries_data_cuda->mass[j] / data.density0;
	);

	gamma_o[i] = gamma;
}

//related paper: An improved particle packing algorithm for complexgeometries
template<bool compute_active_only>
__global__ void particle_packing_negi_2019_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int count_potential_fluid, RealCuda delta_s, RealCuda p_b,
	RealCuda k_r, RealCuda zeta, RealCuda coef_to_compare_v_sq_to, RealCuda c, RealCuda r_limit, RealCuda a_rf_r_limit, RealCuda* max_v_norm_sq,
	ParticlePackingDebug ppd) {
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


	Vector3d a_old = bufferSet->acc[i];

	Vector3d p_i = bufferSet->pos[i];


	RealCuda coef_ab = -p_b;
	Vector3d v_i = bufferSet->vel[i];
	//a_d
	Vector3d a_d = -(1 - zeta) * v_i;
	Vector3d a_b(0);
	Vector3d a_rf(0);

	Vector3d a_b_t(0);
	Vector3d a_rf_t(0);

	Vector3d a_i(0);

	//currently a_d is directly incorporated in the velocity
	//a_i+=a_d;


	//test for a force on the density gradiant itself
	Vector3d a_den(0);

	ppd.a_d[i] = a_d;

	RealCuda gamma = data.W_zero * bufferSet->mass[i] / bufferSet->density[i];


	RealCuda density_estimation = 0;
	Vector3d density_gradiant_estimation(0);
	Vector3d density_gradiant_estimationv2(0);
	RealCuda sum_W = 0;

	//density_estimation += data.W_zero * bufferSet->density[i];
	//sum_W += data.W_zero;

	int count = 0;
	{
		//we will compute the 3 components at once
		ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);

		//itrate on the fluid particles, also only take into account the particle that are not taged for removal
		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
			if (bufferSet->neighborsDataSet->cell_id[j] != TAG_REMOVAL) {
				if (i != j) {
					Vector3d x_ij = p_i - bufferSet->pos[j];

					//a_b
					a_b += KERNEL_GRAD_W(data, x_ij) * bufferSet->mass[j] / bufferSet->density[j];

					//more debug
					a_den += KERNEL_GRAD_W(data, x_ij) * bufferSet->mass[j];
					//gamma 
					gamma += KERNEL_W(data, x_ij) * bufferSet->mass[j] / bufferSet->density[j];

					//even more debug lol
					density_estimation += KERNEL_W(data, x_ij) * bufferSet->density[j];
					density_gradiant_estimation += KERNEL_GRAD_W(data, x_ij) * bufferSet->density[j];
					density_gradiant_estimationv2 += KERNEL_W(data, x_ij) * bufferSet->density[j] * x_ij.unit();
					sum_W += KERNEL_W(data, x_ij);

					RealCuda r = x_ij.norm();
					x_ij /= r;

					//a_rf
					//note at the point x_ij is normalized so it correspond to n_ij in the formula
					//*
					if (r < delta_s) {
						if (r > r_limit) {
							a_rf += x_ij * k_r * ((3 * c * c) / (r * r * r * r) - (2 * c) / (r * r * r));
						}
						else {
							a_rf += x_ij * a_rf_r_limit;
						}
					}
					//*/

					count++;
				}
			}
		);
		a_b *= coef_ab;
		bufferSet->setNumberOfNeighbourgs(count, i, 0);
		ppd.gamma_f[i] = gamma;

		ppd.a_b_f[i] = a_b;
		ppd.a_rf_f[i] = a_rf;

		a_b_t += a_b;
		a_rf_t += a_rf;
		a_b = Vector3d(0);
		a_rf = Vector3d(0);

		density_estimation /= sum_W;
		density_gradiant_estimation *= coef_ab;
		//printf("fluid only den estimatiion (den/grad) %i   %f  //  %f %f %f\n", i, density_estimation, density_gradiant_estimation.x , density_gradiant_estimation.y , density_gradiant_estimation.z);
		density_gradiant_estimation = coef_ab;
		density_estimation *= sum_W;
		//and ont he boundarie
		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
			Vector3d x_ij = p_i - data.boundaries_data_cuda->pos[j];


		//a_b
		a_b += KERNEL_GRAD_W(data, x_ij) * data.boundaries_data_cuda->mass[j] / data.density0;


		a_den += KERNEL_GRAD_W(data, x_ij) * bufferSet->mass[j];

		//gamma for debug
		gamma += KERNEL_W(data, x_ij) * data.boundaries_data_cuda->mass[j] / data.density0;


		density_estimation += KERNEL_W(data, x_ij) * data.density0;
		density_gradiant_estimation += KERNEL_GRAD_W(data, x_ij) * data.density0;
		density_gradiant_estimationv2 += KERNEL_W(data, x_ij) * data.density0 * x_ij.unit();
		sum_W += KERNEL_W(data, x_ij);

		RealCuda r = x_ij.norm();
		x_ij /= r;

		//a_rf
		//note at the point x_ij is normalized so it correspond to n_ij in the formula
		//*
		if (r < delta_s) {
			if (r > r_limit) {
				a_rf += x_ij * k_r * ((3 * c * c) / (r * r * r * r) - (2 * c) / (r * r * r));
			}
			else {
				a_rf += x_ij * a_rf_r_limit;
			}
		}
		count++;
		//*/
		);
	}	
	a_b *= coef_ab;
	bufferSet->setNumberOfNeighbourgs(count, i, 1);

	//a_den *= coef_ab;

	ppd.gamma_b[i] = gamma- ppd.gamma_f[i];


	density_gradiant_estimation *= coef_ab;
	density_gradiant_estimationv2 *= coef_ab;
	density_estimation /= sum_W;
	//printf("Complete   den estimatiion (den/grad) %i   %f  //  %f %f %f\n", i, density_estimation, density_gradiant_estimation.x, density_gradiant_estimation.y, density_gradiant_estimation.z);
	//printf("Complete den estimatiionv2 (den/grad) %i   %f  //  %f %f %f\n", i, density_estimation, density_gradiant_estimationv2.x, density_gradiant_estimationv2.y, density_gradiant_estimationv2.z);

	/*
	if (gamma > 1.01) {
		RealCuda g = MAX_MACRO_CUDA(1.2 - gamma, 0) / (1.2-1.01);
		RealCuda r = 1-g;

		bufferSet->color[i] = Vector3d(r, g, 0);
	}
	//*/
	/*
	Vector3d p_t = p_i;
	p_t.y = 0;

	if (p_t.norm()<0.01) {
		bufferSet->color[i] = Vector3d(1, 0, 0);
		printf("%i   %f\n", i, p_i.y);
	}//*/
	/*
	if (i==5573) {
		bufferSet->color[i] = Vector3d(1, 0, 0);
	}
	//*/




	ppd.a_b_b[i] = a_b;
	ppd.a_rf_b[i] = a_rf;
	a_b_t += a_b;

	//apply gamma on all a_b forces so that only the particles that are currently unstable are moved
	if (false) {
		RealCuda gamma_ponderation = abs(1 - gamma);
		ppd.a_b_f[i] *= gamma_ponderation;
		ppd.a_b_b[i] *= gamma_ponderation;
		a_b_t *= gamma_ponderation;
	}

	//a_den *= abs(data.density0 - bufferSet->density[i]);
	//ppd.a_rf_f[i] = a_den;
	/*
	if (a_b_t.norm()>2300) {
		bufferSet->color[i] = Vector3d(1, 0, 0);
		printf("%i ?? %f ;;  %f  (%f %f %f)\n", i, gamma ,a_b_t.norm() , a_b_t.x, a_b_t.y, a_b_t.z);
	}
	//*/
	//printf("%f %f %f  vs %f (%f)\n", a_b_t.x, a_b_t.y, a_b_t.z, 0.2 * ABS_MACRO_CUDA(coef_ab), coef_ab);

	//so here will be some work to hopefully have smth stable
	//a_b_t.toEpsilonAbsToZero(0.4 * ABS_MACRO_CUDA(coef_ab));


	a_rf_t += a_rf;
	
	a_i += a_b_t +a_rf_t;

	{
		a_i = density_gradiant_estimationv2;
		ppd.a_b_b[i] = density_gradiant_estimationv2;
		ppd.a_b_f[i] = Vector3d(0);
	}

	if (a_i.norm() > 2500) {
		RealCuda factor = 2500 / a_i.norm();
		a_i *= factor;
		ppd.a_b_f[i] *= factor;
		ppd.a_rf_f[i] *= factor;
		ppd.a_b_b[i] *= factor;
		ppd.a_rf_b[i] *= factor;
	}
	//a_i = a_den;
	//a_i = density_gradiant_estimationv2;

	//for now lets not correct any under sampling
	if (false&&gamma < 1.0) {
		a_i = Vector3d(0);
		ppd.a_b_f[i] = Vector3d(0);
		ppd.a_rf_f[i] = Vector3d(0);
		ppd.a_b_b[i] = Vector3d(0);
		ppd.a_rf_b[i] = Vector3d(0);
		ppd.a_d[i] = -1.0f*bufferSet->vel[i];

		//and we block any exiting motion
		bufferSet->vel[i] = Vector3d(0);
	}
	//we cannot drectly modify v_i here since we do not have delta_t
	bufferSet->acc[i] = a_i;

	//do it at the end with some luck the thread ill be desynched enougth for the atomic to not matter
	RealCuda v_i_sq_norm = v_i.squaredNorm();
	if (v_i_sq_norm > coef_to_compare_v_sq_to) {
		atomicToMax(max_v_norm_sq, v_i_sq_norm);
	}

	//bufferSet->color[i] = Vector3d(0, 1, 0);

	//apply the gamma depending damping here
	//this only works in a perfect world where the rest state of every particles is the 1.0
	//however that is not necessarily the case
	if (false) {
		RealCuda limit_gamma_damping_sup = 0.01;
		RealCuda coef_gamma_damping=MIN_MACRO_CUDA(ABS_MACRO_CUDA(1.0 - gamma)/ limit_gamma_damping_sup, 1)  ;
		//printf("damping coef: %f: ", coef_gamma_damping);
		bufferSet->vel[i] *= coef_gamma_damping;
	}
	//this damping is a simple damping that nuke the existing velocity if the new force I have to apply is the oposite of the existing one
	//I'll note that this is garbage but well it will most likely work pretty fine
	if(true){
		RealCuda damping_factor = 0.5;
		Vector3d v = bufferSet->vel[i];
		if ((v.x * a_i.x) < 0) {
			v.x *= damping_factor;
		}
		if ((v.y * a_i.y) < 0) {
			v.y *= damping_factor;
		}
		if ((v.z * a_i.z) < 0) {
			v.z *= damping_factor;
		}
		bufferSet->vel[i]=v;
	}

	{
		bufferSet->vel[i] *= 0.75;
	}
	//and here I'll use a "smart damping scheme" essencially it will use the past acceleration to know if it should slow before reaching the actual 
	//point at which it is too late
	//esencially it is a classical damping depending on the derivative
	//the main problem currently is that it is not consistant in space since the damping coefficient is not the same on the 3 axis
	if(false){
		RealCuda damping_limit = 1;
		RealCuda ratio_a = a_i.x / a_old.x;
		if (ratio_a > 0) {
			//now depending on how much the force has been lowered  I'll damp the past velocity
			//the logic behind it is that if we had a large drop in acceleration that mean we got significantly closer to the target
			if (ratio_a < damping_limit) {
				bufferSet->vel[i].x *= ratio_a;
			}
		}

		ratio_a = a_i.y / a_old.y;
		if (ratio_a > 0) {
			//now depending on how much the force has been lowered  I'll damp the past velocity
			//the logic behind it is that if we had a large drop in acceleration that mean we got significantly closer to the target
			if (ratio_a < damping_limit) {
				bufferSet->vel[i].y *= ratio_a;
			}
		}

		ratio_a = a_i.z / a_old.z;
		if (ratio_a > 0) {
			//now depending on how much the force has been lowered  I'll damp the past velocity
			//the logic behind it is that if we had a large drop in acceleration that mean we got significantly closer to the target
			if (ratio_a < damping_limit) {
				bufferSet->vel[i].z *= ratio_a;
			}
		}
	}

	if(false){
		//ok so I'll try to do smth more
		//I'll estiate the value of next simulation step gamma
		//Though I'll do a lightweight estimation by suposing that I'm the only particle moving
		//the easy way to do that is simply to estimate my future position and rerun the gamma computation
		Vector3d d_t = 0.0001;
		Vector3d p_i_future = p_i + d_t * (bufferSet->vel[i] + d_t * bufferSet->acc[i]);
		Vector3d p_i_future_no_acc = p_i + d_t * (bufferSet->vel[i]);
		RealCuda gamma_future = 0;
		RealCuda density_future = 0;
		RealCuda gamma_future_no_acc = 0;
		RealCuda density_future_no_acc = 0;


		//we will compute the 3 components at once
		ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);

		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
			if (i != j) {
				Vector3d x_ij = p_i_future - bufferSet->pos[j];

				//gamma 
				gamma_future += KERNEL_W(data, x_ij) * bufferSet->mass[j] / bufferSet->density[j];
				density_future += KERNEL_W(data, x_ij) * bufferSet->mass[j];


				x_ij = p_i_future_no_acc - bufferSet->pos[j];

				//gamma 
				gamma_future_no_acc += KERNEL_W(data, x_ij) * bufferSet->mass[j] / bufferSet->density[j];
				density_future_no_acc += KERNEL_W(data, x_ij) * bufferSet->mass[j];
			}
		);

		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
			{
				Vector3d x_ij = p_i_future - data.boundaries_data_cuda->pos[j];

				//gamma for debug
				gamma_future += KERNEL_W(data, x_ij) * data.boundaries_data_cuda->mass[j] / data.density0;
				density_future += KERNEL_W(data, x_ij) * data.boundaries_data_cuda->mass[j];

				x_ij = p_i_future_no_acc - data.boundaries_data_cuda->pos[j];

				//gamma for debug
				gamma_future_no_acc += KERNEL_W(data, x_ij) * data.boundaries_data_cuda->mass[j] / data.density0;
				density_future_no_acc += KERNEL_W(data, x_ij) * data.boundaries_data_cuda->mass[j];
			}
		);
		density_future += data.W_zero * bufferSet->mass[i];
		gamma_future += data.W_zero * bufferSet->mass[i] / density_future;

		density_future_no_acc += data.W_zero * bufferSet->mass[i];
		gamma_future_no_acc += data.W_zero * bufferSet->mass[i] / density_future_no_acc;


		//printf("pos and posfuture_estimation: %f %f %f // %f %f %f \n", p_i.x, p_i.y, p_i.z, p_i_future.x, p_i_future.y, p_i_future.z);
		//printf("density comparison now/future: %f // %f \n", bufferSet->density[i], density_future);
		//bufferSet->kappa[i] = density_future;
		//bufferSet->kappaV[i] = gamma_future;

		//let's put some rules to limit the number of particles that have an acceleration applyed to them
		//first of all let's only correct oversampled areas
		RealCuda gamma_limit = 1.0;
		if (gamma < gamma_limit) {
			bufferSet->vel[i] = 0.0;
			bufferSet->acc[i] = 0.0;
		}

		//then prevent oversampled area to get worse
		//note I'll simplify this by knowing I only have gammas >1.0
		if (gamma_future > gamma) {
			bufferSet->acc[i] = 0.0;
		}

		if (gamma_future_no_acc > gamma) {
			bufferSet->vel[i] = 0.0;
		}
	}

	
	if (gamma < 0.96) {
		//bufferSet->color[i] = Vector3d(0, 1, 0);
	}
}

//this kernel witll poush the fluid particles depending on the boundaries particles in their neighborhood
//essencially I'll use the boundaries particle to compute the normal
template<bool compute_active_only>
__global__ void push_particles_from_boundaries_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int count_potential_fluid, RealCuda coef_push) {
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

	Vector3d p_i = bufferSet->pos[i];
	Vector3d F(0);

	//essencially I'll push the particles relative to their contribution in the density
	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		Vector3d x_ij = p_i - data.boundaries_data_cuda->pos[j];
		F += KERNEL_W(data, x_ij) * data.boundaries_data_cuda->mass[j] * x_ij.unit();
	);

	F *= coef_push;


	bufferSet->acc[i] = F;
	bufferSet->vel[i] = Vector3d(0);
}


//this kernel witll poush the fluid particles depending on the boundaries particles in their neighborhood
//essencially I'll use the boundaries particle to compute the normal
template<bool compute_active_only>
__global__ void low_densities_attraction_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int count_potential_fluid, RealCuda coef_attraction) {
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

	Vector3d p_i = bufferSet->pos[i];
	Vector3d den_i = bufferSet->density[i];
	Vector3d F(0);

	//essencially I'll push the particles relative to their contribution in the density
	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
	
		if (den_i > bufferSet->density[j]) {
			Vector3d x_ij = p_i - bufferSet->pos[j];
			//F += KERNEL_W(data, x_ij) * data.boundaries_data_cuda->mass[j] * x_ij.unit();

			F += -1.0f * x_ij.unit() * (den_i - bufferSet->density[j]);
		}

	);

	F *= coef_attraction;


	bufferSet->acc[i] = F;
	bufferSet->vel[i] = Vector3d(0);
}



__global__ void data_manipulation_debug_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }

	if (i == 5573) {
		bufferSet->pos[i].y += data.particleRadius ;
	}
}

template<bool candidate_only_for_fluid>
__global__ void comp_closest_dist_to_neighbors_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int count_candidates=-1) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }

	Vector3d p_i = bufferSet->pos[i];

	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);

	RealCuda dist = 1000;
	int id = 0;
	bufferSet->kappaV[i] = 0;
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
		if (i != j) {
			if ((!candidate_only_for_fluid) || (j < count_candidates)) {
				Vector3d x_ij = p_i - bufferSet->pos[j];

				if (x_ij.norm() < dist) {
					id = j;
					dist = x_ij.norm();
				}
			}
		}
	);

	//this coef will switch to -1 when the closest is a boundary
	int coef = 1;
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		if (i != j) {
			Vector3d x_ij = p_i - data.boundaries_data_cuda->pos[j];

			if (x_ij.norm() < dist) {
				id = j;
				dist = x_ij.norm();
				bufferSet->kappaV[i] = 1;
				coef = -1;
			}
		}
	);

	bufferSet->kappa[i] = coef*dist;

	/*
	if (dist / data.particleRadius < 1) {
		bufferSet->color[i] = Vector3d(0, 1, 1);
	}
	//*/
}

__global__ void tag_closest_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, int tag_value) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }

	if (bufferSet->neighborsDataSet->cell_id[i] == TAG_ACTIVE) {
		Vector3d p_i = bufferSet->pos[i];
		ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);
		
		RealCuda dist = 1000;
		int id = 0;
		int count = 0;
		ITER_NEIGHBORS_FROM_STRUCTURE_BASE(bufferSet->neighborsDataSet, bufferSet->pos,
			if (i != j) {
				Vector3d x_ij = p_i - bufferSet->pos[j];

				if (x_ij.norm() < dist) {
					id = j;
					dist = x_ij.norm();
				}
				bufferSet->neighborsDataSet->cell_id[j] = tag_value;
				count++;
				//if (count >0) {	return;}
			}
		);

		//bufferSet->neighborsDataSet->cell_id[id] = TAG_ACTIVE;
	}
}






__global__ void evaluate_and_discard_impact_on_neighbors_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet, Vector3d* positions, RealCuda* den,
	int count_samples,	RealCuda density_limit) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count_samples) { return; }

	Vector3d p_i = positions[i];


	ITER_NEIGHBORS_INIT_CELL_COMPUTATION(p_i, data.getKernelRadius(), data.gridOffset);

	//*
	int count_neighbors = 0;
	RealCuda density = 0;

	//check if there is any fluid particle above us
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
		//RealCuda density_delta = (fluidSet->pos[j]-p_i).norm();
		RealCuda density_delta = fluidSet->getMass(j) * KERNEL_W(data, p_i - fluidSet->pos[j]);
		if ((fluidSet->density[i] + density_delta) > density_limit) {
			den[i] = -1;
			return;
		}
	);

}

__global__ void compute_air_particle_mass_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* particleSet, int nbr_fluid_particles) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }
	if (i < nbr_fluid_particles) { return; }

	Vector3d pos_i = particleSet->pos[i];

	//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
	ITER_NEIGHBORS_INIT(data, particleSet, i);

	RealCuda delta = data.W_zero;
	ITER_NEIGHBORS_FLUID(data, particleSet,
		i,
		{  delta += KERNEL_W(data, pos_i - body.pos[neighborIndex]); }
		);


	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////



	ITER_NEIGHBORS_BOUNDARIES(data, particleSet,
		i,
		{ delta += KERNEL_W(data, pos_i - body.pos[neighborIndex]); }
		);


	//////////////////////////////////////////////////////////////////////////
	// Dynamic bodies
	//////////////////////////////////////////////////////////////////////////
	//*
	ITER_NEIGHBORS_SOLIDS(data, particleSet,
		i,
		{ delta += KERNEL_W(data, pos_i - body.pos[neighborIndex]); }
		);
	//*/


	particleSet->mass[i] = data.density0 / delta;
}


__global__ void cuda_get_min_max_sqnorm_v3d_kernel(Vector3d* array, int size, RealCuda* min, RealCuda* max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size) { return; }

	RealCuda norm_i = array[i].squaredNorm();

	if (min != NULL) {
		atomicToMin(min, norm_i);
	}

	if (max != NULL) {
		atomicToMax(max, norm_i);
	}
}

template<bool restrict_to_active>
__global__ void cuda_get_full_velocity_information_kernel(UnifiedParticleSet* particleSet, RealCuda* min, RealCuda* max,
	RealCuda* sum) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	if (restrict_to_active) {
		if (particleSet->neighborsDataSet->cell_id[i] != TAG_ACTIVE) {
			return;
		}
	}
	//*
	RealCuda norm_i = particleSet->vel[i].norm();

	if (min != NULL) {
		atomicToMin(min, norm_i);
	}

	if (max != NULL) {
		atomicToMax(max, norm_i);
	}

	if (sum != NULL) {
		atomicAdd(sum, norm_i);
	}
	//*/
}


void RestFLuidLoader::stabilizeFluid(SPH::DFSPHCData& data, RestFLuidLoaderInterface::StabilizationParameters& params) {
	if(!isInitialized()) {
		std::cout << "RestFLuidLoader::stabilizeFluid Loading impossible data was not initialized" << std::endl;
		return ;
	}

	if (!isDataTagged()) {
		std::cout << "!!!!!!!!!!! RestFLuidLoader::stabilizeFluid you are loading untagged data !!!!!!!!!!!" << std::endl;
		return ;
	}

	params.stabilization_sucess = false;

	if (params.evaluateStabilization) {
		params.stabilzationEvaluation1 = -1;
		params.stabilzationEvaluation2 = -1;
		params.stabilzationEvaluation3 = -1;
	}

	if (params.method == 0) {
		//so the first method will be to actually simulate the fluid while potentially restricting it
		//this need to be fully manipulable to potentially only activate part of the simulation process
		//as for simulation only part of the simulation domain it may be hard but by sorting the data in the right order it may be feasible with the current implementation though the cache hit rate will go down hard
		//worst case I'll have to copy all the functions 
		//a relatively easy way to add a particle restriction to the current implementation would be to use a macro that I set to nothing when  don't consider tagging and set to a return when I do
		//but I'll think more about that later (you can also remove from the simulation all particle that are so far from the usefull ones that they wont have any impact 



		UnifiedParticleSet* particleSet = data.fluid_data;
		int count_fluid_particles = particleSet->numParticles;
		if (params.show_debug) {
			std::cout << "nbr particles before loading: " << particleSet->numParticles << std::endl;
		}

		std::vector<std::string> timing_names{ "init","neighbors_init","update_tag","neighbors_store","divergence",
			"external","pressure","check max velocity","check particles outside boundairies" };
		SPH::SegmentedTiming timings(" RestFLuidLoader::stabilizeFluid method simu + damping", timing_names, true);
		timings.init_step();

		if (params.keep_existing_fluid && params.reloadFluid) {
			if (params.show_debug) {
				std::cout << "Reloading fluid is impossible currently if keeping the existing fluid is required " << std::endl;
			}
			params.reloadFluid = false;
		}

		// I neen to load the data to the simulation however I have to keep the air particles
		if (params.reloadFluid) {
			if (params.show_debug) {
				std::cout << "Reloading asked " << std::endl;
			}
			RestFLuidLoaderInterface::LoadingParameters params_loading;
			params_loading.load_fluid = true;
			params_loading.keep_air_particles = false;
			params_loading.set_up_tagging = true;
			params_loading.keep_existing_fluid = false;
			count_fluid_particles = loadDataToSimulation(data, params_loading);
			if (params.show_debug) {
				std::cout << " test after loading  (current/actualfluid): " << particleSet->numParticles << "   " << count_fluid_particles << std::endl;
			}
		}
		else {
			if (params.show_debug) {
				std::cout << "No reloading asked " << std::endl;
			}
		}


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
		bool reduceDampingAndClamping = params.reduceDampingAndClamping;
		RealCuda reduceDampingAndClamping_val = params.reduceDampingAndClamping_val;

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

		bool simulate_border_only = params.stabilize_tagged_only;
		int restriction_type= 1;
		bool use_tagging = true;
		bool pretag_neighbors = true;
		//I can save the tagging so that I don't have to redo it everytimestep
		static unsigned int* tag_array = NULL;
		static unsigned int tag_array_max_size = 0;
		if (simulate_border_only) {
			if (hasFullTaggingSaved()) {
				std::cout << "using precomputed tagging" << std::endl;
				//ok in some situation I had to allocate dedicated memory to store the backup of the tagging ...
				//so yeah sorry for that "if"
				if (tag_array_with_existing_fluid_size > 0) {
					std::cout << "using specilized storage" << std::endl;
					tag_array = tag_array_with_existing_fluid;
					tag_array_max_size = tag_array_with_existing_fluid_size;
				}
				else {
					std::cout << "using background buffer as storage" << std::endl;
					tag_array = backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted;
					tag_array_max_size = backgroundFluidBufferSet->numParticlesMax;
				}

				for (int i = 0; i < particleSet->numParticles; i++) {
					//tag_array[i] = TAG_UNTAGGED;
				}
			}
			else {
				if (tag_array_max_size < data.fluid_data->numParticles) {
					CUDA_FREE_PTR(tag_array);
				}

				if (tag_array == NULL) {
					cudaMallocManaged(&(tag_array), data.fluid_data->numParticlesMax * sizeof(unsigned int));
					tag_array_max_size = data.fluid_data->numParticlesMax;
				}
				//init the neighbor structure
				//data.fluid_data->initNeighborsSearchData(data, false);

				//for now I'll leave some system to full computation and I'll change them 
				//if their computation time is high enougth 
				cuda_neighborsSearch(data, false);

				//init the tagging and make a backup
				set_buffer_to_value<unsigned int>(data.fluid_data->neighborsDataSet->cell_id, TAG_UNTAGGED, data.fluid_data->numParticles);
				{
					RealCuda tagging_distance = data.getKernelRadius() * 0.99;
					int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
					tag_neighborhood_kernel<false, true> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data_cuda, data.fluid_data->gpu_ptr,
						tagging_distance, count_fluid_particles);
					gpuErrchk(cudaDeviceSynchronize());
				}

				int additional_neighbors_order_tagging = 0;
				for (int i = 0; i < (additional_neighbors_order_tagging); ++i) {
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);

					//tag the first order neighbors
					if (i == 0) {
						tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS);
						gpuErrchk(cudaDeviceSynchronize());
					}



					//then the second order
					tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE_NEIGHBORS, TAG_1);
					gpuErrchk(cudaDeviceSynchronize());


					//then cnvert the tags
					convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_ACTIVE_NEIGHBORS, TAG_ACTIVE);
					gpuErrchk(cudaDeviceSynchronize());
					if (i < (additional_neighbors_order_tagging - 1)) {
						convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_1, TAG_ACTIVE_NEIGHBORS);
					}
					else {
						convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_1, TAG_UNTAGGED);
					}
						gpuErrchk(cudaDeviceSynchronize());
				}

				//count hte number of tagged particles and back the tag array (at the same sime to parralel everything (comp and mem transfer)
				{
					int tag = TAG_ACTIVE;
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
					data.count_active = *(SVS_CU::get()->tagged_particles_count);
				}


				//do a wide tagging of their neighbors
				//by wide I mean you nee to do the tagging like if they had a slightly extended neighborhood
				//*
				if (true) {
					RealCuda tagging_distance = data.getKernelRadius() * 1.1;
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
					tag_neighbors_of_tagged_kernel<true, true, false> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, tagging_distance,
						TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS, count_fluid_particles);
					gpuErrchk(cudaDeviceSynchronize());
				}
				{
					int tag = TAG_ACTIVE_NEIGHBORS;
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
					data.count_active_neighbors = *(SVS_CU::get()->tagged_particles_count);
				}
				//*/

				//sort de data following the tag so that the particles that interest us are stacked at the front
				if (true) {
					//run the sort
					cub::DeviceRadixSort::SortPairs(particleSet->neighborsDataSet->d_temp_storage_pair_sort, particleSet->neighborsDataSet->temp_storage_bytes_pair_sort,
						data.fluid_data->neighborsDataSet->cell_id, data.fluid_data->neighborsDataSet->cell_id_sorted,
						particleSet->neighborsDataSet->p_id, particleSet->neighborsDataSet->p_id_sorted, particleSet->numParticles);
					gpuErrchk(cudaDeviceSynchronize());

					cuda_sortData(*particleSet, particleSet->neighborsDataSet->p_id_sorted);
					gpuErrchk(cudaDeviceSynchronize());

					//and backup the tag
					gpuErrchk(cudaMemcpy(tag_array, data.fluid_data->neighborsDataSet->cell_id_sorted, data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
				}
				else {
					//and backup the tag
					gpuErrchk(cudaMemcpy(tag_array, data.fluid_data->neighborsDataSet->cell_id, data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
				}
					
			}

			//this is the line to reload the tagging from the backup
			//gpuErrchk(cudaMemcpy(data.fluid_data->neighborsDataSet->cell_id, tag_array, data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

		}

		//a test changing the mass of air particles to see it it improve anything
		if(false){
			particleSet->initNeighborsSearchData(data, false);
			cuda_updateNeighborsStorage(data, *particleSet);
			int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
			compute_air_particle_mass_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, count_fluid_particles);
			gpuErrchk(cudaDeviceSynchronize());
		}

		//just to be sure
		//set_buffer_to_value<RealCuda>(data.fluid_data->densityAdv, 0, data.fluid_data->numParticles);
		set_buffer_to_value<RealCuda>(data.fluid_data->density, 0, data.fluid_data->numParticles);
		set_buffer_to_value<RealCuda>(data.fluid_data->kappa, 0, data.fluid_data->numParticles);
		set_buffer_to_value<RealCuda>(data.fluid_data->kappaV, 0, data.fluid_data->numParticles);


		if (false) {
			particleSet->initNeighborsSearchData(data, false);
			cuda_updateNeighborsStorage(data, *particleSet, -1);
			cuda_divergence_warmstart_init(data);

			std::ofstream myfile("temp4.csv", std::ofstream::trunc);
			if (myfile.is_open())
			{
				for (int i_test = 0; i_test < particleSet->numParticles; ++i_test) {
					myfile << i_test << "   " << particleSet->density[i_test] << "  " <<
						data.fluid_data->neighborsDataSet->cell_id[i_test] << std::endl;

				}
			}
			exit(0);
		}
		
		timings.time_next_point();

		data.restriction_mode = restriction_type;

		if (params.show_debug) {
			std::cout << "RestFLuidLoader::stabilizeFluid checking the restriction mode and the true particle count " <<
				data.restriction_mode << "   " << data.true_particle_count << std::endl;
		}

		bool interupt_at_step_end = false;
		int min_stabilization_iter = params.min_stabilization_iter;
		RealCuda stable_velocity_max_target = params.stable_velocity_max_target;
		RealCuda stable_velocity_avg_target = params.stable_velocity_avg_target;
		int count_lost_particles = 0;
		int count_lost_particles_limit = params.countLostParticlesLimit;
		int iter = 0;
		for (iter = 0; iter < params.stabilizationItersCount; iter++) {

			//even though the name is bad but it need to be here so that the iter count is correct
			if (interupt_at_step_end) {
				break;
			}

			if (iter != 0) {
				timings.init_step();
				timings.time_next_point();

				//data.fluid_data->updateActiveParticleNumber(count_fluid_particles);
			}

			if (simulate_border_only) {
				data.computeFluidLevel();
				/*
				if (iter >= 3) {
					//maxErrorD *= 0.8;
					maxErrorD = 0.1;
				}
				//*/
				
				//for now I'll leave some system to full computation and I'll change them if their computation time is high enougth
				//neighborsearch 
				//cuda_neighborsSearch(data, false);
				//I have to use the version separating the init and the storage since I need to scitch the index between the 2
				particleSet->initNeighborsSearchData(data, false);

				//test the fluid level to see how it evolve
				if (params.show_debug) {
					std::cout << "fluid level testing in stabilization: " << data.computeFluidLevel() << std::endl;
				}

				timings.time_next_point();
				
				//recover the tagging
				if (restriction_type == 1) {
					gpuErrchk(cudaMemcpy(data.fluid_data->neighborsDataSet->cell_id, tag_array, data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
					
					//now I need to tag their neighbors and to count the number of tagged particles
					if(!pretag_neighbors){
						int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
						tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE,TAG_ACTIVE_NEIGHBORS);
						gpuErrchk(cudaDeviceSynchronize());
					}
					if(false){
						*(SVS_CU::get()->tagged_particles_count) = 0;
						int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
						count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_ACTIVE_NEIGHBORS, SVS_CU::get()->tagged_particles_count);
						gpuErrchk(cudaDeviceSynchronize());

						data.count_active_neighbors = *(SVS_CU::get()->tagged_particles_count);

						if (params.show_debug) {
							std::cout << "count active/activeneighbors : " << data.count_active << "  " << data.count_active_neighbors << std::endl;
						}
					}
				}


				timings.time_next_point();

				cuda_updateNeighborsStorage(data, *particleSet, iter);


				//and tag the neigbors for physical properties computation
				


				timings.time_next_point();

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

				timings.time_next_point();
				
				if (false) {
					if (iter == 0) {
						SPH::UnifiedParticleSet* studySet = particleSet;
						Vector3d* pos = new Vector3d[studySet->numParticles];
						Vector3d* vel = new Vector3d[studySet->numParticles];
						read_UnifiedParticleSet_cuda(*studySet, pos, vel, NULL, NULL);

						std::ofstream myfile("temp10.csv", std::ofstream::trunc);
						if (myfile.is_open())
						{
							for (int j = 0; j < studySet->numParticles; j++) {

								myfile << j << "  " << studySet->neighborsDataSet->cell_id[j] << "  " << pos[j].toString() << "  " <<
									vel[j].toString() << "  " << studySet->acc[j].toString() << "  " <<
									studySet->density[j] << "  " << std::endl;
							}
							myfile.close();
						}

						delete[] pos;
						delete[] vel;

					}
				}
				
				if (params.show_debug) {
					RealCuda max_density = 0;
					int id_max_density = 0;
					for (int j = 0; j < particleSet->numParticles; ++j) {
						if (max_density < particleSet->density[j]) {
							max_density= particleSet->density[j];
							id_max_density = j;
						}
					}

					std::cout << "max density (id/ density / tag): " <<id_max_density<<" / "<<  max_density <<
						" / "<<particleSet->neighborsDataSet->cell_id[id_max_density]<<std::endl;
				}

				//external forces
				if (useExternalForces) {
					cuda_externalForces(data);

					if (false) {
						if (iter == 0) {
							SPH::UnifiedParticleSet* studySet = particleSet;
							Vector3d* pos = new Vector3d[studySet->numParticles];
							Vector3d* vel = new Vector3d[studySet->numParticles];
							read_UnifiedParticleSet_cuda(*studySet, pos, vel, NULL, NULL);

							std::ofstream myfile("temp11.csv", std::ofstream::trunc);
							if (myfile.is_open())
							{
								for (int j = 0; j < studySet->numParticles; j++) {

									myfile << j << "  " << studySet->neighborsDataSet->cell_id[j] << "  " << pos[j].toString() << "  " <<
										vel[j].toString() << "  " << studySet->acc[j].toString() << "  " <<
										 studySet->density[j] << "  " << std::endl;
								}
								myfile.close();
							}

							delete[] pos;
							delete[] vel;

						}
					}


					cuda_update_vel(data);
				}

				timings.time_next_point();
				
				if (false) {
					if (iter == 0) {
						SPH::UnifiedParticleSet* studySet = particleSet;
						Vector3d* pos = new Vector3d[studySet->numParticles];
						Vector3d* vel = new Vector3d[studySet->numParticles];
						read_UnifiedParticleSet_cuda(*studySet, pos, vel, NULL, NULL);

						std::ofstream myfile("temp12.csv", std::ofstream::trunc);
						if (myfile.is_open())
						{
							for (int j = 0; j < studySet->numParticles; j++) {

								myfile << j << "  " << studySet->neighborsDataSet->cell_id[j] << "  " << pos[j].toString() << "  " <<
									vel[j].toString() << "  " << studySet->acc[j].toString() << "  " <<
									studySet->density[j] << "  " << std::endl;
							}
							myfile.close();
						}

						delete[] pos;
						delete[] vel;

					}
				}


				//density
				if (useDensitySolver) {
					iterD = cuda_pressureSolve(data, maxIterD, maxErrorD);
				}


				timings.time_next_point();

				if (params.useMaxErrorDPreciseAtMinIter) {
					if (iterD <=5) {
						maxErrorD = (maxErrorD+params.maxErrorDPrecise)/2.0f;
					}
				}

				//check the max velocity pre dampings
				//if the maximum velocity is below a threshold then we can trigger the system to end after this stabilization step
				if((iter>min_stabilization_iter)||params.show_debug){
					RealCuda* max_vel_norm = SVS_CU::get()->avg_density_err;
					*max_vel_norm = 0;
					RealCuda* avg_vel_norm = outRealCuda;
					*avg_vel_norm = 0;

					int numBlocks = calculateNumBlocks(particleSet->numParticles);
					cuda_get_full_velocity_information_kernel<true> << <numBlocks, BLOCKSIZE >> > (particleSet->gpu_ptr,
						NULL, max_vel_norm, avg_vel_norm);
					gpuErrchk(cudaDeviceSynchronize());

					if (params.show_debug) {
						std::cout << "max / avg vel norm (relative to particle radius displacement): " << *max_vel_norm << 
							" / " << (*avg_vel_norm) / data.count_active << "   ( " << (*max_vel_norm) / data.particleRadius*data.get_current_timestep() <<
							" / " << (*avg_vel_norm) / data.count_active / data.particleRadius*data.get_current_timestep() <<" )" <<std::endl;
					}

					if (iter>min_stabilization_iter) {
						if (((*max_vel_norm) < stable_velocity_max_target)&&
							(((*avg_vel_norm) / data.count_active)<stable_velocity_avg_target)) {
							interupt_at_step_end = true;
						}
					}
				}


				if (preUpdateVelocityDamping) {
					apply_factor_to_buffer(data.fluid_data->vel, Vector3d(preUpdateVelocityDamping_val), data.fluid_data->numParticles);
				}

				if (preUpdateVelocityClamping) {
					clamp_buffer_to_value<Vector3d, 4>(data.fluid_data->vel, Vector3d(preUpdateVelocityClamping_val), data.fluid_data->numParticles);
				}

				if (false) {
					if (iter == 0) {
						SPH::UnifiedParticleSet* studySet = particleSet;
						Vector3d* pos = new Vector3d[studySet->numParticles];
						Vector3d* vel = new Vector3d[studySet->numParticles];
						read_UnifiedParticleSet_cuda(*studySet, pos, vel, NULL, NULL);

						std::ofstream myfile("temp13.csv", std::ofstream::trunc);
						if (myfile.is_open())
						{
							for (int j = 0; j < studySet->numParticles; j++) {

								myfile << j << "  " << studySet->neighborsDataSet->cell_id[j] << "  " << pos[j].toString() << "  " <<
									vel[j].toString() << "  " << studySet->acc[j].toString() << "  " <<
									studySet->density[j] << "  " << std::endl;
							}
							myfile.close();
						}

						delete[] pos;
						delete[] vel;

					}
				}

				if (false) {
					{
						int numBlocks = calculateNumBlocks(count_potential_fluid);
						advance_in_time_particleSet_kernel<true, false> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, data.h);
						gpuErrchk(cudaDeviceSynchronize());
					}
				}
				else {

					cuda_update_pos(data);
				}




				timings.time_next_point();

				if (params.show_debug) {
					std::cout << "fluid_stabilization internal iters: " << iterV << "  " << iterD << std::endl;
				}

				if(false){
					
					RealCuda min_density = 10000;
					RealCuda max_density = 0;
					RealCuda avg_density = 0;
					RealCuda min_density_all = 10000;
					RealCuda max_density_all = 0;
					RealCuda avg_density_all = 0;
					int count = 0;

					for (int j = 0; j < count_fluid_particles; ++j) {
						if (particleSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE)
						{
							avg_density += particleSet->density[j];
							min_density = std::fminf(min_density, particleSet->density[j]);
							max_density = std::fmaxf(max_density, particleSet->density[j]);
							count++;
						}
						avg_density_all += particleSet->density[j];
						min_density_all = std::fminf(min_density_all, particleSet->density[j]);
						max_density_all = std::fmaxf(max_density_all, particleSet->density[j]);

					}
					avg_density_all /= count_fluid_particles;
					avg_density /= count;
					//*
					std::cout << "avg/min/max density (tagged ? all fluid) : " << avg_density << "  " << min_density << "  " << max_density << " ?? "
						<< avg_density_all << "  " << min_density_all << "  " << max_density_all << std::endl;
				}


				if (postUpdateVelocityDamping) {
					apply_factor_to_buffer(data.fluid_data->vel, Vector3d(postUpdateVelocityDamping_val), data.fluid_data->numParticles);
				}

				if (postUpdateVelocityClamping) {
					clamp_buffer_to_value<Vector3d, 4>(data.fluid_data->vel, Vector3d(postUpdateVelocityClamping_val), data.fluid_data->numParticles);
				}

				//I need to force 0 on the density adv buffer since the neighbors may change between iterations
				//set_buffer_to_value<RealCuda>(data.fluid_data->densityAdv, 0, data.fluid_data->numParticles);


				//this will have to be commented by the end because it is waiting computation time if  the fluid is stable
				//this one is only for debug so you should not bother with it
				if (runCheckParticlesPostion) {
					int c = data.checkParticlesPositions(2);
					if (interuptOnLostParticle) {
						if (c > 0) {
							std::cout << "fluid stabilization interupted du to the loss of particles" << std::endl;
							return;
						}
					}
					if (data.restriction_mode == 2) {
						data.count_active -= c;
					}
				}



				timings.time_next_point();
				timings.end_step();
				//std::cout << "nbr iter div/den: " << iterV << "  " << iterD << std::endl;

				
			}
			else {


				timings.time_next_point();
				timings.time_next_point();

				//neighborsearch 
				cuda_neighborsSearch(data, false);

				timings.time_next_point();

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


				timings.time_next_point();

				//external forces
				if (useExternalForces) {
					cuda_externalForces(data);
					cuda_update_vel(data);
				}

				timings.time_next_point();

				//density
				if (useDensitySolver) {
					iterD = cuda_pressureSolve(data, maxIterD, maxErrorD);
				}


				timings.time_next_point();

				if (preUpdateVelocityDamping) {
					apply_factor_to_buffer(data.fluid_data->vel, Vector3d(preUpdateVelocityDamping_val), data.fluid_data->numParticles);
				}

				if (preUpdateVelocityClamping) {
					clamp_buffer_to_value<Vector3d, 4>(data.fluid_data->vel, Vector3d(preUpdateVelocityClamping_val), data.fluid_data->numParticles);
				}

				
				cuda_update_pos(data);
			



				if (postUpdateVelocityDamping) {
					apply_factor_to_buffer(data.fluid_data->vel, Vector3d(postUpdateVelocityDamping_val), data.fluid_data->numParticles);
				}

				if (postUpdateVelocityClamping) {
					clamp_buffer_to_value<Vector3d, 4>(data.fluid_data->vel, Vector3d(postUpdateVelocityClamping_val), data.fluid_data->numParticles);
				}


				timings.time_next_point();

				if (runCheckParticlesPostion) {
					int c = data.checkParticlesPositions(2);
					if (interuptOnLostParticle) {
						if (c > 0) {
							return;
						}
					}
				}


				timings.time_next_point();
				timings.end_step();

				if (reduceDampingAndClamping) {
					preUpdateVelocityClamping_val *= reduceDampingAndClamping_val;
					postUpdateVelocityClamping_val *= reduceDampingAndClamping_val;
					preUpdateVelocityDamping_val *= reduceDampingAndClamping_val;
					postUpdateVelocityDamping_val *= reduceDampingAndClamping_val;
				}


			}
		}
		gpuErrchk(read_last_error_cuda("check stable after stabilization ", params.show_debug));
		
		std::cout << "RestFLuidLoader::stabilizeFluid checking the restriction mode and the true particle count after end" <<
			data.restriction_mode << "   " << data.true_particle_count << std::endl;

		data.fluid_data->updateActiveParticleNumber(count_fluid_particles);
		//reset that anyway, worse case possible it is already equals to -1
		data.true_particle_count = -1;

		//and remove the restriction if there is one
		data.restriction_mode = 0;

		timings.recap_timings();

		params.count_iter_o = iter;

		//I need to clear the warmstart and velocity buffer
		if (params.clearWarmstartAfterStabilization) {
			set_buffer_to_value<RealCuda>(data.fluid_data->kappa, 0, data.fluid_data->numParticles);
			set_buffer_to_value<RealCuda>(data.fluid_data->kappaV, 0, data.fluid_data->numParticles);
		}
		set_buffer_to_value<Vector3d>(data.fluid_data->vel, Vector3d(0), data.fluid_data->numParticles);

		//set the timestep back to the previous one
		data.updateTimeStep(old_timeStep);
		data.updateTimeStep(old_timeStep);




	}
	else if (params.method == 1) {
		//ok let's try with a particle packing algorithm
		//this algo come from :
		//An improved particle packing algorithm for complexgeometries

		//use that variable to study a single particle
		int id = 5573;//centered particle
		//int id = 16730;//worst particle in a stable fluid

		UnifiedParticleSet* particleSet = data.fluid_data;
		int count_fluid_particles = particleSet->numParticles;
		std::cout << "nbr particles before loading: " << particleSet->numParticles << std::endl;


		// I neen to load the data to the simulation however I have to keep the air particles
		if (params.reloadFluid) {

			std::cout << "Reloading asked " << std::endl;
			RestFLuidLoaderInterface::LoadingParameters params_loading;
			params_loading.load_fluid = true;
			params_loading.keep_air_particles = false;
			params_loading.set_up_tagging = false;
			params_loading.keep_existing_fluid = false;
			count_fluid_particles = loadDataToSimulation(data, params_loading);
			std::cout << " test after loading  (current/actualfluid): " << particleSet->numParticles << "   " << count_fluid_particles << std::endl;
		}
		else {
			std::cout << "No reloading asked " << std::endl;
		}
		particleSet->resetColor();
		//alterate the data for testing sake
		if (false) {
			int numBlocks = calculateNumBlocks(particleSet->numParticles);
			data_manipulation_debug_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr);
			gpuErrchk(cudaDeviceSynchronize());
		}


		//maybe a flull contruction of the neighbor is useless (typicaly storing them is most likely useless
		cuda_neighborsSearch(data, false);
		
		//so first initialize the density for all particles since we are gonna need it
		//also set a density limit way high to be sure no aditional particles get tagged
		{
			int numBlocks = calculateNumBlocks(particleSet->numParticles);
			evaluate_and_tag_high_density_from_buffer_kernel<false, false, false, false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, 
				outInt, 4000, particleSet->numParticles, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
			gpuErrchk(cudaDeviceSynchronize());
		}


		// a debug that show the lowest distance between two particles (with at least one beeing a fluid particle)
		if(true) {
			{
				int numBlocks = calculateNumBlocks(particleSet->numParticles);
				comp_closest_dist_to_neighbors_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr);
				gpuErrchk(cudaDeviceSynchronize());
			}

			//read data to CPU
			static Vector3d* vel = NULL;
			static Vector3d* pos = NULL;
			int size = 0;
			if (data.fluid_data->numParticles > size) {
				if (vel != NULL) {
					delete[] vel;
					delete[] pos;
				}
				vel = new Vector3d[particleSet->numParticlesMax];
				pos = new Vector3d[particleSet->numParticlesMax];
				size = particleSet->numParticlesMax;

			}
			read_UnifiedParticleSet_cuda(*(particleSet), pos, vel, NULL);

			static bool first_time = true;
			if (first_time) {
				first_time = false;
				std::ofstream myfile("temp.csv", std::ofstream::trunc);
				if (myfile.is_open())
				{
					myfile << "type min_dist px py pz" << std::endl;
				}
			}
			std::ofstream myfile("temp.csv", std::ofstream::app);
			if (myfile.is_open())
			{
				for (int i = 0; i < count_fluid_particles; ++i) {
					myfile << particleSet->kappaV[i]  <<"  "<<particleSet->kappa[i]/data.particleRadius << "  "<<pos[i].toString()<<"  "<<particleSet->density[i]<<std::endl;
				}
				myfile.close();
			}
		}
	
		//let's try smth new
		//I will only tag the particles that have a high enougth density
		if (false) {
			set_buffer_to_value<unsigned int>(data.fluid_data->neighborsDataSet->cell_id, TAG_UNTAGGED, data.fluid_data->numParticles);
			{
				int numBlocks = calculateNumBlocks(particleSet->numParticles);
				tag_densities_kernel<true, true, 0> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, 1025, count_fluid_particles, TAG_UNTAGGED, TAG_ACTIVE);
				gpuErrchk(cudaDeviceSynchronize());
			}

			{
				int tag = TAG_ACTIVE;
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
			
		}

		//retag the particles that are near the border
		if (true) {
			set_buffer_to_value<unsigned int>(data.fluid_data->neighborsDataSet->cell_id, TAG_UNTAGGED, data.fluid_data->numParticles);
			{
				int numBlocks = calculateNumBlocks(count_fluid_particles);
				tag_neighborhood_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data_cuda, particleSet->gpu_ptr, data.getKernelRadius() * 1.001, count_fluid_particles);
				gpuErrchk(cudaDeviceSynchronize());
			}
		}

		//add the tag specific to the first and second order neighbors
		if (true) {
			{
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS);
				gpuErrchk(cudaDeviceSynchronize());

				//then the second order
				tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE_NEIGHBORS, TAG_1);
				gpuErrchk(cudaDeviceSynchronize());
			}

			//clear the tagging for air particles
			for (int i = count_fluid_particles; i < (data.fluid_data->numParticles); ++i) {
				particleSet->neighborsDataSet->cell_id[i] = TAG_UNTAGGED;
			}
		}

		//this can be used to tag the n order neighborhood as active
		///TODO: WARNING:: THIS need to be corrected as it currently also tag the air ...
		if(false){
			int additional_neighbors_order_tagging = 1;
			for (int i = 0; i < (additional_neighbors_order_tagging); ++i) {
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);

				//tag the first order neighbors
				if (i == 0) {
					tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS);
					gpuErrchk(cudaDeviceSynchronize());
				}


				//then the second order
				tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE_NEIGHBORS, TAG_1);
				gpuErrchk(cudaDeviceSynchronize());


				//then cnvert the tags
				convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_ACTIVE_NEIGHBORS, TAG_ACTIVE);
				gpuErrchk(cudaDeviceSynchronize());
				if (i < (additional_neighbors_order_tagging - 1)) {
					convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_1, TAG_ACTIVE_NEIGHBORS);
				}
				else {
					convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_1, TAG_UNTAGGED);
				}
				gpuErrchk(cudaDeviceSynchronize());
			}
		}

		//l'ets reverse the flag to do some tests
		if (false) {
			for (int j = 0; j < count_fluid_particles; ++j) {
				if (particleSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE) {
					//particleSet->neighborsDataSet->cell_id[j] = 0;
				}
				else {
					particleSet->neighborsDataSet->cell_id[j] = TAG_ACTIVE;
				}
			}
			for (int j = count_fluid_particles; j < particleSet->numParticles; ++j) {
				particleSet->neighborsDataSet->cell_id[j] = TAG_AIR;
			}
		}

		//tag a single particle
		if (false) {
			set_buffer_to_value<unsigned int>(data.fluid_data->neighborsDataSet->cell_id, 0, data.fluid_data->numParticles);
			data.fluid_data->neighborsDataSet->cell_id[id] = TAG_ACTIVE;
		}


		//show the min max density of taged and all
		if(true){
			show_extensive_density_information(data.fluid_data, count_fluid_particles);
		}



		//evaluate gama at the start for debug purposes
		if(false){
			set_buffer_to_value<RealCuda>(particleSet->kappa, 0, data.fluid_data->numParticles);
			{
				int numBlocks = calculateNumBlocks(particleSet->numParticles);
				compute_gamma_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, count_fluid_particles, particleSet->kappa);
				gpuErrchk(cudaDeviceSynchronize());
			}

			//add some tags for debuging
			if (true) {
				cuda_neighborsSearch(data, false);
				
				//init the tagging and make a backup
				set_buffer_to_value<unsigned int>(data.fluid_data->neighborsDataSet->cell_id, TAG_UNTAGGED, data.fluid_data->numParticles);
				{
					int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
					tag_neighborhood_kernel<false, true> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data_cuda, data.fluid_data->gpu_ptr,
						data.getKernelRadius(), count_fluid_particles);
					gpuErrchk(cudaDeviceSynchronize());
				}

				{
					int tag = TAG_ACTIVE;
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
				{
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);

					//tag the first order neighbors
					tag_neighbors_of_tagged_kernel<true,true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS);
					gpuErrchk(cudaDeviceSynchronize());
					
					//then the second order
					tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE_NEIGHBORS, TAG_1);
					gpuErrchk(cudaDeviceSynchronize());

					//third order
					tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_1, TAG_2);
					gpuErrchk(cudaDeviceSynchronize());

					//forth order
					tag_neighbors_of_tagged_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_2, TAG_3);
					gpuErrchk(cudaDeviceSynchronize());


					//untagg the air particles
					for (int i = count_fluid_particles; i < particleSet->numParticles; i++) {
						particleSet->neighborsDataSet->cell_id[i] = TAG_UNTAGGED;
					}
				}
				{
					int tag = TAG_ACTIVE_NEIGHBORS;
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
				{
					int tag = TAG_1;
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
				{
					int tag = TAG_2;
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
				{
					int tag = TAG_3;
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
				if (true) {
					std::ofstream myfile("temp.csv", std::ofstream::trunc);
					if (myfile.is_open())
					{
						for (int i = 0; i < count_fluid_particles; i++) {
							//if (particleSet->neighborsDataSet->cell_id[i] == TAG_ACTIVE) {
							myfile <<i<< "  " << particleSet->neighborsDataSet->cell_id[i] << "  " << particleSet->density[i] << "   " << particleSet->kappa[i] <<
								 std::endl;;
							//}
						}
						myfile.close();
					}
				}

				exit(0);

			}

		}

		//prepare all cnstants (in the end I'll move them
		//params
		RealCuda delta_s = data.particleRadius * 2;
		RealCuda p_b = params.p_b;//2500 * delta_s;
		RealCuda k_r = params.k_r;// 150 * delta_s * delta_s;
		RealCuda zeta = params.zeta;// 2 * (SQRT_MACRO_CUDA(delta_s) + 1) / delta_s;

		RealCuda dt_pb = 0.1 * data.getKernelRadius() / SQRT_MACRO_CUDA(p_b);
		RealCuda dt_zeta_first = SQRT_MACRO_CUDA(0.1 * data.getKernelRadius() / zeta);
		RealCuda coef_to_compare_v_sq_to = (dt_zeta_first * dt_zeta_first) / (dt_pb * dt_pb);
		coef_to_compare_v_sq_to *= coef_to_compare_v_sq_to;

		RealCuda c = delta_s * 2.0 / 3.0;
		RealCuda r_limit = delta_s / 2;

		//ok so this is pure bullshit
		//I add that factor to make my curve fit with the one the guy fucking drawn in his paper (maybe it will help getting to the stable solution)
		//k_r *= 0.03;
		//and another factor to normalize the force on the same scale as a_b
		//k_r /= 700;


		std::cout << "parameters values p_b/k_r/zeta: " << p_b << "  " << k_r << "  " << zeta << std::endl;


		//I'll itegrate this cofficient inside
		k_r *= 12;

		//this is the parenthesis for the case where r it set to the limit
		RealCuda a_rf_r_limit = k_r * ((3 * c * c) / (r_limit * r_limit * r_limit * r_limit) - (2 * c) / (r_limit * r_limit * r_limit));

		std::cout << "arfrlimit: " << a_rf_r_limit << "  " << a_rf_r_limit / k_r << std::endl;

		//and now we can compute the acceleration
		set_buffer_to_value<Vector3d>(particleSet->vel, Vector3d(0, 0, 0), particleSet->numParticles);

		//OK I'll use a deubug structure to understand what is happening
		ParticlePackingDebug ppd;
		ppd.alloc(particleSet->numParticles);

		//I'll use them to debug
		set_buffer_to_value<RealCuda>(particleSet->kappa, 0, particleSet->numParticles);
		set_buffer_to_value<RealCuda>(particleSet->kappaV, 0, particleSet->numParticles);
		set_buffer_to_value<Vector3d>(particleSet->acc, Vector3d(0, 0, 0), particleSet->numParticles);

		std::vector<std::string> timing_names{ "void","tag","closest_dist","density","void","compute_acc","void","step_pos" };
		static SPH::SegmentedTiming timings("stabilization_method_1 loop", timing_names, true);

		for (int i = 0; i < params.stabilizationItersCount; i++) {
			timings.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)
			particleSet->resetColor();

			timings.time_next_point();//time p1
			// to reevaluate the density I need to rebuild the neighborhood
			//though this would override the tagging I'm using
			//so I nee to backup the tagging and reload it after
			bool use_precomputed_tag = true;
			if(use_precomputed_tag){
				static unsigned int* tag_array = NULL;
				if (tag_array == NULL) {
					cudaMallocManaged(&(tag_array), particleSet->numParticles * sizeof(unsigned int));
				}
				gpuErrchk(cudaMemcpy(tag_array, particleSet->neighborsDataSet->cell_id, particleSet->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

				particleSet->initNeighborsSearchData(data, false);

				gpuErrchk(cudaMemcpy(particleSet->neighborsDataSet->cell_id, tag_array, particleSet->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
			}
			else {
				particleSet->initNeighborsSearchData(data, false);

				set_buffer_to_value<unsigned int>(data.fluid_data->neighborsDataSet->cell_id, TAG_UNTAGGED, data.fluid_data->numParticles);
				{
					int numBlocks = calculateNumBlocks(particleSet->numParticles);
					tag_densities_kernel<true, true, 0> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, 1005, count_fluid_particles, TAG_UNTAGGED, TAG_ACTIVE);
					gpuErrchk(cudaDeviceSynchronize());
				}
			}

			timings.time_next_point();//time p1
			if (true) {
				{
					int numBlocks = calculateNumBlocks(particleSet->numParticles);
					comp_closest_dist_to_neighbors_kernel<false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr);
					gpuErrchk(cudaDeviceSynchronize());
				}

				RealCuda dist = 1000000;
				for (int i = 0; i < count_fluid_particles; ++i) {
					if (dist > particleSet->kappa[i]) {
						dist = particleSet->kappa[i];
					}
				}
				std::cout << "closest dist (relative to particle radius): " << dist / data.particleRadius << std::endl;
			}

			timings.time_next_point();//time p1
			
			//eval the density
			{
				*outInt = 0;
				{
					int numBlocks = calculateNumBlocks(particleSet->numParticles);
					evaluate_and_tag_high_density_from_buffer_kernel<false, false, false, false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, 
						outInt, 4000, particleSet->numParticles, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
					gpuErrchk(cudaDeviceSynchronize());
				}

			}
			timings.time_next_point();//time p1

			ppd.reset();

			*outRealCuda = -1;

			timings.time_next_point();//time p1
			//th particle packing algorithm
			if(false){
				int numBlocks = calculateNumBlocks(count_fluid_particles);
				particle_packing_negi_2019_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, count_fluid_particles,
					delta_s, p_b, k_r, zeta, coef_to_compare_v_sq_to,
					c, r_limit, a_rf_r_limit, outRealCuda, ppd);
				gpuErrchk(cudaDeviceSynchronize());
			}

			//the thing that push the particles from the border
			//it is too risky as it will cause superpositions 
			//and anyway there may be gap in the layer near the boundary
			if (false) {
				int numBlocks = calculateNumBlocks(count_fluid_particles);
				push_particles_from_boundaries_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, count_fluid_particles,p_b);
				gpuErrchk(cudaDeviceSynchronize());
			}

			//ok let's try another thing
			//let's try to make the higher den attacted to the lower densities
			if (true) {
				int numBlocks = calculateNumBlocks(count_fluid_particles);
				low_densities_attraction_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, count_fluid_particles, p_b);
				gpuErrchk(cudaDeviceSynchronize());
			}
			timings.time_next_point();//time p1

			RealCuda dt = *outRealCuda;

			if (params.timeStep > 0) {
				dt = params.timeStep;
			}
			else {
				if (dt > 0) {
					dt = SQRT_MACRO_CUDA(SQRT_MACRO_CUDA(dt)) * dt_zeta_first;
				}
				else {
					dt = dt_pb;
				}

			}

			//the paper tell us to increase zeta by 1% everysteps
			//so we need to recompute the coefficients
			if (params.zetaChangeFrequency > 0) {
				if ((i % params.zetaChangeFrequency) == 0) {
					//for when zeta is a pure damping coef directly on the velocity
					zeta *= params.zetaChangeCoefficient;

					std::cout << "zeta updated to: " << zeta << std::endl;

					/*
					//for when a_d is part of the acceletation
					zeta *= 1.01;
					dt_zeta_first = SQRT_MACRO_CUDA(0.1 * data.getKernelRadius() / zeta);
					coef_to_compare_v_sq_to = (dt_zeta_first * dt_zeta_first) / (dt_pb * dt_pb);
					coef_to_compare_v_sq_to *= coef_to_compare_v_sq_to;
					//*/
				}
			}

			timings.time_next_point();//time p1
			//std::cout << "test computations dt: " << *outRealCuda << "  " << coef_to_compare_v_sq_to << "  " << dt_pb << "  " << dt_zeta_first << std::endl;

			{
				int numBlocks = calculateNumBlocks(count_fluid_particles);
				advance_in_time_particleSet_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (particleSet->gpu_ptr, dt, zeta, true);
				gpuErrchk(cudaDeviceSynchronize());
			}

			timings.time_next_point();//time p1

			timings.end_step();//end point of the current step (if measuring avgs you need to call it at every end of the loop)

			//writte gamma info to file
			if (false) {
				std::ofstream myfile("temp.csv", std::ofstream::trunc);
				if (myfile.is_open())
				{
					for (int i = 0; i < count_fluid_particles; i++) {
						//if (particleSet->neighborsDataSet->cell_id[i] == TAG_ACTIVE) {
						myfile << particleSet->neighborsDataSet->cell_id[i] << "  " << ppd.gamma_f[i] << "  " << ppd.gamma_b[i] << "  " << ppd.gamma_f[i] + ppd.gamma_b[i] << "  " <<
							(ppd.gamma_f[i] + ppd.gamma_b[i] - 1) / ppd.gamma_b[i] << "  " <<
							particleSet->getNumberOfNeighbourgs(i, 0) << "   " << particleSet->getNumberOfNeighbourgs(i, 1) <<
							"  " << particleSet->getNumberOfNeighbourgs(i, 0) + particleSet->getNumberOfNeighbourgs(i, 1) << std::endl;;
						//}
					}
					myfile.close();
				}
			}

			//and some other info based one gamma + density
			if (false) {
				std::ofstream myfile("temp.csv", std::ofstream::trunc);
				if (myfile.is_open())
				{
					for (int i = 0; i < count_fluid_particles; i++) {
						//if (particleSet->neighborsDataSet->cell_id[i] == TAG_ACTIVE) {
						myfile << i << "  " << particleSet->neighborsDataSet->cell_id[i] << "  " << ppd.gamma_f[i] + ppd.gamma_b[i] << "  " << particleSet->density[i] << "  " <<
							particleSet->acc[i].toString() << "  " << particleSet->acc[i].norm() << std::endl;;
						//}
					}
					myfile.close();
				}
			}


			//to reevaluate the density I need to rebuild the neighborhood
			//though this would override the tagging I'm using
			//so I nee to backup the tagging and reload it after
			static unsigned int* tag_array = NULL;
			if (tag_array == NULL) {
				cudaMallocManaged(&(tag_array), particleSet->numParticles * sizeof(unsigned int));
			}
			gpuErrchk(cudaMemcpy(tag_array, particleSet->neighborsDataSet->cell_id, particleSet->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

			particleSet->initNeighborsSearchData(data, false);

			gpuErrchk(cudaMemcpy(particleSet->neighborsDataSet->cell_id, tag_array, particleSet->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
			{
				int numBlocks = calculateNumBlocks(count_fluid_particles);
				evaluate_and_tag_high_density_from_buffer_kernel<false, false, false, false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, 
					outInt, 4000, count_fluid_particles, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
				gpuErrchk(cudaDeviceSynchronize());
			}

			if (true) {
				std::cout << "iter: " << i << "  dt: " << dt << std::endl;

				show_extensive_density_information(data.fluid_data, count_fluid_particles);

				if (true) {

					Vector3d max_displacement(0);
					Vector3d min_displacement(10000000);
					Vector3d avg_displacement(0);
					Vector3d avg_signed_displacement(0);
					int count = 0;

					for (int j = 0; j < count_fluid_particles; ++j) {
						if (particleSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE)
						{
							avg_displacement += (particleSet->acc[j]).abs();//dt * dt *
							avg_signed_displacement += particleSet->acc[j];
							min_displacement.toMin(particleSet->acc[j]);//dt * dt *
							max_displacement.toMax(particleSet->acc[j]);//dt * dt *
							count++;
						}

					}
					avg_displacement /= count;
					avg_signed_displacement /= count;
					//*
					std::cout << avg_signed_displacement.toString() << " // " << avg_displacement.toString() << " // " << min_displacement.toString() << " // " << max_displacement.toString() << std::endl;
					//*/

					//std::cout << "info for p_id den//acc : " << ppd.gamma_f[id] + ppd.gamma_b[id] << "  //  " << particleSet->density[id] << "  //  " << particleSet->acc[id].toString() << std::endl;;
				}



				if (true) {
					ppd.readAvgAndMax(particleSet->neighborsDataSet->cell_id);
					//std::cout << ppd.avgAndMaxToString(false, false, false, false, false, true);
					std::cout << ppd.avgAndMaxToString();
				}

				if (false) {
					std::cout << ppd.particleInfoToString(id);

				}

				if (false) {
					//read data to CPU
					static Vector3d* vel = NULL;
					static Vector3d* pos = NULL;
					int size = 0;
					if (data.fluid_data->numParticles > size) {
						if (vel != NULL) {
							delete[] vel;
							delete[] pos;
						}
						vel = new Vector3d[particleSet->numParticlesMax];
						pos = new Vector3d[particleSet->numParticlesMax];
						size = particleSet->numParticlesMax;

					}
					read_UnifiedParticleSet_cuda(*(particleSet), pos, vel, NULL);

					if (true) {
						RealCuda avg_density = 0;
						RealCuda min_density = 10000;
						RealCuda max_density = 0;
						int count = 0;
						for (int j = 0; j < count_fluid_particles; j++) {
							if (data.fluid_data->neighborsDataSet->cell_id[j] == (TAG_ACTIVE + 2)) {
								count++;
								avg_density += particleSet->density[j];
								min_density = std::fminf(min_density, particleSet->density[j]);
								max_density = std::fmaxf(max_density, particleSet->density[j]);
							}
						}
						avg_density /= count;
						std::cout << "avg/min/max density (neighbor of interest particle) :" << avg_density << "  " << min_density << "  " << max_density << std::endl;

						avg_density *= count;
						avg_density += particleSet->density[id];
						min_density = std::fminf(min_density, particleSet->density[id]);
						max_density = std::fmaxf(max_density, particleSet->density[id]);
						count++;
						avg_density /= count;
						std::cout << "avg/min/max density (neighbor of interest + particle):" << avg_density << "  " << min_density << "  " << max_density << std::endl;
					}

					{
						int count = 0;
						Vector3d avg(0);
						Vector3d avg_signed(0);
						Vector3d max(0);
						Vector3d min(1000000);
						RealCuda avg_norm = 0;
						RealCuda max_norm = 0;
						for (int j = 0; j < count_fluid_particles; ++j) {
							if (particleSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE)
							{
								//std::cout << j << std::endl;
								avg += vel[j].abs();
								avg_signed += vel[j];
								max.toMax(vel[j]);
								min.toMin(vel[j]);
								avg_norm += vel[j].norm();
								max_norm = MAX_MACRO_CUDA(max_norm, vel[j].norm());
								count++;
							}
						}
						avg_norm /= count;
						avg /= count;
						avg_signed /= count;

						//std::cout << "vel for p_id: " << vel[id].toString()<<std::endl;
						std::cout << "velocities norm avg/max :" << avg_norm << "  " << max_norm << std::endl;
						//std::cout << avg_signed.toString() << " // " << avg.toString() << " // " << min.toString() << " // " << max.toString() << std::endl;

					}

					if (true) {

						static bool first_time = true;
						if (first_time) {
							first_time = false;
							std::ofstream myfile("temp.csv", std::ofstream::trunc);
							if (myfile.is_open())
							{
								myfile << "gamma density vx vy vz fx fy fz" << std::endl;
							}
						}
						std::ofstream myfile("temp.csv", std::ofstream::app);
						if (myfile.is_open())
						{

							myfile << ppd.gamma_f[id] + ppd.gamma_b[id] << "  " << particleSet->density[id] << "  " << vel[id].toString() << "  " << 
								particleSet->acc[id].toString() << "  " <<particleSet->kappa[id]<< "  " << particleSet->kappaV[id] << std::endl;
							myfile.close();
						}
					}

				}
				std::cout << std::endl;

			}

		}

		timings.recap_timings();//writte timming to cout

		//output the evaluation of the last step
		if (false) {
			std::ofstream myfile("temp.csv", std::ofstream::trunc);
			if (myfile.is_open())
			{
				for (int i = 0; i < count_fluid_particles; i++) {
					//if (particleSet->neighborsDataSet->cell_id[i] == TAG_ACTIVE) {
					myfile << i << "  " << particleSet->neighborsDataSet->cell_id[i] << "  " << ppd.gamma_f[i] + ppd.gamma_b[i] << "  " << particleSet->density[i] << "  " <<
						particleSet->acc[i].toString() << "  " << particleSet->acc[i].norm() << std::endl;;
					//}
				}
				myfile.close();
			}
		}

		//when I'm done I need to remove the air particles (but since the buffer is still sorted it's just a question of changing the number of active particles
		std::cout << "removing all but fluid (before/after/potential(for reference)//theorical): " << particleSet->numParticles << "   " << count_fluid_particles << "   " <<
			count_potential_fluid << "   " << count_potential_fluid - count_high_density_tagged_in_potential << "   " << std::endl;
		particleSet->updateActiveParticleNumber(count_fluid_particles);



		set_buffer_to_value<Vector3d>(particleSet->vel, Vector3d(0, 0, 0), particleSet->numParticles);
		set_buffer_to_value<Vector3d>(particleSet->acc, Vector3d(0, 0, 0), particleSet->numParticles);
		set_buffer_to_value<RealCuda>(particleSet->kappa, 0, particleSet->numParticles);
		set_buffer_to_value<RealCuda>(particleSet->kappaV, 0, particleSet->numParticles);

		data.checkParticlesPositions(2);

		//loadDataToSimulation(data);

	}
	else if (params.method == 2) {
	//ok since displacing the particle from an overdensity is so fucking hard maybe placing particles in an undersampled space will be easier
	//I know I'm missing around 700 partiles to maintain volume so first let's find a solution to find position where I can put those particles
		int count_fluid_particles=0;
		{
			RestFLuidLoaderInterface::LoadingParameters params_loading;
			params_loading.load_fluid = true;
			params_loading.keep_air_particles = true;
			params_loading.set_up_tagging = false;
			params_loading.keep_existing_fluid = false;
			count_fluid_particles = loadDataToSimulation(data, params_loading);
		}
		UnifiedParticleSet* particleSet = data.fluid_data;
		std::cout << " test after loading  (current/actualfluid): " << particleSet->numParticles << "   " << count_fluid_particles << std::endl;

	//maybe a flull contruction of the neighbor is useless (typicaly storing them is most likely useless
	cuda_neighborsSearch(data, false);

	set_buffer_to_value<unsigned int>(data.fluid_data->neighborsDataSet->cell_id, TAG_UNTAGGED, data.fluid_data->numParticles);
	{
		int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
		tag_neighborhood_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data_cuda, particleSet->gpu_ptr, data.getKernelRadius() * 1.001, count_fluid_particles);
		gpuErrchk(cudaDeviceSynchronize());
	}

	*outInt = 0;
	{
		int numBlocks = calculateNumBlocks(particleSet->numParticles);
		evaluate_and_tag_high_density_from_buffer_kernel<false, false, false, false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr,
			outInt, 4000, particleSet->numParticles, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
		gpuErrchk(cudaDeviceSynchronize());
	}
	//let's back the old density
	//then all be able to compare it to the new
	set_buffer_to_value<RealCuda>(particleSet->kappa, 0, particleSet->numParticles);
	gpuErrchk(cudaMemcpy(particleSet->kappa, particleSet->density, (count_fluid_particles) * sizeof(RealCuda), cudaMemcpyDeviceToDevice));


	if (true) {
		RealCuda min_density = 10000;
		RealCuda max_density = 0;
		RealCuda avg_density = 0;
		RealCuda min_density_all = 10000;
		RealCuda max_density_all = 0;
		RealCuda avg_density_all = 0;
		int count = 0;

		for (int j = 0; j < count_fluid_particles; ++j) {
			if (particleSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE)
			{
				avg_density += particleSet->density[j];
				min_density = std::fminf(min_density, particleSet->density[j]);
				max_density = std::fmaxf(max_density, particleSet->density[j]);
				count++;
			}
			avg_density_all += particleSet->density[j];
			min_density_all = std::fminf(min_density_all, particleSet->density[j]);
			max_density_all = std::fmaxf(max_density_all, particleSet->density[j]);

		}
		avg_density_all /= count_fluid_particles;
		avg_density /= count;
		//*
		std::cout << "avg/min/max density (tagged ? all fluid) : " << avg_density << "  " << min_density << "  " << max_density << " ?? "
			<< avg_density_all << "  " << min_density_all << "  " << max_density_all << std::endl;
		//*/
	}

	//*
	BufferFluidSurface S;
	S.setCuboid(Vector3d(0, 1, 0), Vector3d(0.5, 1, 0.5));
	RealCuda height_cap = 2;

	int count_samples = 0;
	Vector3d* pos = NULL;
	RealCuda* den = NULL;
	RealCuda affected_range = data.getKernelRadius();

	//I'll do a sampling on a regular grid
	RealCuda spacing = data.particleRadius / 2;

	Vector3d min, max;
	get_UnifiedParticleSet_min_max_naive_cuda(*(data.boundaries_data), min, max);
	std::cout << "min/ max: " << min.toString() << " " << max.toString() << std::endl;
	min += 2 * data.particleRadius;
	max -= 2 * data.particleRadius;
	Vector3i count_dim = (max - min) / spacing;
	count_dim += 1;

	std::cout << "count samples base :" << count_dim.x * count_dim.y * count_dim.z << std::endl;

	//only keep the samples that are near the plane
	int real_count = 0;
	for (int i = 0; i < count_dim.x; ++i) {
		for (int j = 0; j < count_dim.y; ++j) {
			for (int k = 0; k < count_dim.z; ++k) {
				Vector3d p_i = min + Vector3d(i, j, k) * spacing;
				//if (S.distanceToSurface(p_i) < affected_range)
				if (S.isinside(p_i))
				{
					if (p_i.y < height_cap) {
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
				//if (S.distanceToSurface(p_i) < affected_range)
				if (S.isinside(p_i))
				{
					if (p_i.y < height_cap) {
						pos[real_count] = p_i;
						real_count++;
					}
				}
			}
		}
	}

	//evluate the sampling density
	{
		int numBlocks = calculateNumBlocks(count_samples);
		DFSPH_evaluate_density_field_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, pos, den, count_samples);
		gpuErrchk(cudaDeviceSynchronize());
	}
	//since those are potential particles let's add their own wieght to their computation
	for (int i = 0; i < count_samples; ++i) {
		den[i] += particleSet->getMass(0) * data.W_zero;
	}

	//also I need to evaluate the impact tof the sampling on the existing particles
	//and never consiser the ones that make the density of existing particles too high
	//this will be fused withthe density evluation in the final version
	//evluate the sampling density
	{
		// (SPH::DFSPHCData data, SPH::UnifiedParticleSet* bufferSet, Vector3d* pos, int count_samples) {
		int numBlocks = calculateNumBlocks(count_samples);
		evaluate_and_discard_impact_on_neighbors_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, pos, den, count_samples, 1050);
		gpuErrchk(cudaDeviceSynchronize());
	}

	//*
	//read data to CPU
	static Vector3d* pos_f = NULL;
	int size = 0;
	if (particleSet->numParticlesMax > size) {
		if (pos_f != NULL) {
			delete[] pos_f;
		}
		pos_f = new Vector3d[particleSet->numParticlesMax];
		size = particleSet->numParticlesMax;

	}
	read_UnifiedParticleSet_cuda(*(particleSet), pos_f, NULL, NULL);
	//*/

	int count_potential = 0;
	int avg = 0;
	std::vector<Vector3d> added;
	std::vector<int> added_idxs;
	int density_limit = 1100;
	for (int i = 0; i < count_samples; ++i) {
		if (den[i] > 0 && den[i] < density_limit) {
			for (int j = 0; j < added.size(); ++j) {
				Vector3d x_ij = pos[i] - added[j];
				if (x_ij.norm() < data.getKernelRadius()) {
					RealCuda density_delta = particleSet->getMass(0) * KERNEL_W(data, x_ij);
					den[i] += density_delta;
				}
			}

			if (den[i] < density_limit) {
				bool valid_particle = true;
				//check if the new particle does not cause any problem with the already  added new particles
				//*
				for (int j = 0; j < added.size(); ++j) {
					Vector3d x_ij = pos[i] - added[j];
					if (x_ij.norm() < data.getKernelRadius()) {
						RealCuda density_delta = particleSet->getMass(0) * KERNEL_W(data, x_ij);
						if ((den[added_idxs[j]] + density_delta) > density_limit) {
							valid_particle = false;
							break;
						}
					}
				}
				//*/
				//*
				//do the same for the fluid particles
				if (valid_particle) {
					for (int j = 0; j < count_fluid_particles; ++j) {
						Vector3d x_ij = pos[i] - pos_f[j];
						if (x_ij.norm() < data.getKernelRadius()) {
							RealCuda density_delta = particleSet->getMass(0) * KERNEL_W(data, x_ij);
							if ((particleSet->density[j] + density_delta) > density_limit) {
								valid_particle = false;
								break;
							}
						}
					}
				}
				//*/

				if (valid_particle) {
					//I need to actually apply the impact on existing addded
					//*
					for (int j = 0; j < added.size(); ++j) {
						Vector3d x_ij = pos[i] - added[j];
						if (x_ij.norm() < data.getKernelRadius()) {
							RealCuda density_delta = particleSet->getMass(0) * KERNEL_W(data, x_ij);
							den[added_idxs[j]] += density_delta;
						}
					}
					//*/
					//and on the existing fluid
					//*
					for (int j = 0; j < count_fluid_particles; ++j) {
						Vector3d x_ij = pos[i] - pos_f[j];
						if (x_ij.norm() < data.getKernelRadius()) {
							RealCuda density_delta = particleSet->getMass(0) * KERNEL_W(data, x_ij);
							particleSet->density[j] += density_delta;

						}
					}
					//*/


					count_potential++;
					added.push_back(pos[i]);
					added_idxs.push_back(i);
				}
			}
		}
		avg += den[i];
	}
	avg /= count_samples;
	std::cout << "avg den: " << avg << std::endl;
	std::cout << "potential spaces count: " << count_potential << "  from this number of samples: " << count_samples << std::endl;
	//*/


	//and for this method add the new particles 
	//Since I want to do some debug I'll have to use an insertion method but for the end product I'll be able to do a simple 
	//replacement of the air particles positions by the new positions
	bool use_particle_insert_approach = true;
	if (use_particle_insert_approach) {
		//first since I want to insert the new positions at the end of the fluid positions and before the air positions
		int nbr_air_particles = particleSet->numParticles - count_fluid_particles;

		//I need to increase the number of active particles
		particleSet->updateActiveParticleNumber(particleSet->numParticles + count_potential);

		//displace the air particles
		//there is no easy way to displace values in a buffer without intermediary
		//so I'll use an additional buffer as an intermediary since this will only be used here for debug
		Vector3d* temp = NULL;
		cudaMallocManaged(&(temp), particleSet->numParticles * sizeof(Vector3d));

		gpuErrchk(cudaMemcpy(temp, particleSet->pos, particleSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		//displace 
		for (int j = 0; j < nbr_air_particles; ++j) {
			temp[(particleSet->numParticles - 1) - j] = temp[(particleSet->numParticles - 1) - j - count_potential];
		}

		//insert the positions
		for (int j = 0; j < count_potential; ++j) {
			temp[count_fluid_particles + j] = added[j];
		}

		//and copy back to gpu
		gpuErrchk(cudaMemcpy(particleSet->pos, temp, particleSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		count_fluid_particles += count_potential;

		//reinitialize the mass buffer
		set_buffer_to_value<RealCuda>(particleSet->mass, particleSet->mass[0], particleSet->numParticles);

		cudaFree(temp);

		{
			//ok let's do smth

			//and now retest the density to see where we are at
			//retag the particles that are near the border
			particleSet->initNeighborsSearchData(data, false);
			set_buffer_to_value<unsigned int>(data.fluid_data->neighborsDataSet->cell_id, TAG_UNTAGGED, data.fluid_data->numParticles);
			{
				int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
				tag_neighborhood_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data_cuda, particleSet->gpu_ptr, data.getKernelRadius() * 1.001, count_fluid_particles);
				gpuErrchk(cudaDeviceSynchronize());
			}

			*outInt = 0;
			{
				int numBlocks = calculateNumBlocks(particleSet->numParticles);
				evaluate_and_tag_high_density_from_buffer_kernel<false, false, false, false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr,
					outInt, 4000, particleSet->numParticles, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
				gpuErrchk(cudaDeviceSynchronize());
			}


			if (true) {
				std::ofstream myfile("temp.csv", std::ofstream::trunc);
				if (myfile.is_open())
				{
					for (int i = 0; i < count_fluid_particles; i++) {
						myfile << i << "  " << particleSet->neighborsDataSet->cell_id[i] << "  " << (i > (count_fluid_particles - count_potential)) << "  " <<
							particleSet->density[i] << "  " << particleSet->kappa[i] << "  " <<
							std::endl;
					}
					myfile.close();
				}
			}

			RealCuda min_density = 10000;
			RealCuda max_density = 0;
			RealCuda avg_density = 0;
			RealCuda min_density_all = 10000;
			RealCuda max_density_all = 0;
			RealCuda avg_density_all = 0;
			int count = 0;

			for (int j = 0; j < count_fluid_particles; ++j) {
				if (particleSet->neighborsDataSet->cell_id[j] == TAG_ACTIVE)
				{
					avg_density += particleSet->density[j];
					min_density = std::fminf(min_density, particleSet->density[j]);
					max_density = std::fmaxf(max_density, particleSet->density[j]);
					count++;
				}
				avg_density_all += particleSet->density[j];
				min_density_all = std::fminf(min_density_all, particleSet->density[j]);
				max_density_all = std::fmaxf(max_density_all, particleSet->density[j]);

			}
			avg_density_all /= count_fluid_particles;
			avg_density /= count;
			//*
			std::cout << "avg/min/max density (tagged ? all fluid) : " << avg_density << "  " << min_density << "  " << max_density << " ?? "
				<< avg_density_all << "  " << min_density_all << "  " << max_density_all << std::endl;
			//*/

			//std::cout << "info for p_id den//acc : " << ppd.gamma_f[id] + ppd.gamma_b[id] << "  //  " << particleSet->density[id] << "  //  " << particleSet->acc[id].toString() << std::endl;;
		}

	}

	//when I'm done I need to remove the air particles (but since the buffer is still sorted it's just a question of changing the number of active particles
	std::cout << "removing all but fluid (before/after/potential(for reference)//theorical): " << particleSet->numParticles << "   " << count_fluid_particles << "   " <<
		count_potential_fluid << "   " << count_potential_fluid - count_high_density_tagged_in_potential + count_potential << "   " << std::endl;
	particleSet->updateActiveParticleNumber(count_fluid_particles);


	set_buffer_to_value<Vector3d>(particleSet->vel, Vector3d(0, 0, 0), particleSet->numParticles);
	set_buffer_to_value<Vector3d>(particleSet->acc, Vector3d(0, 0, 0), particleSet->numParticles);
	set_buffer_to_value<RealCuda>(particleSet->kappa, 0, particleSet->numParticles);
	set_buffer_to_value<RealCuda>(particleSet->kappaV, 0, particleSet->numParticles);

	data.checkParticlesPositions(2);

	}
	else if (params.method == 3) {
		//here I'll do the approach using the low density particles as attraction points
		//The mains advantages of this approach are:
		//	it is extremely simple, so it should be pretty fucking fast
		//	since it is only an attraction rule, there is no need to consider the boundaries or the free surface
		if (!hasFullTaggingSaved()) {
			throw("I don't even want to handle that cas now, just use the damn tag loader");
		}
		int count_fluid_particles = data.count_active;
		unsigned int* tag_array = backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted;
		SPH::UnifiedParticleSet* particleSet = data.fluid_data;

		RealCuda dt = params.timeStep;

		bool debug_mode = true;
		bool debug_mode_messages = true;

		std::vector<std::string> timing_names{ "init","void","tag","closest_dist","density","compute_acc","step_pos" };
		static SPH::SegmentedTiming timings("stabilization_method_1 loop", timing_names, true);

		timings.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)

		// I neen to load the data to the simulation however I have to keep the air particles
		std::cout << "nbr particles before loading: " << particleSet->numParticles << std::endl;
		if (params.reloadFluid) {

			std::cout << "Reloading asked " << std::endl;
			RestFLuidLoaderInterface::LoadingParameters params_loading;
			params_loading.load_fluid = true;
			params_loading.keep_air_particles = true;
			params_loading.set_up_tagging = true;
			params_loading.keep_existing_fluid = false;
			count_fluid_particles = loadDataToSimulation(data, params_loading);
			std::cout << " test after loading  (current/actualfluid): " << particleSet->numParticles << "   " << count_fluid_particles << std::endl;
		}
		else {
			std::cout << "No reloading asked " << std::endl;
		}
		timings.time_next_point();//time p1

		for (int i = 0; i < params.stabilizationItersCount; i++) {
			if (i != 0) {
				timings.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)
				timings.time_next_point();//time p1
			}

			if (debug_mode) {
				particleSet->resetColor();
			}

			timings.time_next_point();//time p1
			// to reevaluate the density I need to rebuild the neighborhood
			//though this would override the tagging I'm using
			//so I nee to backup the tagging and reload it after
			bool use_precomputed_tag = true;
			if (use_precomputed_tag) {
				particleSet->initNeighborsSearchData(data, false);

				gpuErrchk(cudaMemcpy(particleSet->neighborsDataSet->cell_id, tag_array, particleSet->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
			}
			
			timings.time_next_point();//time p1
			if (debug_mode_messages) {
				{
					int numBlocks = calculateNumBlocks(particleSet->numParticles);
					comp_closest_dist_to_neighbors_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, count_fluid_particles);
					gpuErrchk(cudaDeviceSynchronize());
				}

				RealCuda dist = 1000000;
				bool is_closest_boundary = false;
				int id_particle = 0;
				for (int i = 0; i < count_fluid_particles; ++i) {
					if (dist > abs(particleSet->kappa[i])) {
						dist = abs(particleSet->kappa[i]);
						id_particle = i;
						if(particleSet->kappa[i]<0){
							is_closest_boundary = true;
						}
					}
				}
				std::cout << "closest dist (relative to particle radius): " << dist / data.particleRadius << "  "<<(is_closest_boundary?"is_boundry":"is_fluid")<<
					"   "<<id_particle<<std::endl;
				if (false) {
					std::cout << "more info about that particle: " << particleSet->density[id_particle] <<"   "<<particleSet->neighborsDataSet->cell_id[id_particle]<<
						"  //  "<<particleSet->acc[id_particle].toString()<<std::endl;

					static Vector3d* pos_temp = NULL;
					int size = 0;
					if (data.fluid_data->numParticles > size) {
						if (pos_temp != NULL) {
							delete[] pos_temp;
						}
						pos_temp = new Vector3d[data.fluid_data->numParticlesMax];
						size = data.fluid_data->numParticlesMax;

					}
					read_UnifiedParticleSet_cuda(*(data.fluid_data), pos_temp, NULL, NULL);


					std::cout << "even more info about that particle: " << pos_temp[id_particle].toString()<<std::endl;
				}
			}

			timings.time_next_point();//time p1

			//eval the density
			{
				*outInt = 0;
				{
					int numBlocks = calculateNumBlocks(particleSet->numParticles);
					evaluate_and_tag_high_density_from_buffer_kernel<false, false, false, false> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr,
						outInt, 4000, particleSet->numParticles, NULL, (params.keep_existing_fluid ? data.fluid_data->gpu_ptr : NULL));
					gpuErrchk(cudaDeviceSynchronize());
				}

			}

			if (false&&debug_mode_messages) {
				show_extensive_density_information(data.fluid_data, count_fluid_particles);
			}
			timings.time_next_point();//time p1


			*outRealCuda = -1;

			//ok let's try another thing
			//let's try to make the higher den attacted to the lower densities
			if (true) {
				int numBlocks = calculateNumBlocks(count_fluid_particles);
				low_densities_attraction_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, count_fluid_particles, params.p_b);
				gpuErrchk(cudaDeviceSynchronize());
			}
			timings.time_next_point();//time p1

			
			{
				int numBlocks = calculateNumBlocks(count_fluid_particles);
				advance_in_time_particleSet_kernel<true, true> << <numBlocks, BLOCKSIZE >> > (particleSet->gpu_ptr, dt);
				gpuErrchk(cudaDeviceSynchronize());
			}

			timings.time_next_point();//time p1

			timings.end_step();//end point of the current step (if measuring avgs you need to call it at every end of the loop)

			
		
		
		}

		timings.recap_timings();//writte timming to cout

		particleSet->updateActiveParticleNumber(count_fluid_particles);

	}
	else {
		std::cout << " RestFLuidLoader::stabilizeFluid no stabilization method selected" << std::endl;
		return;
	}


	params.stabilization_sucess = true;
	   
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


		RealCuda stabilzationEvaluation1 = -1;
		RealCuda stabilzationEvaluation2 = -1;
		RealCuda stabilzationEvaluation3 = -1;
		

		for (int i = 0; i < params.max_iterEval;++i) {
			//for now I'll use the solution of checking the max
			cuda_neighborsSearch(data, false);

			cuda_divergenceSolve(data, params.maxIterVEval, params.maxErrorVEval);

			cuda_externalForces(data);
			cuda_update_vel(data);

			cuda_pressureSolve(data, params.maxIterDEval, params.maxErrorDEval);

			cuda_update_pos(data);


			data.checkParticlesPositions(2);

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

			{
				//check the maximum velocity
				for (int i = 0; i < data.fluid_data->numParticles; ++i) {
					stabilzationEvaluation1 = MAX_MACRO_CUDA(stabilzationEvaluation1, vel[i].squaredNorm());
				}

				//check the average velocty near boundary
				{
					//first i have to tag the particles in question
					set_buffer_to_value<unsigned int>(data.fluid_data->neighborsDataSet->cell_id, TAG_UNTAGGED, data.fluid_data->numParticles);
					{
						int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
						tag_neighborhood_kernel<false, true> << <numBlocks, BLOCKSIZE >> > (data, data.boundaries_data_cuda, data.fluid_data_cuda, data.getKernelRadius(), -1);
						gpuErrchk(cudaDeviceSynchronize());
					}

					//then sum them
					RealCuda avg_vel = 0;
					int count_tagged = 0;
					for (int i = 0; i < data.fluid_data->numParticles; ++i) {
						if (data.fluid_data->neighborsDataSet->cell_id[i] != TAG_ACTIVE) {
							avg_vel += vel[i].norm();
							count_tagged++;
						}
					}
					avg_vel /= count_tagged;
					stabilzationEvaluation2 = MAX_MACRO_CUDA(stabilzationEvaluation2, avg_vel);
				}

				//and the last evaluation is the error on the density
				{
					RealCuda max_density_err = 0;
					for (int i = 0; i < data.fluid_data->numParticles; ++i) {
						max_density_err = MAX_MACRO_CUDA(max_density_err, (data.fluid_data->density[i]-data.density0));
					}
					stabilzationEvaluation3 = MAX_MACRO_CUDA(stabilzationEvaluation3, max_density_err);
				}
			}


		}
		//set the timestep back to the previous one
		data.updateTimeStep(old_timeStep);
		data.updateTimeStep(old_timeStep);

		//store the max velocity evaluation
		stabilzationEvaluation1=SQRT_MACRO_CUDA(stabilzationEvaluation1);
		params.stabilzationEvaluation1 = stabilzationEvaluation1;


		params.stabilzationEvaluation2 = stabilzationEvaluation2;

		params.stabilzationEvaluation3 = stabilzationEvaluation3;

	}


}