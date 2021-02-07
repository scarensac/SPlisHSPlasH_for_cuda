#include "OpenBoundariesSimple.h"
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
	class OpenBoundariesSimple {
	public:
		bool _isinitialized;

		//reprent the actual simulation boundary
		BufferFluidSurface S_boundary;

		//is slighly smaller than the actual boundary
		//is is used to quickly extract the layer of particles that are right next to the boundary
		BufferFluidSurface S_fluidInterior;

		//should represent the fluid surface/height
		BufferFluidSurface S_fluidSurface;

		//technicaly I could use a simple vector3d* but since I have to use
		//an unified particle set to load it I might as well keep it it may be usefull oneday
		UnifiedParticleSet* inflowPositionsSet;

		OpenBoundariesSimple() {
			_isinitialized = false;
			inflowPositionsSet = NULL;
		};

		~OpenBoundariesSimple() {

		};

		static OpenBoundariesSimple& getStructure() {
			static OpenBoundariesSimple obs;
			return obs;
		}

		bool isInitialized() { return _isinitialized; }

		void init(DFSPHCData& data, OpenBoundariesSimpleInterface::InitParameters& params);

		void applyOpenBoundary(DFSPHCData& data, OpenBoundariesSimpleInterface::ApplyParameters& params);
	};
}


void OpenBoundariesSimpleInterface::init(DFSPHCData& data, OpenBoundariesSimpleInterface::InitParameters& params) {
	OpenBoundariesSimple::getStructure().init(data, params);
}

void OpenBoundariesSimpleInterface::applyOpenBoundary(DFSPHCData& data, OpenBoundariesSimpleInterface::ApplyParameters& params) {
	OpenBoundariesSimple::getStructure().applyOpenBoundary(data, params);
}


__global__ void inflow_compute_and_store_constant_density_contribution_kernel(DFSPHCData data, SPH::UnifiedParticleSet* particleSet) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	Vector3d p_i = particleSet->pos[i];

	RealCuda density = particleSet->getMass(i) * data.W_zero;

		ITER_NEIGHBORS_INIT(data, particleSet, i);
	SPH::UnifiedParticleSet* otherSet;

	//I need to skip the fluid particles from the buffer since I only whan the constant contribution
	ADVANCE_END_PTR(end_ptr, particleSet->getNumberOfNeighbourgs(i, 0));
	neighbors_ptr = end_ptr;


	//for boundaries and solids since they do not move it is only needed to compute it once at the start
	//then boundaires
	otherSet = data.boundaries_data_cuda;
	ITER_NEIGHBORS_FROM_STORAGE(data, particleSet, i, 1,
		{
			RealCuda density_delta = otherSet->getMass(neighborIndex) * KERNEL_W(data, p_i - otherSet->pos[neighborIndex]);
			density += density_delta;
		}
	);

	particleSet->densityAdv[i] = density;
}

void OpenBoundariesSimple::init(DFSPHCData& data, OpenBoundariesSimpleInterface::InitParameters& params) {
	if (isInitialized()) {
		std::cout << "OpenBoundariesSimple::init was already initialized" << std::endl;
		return;
	}



	//init the surfaces
	if (params.simulation_config == 0) {
		S_boundary.setCylinder(Vector3d(0, 0, 0), 10, 1.5);
		S_fluidInterior.setCylinder(Vector3d(0, 0, 0), 10, S_boundary.getRadius() - data.particleRadius * 3);
		S_fluidSurface.setPlane(Vector3d(0, 1, 0), Vector3d(0, -1, 0));
	}
	else if (params.simulation_config == 1) {
		//S_boundary.setCylinder(Vector3d(0, 0, 0), 10, 1.5);
		//S_fluidInterior.setCylinder(Vector3d(0, 0, 0), 10, S_boundary.getradius() - data.particleRadius * 3);

		S_boundary.setCuboid(Vector3d(0, 0, 0), Vector3d(1, 10, 5));
		S_fluidInterior.setCuboid(Vector3d(0, 0, -1), Vector3d(1, 10, 1+S_boundary.getHalfLength().z - data.particleRadius * 5));
		
		S_fluidSurface.setPlane(Vector3d(0, 1, 0), Vector3d(0, -1, 0));
	}
	else {
		std::cout << "OpenBoundariesSimple::init no existing config detected" << std::endl;
		exit(5986);
	}

	//load the positions for the inflow
	Vector3d min_fluid_buffer;
	Vector3d max_fluid_buffer;
	SPH::UnifiedParticleSet* dummy = NULL;
	inflowPositionsSet = new SPH::UnifiedParticleSet();
	inflowPositionsSet->load_from_file(data.fluid_files_folder + "inflowPositionsSet_file.txt", false, &min_fluid_buffer, &max_fluid_buffer, false);
	allocate_and_copy_UnifiedParticleSet_vector_cuda(&dummy, inflowPositionsSet, 1);



	//we need to remove any particle that will not be part of the infow
	//the inflow is a single layer of particle near the boundary
	//so I can use S_fluidInterior for that
	if(true){
		int* outInt = SVS_CU::get()->count_invalid_position;
		*outInt = 0;

		//we have to reverse that surface here 
		S_fluidInterior.setReversedSurface(true);
		
		//clear the buffer used for tagging
		set_buffer_to_value<unsigned int>(inflowPositionsSet->neighborsDataSet->cell_id, TAG_UNTAGGED, inflowPositionsSet->numParticles);

		//find the particles to rmv
		//by limiting to the area near boundary
		{
			int numBlocks = calculateNumBlocks(inflowPositionsSet->numParticles);
			tag_outside_of_surface_kernel<false> << <numBlocks, BLOCKSIZE >> > (inflowPositionsSet->gpu_ptr, S_fluidInterior, outInt, TAG_REMOVAL);
			gpuErrchk(cudaDeviceSynchronize());
		}

		//and restricting it to the height decided by the inflow
		///TODO move this so that th inflow height can by dynamic through the simulation if desired
		{
			int numBlocks = calculateNumBlocks(inflowPositionsSet->numParticles);
			tag_outside_of_surface_kernel<false> << <numBlocks, BLOCKSIZE >> > (inflowPositionsSet->gpu_ptr, S_fluidSurface, outInt, TAG_REMOVAL);
			gpuErrchk(cudaDeviceSynchronize());
		}

		gpuErrchk(read_last_error_cuda("OpenBoundariesSimple::init before callign removal function: ", params.show_debug));

		//and remove the particle if needed
		int count_to_rmv = *outInt;
		if (count_to_rmv > 0) {
			//and remove the particles	
			//*
			remove_tagged_particles(inflowPositionsSet, inflowPositionsSet->neighborsDataSet->cell_id,
				inflowPositionsSet->neighborsDataSet->cell_id_sorted, count_to_rmv,false, false);
			//*/
		}

		//return the surface to normal
		S_fluidInterior.setReversedSurface(false);
	}




	//a test that replaces the fluid data with the inflow buffer data to see what is hapening
	if(false){
		data.fluid_data->updateActiveParticleNumber(inflowPositionsSet->numParticles);

		gpuErrchk(cudaMemcpy(data.fluid_data->mass, inflowPositionsSet->mass, 
			inflowPositionsSet->numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(data.fluid_data->pos, inflowPositionsSet->pos,
			inflowPositionsSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(data.fluid_data->vel, inflowPositionsSet->vel,
			inflowPositionsSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(data.fluid_data->color, inflowPositionsSet->color,
			inflowPositionsSet->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

	}


	gpuErrchk(read_last_error_cuda("OpenBoundariesSimple::init before computing constant contrib: ", params.show_debug));

	//compute the contribution of the boundaries and store it
	//no need or that a density based condition for adding the particle does not work...
	if (false){
		int numBlocks = calculateNumBlocks(inflowPositionsSet->numParticles);
		inflow_compute_and_store_constant_density_contribution_kernel << <numBlocks, BLOCKSIZE >> > (data,
			inflowPositionsSet->gpu_ptr);
		gpuErrchk(cudaDeviceSynchronize());
		
	}


	_isinitialized = true;

	gpuErrchk(read_last_error_cuda("OpenBoundariesSimple::init end: ", params.show_debug));

}

__global__ void inflow_with_predefined_positions_kernel(DFSPHCData data, SPH::UnifiedParticleSet* particleSet, 
	int* countAdded, RealCuda allowedNewDistance, BufferFluidSurface S_boundary){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }
	
	Vector3d p_i = particleSet->pos[i];

	RealCuda min_dist=100;
	
	//compute the contribution that change every time step
	{
		//i don't have the neighbors stored since the structure will only be used once every timestep
		ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, particleSet, i);

		//search the fluid neighbors
		UnifiedParticleSet* otherSet = data.fluid_data_cuda;
		ITER_NEIGHBORS_FROM_STRUCTURE(otherSet->neighborsDataSet, otherSet->pos,
			{
				RealCuda dist= (p_i - otherSet->pos[j]).norm();
				min_dist = MIN_MACRO_CUDA(min_dist, dist);
			}
		);


		//the dynamic bodies
		if (data.numDynamicBodies > 0) {
			for (int id_body = 0; id_body < data.numDynamicBodies; ++id_body) {
				otherSet = &data.vector_dynamic_bodies_data_cuda[id_body];
				ITER_NEIGHBORS_FROM_STRUCTURE(otherSet->neighborsDataSet, otherSet->pos,
					{
						RealCuda dist = (p_i - otherSet->pos[j]).norm();
						min_dist = MIN_MACRO_CUDA(min_dist, dist);
					}
				);
			}
		}
	}

	//writte the particle to the memory and initialize whatever is needed
	if (min_dist > allowedNewDistance) {
		//first compute the velocity of the new particle
		//do it with a pondered avg
		Vector3d v(0, 0, 0);
		bool add_particle = true;
		if(false){
			RealCuda sum_weights = 0;
			ITER_NEIGHBORS_INIT_FROM_STRUCTURE(data, particleSet, i);

			//iter over existing fluid particles
			UnifiedParticleSet* otherSet = data.fluid_data_cuda;
			ITER_NEIGHBORS_FROM_STRUCTURE(otherSet->neighborsDataSet, otherSet->pos,
				{
					RealCuda weight = KERNEL_W(data, p_i - otherSet->pos[j]);
					v += otherSet->vel[j] * weight;
					sum_weights += weight;
				}
			);
			if (sum_weights > 0) {
				v /= sum_weights;
			}

			//check that the velocity is not toward the boundary
			Vector3d surface_normal=S_boundary.getNormal(p_i);
			
			if (v.dot(surface_normal) < 0){
				add_particle = false;
			}
		}


		if(add_particle){
			//if all fine add the new particle

			//first get a unique index
			int id= atomicAdd(countAdded, 1);
			id += data.fluid_data_cuda->numParticles;

			//and writte the information 
			data.fluid_data_cuda->pos[id] = p_i;
			data.fluid_data_cuda->vel[id] = v;
			data.fluid_data_cuda->mass[id] = data.fluid_data_cuda->mass[0];
			data.fluid_data_cuda->kappa[id] = 0;
			data.fluid_data_cuda->kappaV[id] = 0;
			if (data.fluid_data_cuda->has_color_buffer) {
				data.fluid_data_cuda->color[id] = Vector3d(-1,-1,-1);
			}
		}


	}

	
}



__global__ void outflow_basic_kernel(DFSPHCData data, SPH::UnifiedParticleSet* particleSet,
	int* countToRmv, BufferFluidSurface S_fluidInterior, BufferFluidSurface S_fluidSurface) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particleSet->numParticles) { return; }

	Vector3d p_i = particleSet->pos[i];

	if (!S_fluidInterior.isinside(p_i)) 
	{
		if (!S_fluidSurface.isinside(p_i)) 
		{
			particleSet->neighborsDataSet->cell_id[i] = TAG_REMOVAL;
			atomicAdd(countToRmv, 1);
		}
	}

}



/*

*/


void OpenBoundariesSimple::applyOpenBoundary(DFSPHCData& data, OpenBoundariesSimpleInterface::ApplyParameters& params) {
	if (!isInitialized()) {
		std::cout << "OpenBoundariesSimple::applyOpenBoundary the structure need to be initialized before" << std::endl;
		return;
	}

	//first let's apply the inflow
	//for this version of the inflow, the strategy is to check at every positions of the inflow buffer
	//then check if there is enougth space for a new particle, and if there is compute the velocity of the new particle
	
	int* outInt = SVS_CU::get()->count_invalid_position;
	if(params.useInflow){
		if (params.allowedNewDistance <= 0) {
			std::cout << "OpenBoundariesSimple::applyOpenBoundary: an invalid min distance was spacified for inflow: " <<
				params.allowedNewDistance << std::endl;
			exit(1256);
		}


		//add more particle in case there might be a max near the curretn number
		if (data.fluid_data->numParticles > (data.fluid_data->numParticlesMax*0.75)) {
			data.fluid_data->changeMaxParticleNumber(data.fluid_data->numParticlesMax * 2);
		}

		data.fluid_data->initNeighborsSearchData(data, false, false);


		gpuErrchk(read_last_error_cuda("OpenBoundariesSimple::applyOpenBoundary before applying inflow: ", params.show_debug))


		*outInt = 0;
		{
			int numBlocks = calculateNumBlocks(inflowPositionsSet->numParticles);
			inflow_with_predefined_positions_kernel << <numBlocks, BLOCKSIZE >> > (data, inflowPositionsSet->gpu_ptr, 
				outInt, params.allowedNewDistance, S_boundary);
			gpuErrchk(cudaDeviceSynchronize());
		}
		int count_to_add = *outInt;


		gpuErrchk(read_last_error_cuda("OpenBoundariesSimple::applyOpenBoundary after applying inflow: ", params.show_debug))

		//if some particles have been added change the count
		if (count_to_add > 0) {
			data.fluid_data->updateActiveParticleNumber(data.fluid_data->numParticles + count_to_add);
		}
	}


	//next the outflow
	//for this version of the outflow I'll simply remove any particle too close from the boundary and above the desired fluid surface
	//clear the buffer used for tagging
	if (params.useOutflow) {
		set_buffer_to_value<unsigned int>(data.fluid_data->neighborsDataSet->cell_id, TAG_UNTAGGED, inflowPositionsSet->numParticles);
		*outInt = 0;
		{
			int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
			outflow_basic_kernel << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr,
				outInt, S_fluidInterior, S_fluidSurface);
			gpuErrchk(cudaDeviceSynchronize());
		}
		int count_to_remove = *outInt;

		if (count_to_remove > 0) {
			remove_tagged_particles(data.fluid_data, data.fluid_data->neighborsDataSet->cell_id,
				data.fluid_data->neighborsDataSet->cell_id_sorted, count_to_remove);
		}
	}

	gpuErrchk(read_last_error_cuda("OpenBoundariesSimple::applyOpenBoundary end: ", params.show_debug));
}