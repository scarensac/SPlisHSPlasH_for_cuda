#include "DynamicWindow.h"
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
	class DynamicWindow {
	public:
		bool _isInitialized; //mean we read data from disk
		bool _isDataTagged; //mean the data is now tagged to fit the desired conditions

		//those to variables are here to signal to the the stabilization phase what kind of tagging is already done
		bool _hasLoadingStoredActiveTagging; //mean the loading phase created a buffer the the activ particles tagged
		bool _hasLoadingStoredActiveNeighborsTagging; //mean the loading phase created a buffer the the neigbors of active particles tagged


		//the simulation boundary
		BufferFluidSurface S_simulation;
		//slightly smaller than S_simulation (exactly at 1 kernel radius fromit normaly
		BufferFluidSurface S_boundaryRange;

		//the fluid surface
		BufferFluidSurface S_fluid;

		//this one describe the interior of the fluid when the one layer of particles that is close 
		//to the boundaries are removed 
		BufferFluidSurface S_fluidInterior;
		
		//this one contains the surface that define the interior limit of the buffer
		//combine it to S_simulation_to obtain a surface that define the wholebuffer
		BufferFluidSurface S_bufferInterior;



		//donc worry about that it's an intermediary structure to fuse surfaces
		//but currently they only ontain on surface
		SurfaceAggregation S_simulation_aggr;
		SurfaceAggregation S_fluid_aggr;


		//this aggregation represent the existing fluid area
		//it is only set here to prevent having to allocate memory during execution
		//it HAS to contain 2 surfaces currently (maybe not if the system has been changed
		//the first should be S_fluidInterior pre displacement
		//and the second should be S_boundaryRange post displacement
		SurfaceAggregation SA_keptExistingFluidArea;


		int* outInt;
		RealCuda* outRealCuda;


		SPH::UnifiedParticleSet* backgroundFluidBufferSet;
		int count_potential_fluid;

		unsigned int* tag_array_with_existing_fluid;
		unsigned int tag_array_with_existing_fluid_size;

		//if this is equals to 0 after tagging this mean the tagging phase extracted them
		int count_high_density_tagged_in_potential;
		int count_high_density_tagged_in_air;
		int count_to_remove_in_existing_fluid;



		DynamicWindow() {
			_isInitialized = false;
			_isDataTagged = false;
			_hasLoadingStoredActiveTagging = false;
			_hasLoadingStoredActiveNeighborsTagging = false;

			cudaMallocManaged(&(outInt), sizeof(int));
			cudaMallocManaged(&(outRealCuda), sizeof(RealCuda));
			
			backgroundFluidBufferSet = NULL;
			count_potential_fluid = 0;
			count_high_density_tagged_in_potential = 0;
			count_high_density_tagged_in_air = 0;
			count_to_remove_in_existing_fluid = 0;

			tag_array_with_existing_fluid=NULL;
			tag_array_with_existing_fluid_size=0;
		}

		~DynamicWindow() {

		}

		static DynamicWindow& getStructure() {
			static DynamicWindow rfl;
			return rfl;
		}

		void clear();

		////!!! WARNING after this function is executed we must NEVER sort the particle data in the backgroundBuffer
		//an explanation for the air particle range
		//it is used to limit the amount of air particles that are kept 
		//-1: no restriction; 0: no air(don't use that...), 1: air that are neighbors to fluid, 2: air that are neigbors to air particles that are neighbors to fluid
		void init(DFSPHCData& data, DynamicWindowInterface::InitParameters& params);

		bool isInitialized() { return _isInitialized; }

		bool isDataTagged() { return _isDataTagged; }

		bool hasLoadingStoredActiveTagging() {	return _hasLoadingStoredActiveTagging;	};
		bool hasLoadingStoredActiveNeighborsTagging() {return _hasLoadingStoredActiveNeighborsTagging;}

		//ok here I'll test a system to initialize a volume of fluid from
		//a large wolume of fluid (IE a technique to iinit the fluid at rest)
		void tagDataToSurface(SPH::DFSPHCData& data, DynamicWindowInterface::TaggingParameters& params);

		//ok here I'll test a system to initialize a volume of fluid from
		//a large wolume of fluid (IE a technique to iinit the fluid at rest)
		//return the number of FLUID particles (it may be less than the number of loaded particle since there are air particles)
		int loadDataToSimulation(SPH::DFSPHCData& data, DynamicWindowInterface::LoadingParameters& params);

		//so this is a function that will be used to move around the particles in the fluid to 
		//improove the stability of the fluid when the first time step is ran
		//Warning this function will erase the current fluid data no mather what
		void stabilizeFluid(SPH::DFSPHCData& data, DynamicWindowInterface::StabilizationParameters& params);
	};
}

void DynamicWindowInterface::clear() {
	DynamicWindow::getStructure().clear();
}

void DynamicWindowInterface::init(DFSPHCData& data, InitParameters& params) {
	DynamicWindow::getStructure().init(data, params);
}

bool DynamicWindowInterface::isInitialized() {
	return DynamicWindow::getStructure().isInitialized();
}


void DynamicWindowInterface::initializeFluidToSurface(SPH::DFSPHCData& data, TaggingParameters& params, 
	LoadingParameters& params_loading) {
	
	if (!isInitialized()) {
		std::cout << "DynamicWindow::initializeFluidToSurface Loading impossible data was not initialized" << std::endl;
		return;
	}

	std::vector<std::string> timing_names{ "init","tag","load" };
	SPH::SegmentedTiming timings("DynamicWindowInterface::initializeFluidToSurface", timing_names, true);
	timings.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)



	timings.time_next_point();//time p1
	
	DynamicWindow::getStructure().tagDataToSurface(data,params);
	
	timings.time_next_point();//time p2
	
	if (params_loading.load_fluid) {
		int count_fluid = DynamicWindow::getStructure().loadDataToSimulation(data, params_loading);
	}
	
	timings.time_next_point();//time p3
	timings.end_step();//end point of the current step (if measuring avgs you need to call it at every end of the loop)
	timings.recap_timings();//writte timming to cout

	if (params.output_timming_information) {
		params.time_total = timings.getTimmingAvg(1);
	}

	//the idea is that I'll get the min max density here
	//since they are already computed it should be fine since this is outside of the timmings
	if (params.output_density_information) {
		RealCuda min_density = 10000;
		RealCuda max_density = 0;
		RealCuda avg_density = 0;
		UnifiedParticleSet* particleSet = DynamicWindow::getStructure().backgroundFluidBufferSet;
		int count = 0;
		for (int j = 0; j < DynamicWindow::getStructure().count_potential_fluid; ++j) {
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
		for (int j = 0; j < DynamicWindow::getStructure().count_potential_fluid; ++j) {
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

void DynamicWindowInterface::stabilizeFluid(SPH::DFSPHCData& data, DynamicWindowInterface::StabilizationParameters& params) {
	DynamicWindow::getStructure().stabilizeFluid(data, params);
}


void DynamicWindow::clear() {
	_isInitialized = false;
	_isDataTagged = false;
	_hasLoadingStoredActiveTagging = false;
	_hasLoadingStoredActiveNeighborsTagging = false;


	S_simulation.clear();
	S_fluid.clear();
	S_fluidInterior.clear();
	S_simulation_aggr.clear();
	S_fluid_aggr.clear();


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



void DynamicWindow::init(DFSPHCData& data, DynamicWindowInterface::InitParameters& params) {
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

	std::vector<std::string> timing_names{ "mesh load","void","load", "center background","restict to simulation","restrict distance to boundary particles","tag to fluid"," sort by tagging" };
	SPH::SegmentedTiming timings("fluid loader init", timing_names, true);
	timings.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)

	std::string backgroundFileFolder = "folderDynamicWindow/";
	std::string backgroundFileName = "";


	//init the surfaces
	if (params.simulation_config == 0) {
		//cylinder with 1.5m radius

		S_simulation.setCylinder(Vector3d(0, 0, 0), 10, 1.5);

		if ((S_simulation.getRadius() - params.max_allowed_displacement) < data.getKernelRadius()*4) {
			std::cout << "the simulation area is too small relative to the required buffers size for the dynamix window area/buffer_size/reuired_min_diff" << 
				S_simulation.getRadius() << " / " << params.max_allowed_displacement << " / " << data.getKernelRadius() * 4 << " / " << std::endl;
			gpuErrchk(cudaError_t::cudaErrorUnknown);
		}

		//those are mostly used to remove some of the existing fluid
		S_boundaryRange.setCylinder(Vector3d(0, 0, 0), 10, S_simulation.getRadius() - (data.getKernelRadius()*1.5));
		S_fluidInterior.setCylinder(Vector3d(0, 0, 0), 10, S_simulation.getRadius() - data.particleRadius * 3);

		//those are specific the the bancground buffer particles
		S_bufferInterior.setCylinder(Vector3d(0, 0, 0), 10, 
			S_fluidInterior.getRadius() - (params.max_allowed_displacement + data.getKernelRadius()*1.5));
		S_bufferInterior.setReversedSurface(true);

		
		S_fluid.setPlane(Vector3d(0, 1, 0), Vector3d(0, -1, 0));


		SA_keptExistingFluidArea.addSurface(S_fluidInterior);
		SA_keptExistingFluidArea.addSurface(S_boundaryRange);

		backgroundFileName = "dynamicWindow_cylinder.txt";

	}
	else if (params.simulation_config == 1) {

		//the 1.5m sphere config

		S_simulation.setSphere(Vector3d(0, 1, 0), 1.5);
		//S_fluid.setCuboid(Vector3d(0, 1.25, 0), Vector3d(2, 2, 2));
		S_fluid.setPlane(Vector3d(0, 1, 0), Vector3d(0, -1, 0));


		if ((S_simulation.getRadius() - params.max_allowed_displacement) < data.getKernelRadius() * 4) {
			std::cout << "the simulation area is too small relative to the required buffers size for the dynamix window area/buffer_size/reuired_min_diff" <<
				S_simulation.getRadius() << " / " << params.max_allowed_displacement << " / " << data.getKernelRadius() * 4 << " / " << std::endl;
			gpuErrchk(cudaError_t::cudaErrorUnknown);
		}

		//those are mostly used to remove some of the existing fluid
		S_boundaryRange.setSphere(S_simulation.getCenter(), S_simulation.getRadius() - (data.getKernelRadius()*1.5));
		S_fluidInterior.setSphere(S_simulation.getCenter(), S_simulation.getRadius() - data.particleRadius * 3);

		//those are specific the the bancground buffer particles
		S_bufferInterior.setSphere(S_fluidInterior.getCenter(),
			S_fluidInterior.getRadius() - (params.max_allowed_displacement + data.getKernelRadius()*1.5));
		S_bufferInterior.setReversedSurface(true);




		SA_keptExistingFluidArea.addSurface(S_fluidInterior);
		SA_keptExistingFluidArea.addSurface(S_boundaryRange);


		backgroundFileName = "dynamicWindow_sphere.txt";

	}
	else if (params.simulation_config == 2) {
		//2.5m cylinder

		S_simulation.setCylinder(Vector3d(0, 0, 0), 10, 2.5);

		if ((S_simulation.getRadius() - params.max_allowed_displacement) < data.getKernelRadius() * 4) {
			std::cout << "the simulation area is too small relative to the required buffers size for the dynamix window area/buffer_size/reuired_min_diff" <<
				S_simulation.getRadius() << " / " << params.max_allowed_displacement << " / " << data.getKernelRadius() * 4 << " / " << std::endl;
			gpuErrchk(cudaError_t::cudaErrorUnknown);
		}

		//those are mostly used to remove some of the existing fluid
		S_boundaryRange.setCylinder(Vector3d(0, 0, 0), 10, S_simulation.getRadius() - (data.getKernelRadius()*1.5));
		S_fluidInterior.setCylinder(Vector3d(0, 0, 0), 10, S_simulation.getRadius() - data.particleRadius * 3);

		//those are specific the the bancground buffer particles
		S_bufferInterior.setCylinder(Vector3d(0, 0, 0), 10,
			S_simulation.getRadius() - (params.max_allowed_displacement + data.getKernelRadius()*1.5));
		S_bufferInterior.setReversedSurface(true);


		S_fluid.setPlane(Vector3d(0, 1, 0), Vector3d(0, -1, 0));


		SA_keptExistingFluidArea.addSurface(S_fluidInterior);
		SA_keptExistingFluidArea.addSurface(S_boundaryRange);

		backgroundFileName = "dynamicWindow_cylinder_r2_5m.txt";

	}
	else if (params.simulation_config == 3) {

		//the star shaped border here
		//the parameters for the star are 5 points, re=3.5/2, ri=2/2, direction=(0,0,1), h=5

		RealCuda ri = 2 / 2.0;
		RealCuda re = 3.5 / 2.0;
		int pointCount = 5;
		Vector3d firstPointDir = Vector3d(0, 0, 1);
		S_simulation.setStar(Vector3d(0, 0, 0), 10, re, ri, pointCount, firstPointDir);

		if ((S_simulation.getRadiusInternal() - params.max_allowed_displacement) < data.getKernelRadius() * 4) {
			std::cout << "the simulation area is too small relative to the required buffers size for the dynamix window area/buffer_size/reuired_min_diff" <<
				S_simulation.getRadiusInternal() << " / " << params.max_allowed_displacement << " / " << data.getKernelRadius() * 4 << " / " << std::endl;
			gpuErrchk(cudaError_t::cudaErrorUnknown);
		}

		//those are mostly used to remove some of the existing fluid
		S_boundaryRange.setStar(Vector3d(0, 0, 0), 10, re - (data.getKernelRadius()*1.5), 
			ri - (data.getKernelRadius()*1.5), pointCount, firstPointDir);

		S_fluidInterior.setStar(Vector3d(0, 0, 0), 10, re - data.particleRadius * 3,
			ri - data.particleRadius * 3, pointCount, firstPointDir);

		//those are specific the the bancground buffer particles

		S_bufferInterior.setStar(Vector3d(0, 0, 0), 10, re - (params.max_allowed_displacement + data.getKernelRadius()*1.5), 
			ri - (params.max_allowed_displacement + data.getKernelRadius()*1.5), pointCount, firstPointDir);
		S_bufferInterior.setReversedSurface(true);


		S_fluid.setPlane(Vector3d(0, 1, 0), Vector3d(0, -1, 0));


		SA_keptExistingFluidArea.addSurface(S_fluidInterior);
		SA_keptExistingFluidArea.addSurface(S_boundaryRange);

		backgroundFileName = "dynamicWindow_star_re1_75_ri1.txt";

	}
	else if (params.simulation_config == 4) {

		//the 2.5m sphere config

		S_simulation.setSphere(Vector3d(0, 1, 0), 2.5);
		//S_fluid.setCuboid(Vector3d(0, 1.25, 0), Vector3d(2, 2, 2));
		S_fluid.setPlane(Vector3d(0, 1, 0), Vector3d(0, -1, 0));


		if ((S_simulation.getRadius() - params.max_allowed_displacement) < data.getKernelRadius() * 4) {
			std::cout << "the simulation area is too small relative to the required buffers size for the dynamix window area/buffer_size/reuired_min_diff" <<
				S_simulation.getRadius() << " / " << params.max_allowed_displacement << " / " << data.getKernelRadius() * 4 << " / " << std::endl;
			gpuErrchk(cudaError_t::cudaErrorUnknown);
		}

		//those are mostly used to remove some of the existing fluid
		S_boundaryRange.setSphere(S_simulation.getCenter(), S_simulation.getRadius() - (data.getKernelRadius()*1.5));
		S_fluidInterior.setSphere(S_simulation.getCenter(), S_simulation.getRadius() - data.particleRadius * 3);

		//those are specific the the bancground buffer particles
		S_bufferInterior.setSphere(S_fluidInterior.getCenter(),
			S_fluidInterior.getRadius() - (params.max_allowed_displacement + data.getKernelRadius()*1.5));
		S_bufferInterior.setReversedSurface(true);




		SA_keptExistingFluidArea.addSurface(S_fluidInterior);
		SA_keptExistingFluidArea.addSurface(S_boundaryRange);


		backgroundFileName = "dynamicWindow_sphere_2_5m.txt";

	}
	else if (params.simulation_config == 5) {

		//the star shaped border here
		//the parameters for the star are 5 points, re=3.5, ri=2, direction=(0,0,1), h=5

		RealCuda ri = 2 ;
		RealCuda re = 3.5 ;
		int pointCount = 5;
		Vector3d firstPointDir = Vector3d(0, 0, 1);
		S_simulation.setStar(Vector3d(0, 0, 0), 10, re, ri, pointCount, firstPointDir);

		if ((S_simulation.getRadiusInternal() - params.max_allowed_displacement) < data.getKernelRadius() * 4) {
			std::cout << "the simulation area is too small relative to the required buffers size for the dynamix window area/buffer_size/reuired_min_diff" <<
				S_simulation.getRadiusInternal() << " / " << params.max_allowed_displacement << " / " << data.getKernelRadius() * 4 << " / " << std::endl;
			gpuErrchk(cudaError_t::cudaErrorUnknown);
		}

		//those are mostly used to remove some of the existing fluid
		S_boundaryRange.setStar(Vector3d(0, 0, 0), 10, re - (data.getKernelRadius()*1.5),
			ri - (data.getKernelRadius()*1.5), pointCount, firstPointDir);

		S_fluidInterior.setStar(Vector3d(0, 0, 0), 10, re - data.particleRadius * 3,
			ri - data.particleRadius * 3, pointCount, firstPointDir);

		//those are specific the the bancground buffer particles

		S_bufferInterior.setStar(Vector3d(0, 0, 0), 10, re - (params.max_allowed_displacement + data.getKernelRadius()*1.5),
			ri - (params.max_allowed_displacement + data.getKernelRadius()*1.5), pointCount, firstPointDir);
		S_bufferInterior.setReversedSurface(true);


		S_fluid.setPlane(Vector3d(0, 1, 0), Vector3d(0, -1, 0));


		SA_keptExistingFluidArea.addSurface(S_fluidInterior);
		SA_keptExistingFluidArea.addSurface(S_boundaryRange);

		backgroundFileName = "dynamicWindow_star_re3_5_ri2.txt";

	}
	else {
		std::cout << "DynamicWindow::init no existing config detected(requested config) : " <<params.simulation_config<<  std::endl;
		gpuErrchk(cudaError_t::cudaErrorUnknown);
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
		backgroundFluidBufferSet->load_from_file(data.fluid_files_folder + backgroundFileFolder +backgroundFileName,
			false, &min_fluid_buffer, &max_fluid_buffer, false);
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

	timings.time_next_point();//time 


	//now I need to apply a restriction on the domain to limit it to the current simulated domain
	//so tag the particles
	*outInt = 0;
	if (air_particles_restriction > 0) {
		if (params.show_debug) {
			std::cout << "with added air restriction" << std::endl;
		}

		//no need for an offset on the boundary since this system use a fluid as rest that fit the boundary as background
		RealCuda offset_simu_space = 0;
		RealCuda air_particle_conservation_distance = -1.0f*air_particles_restriction * data.getKernelRadius();
		
		//essencially since I load a fluid that already fit the boundaries this will only restrict to the air
		{
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			surface_restrict_particleset_kernel<< <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_simulation_aggr, offset_simu_space, 
				S_fluid, air_particle_conservation_distance, outInt);
			gpuErrchk(cudaDeviceSynchronize());
		}

		if (params.show_debug) {
			UnifiedParticleSet* tempSet = backgroundFluidBufferSet;
			//for debug purposes check the numbers
			std::cout << "check after resttriction to boundaries + fluid+air" << std::endl;

			{
				int tag = TAG_REMOVAL;
				int numBlocks = calculateNumBlocks(tempSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
		}

		if (params.show_debug) {
			std::cout << "Restricting to fluid height+air existing(count tagged): " << backgroundFluidBufferSet->numParticles <<
				" (" << *outInt << ")" << std::endl;
		}

		//remove the interior since I'll keep the existing fluid
		if (params.show_debug) {
			std::cout << "s_buffer_interior: " << S_bufferInterior.toString() << std::endl;
		}
		
		{
			int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
			tag_outside_of_surface_kernel<false> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, S_bufferInterior, outInt, TAG_REMOVAL);
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
		std::cout << "you should not be reaching here, throwing a cuda error a easysly know the line in the code" << std::endl;
		gpuErrchk(cudaError_t::cudaErrorUnknown);
	}

	int count_to_rmv = *outInt;

	if (params.show_debug) {
		UnifiedParticleSet* tempSet = backgroundFluidBufferSet;
		//for debug purposes check the numbers

		{
			int tag = TAG_REMOVAL;
			int numBlocks = calculateNumBlocks(tempSet->numParticles);
			*(SVS_CU::get()->tagged_particles_count) = 0;
			count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
			gpuErrchk(cudaDeviceSynchronize());

			std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
		}
	}



	//and remove the particles	
	remove_tagged_particles(backgroundFluidBufferSet, backgroundFluidBufferSet->neighborsDataSet->cell_id,
		backgroundFluidBufferSet->neighborsDataSet->cell_id_sorted, count_to_rmv);

	if (params.show_debug) {
		std::cout << "Restricting to simulation area count remaining(count removed): " << backgroundFluidBufferSet->numParticles <<
			" (" << count_to_rmv << ")" << std::endl;
	}

	timings.time_next_point();//time 



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

	//a test that replaces the fluid data with the inflow buffer data to see what is hapening
	if (false) {
		UnifiedParticleSet* setToLoad = backgroundFluidBufferSet;
		data.fluid_data->updateActiveParticleNumber(setToLoad->numParticles);

		gpuErrchk(cudaMemcpy(data.fluid_data->mass, setToLoad->mass,
			setToLoad->numParticles * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(data.fluid_data->pos, setToLoad->pos,
			setToLoad->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(data.fluid_data->vel, setToLoad->vel,
			setToLoad->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(data.fluid_data->color, setToLoad->color,
			setToLoad->numParticles * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

	}


	gpuErrchk(read_last_error_cuda("check cuda error end init ", params.show_debug));
	
}


void DynamicWindow::tagDataToSurface(SPH::DFSPHCData& data, DynamicWindowInterface::TaggingParameters& params) {
	if (!isInitialized()) {
		std::cout << "DynamicWindow::tagDataToSurface tagging impossible data was not initialized" << std::endl;
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
	

	//*
	std::vector<std::string> timing_names{"resize_fluid_data_buffer_if_there_is_a_risk", "displace_boundaries_and_surfaces", 
		"init_neighbors_fluid","tag_existing_fluid_to_remove",
		"neighbor","void","tagging","void","constant density based extraction",
		"loop","count_tagged_for_removal","void","cleartag" };
	SPH::SegmentedTiming timings("tag data", timing_names, true);
	timings.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)



	int count_existing_fluid_particles = data.fluid_data->numParticles;
	if (data.fluid_data->numParticlesMax < (count_potential_fluid + count_existing_fluid_particles)) {
		data.fluid_data->changeMaxParticleNumber((count_potential_fluid + count_existing_fluid_particles) * 2);
	}
	timings.time_next_point();//time 

	//so we need to remove part of the existing fluid
	{
		//save the surface that define the particle layer next to the boundaries in the aggregation of existing fluid area
		SA_keptExistingFluidArea.setSurface(0, S_fluidInterior);


		//also move the simulation window here
		if (!params.displacement.isZero()) {
			data.dynamicWindowTotalDisplacement += params.displacement;

			//move the boundaries particles
			{
				int numBlocks = calculateNumBlocks(data.boundaries_data->numParticles);
				apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (data.boundaries_data->pos,
					params.displacement, data.boundaries_data->numParticles);
				gpuErrchk(cudaDeviceSynchronize());
			}
			//and don't forget to update the neighbors storage
			data.boundaries_data->initNeighborsSearchData(data, false, false);

			//also move the background buffer for an easier integration
			{
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				apply_delta_to_buffer_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->pos,
					params.displacement, backgroundFluidBufferSet->numParticles);
				gpuErrchk(cudaDeviceSynchronize());
			}

			//move the surfaces
			S_simulation.move(params.displacement);
			S_boundaryRange.move(params.displacement);
			S_fluid.move(params.displacement);
			S_fluidInterior.move(params.displacement);
			S_bufferInterior.move(params.displacement);
		}


		timings.time_next_point();//time 

		//init fluid neighbor structure
		data.fluid_data->initNeighborsSearchData(data, false);


		timings.time_next_point();//time 

		//save the surface that define the particles that are within effective range of the boundary post displacement
		SA_keptExistingFluidArea.setSurface(1, S_boundaryRange);

		//now that the surfaces have been moved
		//I will remove the fluid particles that are too close to the boundary
		//two ways to do it, simply tag the neighbors of the boundary, but this may take time
		//or define a surface that is at 1 kernelradius of the boundary (should be way faster)
		//here is the surface based approach
		//also to go faster I fused that with the removal of particle too close to the boundary pre movement (see coment below)
		//				first tag the layer that is extremely close to the boundary 
		//				I remove them because they have a specific particle arrangment
		set_buffer_to_value<unsigned int>(data.fluid_data->neighborsDataSet->cell_id, TAG_UNTAGGED, data.fluid_data->numParticles);
		*outInt = 0;
		{
			int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
			tag_outside_of_surface_kernel<false> << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, SA_keptExistingFluidArea, outInt, TAG_REMOVAL);
			gpuErrchk(cudaDeviceSynchronize());
		}

		//and store the count that have been taggded so I don't have to recompute it...
		count_to_remove_in_existing_fluid = *outInt;

		if (params.show_debug) {
			std::cout << "nbr existing fluid to remove total: " << count_to_remove_in_existing_fluid << std::endl;
		}


		gpuErrchk(read_last_error_cuda("check error after displacement and existing fluid tagging: ", params.show_debug));
	}


	timings.time_next_point();//time 

	//*/
	//reinitialise the neighbor structure (might be able to delete it if I never use it
	//though I will most likely use it
	//backgroundFluidBufferSet->initNeighborsSearchData(data, false);
	backgroundFluidBufferSet->initAndStoreNeighbors(data, false);



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

		if (params.show_debug) {
			std::cout << "tag info pre actual tagging" << std::endl;
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
		}

		//for the dynamic window I also do not want that the buffer particles that are above existing fluid may be added to the simulation
		//however they may be important to have a complete neighborhood at some point
		//so the solution is to tag as air any buffer aprticle that is in the area that correspond to the existing fluid I'm keeping 
		//this area is the intersaction of S_fluidInterior predisplacement and S_boundaryRange post displacement
		//since the structure I have tag the particles on the exterior of the area I have to reverse it
		if (true) {
			SA_keptExistingFluidArea.setReversedSurface(true);
		
			{
				int numBlocks = calculateNumBlocks(backgroundFluidBufferSet->numParticles);
				tag_outside_of_surface_kernel<false> << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->gpu_ptr, 
					SA_keptExistingFluidArea, NULL, TAG_AIR);
				gpuErrchk(cudaDeviceSynchronize());
			}

			//restore the surface aggregation to its normal state
			SA_keptExistingFluidArea.setReversedSurface(false);
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
		bool tag_neigbors = false;
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

			std::cout << "tag info post actual tagging" << std::endl;
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

		//I need to convert all the remaining air tagged particles that were initialy potentially fluid particles
		//those are easy to identify that are all air tagged particle with an id< to count_potential_fluids
		if(true){
			int numBlocks = calculateNumBlocks(count_potential_fluid);
			convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (backgroundFluidBufferSet->neighborsDataSet->cell_id,
				count_potential_fluid, TAG_AIR, TAG_REMOVAL);
			gpuErrchk(cudaDeviceSynchronize());
		}


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


int DynamicWindow::loadDataToSimulation(SPH::DFSPHCData& data, DynamicWindowInterface::LoadingParameters& params) {
	if (!isInitialized()) {
		std::cout << "DynamicWindow::loadDataToSimulation Loading impossible data was not initialized" << std::endl;
		return -1;
	}

	if (!isDataTagged()) {
		std::cout << "!!!!!!!!!!! DynamicWindow::loadDataToSimulation you are loading untagged data !!!!!!!!!!!" << std::endl;
		return -1;
	}

	if (!params.load_fluid) {
		return data.fluid_data->numParticles;
	}

	//reset the tagging flags
	_hasLoadingStoredActiveTagging = false;
	_hasLoadingStoredActiveNeighborsTagging = false;



	gpuErrchk(read_last_error_cuda("check error before loading: ", params.show_debug));

	std::vector<std::string> timing_names{ "copy","init_boundary_neighbors","alloc_tagging_storage","init_neighbors_and_recover_active_tag" ,
		"tagging_active_neighbors","count_active","void" };
	SPH::SegmentedTiming timings("DynamicWindow::loadDataToSimulation", timing_names, true);
	timings.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)

	
	int nbr_fluid_particles;

	if (params.show_debug) {
		std::cout << "count nbr particle in buffers existing/fluid/air " <<
			data.fluid_data->numParticles << "  " <<
			count_potential_fluid << "  " <<
			backgroundFluidBufferSet->numParticles-count_potential_fluid << "  " << std::endl;

		std::cout << "count to rmv in existing/fluid/air " <<
			count_to_remove_in_existing_fluid << "  " <<
			count_high_density_tagged_in_potential << "  " <<
			count_high_density_tagged_in_air << std::endl;
	}

	if (params.keep_existing_fluid) {
		if (params.keep_air_particles) {
			throw("DynamicWindow::loadDataToSimulation keeping the air particles while keeping existing fluid is not currently allowed");
		}

	
	

		//just copy all the potential values to the fluid
		//fm fucking l
		//if this shit modifies the maximum number of particles I loose the fucking index
		int count_existing_fluid_particles = data.fluid_data->numParticles;
		if (data.fluid_data->numParticlesMax < (count_potential_fluid + count_existing_fluid_particles)) {
			std::cout << "ok if this happens I'm fucked because I loose the tagging of the fluid..." <<
				"  handling it with a buffer would be an absurd loss of performance so I do a special rezize before tagging"<<std::endl;
			gpuErrchk(cudaError_t::cudaErrorUnknown);
		}
		data.fluid_data->updateActiveParticleNumber(count_potential_fluid + count_existing_fluid_particles);

		//let's set all particles from the buffer to a random color to see where they are placed
		if (true && params.show_debug) {
			set_buffer_to_value<Vector3d>(backgroundFluidBufferSet->color, Vector3d(0,1,0), backgroundFluidBufferSet->numParticles);
		}

		//set the buffer velocities to 0 to be sure
		set_buffer_to_value<Vector3d>(backgroundFluidBufferSet->vel, Vector3d(0, 0, 0), count_existing_fluid_particles);

		gpuErrchk(cudaMemcpy(data.fluid_data->mass + count_existing_fluid_particles, backgroundFluidBufferSet->mass,
			count_potential_fluid * sizeof(RealCuda), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(data.fluid_data->pos + count_existing_fluid_particles, backgroundFluidBufferSet->pos,
			count_potential_fluid * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(data.fluid_data->vel + count_existing_fluid_particles, backgroundFluidBufferSet->vel,
			count_potential_fluid * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(data.fluid_data->color + count_existing_fluid_particles, backgroundFluidBufferSet->color,
			count_potential_fluid * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

		//the main problem is that I have ot extract the particles to rmv
		//however now I don't have a cell index for all existing particles so I have to first build it
		if (params.show_debug) {
			UnifiedParticleSet* tempSet = data.fluid_data;
			//for debug purposes check the numbers
			{
				int tag = TAG_ACTIVE;
				int numBlocks = calculateNumBlocks(tempSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
			{
				int tag = TAG_ACTIVE_NEIGHBORS;
				int numBlocks = calculateNumBlocks(tempSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
			{
				int tag = TAG_AIR;
				int numBlocks = calculateNumBlocks(tempSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}

			{
				int tag = TAG_UNTAGGED;
				int numBlocks = calculateNumBlocks(tempSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}

			{
				int tag = TAG_REMOVAL;
				int numBlocks = calculateNumBlocks(tempSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
		}




		//so let's fuse the two trustures tagging values
		gpuErrchk(cudaMemcpy(data.fluid_data->neighborsDataSet->cell_id + count_existing_fluid_particles, 
			backgroundFluidBufferSet->neighborsDataSet->cell_id,
			count_potential_fluid * sizeof(unsigned int), cudaMemcpyDeviceToDevice));


		if (params.show_debug) {
			UnifiedParticleSet* tempSet = data.fluid_data;
			//for debug purposes check the numbers
			{
				int tag = TAG_ACTIVE;
				int numBlocks = calculateNumBlocks(tempSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
			{
				int tag = TAG_ACTIVE_NEIGHBORS;
				int numBlocks = calculateNumBlocks(tempSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
			{
				int tag = TAG_AIR;
				int numBlocks = calculateNumBlocks(tempSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}

			{
				int tag = TAG_UNTAGGED;
				int numBlocks = calculateNumBlocks(tempSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}

			{
				int tag = TAG_REMOVAL;
				int numBlocks = calculateNumBlocks(tempSet->numParticles);
				*(SVS_CU::get()->tagged_particles_count) = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
			}
		}



		//and now remove the partifcles that were tagged for the fitting
		int count_to_rmv = count_high_density_tagged_in_potential + count_to_remove_in_existing_fluid;

		remove_tagged_particles(data.fluid_data, data.fluid_data->neighborsDataSet->cell_id,
			data.fluid_data->neighborsDataSet->cell_id_sorted, count_to_rmv);


		//clearing the warmstart values is necessary
		set_buffer_to_value<RealCuda>(data.fluid_data->kappa, 0, data.fluid_data->numParticles);
		set_buffer_to_value<RealCuda>(data.fluid_data->kappaV, 0, data.fluid_data->numParticles);

		

		nbr_fluid_particles = data.fluid_data->numParticles;

		timings.time_next_point();//time

		if (params.show_debug) {
			std::cout << "count particle after fusion actual/expected" <<
				data.fluid_data->numParticles << "  " <<
				count_existing_fluid_particles+ count_potential_fluid -count_to_rmv << "  " << std::endl;
		}
		//and now I can update the offset that is used for the neighbors grid
		//for that I can simply look at the total displacement that the dynamic windo has ever made
		//also have to update the bounding box btw maybe I should simply redo the init of the offset and bounding box
		//but it would take way more time that those simples add
		//note: updating that offset means I also have to update the neighbors structure of the boundaries
		data.gridOffset = data.gridOffsetAfterLastLoading- ((data.dynamicWindowTotalDisplacement / data.getKernelRadius()).toFloor());
		data.boundingBoxMin = data.boundingBoxMinAfterLastLoading+ data.dynamicWindowTotalDisplacement;
		data.boundingBoxMax = data.boundingBoxMaxAfterLastLoading+ data.dynamicWindowTotalDisplacement;


		if (params.show_debug) {
			std::cout << "offsets after displacement offset: " << data.gridOffset.toString() << std::endl;
		}

		//since the offset was changed I need to reupdate the oundaries neighbor structure
		data.boundaries_data->initNeighborsSearchData(data, false, false);
		

		timings.time_next_point();//time

		//ok so when keeping existing fluid i can't simply stock it in the background structure
		if (params.set_up_tagging) {
			if (params.show_debug) {
				std::cout << "DynamicWindow::loadDataToSimulation setting up the tagging for stabilization step" << std::endl;
			}

			//I have to allocate dedicated memory since I can't store it into the background arrays because ofthe preexisting particles
			if (tag_array_with_existing_fluid_size < data.fluid_data->numParticles) {
				CUDA_FREE_PTR(tag_array_with_existing_fluid);
				std::cout << "DynamicWindow::loadDataToSimulation have to realoc tagging storage (existing_size/new): " <<
					tag_array_with_existing_fluid_size << "  " << data.fluid_data->numParticles << std::endl;
			}

			if (tag_array_with_existing_fluid == NULL) {
				cudaMallocManaged(&(tag_array_with_existing_fluid), data.fluid_data->numParticlesMax * sizeof(unsigned int));
				tag_array_with_existing_fluid_size = data.fluid_data->numParticlesMax;
			}


			timings.time_next_point();//time


			//so the main difficulty is that we have to also simulate the border with the fluid
			//although since the cell id sorted currently old the previously used tag we can reuse it
			//so first we recover the tagging to recover the active tagging
			gpuErrchk(cudaMemcpy(tag_array_with_existing_fluid,
				data.fluid_data->neighborsDataSet->cell_id_sorted,
				data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

			//first init the neighbor structure
			//although if I don't need to tag the active neighbors, I don't actulayy need to init the neighbros
			if (params.tag_active_neigbors) {
				data.fluid_data->initNeighborsSearchData(data, false);
			}

			//load the backuped tag
			gpuErrchk(cudaMemcpy(data.fluid_data->neighborsDataSet->cell_id,
				tag_array_with_existing_fluid,
				data.fluid_data->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice));


			timings.time_next_point();//time


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
			if (params.tag_active_neigbors) {
				//data.fluid_data->resetColor();
				if (params.tag_active_neigbors_use_repetition_approach) {
					//the direct approach
					RealCuda tagging_distance = data.getKernelRadius() * params.neighbors_tagging_distance_coef;
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
					tag_neighbors_of_tagged_kernel<false, false, false> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, tagging_distance,
						TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS);
					gpuErrchk(cudaDeviceSynchronize());
				}
				else {
					//the repetition approach
					//the repetitive approach simply tg hte normal neighborhood, then tag it's negihbors hood and repeat as much as necessary
					//first convert the distance into a integer
					unsigned int count_repetition = static_cast<unsigned int>(params.neighbors_tagging_distance_coef);
					if ((params.neighbors_tagging_distance_coef - count_repetition) > 0.01) {
						count_repetition++;
					}

					//do the acutal tagging
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
					unsigned int tag_to_tag_neighbors_of = TAG_ACTIVE;
					unsigned int tag_to_set = TAG_1;
					unsigned int tag_to_set_next = TAG_2;
					for (int k = 0; k < count_repetition; ++k) {

						tag_neighbors_of_tagged_kernel<true, false> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr,
							tag_to_tag_neighbors_of, TAG_1);
						gpuErrchk(cudaDeviceSynchronize());

						//after the second pass I need to tart to convert the tags into the neighbors tagging so I cna reuse the tag
						if (k > 0) {
							convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag_to_tag_neighbors_of, TAG_ACTIVE_NEIGHBORS);
							gpuErrchk(cudaDeviceSynchronize());
						}

						//progress the tagging
						tag_to_tag_neighbors_of = tag_to_set;
						unsigned int temp = tag_to_set;
						tag_to_set = tag_to_set_next;
						tag_to_set_next = tag_to_set;
					}

					//and at the end I need to convert the tag of the ones that were tagged in the last pass
					convert_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag_to_tag_neighbors_of, TAG_ACTIVE_NEIGHBORS);
					gpuErrchk(cudaDeviceSynchronize());
				}
			
				_hasLoadingStoredActiveNeighborsTagging = true;
			}


			if (params.show_debug) {
				int tag = TAG_ACTIVE_NEIGHBORS;
				int* count_tagged_candidates = (SVS_CU::get()->tagged_particles_count);
				int* count_tagged_other = SVS_CU::get()->count_created_particles;
				*count_tagged_candidates = 0;
				*count_tagged_other = 0;
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, tag, count_tagged_other);
				gpuErrchk(cudaDeviceSynchronize());

				std::cout << "tag: " << tag << "   count tagged (candidate/others): " << *count_tagged_candidates << "   " << *count_tagged_other << std::endl;
			}


			timings.time_next_point();//time

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
				int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
				int* outInt = SVS_CU::get()->tagged_particles_count;
				int* outInt2 = SVS_CU::get()->count_invalid_position;
				*outInt = 0;
				*outInt2 = 0;
				count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_ACTIVE, outInt);
				if (hasLoadingStoredActiveNeighborsTagging()) {
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_ACTIVE_NEIGHBORS, outInt2);
				}
				gpuErrchk(cudaDeviceSynchronize());
				
				data.count_active = *outInt;
				data.count_active_neighbors = *outInt2;


				if (params.show_debug) {
					std::cout << "count active // active neighbors: " << data.count_active << " // " << data.count_active_neighbors << std::endl;
				}

			}


			timings.time_next_point();//time

			//set a bool to indicate the fullowing system they don't have to recompute the tagging
			_hasLoadingStoredActiveTagging = true;

			if (params.show_debug) {
				std::cout << "tagging state at the end of loadign function" << std::endl;

				UnifiedParticleSet* tempSet = data.fluid_data;
				//for debug purposes check the numbers
				{
					int tag = TAG_ACTIVE;
					int numBlocks = calculateNumBlocks(tempSet->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
				{
					int tag = TAG_ACTIVE_NEIGHBORS;
					int numBlocks = calculateNumBlocks(tempSet->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
				{
					int tag = TAG_AIR;
					int numBlocks = calculateNumBlocks(tempSet->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}

				{
					int tag = TAG_UNTAGGED;
					int numBlocks = calculateNumBlocks(tempSet->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}

				{
					int tag = TAG_REMOVAL;
					int numBlocks = calculateNumBlocks(tempSet->numParticles);
					*(SVS_CU::get()->tagged_particles_count) = 0;
					count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (tempSet->gpu_ptr, tag, SVS_CU::get()->tagged_particles_count);
					gpuErrchk(cudaDeviceSynchronize());

					std::cout << "tag: " << tag << "   count tagged: " << *(SVS_CU::get()->tagged_particles_count) << std::endl;
				}
			}

		}
		else {
			//just some lines so he timer don't break
			timings.time_next_point();//time p2
			timings.time_next_point();//time p2
			timings.time_next_point();//time p2
			timings.time_next_point();//time p2
		}
		
	}
	else {
		throw("DynamicWindow::loadDataToSimulation the dynamic windowsystem HAS to keep the existing fluid");
	}



	timings.time_next_point();//time p3
	timings.end_step();//end point of the current step (if measuring avgs you need to call it at every end of the loop)
	timings.recap_timings();//writte timming to cout

	return nbr_fluid_particles;
}


template<class T> __global__ void cuda_applyFactorToTaggedParticles_kernel(T* buff, T value, unsigned int buff_size,
	unsigned int* tag_array, unsigned int tag) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= buff_size) { return; }

	if (tag_array[i] == tag) {
		buff[i] *= value;
	}
}
template __global__ void cuda_applyFactorToTaggedParticles_kernel<Vector3d>(Vector3d* buff, Vector3d value, unsigned int buff_size,
	unsigned int* tag_array, unsigned int tag);
template __global__ void cuda_applyFactorToTaggedParticles_kernel<int>(int* buff, int value, unsigned int buff_size,
	unsigned int* tag_array, unsigned int tag);
template __global__ void cuda_applyFactorToTaggedParticles_kernel<RealCuda>(RealCuda* buff, RealCuda value, unsigned int buff_size,
	unsigned int* tag_array, unsigned int tag);




void DynamicWindow::stabilizeFluid(SPH::DFSPHCData& data, DynamicWindowInterface::StabilizationParameters& params) {
	if(!isInitialized()) {
		std::cout << "DynamicWindow::stabilizeFluid Loading impossible data was not initialized" << std::endl;
		return ;
	}

	if (!isDataTagged()) {
		std::cout << "!!!!!!!!!!! DynamicWindow::stabilizeFluid you are loading untagged data !!!!!!!!!!!" << std::endl;
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
			"external","pressure","check max velocity","update pos","check particles outside boundairies" };
		SPH::SegmentedTiming timings(" DynamicWindow::stabilizeFluid method simu + damping", timing_names, true);
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
			DynamicWindowInterface::LoadingParameters params_loading;
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
		bool pretag_neighbors = hasLoadingStoredActiveNeighborsTagging();
		//I can save the tagging so that I don't have to redo it everytimestep
		static unsigned int* tag_array = NULL;
		static unsigned int tag_array_max_size = 0;
		if (simulate_border_only) {
			if (hasLoadingStoredActiveTagging()) {
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
				std::cout << "DynamicWindow::stabilizeFluid running a damped simulation requires to have set up the tagging in the loading function" << std::endl;
				gpuErrchk(cudaError_t::cudaErrorUnknown);
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
			std::cout << "DynamicWindow::stabilizeFluid checking the restriction mode and the true particle count " <<
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
						*(SVS_CU::get()->tagged_particles_count) = 0;
						if (true) {
							//tag the neighbors
							tag_neighbors_of_tagged_kernel<true, false> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS);
							gpuErrchk(cudaDeviceSynchronize());

							//and fucking count them
							count_particle_with_tag_kernel << <numBlocks, BLOCKSIZE >> > (data.fluid_data->gpu_ptr, TAG_ACTIVE_NEIGHBORS, SVS_CU::get()->tagged_particles_count);
							gpuErrchk(cudaDeviceSynchronize());
							data.count_active_neighbors = *(SVS_CU::get()->tagged_particles_count);
						}
						else {
							tag_and_count_neighbors_of_tagged_kernel<true, false> << <numBlocks, BLOCKSIZE >> > (data, data.fluid_data->gpu_ptr, 
								TAG_ACTIVE, TAG_ACTIVE_NEIGHBORS, (SVS_CU::get()->tagged_particles_count));
							gpuErrchk(cudaDeviceSynchronize());
						}
						data.count_active_neighbors = *(SVS_CU::get()->tagged_particles_count);
					}
					
					if (params.show_debug) 
					{
						std::cout << "count active // active neighbors: " << data.count_active << " // " << data.count_active_neighbors << std::endl; 
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


				timings.time_next_point();
				
				if (preUpdateVelocityDamping) {
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
					cuda_applyFactorToTaggedParticles_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (data.fluid_data->vel,
						Vector3d(preUpdateVelocityDamping_val), data.fluid_data->numParticles,
						data.fluid_data->neighborsDataSet->cell_id, TAG_ACTIVE);
					gpuErrchk(cudaDeviceSynchronize());
				}

				if (preUpdateVelocityClamping) {
					std::cout << "currently unsusable, need to create a function that clamp tagged only to be reactivated" << std::endl;
					gpuErrchk(cudaError_t::cudaErrorUnknown);
					//clamp_buffer_to_value<Vector3d, 4>(data.fluid_data->vel, Vector3d(preUpdateVelocityClamping_val), data.fluid_data->numParticles);
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
					int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
					cuda_applyFactorToTaggedParticles_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (data.fluid_data->vel,
						Vector3d(postUpdateVelocityDamping_val), data.fluid_data->numParticles,
						data.fluid_data->neighborsDataSet->cell_id, TAG_ACTIVE);
					gpuErrchk(cudaDeviceSynchronize());
				}

				if (postUpdateVelocityClamping) {
					std::cout << "currently unsusable, need to create a function that clamp tagged only to be reactivated" << std::endl;
					gpuErrchk(cudaError_t::cudaErrorUnknown);
					//clamp_buffer_to_value<Vector3d, 4>(data.fluid_data->vel, Vector3d(postUpdateVelocityClamping_val), data.fluid_data->numParticles);
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
				std::cout << "DynamicWindow::stabilizeFluid stabilizations for all particles is not possible "<< 
					"please restrict the stabilization to active particles only" << std::endl;
				gpuErrchk(cudaError_t::cudaErrorUnknown);

			}
		}
		gpuErrchk(read_last_error_cuda("check stable after stabilization ", params.show_debug));
		
		if(params.show_debug) {
			std::cout << "DynamicWindow::stabilizeFluid checking the restriction mode and the true particle count after end" <<
				data.restriction_mode << "   " << data.true_particle_count << std::endl;
		}

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

		//zero the velocity of actives particles
		{
			int numBlocks = calculateNumBlocks(data.fluid_data->numParticles);
			cuda_applyFactorToTaggedParticles_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (data.fluid_data->vel,
				Vector3d(0), data.fluid_data->numParticles,
				data.fluid_data->neighborsDataSet->cell_id, TAG_ACTIVE);
			gpuErrchk(cudaDeviceSynchronize());
		}


		//set the timestep back to the previous one
		data.updateTimeStep(old_timeStep);
		data.updateTimeStep(old_timeStep);




	}
	else {
		std::cout << " DynamicWindow::stabilizeFluid no stabilization method selected" << std::endl;
		return;
	}


	params.stabilization_sucess = true;
	   
	
}