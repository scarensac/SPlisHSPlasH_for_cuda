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




//NOTE1:	seems that virtual function can't be used with managed allocation
//			so I'll use a template to have an equivalent solution
//			the  template parameter allow the représentation of various shapes
//NOTE2:	0 ==> plane: only use this for paralel planes (so max 2)
//			it's just a fast to compute solution if you need bands of fluid near the borders of the simulation
//			however since as soon as you get more than 2 planes the distance computation become pretty heavy
//			it's better to use another solution to represent the same surface (in particular in the case of boxes
//NOTE3:	1 ==> rectangular cuboid. 


template<int type>
class BufferFluidSurfaceBase
{
	//this is for the planes
	int count_planes;
	Vector3d* o;
	Vector3d* n;

	//this is for the cuboid
	Vector3d center;
	Vector3d halfLengths;

	//this for radius based geometries
	RealCuda radius;
public:
	bool destructor_activated;

	inline BufferFluidSurfaceBase() {
		destructor_activated = false;
		switch (type) {
		case 0: {
			count_planes = 0;
			cudaMallocManaged(&(o), sizeof(Vector3d) * 2);
			cudaMallocManaged(&(n), sizeof(Vector3d) * 2);
			break;
		}
		case 1: {break; }
		case 2: {break; }
		default: {; }//it should NEVER reach here
		}
	}


	inline ~BufferFluidSurfaceBase() {
		if (destructor_activated) {
			switch (type) {
			case 0: {
				CUDA_FREE_PTR(o);
				CUDA_FREE_PTR(n);
				break;
			}
			case 1: {break; }
			case 2: {break; }
			default: {; }//it should NEVER reach here
			};
		}
	}


	inline int addPlane(Vector3d o_i, Vector3d n_i) {
		if (type != 0) {
			return -1;
		}
		if (count_planes >= 2) {
			return 1;
		}
		o[count_planes] = o_i;
		n[count_planes] = n_i;
		count_planes++;
		return 0;
	}

	inline int setCuboid(Vector3d c_i, Vector3d hl_i) {
		if (type != 1) {
			return -1;
		}
		center = c_i;
		halfLengths = hl_i;
		return 0;
	}

	inline int setCylinder(Vector3d c_i, RealCuda half_h, RealCuda r) {
		if (type != 2) {
			return -1;
		}
		center = c_i;
		halfLengths = Vector3d(0,half_h,0);
		radius = r;
		return 0;
	}

	inline void copy (const BufferFluidSurfaceBase& o) {
		switch (type) {
		case 0: {
			count_planes = 0;
			for (int i = 0; i < o.count_planes; ++i) {
				addPlane(o.o[i], o.n[i]);
			}
			break;
		}
		case 1: {
			center = o.center;
			halfLengths = o.halfLengths;
			break;
		}
		case 2: {
			center = o.center;
			halfLengths = o.halfLengths;
			radius = o.radius;
			break;
		}
		default: {; }//it should NEVER reach here
		};
	}

	inline void move(const Vector3d& d) {
		switch (type) {
		case 0: {
			for (int i = 0; i < count_planes; ++i) {
				o[i]+=d;
			}
			break;
		}
		case 1: {
			center += d;
			break;
		}
		case 2: {
			center += d;
			break;
		}
		default: {; }//it should NEVER reach here
		};
	}

	std::string toString() {
		std::ostringstream oss;
		//*
		switch (type) {
		case 0: {
			if (count_planes > 0) {
				for (int i = 0; i < count_planes; ++i) {
					oss << "plane " << i << " (o,n)  " << o[i].x << "   " << o[i].y << "   " << o[i].z << 
						"   " << n[i].x << "   " << n[i].y << "   " << n[i].z<< std::endl;
				}
			}
			else {
				oss << "No planes" << std::endl;
			}
			break;
		}
		case 1: {
			oss << "Cuboid center: " << center.toString() << "   halfLengths: " << halfLengths.toString() << std::endl;
			break;
		}
		case 2: {
			oss << "Cylinder center: " << center.toString() << "  radius: "<<radius<<"   height: " << halfLengths.y << std::endl;
			break;
		}
		default: {; }//it should NEVER reach here
		};
		//*/
		return oss.str();
	}



	//to know if we are on the inside of each plane we can simply use the dot product*
	FUNCTION inline bool isInsideFluid(Vector3d p) {
		//*
		switch (type) {
		case 0: {
			for (int i = 0; i < count_planes; ++i) {
				Vector3d v = p - o[i];
				if (v.dot(n[i]) < 0) {
					return false;
				}
			}
			break;
		}
		case 1: {
			Vector3d v = p - center;
			v.toAbs();
			return (v.x < halfLengths.x) && (v.y < halfLengths.y) && (v.z < halfLengths.z);
			break;
		}
		case 2: {
			Vector3d v = p - center;
			if (abs(v.y) > halfLengths.y) { return false; }

			v.y = 0;
			return (v.norm()< radius);
			break;
		}
		default: {; }//it should NEVER reach here
		}
		//*/
		return true;
	}

	FUNCTION inline RealCuda distanceToSurface(Vector3d p) {
		RealCuda dist;
		//*
		switch (type) {
		case 0: {
			dist = abs((p - o[0]).dot(n[0]));
			for (int i = 1; i < count_planes; ++i) {
				Vector3d v = p - o[i];
				RealCuda l = abs(v.dot(n[i]));
				dist = MIN_MACRO_CUDA(dist, l);
			}
			break;
		}
		case 1: {
			Vector3d v = p - center;
			if (isInsideFluid(p)) {
				v.toAbs();

				dist = MIN_MACRO_CUDA(MIN_MACRO_CUDA((halfLengths.x - v.x), (halfLengths.y - v.y)), (halfLengths.z - v.z));

			}
			else {
				///TODO I'm nearly sure I can simply use the abs here but it would not realy impact the globals performances
				///		so I won't take the risk
				Vector3d v2 = v;
				if (v.x < -halfLengths.x) { v.x = -halfLengths.x; }
				else if (v.x > halfLengths.x) { v.x = halfLengths.x; }
				if (v.y < -halfLengths.y) { v.y = -halfLengths.y; }
				else if (v.y > halfLengths.y) { v.y = halfLengths.y; }
				if (v.z < -halfLengths.z) { v.z = -halfLengths.z; }
				else if (v.z > halfLengths.z) { v.z = halfLengths.z; }
				dist = (v - v2).norm();
			}

			break;
		}
		case 2: {
			Vector3d v = p - center;
			v.toAbs();
			if (isInsideFluid(p)) {

				//faster to compute that waty but you cound express it the same way as for the outside distance
				dist = halfLengths.y - v.y;
				v.y = 0;

				dist = MIN_MACRO_CUDA(dist,radius-v.norm());

			}
			else {
				Vector3d temp_v(v.x, 0, v.z);
				temp_v = Vector3d(MAX_MACRO_CUDA(temp_v.norm()-radius,0),MAX_MACRO_CUDA(v.y-halfLengths.y,0),0);
				dist = temp_v.norm();
			}

			break;
		}
		default: {; }//it should NEVER reach here
		}
		//*/
		return dist;
	}

	FUNCTION inline RealCuda distanceToSurfaceSigned(Vector3d p) {
		//you could most likely optimize the implementation for some of the types but it's not critical so why bother
		return  distanceToSurface(p) * ((isInsideFluid(p)) ? 1 : -1);;
	}

};

constexpr int surfaceType = 1;
using BufferFluidSurface = BufferFluidSurfaceBase<surfaceType>;


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

		// this function will handle to control of the y component of the dynamic buffer when there is no ovement specified
		// NOTE: the goal is to not have a reset in the function so that the simulation stays realistic
		void handleOceanBoundariesTest(SPH::DFSPHCData& data);


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
	DynamicWindow::getStructure().handleOceanBoundariesTest(data);
}





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
	bool keep = S.isInsideFluid(particleSet->pos[i]);
	keep = (keep_inside) ? keep : (!keep);
	if (!keep_inside) {
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
__global__ void DFSPH_evaluate_density_in_buffer_kernel(SPH::DFSPHCData data, SPH::UnifiedParticleSet* fluidSet,
	SPH::UnifiedParticleSet* backgroundBufferSet, SPH::UnifiedParticleSet* bufferSet, int* countRmv, 
	BufferFluidSurface S, RealCuda limit_density ) {
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
	

	//compute the fluid contribution
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(fluidSet->neighborsDataSet, fluidSet->pos,
		RealCuda density_delta = fluidSet->mass[j] * KERNEL_W(data, p_i - fluidSet->pos[j]);
	if (density_delta > 0) {
		if (S.isInsideFluid(fluidSet->pos[j])) {
			density += density_delta;
			count_neighbors++;
			if (fluidSet->pos[j].y > (p_i.y)) {
				keep_particle = true;
			}
		}
	}
	);

	keep_particle = keep_particle || (!S.isInsideFluid(p_i));
	

	//keep_particle = true;
	if (keep_particle) {
		//*
		//compute the buffer contribution
		//if (iter == 0)
		if (false){
			//Note: the strange if is just here to check if p_i!=p_j but since doing this kind of operations on float
			//		that are read from 2 files might not be that smat I did some bastard check verifying if the particle are pretty close
			//		since the buffer we are evaluating is a subset of the background buffer, the only possibility to have particle 
			//		that close is that they are the same particle.
			//*
			//no need for that since I lighten the buffers before now
			float limit = data.particleRadius / 10.0f;
			limit *= limit;
			ITER_NEIGHBORS_FROM_STRUCTURE_BASE(backgroundBufferSet->neighborsDataSet, backgroundBufferSet->pos,
				if ((p_i - backgroundBufferSet->pos[j]).squaredNorm()> (limit)) {
					RealCuda density_delta = backgroundBufferSet->mass[j] * KERNEL_W(data, p_i - backgroundBufferSet->pos[j]);
					density_after_buffer += density_delta;
					count_neighbors++;
				}
			);
			//*/
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
	if ((dist<data.particleRadius)&&(backgroundSet->pos[i].y<1.6)) {
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
	int* countRmvBuffer, BufferFluidSurface S) {
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
					if (S.isInsideFluid(fluidSet->pos[j])) {
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

	//*
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		RealCuda density_delta = data.boundaries_data_cuda->mass[j] * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
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
			RealCuda concentration_delta = (fluidSet->mass[j]/ fluidSet->density[j])* KERNEL_W(data, p_i - fluidSet->pos[j]);
			concentration += concentration_delta;
			count_neighbors++;
		}
	);

	//supose that the density of boundaries particles is the rest density
	ITER_NEIGHBORS_FROM_STRUCTURE_BASE(data.boundaries_data_cuda->neighborsDataSet, data.boundaries_data_cuda->pos,
		RealCuda concentration_delta = (data.boundaries_data_cuda->mass[j] / data.density0) * KERNEL_W(data, p_i - data.boundaries_data_cuda->pos[j]);
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
			Vector3d displacement_delta = (fluidSet->densityAdv[j]- fluidSet->densityAdv[i])* (fluidSet->mass[j] / fluidSet->density[j]) * KERNEL_GRAD_W(data, p_i - fluidSet->pos[j]);
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


void DynamicWindow::init(DFSPHCData& data) {
	if (initialized) {
		throw("DynamicWindow::init the structure has already been initialized");
	}

	//define the surface
	if (surfaceType == 0) {
		/*
		std::ostringstream oss;
		oss << "DynamicWindow::init The initialization for this type of surface has not been defined (type=" << surfaceType << std::endl;
		throw(oss.str());
		//*/
		S_initial.addPlane(Vector3d(-1.5, 0, 0), Vector3d(1, 0, 0));
		
	}
	else if (surfaceType == 1) {

		//*
		Vector3d center(0, 0, 0);
		Vector3d halfLength(100);
		halfLength.x = 1.6;
		halfLength.z = 0.5;
		S_initial.setCuboid(center, halfLength);
		//*/
	}
	else if (surfaceType == 2) {
		Vector3d center(0, 2.2, 0);
		S_initial.setCylinder(center, 2, 1.3 - 0.1);
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
		
		//and move the surface
		S.move(data.dynamicWindowTotalDisplacement);
		std::cout << S.toString() << std::endl;

	}

	//thoise some buffers depends on the totla displacement and might have to be displaced even though there is no displacement for this step
	if (data.dynamicWindowTotalDisplacement.norm() > 0.0005) {
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
		countRmv, countRmv2, S);

	gpuErrchk(cudaDeviceSynchronize());

	std::cout << "ligthen estimation (background // buffer): " << *countRmv << "  //  " << *countRmv2 << std::endl;

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
		std::cout << "effcount: " << effective_count << "from:  " << numParticles << std::endl;

		pos = pos_base;
		numParticles = numParticles_base;
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
		}

		Vector3d* pos_cpu = new Vector3d[particleSet->numParticles];
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


	for (int i = 4; i < 17; ++i) {
		//we need to reinit the neighbors struct for the fluidbuffer since we removed some particles
		fluidBufferSet->initNeighborsSearchData(data, false);

		RealCuda target_density = 1500 - i * 25;

		*countRmv = 0;
		{
			int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
			DFSPH_evaluate_density_in_buffer_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, backgroundFluidBufferSet->gpu_ptr, fluidBufferSet->gpu_ptr,
				countRmv, S, target_density);

			gpuErrchk(cudaDeviceSynchronize());
		}

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
		remove_particles(fluidBufferSet, fluidBufferSet->neighborsDataSet->cell_id, *countRmv);
		std::cout << "handle_fluid_boundries_cuda: (iter: " << i << ") changing num particles in the buffer: " << fluidBufferSet->numParticles << "   nb removed : " << *countRmv << std::endl;

	}

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
		std::cout << "handle_fluid_boundries_cuda: changing num particles: " << new_num_particles << "   nb removed : " << *countRmv << std::endl;
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
		std::cout << "handle_fluid_boundries_cuda: changing num particles: " << new_num_particles << "   nb added : " << fluidBufferSet->numParticles << std::endl;
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

	RealCuda diffusion_coefficient = 1000 * 0.5 * data.getKernelRadius() * data.getKernelRadius() / data.density0;
	particleSet->initNeighborsSearchData(data, false);


	//for the explanaition of eahc kernel please see the paper that is referenced in the function description
	//NOTE: I also use those kernel to do a simple surface detection with the following algorithm
	//		Use a low neighbors cap for sampling the surface
	//		then for each particel that I detected I increase a value in it's neighbor particles depending on the kernel value
	//		I then use that value to limit the vertical component of the particle shifting
	{
		int numBlocks = calculateNumBlocks(particleSet->numParticles);
		DFSPH_evaluate_density_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, S);
		gpuErrchk(cudaDeviceSynchronize());
	}

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


void DynamicWindow::handleFluidBoundaries(SPH::DFSPHCData& data, Vector3d movement) {
	if (!isInitialized()) {
		throw("DynamicWindow::handleFluidBoundaries the structure must be initialized before calling this function");
	}


	std::vector<std::string> timing_names{"color_reset","reset pos","apply displacement to buffer","init neightbors","compute density",
		"reduce buffer","cpy_velocity","reduce fluid", "apply buffer"};
	static SPH::SegmentedTiming timings("handle_fluid_boundries_cuda",timing_names,false);
	timings.init_step();



	timings.time_next_point();


	{
		timings.time_next_point();

		
		//reset the buffers
		DynamicWindow::getStructure().initStep(data, movement, false);

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
		if (surfaceType < 2) {
			add_border_to_damp_planes_cuda(data, abs(movement.x)>0.5, abs(movement.z)>0.5);
			data.damp_borders_steps_count = 5;
			data.damp_borders = true;
		}
		//*/
	}

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

		a = 0.2;
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

	std::cout << "sampling count: (x y)" << countSampling.toString() << std::endl;

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
	StokesWaveGenerator waveGenerator, int* count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bufferSet->numParticles) { return; }

	Vector3d pos_i = bufferSet->pos[i];
	bufferSet->neighborsDataSet->cell_id[i] = 0;
	
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

	if ((!S.isInsideFluid(particleSet->pos[i]))|| checkFreeSurfaceForAll) {

		RealCuda h;
		waveGenerator.getHeight(pos, h);


		if (pos.y > h) {
			atomicAdd(count, 1);
			particleSet->neighborsDataSet->cell_id[i] = 25000000;
			return;
		}

		//I'll do the velocity control here
		//if (!checkFreeSurfaceForAll) 
		{
			Vector3d vel;
			waveGenerator.getvel(pos, vel);
			particleSet->vel[i]=vel;
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


void DynamicWindow::handleOceanBoundariesTest(SPH::DFSPHCData& data) {
	UnifiedParticleSet* particleSet = data.fluid_data;
	static int* countRmv = NULL;
	static StokesWaveGenerator waveGenerator;
	static RealCuda time = 0;
	time += 0.003;

	if (!countRmv) {
		cudaMallocManaged(&(countRmv), sizeof(int));

		//I don't need the z component
		waveGenerator.init(data.particleRadius * 2, Vector3d(-2.0,0,0), Vector3d(2.0,2.0,0));
		
		if (false) {
			std::remove("yolo.csv");
			std::ofstream myfile("yolo.csv", std::ofstream::app);
			if (myfile.is_open())
			{
				/*
				myfile << "t" << "  ";
				for (int i = 0; i < waveGenerator.cellCount.x; ++i) {
					myfile << waveGenerator.minPos.x + i*waveGenerator.cellSize << "  ";
				}
				myfile << std::endl;
				//*/
			}

		}

	}

	std::vector<std::string> timing_names{ "compute wave properties","first init","init step","copy velocities","clean fluid",
	"tag buffer to add","add tagged buffer"};
	static SPH::SegmentedTiming timings("DynamicWindow::handleOceanBoundariesTest", timing_names, true);
	timings.init_step();




	waveGenerator.computeWaveState(time, 0.8, 0.003, data.particleRadius/4.0);

	timings.time_next_point();

	static bool first_time = true;
	if (first_time) {

		{


			*countRmv = 0;
			{
				int numBlocks = calculateNumBlocks(particleSet->numParticles);
				DFSPH_tag_above_desired_free_surface_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, S, waveGenerator, countRmv, true);
				gpuErrchk(cudaDeviceSynchronize());
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


			int new_num_particles = particleSet->numParticles - *countRmv;
			std::cout << "handleOceanBoundariesTest removing particles: " << new_num_particles << "   nb removed : " << *countRmv << std::endl;
			particleSet->updateActiveParticleNumber(new_num_particles);
			//*/

		}

		first_time = false;
	}


	timings.time_next_point();

	//init everything for now
	initStep(data, Vector3d(0, 0, 0), false, false);


	timings.time_next_point();

	if (false) {
		//here a test to see if the wave formula don't break
		//to test that I'll just change the position of the particles taht are inside the buffer*countRmv = 0;
		if (false) {
			int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
			DFSPH_buffer_mimic_stockes_waves_kernel << <numBlocks, BLOCKSIZE >> > (data, fluidBufferSet->gpu_ptr, waveGenerator, countRmv);
			gpuErrchk(cudaDeviceSynchronize());
		}

		if (true) {
			std::ofstream myfile("yolo.csv", std::ofstream::app);
			if (myfile.is_open())
			{
				int count2 = 0;
				for (int j = 0; j < waveGenerator.cellCount.y; ++j) {
					for (int i = 0; i < waveGenerator.cellCount.x; ++i) {
						int cell_id = i + j * waveGenerator.cellCount.x;
						myfile << waveGenerator.waveVelocityFieldCount[cell_id] << "  ";
						count2 += waveGenerator.waveVelocityFieldCount[cell_id];
					}
					myfile << std::endl;
				}
				/*
				myfile << time << "  ";
				for (int i = 0; i < waveGenerator.cellCount.x; ++i) {
					myfile << waveGenerator.waveHeightField[i] << "  ";
				}
				myfile << std::endl;
				//*/
				std::cout << "test: " << count2 << std::endl;
			}

		}
	}



	//ok so the point here is to have a system a fast as possible (even is jank as fuck to get the adaptation of the fluid height iside the buffer to the desired height
	//solution number 1:
	//since I have the buffer I technicaly have particles that are dsposed near their rest distribution
	//si I explore the whole buffer do a simple check on their number of neighbors to know if I should spawn them
	if (true) {
		//copy the vel field I'll use the one from the wave generator
		if (false){
			int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
			DFSPH_init_buffer_velocity_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->pos, fluidBufferSet->vel, fluidBufferSet->numParticles);
			gpuErrchk(cudaDeviceSynchronize());
		}

		timings.time_next_point();
		//now remove the fluid particles that are above the desired surface
		{
			*countRmv = 0;
			{
				int numBlocks = calculateNumBlocks(particleSet->numParticles);
				DFSPH_tag_above_desired_free_surface_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr,S, waveGenerator, countRmv);
				gpuErrchk(cudaDeviceSynchronize());
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


			int new_num_particles = particleSet->numParticles - *countRmv;
			//std::cout << "handleOceanBoundariesTest removing particles above  free surface: " << new_num_particles << "   nb removed : " << *countRmv << std::endl;
			particleSet->updateActiveParticleNumber(new_num_particles);
			//*/

		}

		timings.time_next_point();

		particleSet->initNeighborsSearchData(data, false);
		//tag the particles that need to be integrated
		*countRmv = 0;
		{
			int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
			DFSPH_tag_buffer_to_add_kernel<true> << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->gpu_ptr, waveGenerator,countRmv);
			gpuErrchk(cudaDeviceSynchronize());
		}

		timings.time_next_point();
	}

	//std::cout << "handleOceanBoundariesTest: detected I could add: "<<*countRmv<<"  particles  (nbr particles in buffer: " << fluidBufferSet->numParticles<<")"<<std::endl;

	//let's add the taged particles for debugging
	int newNbrParticles = particleSet->numParticles + *countRmv;
	if (newNbrParticles > particleSet->numParticlesMax) {
		particleSet->changeMaxParticleNumber(newNbrParticles * 1.25);
	}

	{
		*countRmv = 0;
		int numBlocks = calculateNumBlocks(fluidBufferSet->numParticles);
		DFSPH_add_tagged_particles_to_fluid_kernel << <numBlocks, BLOCKSIZE >> > (data, particleSet->gpu_ptr, fluidBufferSet->gpu_ptr, countRmv);
		gpuErrchk(cudaDeviceSynchronize());
	}


	//std::cout << "handleOceanBoundariesTest: changing num particles: " << newNbrParticles << "   nb before : " << particleSet->numParticles << std::endl;
	particleSet->updateActiveParticleNumber(newNbrParticles);

	timings.time_next_point();
	timings.end_step();
	timings.recap_timings();
}