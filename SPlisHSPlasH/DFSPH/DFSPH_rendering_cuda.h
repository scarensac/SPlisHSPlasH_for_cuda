#ifndef DFSPH_RENDERING_CUDA
#define DFSPH_RENDERING_CUDA

#include <GL/glew.h>
#include <cuda_gl_interop.h>

class ParticleSetRenderingData {
public:
	cudaGraphicsResource_t pos;
	cudaGraphicsResource_t vel;
	cudaGraphicsResource_t color;

	GLuint vao;
	GLuint pos_buffer;
	GLuint vel_buffer;
	GLuint color_buffer; //not used all of the tme it's mostly a debug fonctionality

	ParticleSetRenderingData() {
		vao=-1;
		pos_buffer=-1;
		vel_buffer=-1;
		color_buffer=-1;
	}
};

#include "SPlisHSPlasH\Vector.h"

namespace SPH{
	class DFSPHCData;
	class UnifiedParticleSet;
}

using namespace SPH;

void cuda_opengl_initParticleRendering(ParticleSetRenderingData& renderingData, unsigned int numParticles,
	Vector3d** pos, Vector3d** vel, bool need_color_buffer=false, Vector3d** color=NULL);
void cuda_opengl_releaseParticleRendering(ParticleSetRenderingData& renderingData);

void cuda_opengl_renderParticleSet(ParticleSetRenderingData& renderingData, unsigned int numParticles);
void cuda_renderFluid(SPH::DFSPHCData* data);
void cuda_renderBoundaries(SPH::DFSPHCData* data, bool renderWalls);

void cuda_reset_color(SPH::UnifiedParticleSet* particleSet);

#endif //DFSPH_RENDERING_CUDA