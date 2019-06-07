#ifndef DFSPH_RENDERING_CUDA
#define DFSPH_RENDERING_CUDA

#include <GL/glew.h>
#include <cuda_gl_interop.h>

class ParticleSetRenderingData {
public:
	cudaGraphicsResource_t pos;
	cudaGraphicsResource_t vel;

	GLuint vao;
	GLuint pos_buffer;
	GLuint vel_buffer;

};

#include "SPlisHSPlasH\Vector.h"

namespace SPH{
	class DFSPHCData;
}

using namespace SPH;

void cuda_opengl_initParticleRendering(ParticleSetRenderingData& renderingData, unsigned int numParticles,
	Vector3d** pos, Vector3d** vel);
void cuda_opengl_releaseParticleRendering(ParticleSetRenderingData& renderingData);

void cuda_opengl_renderParticleSet(ParticleSetRenderingData& renderingData, unsigned int numParticles);
void cuda_renderFluid(SPH::DFSPHCData* data);
void cuda_renderBoundaries(SPH::DFSPHCData* data, bool renderWalls);

#endif //DFSPH_RENDERING_CUDA