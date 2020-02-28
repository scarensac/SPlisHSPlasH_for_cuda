#ifndef DFSPH_RENDERING_CUDA
#define DFSPH_RENDERING_CUDA

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include "SPlisHSPlasH/Vector.h"


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

namespace SPH{
	class Shader;
	class DFSPHCData;
}
class AdvancedRenderingData {
public:
	GLuint FramebufferName = 0;
	GLuint renderedTexture;
	GLuint depthRenderbuffer;
	GLuint rgbaRenderbuffer;
	SPH::Vector3d eye, lookAt;
	int w, h;
	SPH::Shader* shader;

	AdvancedRenderingData() {
		FramebufferName = 0;
		renderedTexture = -1;
		depthRenderbuffer = -1;
		rgbaRenderbuffer = -1;
		w = -1;
		h = -1;
		shader = NULL;
	}

	void init(int width, int height);

	void computeDepthBuffer(SPH::DFSPHCData* data, SPH::Vector3d eye_i, SPH::Vector3d lookAt_i);

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