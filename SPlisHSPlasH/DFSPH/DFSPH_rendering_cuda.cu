#include "DFSPH_rendering_cuda.h"

#include "DFSPH_define_cuda.h"
#include "DFSPH_macro_cuda.h"

#include "DFSPH_c_arrays_structure.h"
#include "SPH_other_systems_cuda.h"

#include "Visualization/Shader.h"
#include "Utilities/FileSystem.h"
#include "SPlisHSPlasH/DFSPH/DFSPH_c_arrays_structure.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "SPlisHSPlasH/Utilities/ImgWriter.h"


#include <sstream>
#include <iostream>

#include <string>
#include <cmath>
#include <cstdint>



namespace RenderingCuda
{
	__global__ void init_buffer_kernel(Vector3d* buff, unsigned int size,Vector3d val) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= size) { return; }

		buff[i] = (i>(size/2))?val+val:val;

	}
}

void AdvancedRenderingData::init(int width, int height)
{
	w = width;
	h = height;
	bool use_texture = false;
	bool depth_texture = true;

	// The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
	glGenFramebuffers(1, &FramebufferName);
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

	if (use_texture) {

		// The texture we're going to render to
		glGenTextures(1, &renderedTexture);

		// "Bind" the newly created texture : all future texture functions will modify this texture
		glBindTexture(GL_TEXTURE_2D, renderedTexture);

		// Give an empty image to OpenGL ( the last "0" )
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

		// Poor filtering. Needed !
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);



		// Set the list of draw buffers.
		if (depth_texture) {
			//bind the depth texture directly
			glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, renderedTexture, 0);
			glDrawBuffer(GL_NONE);
		}
		else {
			// The depth buffer
			glGenRenderbuffers(1, &depthRenderbuffer);
			glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
			glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);
		
			//the texture bindings
			glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);
			GLenum DrawBuffers[1] = {  GL_COLOR_ATTACHMENT0 };
			glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers
		}
	}
	else {
		// The depth buffer
		glGenRenderbuffers(1, &depthRenderbuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);

		// the rgb bu since I won't use it I may not need that
		glGenRenderbuffers(1, &rgbaRenderbuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, rgbaRenderbuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, width, height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rgbaRenderbuffer);

		glDrawBuffer(GL_NONE);
		//faire un glreadpixels pr lire les valeurs
	}
	//*




	// Always check that our framebuffer is ok
	GLenum result = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (result != GL_FRAMEBUFFER_COMPLETE) {
		std::ostringstream oss;
		oss<< "AdvancedRenderingData::init failed, error: " <<result;
		std::cout << oss.str() << std::endl;
		throw(oss.str()) ;
	}
	//*/

	//cleanup
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	//Now to be able to have that render work I need a program with only a basic vertex shader (no need frag shader since I don't need the color only the depth
	//*
	shader=new SPH::Shader();


	std::string exePath = FileSystem::getProgramPath();
	std::string dataPath = FileSystem::normalizePath(exePath + "/" + std::string(SPH_DATA_PATH));

	//std::string vertFile = dataPath + "/shaders/vs_points_basic.glsl";
	//std::string fragFile = dataPath + "/shaders/fs_points_basic.glsl";
	std::string vertFile = dataPath + "/shaders/vs_points_manual_color.glsl";
	std::string fragFile = dataPath + "/shaders/fs_points_manual_color.glsl";

	shader->compileShaderFile(GL_VERTEX_SHADER, vertFile);
	shader->compileShaderFile(GL_FRAGMENT_SHADER, fragFile);
	shader->createAndLinkProgram();
	shader->begin();
	shader->addUniform("modelview_matrix");
	shader->addUniform("projection_matrix");
	shader->addUniform("radius");
	shader->addUniform("viewport_width");
	shader->addUniform("color");
	shader->addUniform("projection_radius");
	shader->addUniform("max_velocity");
	shader->end();
	//*/
}


void AdvancedRenderingData::computeDepthBuffer(SPH::DFSPHCData* data, SPH::Vector3d eye_i, SPH::Vector3d lookAt_i) {
	return;

	if (data->fluid_data == NULL) {
		return;
	}
	std::cout << "reached here" << std::endl;
	eye = eye_i;
	lookAt = lookAt_i;


	//set the render to the texture
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

	glClearColor(0.4f, 0.4f, 0.4f, 0.4f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//I don' even know if this shoudl not be done at the start or chen the window is resized....
	//glRenderMode(GL_RENDER);
	//glViewport(0, 0, w, h);

	shader->begin();
	
	//set the uniforms
	glUniform1f(shader->getUniform("viewport_width"), w);
	glUniform1f(shader->getUniform("radius"), data->particleRadius);
	float fluidColor[4] = { 0.3f, 0.5f, 0.9f, 1.0f };
	glUniform3fv(shader->getUniform("color"), 1, fluidColor);
	glUniform1f(shader->getUniform("max_velocity"), 25);


	//set the matrixes
	glm::mat4 projMat = glm::perspective(40.0f, static_cast<float>(w)/ static_cast<float>(h), 0.1f, 500.0f);
	glm::mat4 viewMat = glm::lookAt(glm::vec3(eye.x, eye.y, eye.z), glm::vec3(lookAt.x, lookAt.y, lookAt.z), glm::vec3(0, 1, 0));
	glm::mat4 modelMat; //identity

	glUniformMatrix4fv(shader->getUniform("modelview_matrix"), 1, GL_FALSE, glm::value_ptr(viewMat * modelMat));
	glUniformMatrix4fv(shader->getUniform("projection_matrix"), 1, GL_FALSE, glm::value_ptr(projMat));

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glPointParameterf(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);



	//render the fluid to generate the depth buffer
	//cuda_renderFluid(data);

	shader->end();

	//now a test to export the rgba to an image
	//*
	static float* pixels = new float[w * h * 4];
	glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, pixels);
	save_image("test_img", pixels, w, h,4,true);
	//*/
	
	glFlush();

	//now go back to the screen
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

}



void cuda_opengl_initParticleRendering(ParticleSetRenderingData& renderingData, unsigned int numParticles,
	Vector3d** pos, Vector3d** vel, bool need_color_buffer, Vector3d** color) {

	//read_last_error_cuda("before alloc rendering on gpu: ");

	glGenVertexArrays(1, &renderingData.vao); // Créer le VAO
	glBindVertexArray(renderingData.vao); // Lier le VAO pour l'utiliser


	glGenBuffers(1, &renderingData.pos_buffer);
	// selectionne le buffer pour l'initialiser
	glBindBuffer(GL_ARRAY_BUFFER, renderingData.pos_buffer);
	// dimensionne le buffer actif sur array_buffer, l'alloue et l'initialise avec les positions des sommets de l'objet
	glBufferData(GL_ARRAY_BUFFER,
		/* length */	numParticles * sizeof(Vector3d),
		/* data */      NULL,
		/* usage */     GL_DYNAMIC_DRAW);
	//set it to the attribute
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FORMAT, GL_FALSE, 0, 0);

	glGenBuffers(1, &renderingData.vel_buffer);
	// selectionne le buffer pour l'initialiser
	glBindBuffer(GL_ARRAY_BUFFER, renderingData.vel_buffer);
	// dimensionne le buffer actif sur array_buffer, l'alloue et l'initialise avec les positions des sommets de l'objet
	glBufferData(GL_ARRAY_BUFFER,
		/* length */	numParticles * sizeof(Vector3d),
		/* data */      NULL,
		/* usage */     GL_DYNAMIC_DRAW);
	//set it to the attribute
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FORMAT, GL_FALSE, 0, 0);

	if (need_color_buffer) {
		glGenBuffers(1, &renderingData.color_buffer);
		// selectionne le buffer pour l'initialiser
		glBindBuffer(GL_ARRAY_BUFFER, renderingData.color_buffer);
		// dimensionne le buffer actif sur array_buffer, l'alloue et l'initialise avec les positions des sommets de l'objet
		glBufferData(GL_ARRAY_BUFFER,
			/* length */	(numParticles) * sizeof(Vector3d),
			/* data */      NULL,
			/* usage */     GL_DYNAMIC_DRAW);
		//set it to the attribute
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 3, GL_FORMAT, GL_FALSE, 0, 0);
	}

	// nettoyage
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Registration with CUDA.
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&renderingData.pos, renderingData.pos_buffer, cudaGraphicsRegisterFlagsNone));
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&renderingData.vel, renderingData.vel_buffer, cudaGraphicsRegisterFlagsNone));
	if (need_color_buffer) {
		gpuErrchk(cudaGraphicsGLRegisterBuffer(&renderingData.color, renderingData.color_buffer, cudaGraphicsRegisterFlagsNone));
	}

	//link the pos and vel buffer to cuda
	gpuErrchk(cudaGraphicsMapResources(1, &renderingData.pos, 0));
	gpuErrchk(cudaGraphicsMapResources(1, &renderingData.vel, 0));
	if (need_color_buffer) {
		gpuErrchk(cudaGraphicsMapResources(1, &renderingData.color, 0));
	}

	//set the openglbuffer for direct use in cuda
	Vector3d* vboPtr = NULL;
	size_t size = 0;

	// pos
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &size, renderingData.pos));//get cuda ptr
	*pos = vboPtr;

	// vel
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &size, renderingData.vel));//get cuda ptr
	*vel = vboPtr;

	if (need_color_buffer) {
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &size, renderingData.color));//get cuda ptr
		*color = vboPtr;
	}
}

void cuda_opengl_releaseParticleRendering(ParticleSetRenderingData& renderingData) {
	//unlink the pos and vel buffer from cuda
	gpuErrchk(cudaGraphicsUnmapResources(1, &(renderingData.pos), 0));
	gpuErrchk(cudaGraphicsUnmapResources(1, &(renderingData.vel), 0));
	if (renderingData.color_buffer <= 100000) {
		gpuErrchk(cudaGraphicsUnmapResources(1, &renderingData.color, 0));
	}

	//delete the opengl buffers
	glDeleteBuffers(1, &renderingData.vel_buffer);
	glDeleteBuffers(1, &renderingData.pos_buffer);
	if (renderingData.color_buffer <= 100000) {
		glDeleteBuffers(1, &renderingData.color_buffer);
	}
	glDeleteVertexArrays(1, &renderingData.vao);
}

void cuda_opengl_renderParticleSet(ParticleSetRenderingData& renderingData, unsigned int numParticles) {


	//unlink the pos and vel buffer from cuda
	gpuErrchk(cudaGraphicsUnmapResources(1, &(renderingData.pos), 0));
	gpuErrchk(cudaGraphicsUnmapResources(1, &(renderingData.vel), 0));
	if (renderingData.color_buffer<=100000) {
		gpuErrchk(cudaGraphicsUnmapResources(1, &(renderingData.color), 0));
	}


	//Actual opengl rendering
	// link the vao
	glBindVertexArray(renderingData.vao);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//show it
	glDrawArrays(GL_POINTS, 0, numParticles);

	// unlink the vao
	glBindVertexArray(0);

	//link the pos and vel buffer to cuda
	gpuErrchk(cudaGraphicsMapResources(1, &renderingData.pos, 0));
	gpuErrchk(cudaGraphicsMapResources(1, &renderingData.vel, 0));
	if (renderingData.color_buffer <= 100000) {
		gpuErrchk(cudaGraphicsMapResources(1, &renderingData.color, 0));
	}

}

void cuda_renderFluid(SPH::DFSPHCData* data) {
	cuda_opengl_renderParticleSet(*data->fluid_data->renderingData, data->fluid_data[0].numParticles);
}



void cuda_renderBoundaries(SPH::DFSPHCData* data, bool renderWalls) {
	if (renderWalls) {
		cuda_opengl_renderParticleSet(*(data->boundaries_data->renderingData), data->boundaries_data->numParticles);
	}

	for (int i = 0; i < data->numDynamicBodies; ++i) {
		SPH::UnifiedParticleSet& body = data->vector_dynamic_bodies_data[i];
		cuda_opengl_renderParticleSet(*body.renderingData, body.numParticles);
	}
}



void cuda_reset_color(SPH::UnifiedParticleSet* particleSet) {
	if (particleSet->has_color_buffer) {
		int numBlocks = calculateNumBlocks(particleSet->numParticles);
		RenderingCuda::init_buffer_kernel << <numBlocks, BLOCKSIZE >> > (particleSet->color, particleSet->numParticles, Vector3d(-1));
		gpuErrchk(cudaDeviceSynchronize());
	}
}

