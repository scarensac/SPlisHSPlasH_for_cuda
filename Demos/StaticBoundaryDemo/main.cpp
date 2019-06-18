#include "SPlisHSPlasH/Common.h"
#include "GL/glew.h"
#include "Visualization/MiniGL.h"
#include "GL/glut.h"
#include "SPlisHSPlasH/TimeManager.h"
#include <Eigen/Dense>
#include <iostream>
#include "SPlisHSPlasH/Utilities/Timing.h"
#include "Utilities/PartioReaderWriter.h"
#include "SPlisHSPlasH/StaticRigidBody.h"
#include "Utilities/OBJLoader.h"
#include "SPlisHSPlasH/Utilities/PoissonDiskSampling.h"
#include "Demos/Common/DemoBase.h"
#include "Utilities/FileSystem.h"
#include <fstream>
#include "SPlisHSPlasH/DFSPH/DFSPH_CUDA.h"

// Enable memory leak detection
#ifdef _DEBUG
#ifndef EIGEN_ALIGN
	#define new DEBUG_NEW 
#endif
#endif

#define FFMPEG_RENDER
#ifdef FFMPEG_RENDER
//#define USE_MULTIPLES_SHADER
#endif
using namespace SPH;
using namespace Eigen;
using namespace std;

void timeStep ();
void initBoundaryData();
void render ();
void renderBoundary();
void reset();
void simulationMethodChanged();
void partioExport();

DemoBase base;
Real nextFrameTime = 0.0;
unsigned int frameCounter = 1;
FILE* ffmpeg = NULL;

// main 
int main( int argc, char **argv )
{
	REPORT_MEMORY_LEAKS;

	base.init(argc, argv, "StaticBoundaryDemo");
	initBoundaryData();
	base.buildModel();
	base.setSimulationMethodChangedFct(simulationMethodChanged);

	MiniGL::setClientIdleFunc(50, timeStep);
	MiniGL::setKeyFunc(0, 'r', reset);
	MiniGL::setClientSceneFunc(render);

	glutMainLoop ();	

	base.cleanup ();

	Timing::printAverageTimes();
	Timing::printTimeSums();
	
	return 0;
}

void reset()
{
	Timing::printAverageTimes();
	Timing::reset();

	base.getSimulationMethod().simulation->reset();
	TimeManager::getCurrent()->setTime(0.0);
	base.getSelectedParticles().clear();

	nextFrameTime = 0.0;
	frameCounter = 1;
}

void simulationMethodChanged()
{
	reset();
}

void timeStep ()
{
	if ((base.getPauseAt() > 0.0) && (base.getPauseAt() < TimeManager::getCurrent()->getTime()))
		base.setPause(true);

	if (base.getSimulationMethod().simulationMethod == DemoBase::SimulationMethods::DFSPH_CUDA) {
		DFSPHCUDA* sim = dynamic_cast<DFSPHCUDA*>(base.getSimulationMethod().simulation);
		sim->handleDynamicBodiesPause(base.getRbPause());

		//save th simulation state if asked
		sim->handleSimulationSave(base.getSaveLiquid() || base.getSaveSimulation(), base.getSaveSimulation(), base.getSaveSimulation());
		//I'll handle the save as token so I need to consume them
		base.setSaveLiquid(false);
		base.setSaveSimulation(false);



		//load the simulation state if asked
		sim->handleSimulationLoad(base.getLoadLiquid() || base.getLoadSimulation(), false, base.getLoadSimulation(), false, base.getLoadSimulation(), false);
		//I'll handle the save as token so I need to consume them
		base.setLoadLiquid(false);
		base.setLoadSimulation(false);

		//ontrol the fluid height if required
		sim->handleFLuidLevelControl(base.getFLuidLevelTarget());
		//ok this is a problem... because when the sim is posed I only need to do it once but if it is not I need to do it countinuously
		//I'll need to handle it with some bool in the fluid class
		base.setFluidLevelControl(false);

		//move the simulation area
		//carefull this means the neighbo searh result (cell_stat_end) doesn't mean anything after
		sim->handleSimulationMovement(vector3rTo3d(base.getSimulationMovement()));
		base.resetSimulationMovement();

		//used if setting the fluid vel to zero is usefull
		if (base.getZeroVelocities()) {
			sim->zeroFluidVelocities();
		}
		base.setZeroVelocities(false);
	}


	if (base.getPause())
		return;

	// Simulation code
	for (unsigned int i = 0; i < base.getNumberOfStepsPerRenderUpdate(); i++)
	{
		START_TIMING("SimStep");
		base.getSimulationMethod().simulation->step();
		STOP_TIMING_AVG;

		if (base.getEnablePartioExport())
		{
			if (TimeManager::getCurrent()->getTime() >= nextFrameTime)
			{
				nextFrameTime += 1.0 / base.getFramesPerSecond();
				partioExport();
				frameCounter++;
			}
		}
		if (TimeManager::getCurrent()->getTime() > 0.5) {
			//MiniGL::move(-0.7*TimeManager::getCurrent()->getTimeStepSize(), 0, 0);
		}
	}

#ifdef FFMPEG_RENDER
	//the part to save to file
	static int steps=0;
	if (ffmpeg != NULL) {
		if (TimeManager::getCurrent()->getTime()>10) {
			_pclose(ffmpeg);
			exit(0);
		}
		steps++;
	}
#endif
}

#include <sstream>

void render()
{
	static int width = glutGet(GLUT_WINDOW_WIDTH);
	static int height = glutGet(GLUT_WINDOW_HEIGHT);


	static int printed_timing = 0;
	static float old_time = 0;
	static std::chrono::steady_clock::time_point old_real_time = std::chrono::steady_clock::now();

	if (old_time == 0 && TimeManager::getCurrent()->getTime() > 0) {
		old_time = TimeManager::getCurrent()->getTime();
		old_real_time = std::chrono::steady_clock::now();
	}

	if (TimeManager::getCurrent()->getTime() > old_time + 0.25) {
		float count_steps = (TimeManager::getCurrent()->getTime() - old_time) / TimeManager::getCurrent()->getTimeStepSize();
		std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
		float time = std::chrono::duration_cast<std::chrono::nanoseconds> (now - old_real_time).count() / 1000000.0f;
		printed_timing = time / count_steps;

		old_real_time = now;
		old_time = TimeManager::getCurrent()->getTime();
	}

	glDisable(GL_LIGHTING);
	glColor3f(1, 0, 0);
	std::ostringstream oss;
	oss << printed_timing;

	int w;
	w = glutBitmapLength(GLUT_BITMAP_TIMES_ROMAN_24, (const unsigned char*)(oss.str().c_str()));

	oss << " ms";


	float x = .5; /* Centre in the middle of the window */
	glWindowPos2i(width / 2 - w, height - 100);

	int len = strlen(oss.str().c_str());
	for (int i = 0; i < len; i++) {
		glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, oss.str().c_str()[i]);
	}
	glEnable(GL_LIGHTING);

#ifdef FFMPEG_RENDER
	static int* buffer = new int[width*height];
	

	if (ffmpeg == NULL) {
		int framerate = (1 / TimeManager::getCurrent()->getTimeStepSize())/2;
		std::cout<<"video framerate: "<<framerate<< std::endl;
		std::ostringstream oss;
		// start ffmpeg telling it to expect raw rgba 720p-60hz frames
		// -i - tells it to read frames from stdin
		oss << "D:\\ffmpeg-4.1.3-win64-static\\bin\\ffmpeg " <<
			" -r "<<framerate<<" -f rawvideo -pix_fmt rgba -s "<<width<<"x"<<height<<" -i - " <<
			"-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4";



		// open pipe to ffmpeg's stdin in binary write mode
		ffmpeg = _popen(oss.str().c_str(), "wb");
	}
#endif

	//activate this if you want the axis
	//MiniGL::coordinateSystem();

	base.renderFluid();

#ifdef FFMPEG_RENDER
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);

	fwrite(buffer, sizeof(int)*width*height, 1, ffmpeg);


#ifdef USE_MULTIPLES_SHADER
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_ACCUM_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	base.renderFluid(1);

	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
	fwrite(buffer, sizeof(int)*width*height, 1, ffmpeg);
#endif
#endif

	renderBoundary();



	
}

void renderBoundary()
{
	DemoBase::SimulationMethod &simulationMethod = base.getSimulationMethod();
	Shader &shader = base.getShader();
	Shader &meshShader = base.getMeshShader();
	SceneLoader::Scene &scene = base.getScene();
	const int renderWalls = base.getRenderWalls();
	GLint context_major_version = base.getContextMajorVersion();

	float wallColor[4] = { 0.1f, 0.6f, 0.6f, 1.0f };
	if ((renderWalls == 1) || (renderWalls == 2))
	{
		if (context_major_version > 3)
		{
			shader.begin();
			glUniform3fv(shader.getUniform("color"), 1, &wallColor[0]);

			
			if (base.isDFSPH()) {
				base.renderBoundariesDFSPH_CUDA(renderWalls == 1);
			}
			else 
			{
				//std::cout << "using original" << std::endl;
				glEnableVertexAttribArray(0);
				for (int body = simulationMethod.model.numberOfRigidBodyParticleObjects() - 1; body >= 0; body--)
				{
					if ((renderWalls == 1) || (!scene.boundaryModels[body]->isWall))
					{
						FluidModel::RigidBodyParticleObject *rb = simulationMethod.model.getRigidBodyParticleObject(body);
						glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 0, &simulationMethod.model.getPosition(body + 1, 0));
						glDrawArrays(GL_POINTS, 0, rb->numberOfParticles());
					}
				}
				glDisableVertexAttribArray(0);
			}
			shader.end();
		}
		else
		{
			glDisable(GL_LIGHTING);
			glPointSize(4.0f);
 
			glBegin(GL_POINTS);
			for (int body = simulationMethod.model.numberOfRigidBodyParticleObjects() - 1; body >= 0; body--)
			{
				if ((renderWalls == 1) || (!scene.boundaryModels[body]->isWall))
				{
					FluidModel::RigidBodyParticleObject *rb = simulationMethod.model.getRigidBodyParticleObject(body);
					for (unsigned int i = 0; i < rb->numberOfParticles(); i++)
					{
						glColor3fv(wallColor);
						glVertex3v(&simulationMethod.model.getPosition(body + 1, i)[0]);
					}
				}
			}
			glEnd();
			glEnable(GL_LIGHTING);
		}
	}
	else if ((renderWalls == 3) || (renderWalls == 4))
	{
		for (int body = simulationMethod.model.numberOfRigidBodyParticleObjects() - 1; body >= 0; body--)
		{
			if ((renderWalls == 3) || (!scene.boundaryModels[body]->isWall))
			{
				meshShader.begin();
				glUniform1f(meshShader.getUniform("shininess"), 5.0f);
				glUniform1f(meshShader.getUniform("specular_factor"), 0.2f);

				GLfloat matrix[16];
				glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
				glUniformMatrix4fv(meshShader.getUniform("modelview_matrix"), 1, GL_FALSE, matrix);
				GLfloat pmatrix[16];
				glGetFloatv(GL_PROJECTION_MATRIX, pmatrix);
				glUniformMatrix4fv(meshShader.getUniform("projection_matrix"), 1, GL_FALSE, pmatrix);

				glUniform3fv(meshShader.getUniform("surface_color"), 1, wallColor);

				FluidModel::RigidBodyParticleObject *rb = simulationMethod.model.getRigidBodyParticleObject(body);
				MiniGL::drawMesh(((StaticRigidBody*)rb->m_rigidBody)->getGeometry(), wallColor);

				meshShader.end();
			}
		}		
	}
}


void initBoundaryData()
{
	std::string base_path = FileSystem::getFilePath(base.getSceneFile());
	SceneLoader::Scene &scene = base.getScene();
	const bool useCache = base.getUseParticleCaching();

	for (unsigned int i = 0; i < scene.boundaryModels.size(); i++)
	{
		string meshFileName = FileSystem::normalizePath(base_path + "/" + scene.boundaryModels[i]->meshFile);

		std::vector<Vector3r> boundaryParticles;
		if (scene.boundaryModels[i]->samplesFile != "")
		{
			string particleFileName = base_path + "/" + scene.boundaryModels[i]->samplesFile;
			PartioReaderWriter::readParticles(particleFileName, scene.boundaryModels[i]->translation, scene.boundaryModels[i]->rotation, scene.boundaryModels[i]->scale[0], boundaryParticles);
		}

		StaticRigidBody *rb = new StaticRigidBody();
		TriangleMesh &geo = rb->getGeometry();
		OBJLoader::loadObj(meshFileName, geo, scene.boundaryModels[i]->scale);
		for (unsigned int j = 0; j < geo.numVertices(); j++)
			geo.getVertices()[j] = scene.boundaryModels[i]->rotation * geo.getVertices()[j] + scene.boundaryModels[i]->translation;

		geo.updateNormals();
		geo.updateVertexNormals();

		if (scene.boundaryModels[i]->samplesFile == "")
		{
			// Cache sampling
			std::string mesh_base_path = FileSystem::getFilePath(scene.boundaryModels[i]->meshFile);
			std::string mesh_file_name = FileSystem::getFileName(scene.boundaryModels[i]->meshFile);
			std::string scene_path = FileSystem::getFilePath(base.getSceneFile());
			std::string scene_file_name = FileSystem::getFileName(base.getSceneFile());
			string cachePath = scene_path + "/" + mesh_base_path + "/Cache";
			string particleFileName = FileSystem::normalizePath(cachePath + "/" + scene_file_name + "_" + mesh_file_name + "_" + std::to_string(i) + ".bgeo");

			// check MD5 if cache file is available
			bool foundCacheFile = false;
			bool md5 = false;

			std::string md5FileName = FileSystem::normalizePath(cachePath + "/" + scene_file_name + ".md5");
			if (useCache)
			{
				foundCacheFile = FileSystem::fileExists(particleFileName);
				if (foundCacheFile)
				{
					string md5Str = FileSystem::getFileMD5(base.getSceneFile());
					md5 = FileSystem::checkMD5(md5Str, md5FileName);
				}
			}

			
			if (useCache && foundCacheFile && md5)
			{
				PartioReaderWriter::readParticles(particleFileName, Vector3r::Zero(), Matrix3r::Identity(), 1.0, boundaryParticles);
				std::cout << "Loaded cached boundary sampling: " << particleFileName << "\n";
			}

			if (!useCache || !foundCacheFile || !md5)
			{
				std::cout << "Surface sampling of " << meshFileName << "\n";
				START_TIMING("Poisson disk sampling");
				PoissonDiskSampling sampling;
				sampling.sampleMesh(geo.numVertices(), geo.getVertices().data(), geo.numFaces(), geo.getFaces().data(), scene.particleRadius, 10, 1, boundaryParticles);
				STOP_TIMING_AVG;

				// Cache sampling
				if (useCache && (FileSystem::makeDir(cachePath) == 0))
				{
					std::cout << "Save particle sampling: " << particleFileName << "\n";
					PartioReaderWriter::writeParticles(particleFileName, (unsigned int)boundaryParticles.size(), boundaryParticles.data(), NULL, scene.particleRadius);

					FileSystem::writeMD5File(base.getSceneFile(), md5FileName);
				}
			}
		}
		base.getSimulationMethod().model.addRigidBodyObject(rb, static_cast<unsigned int>(boundaryParticles.size()), &boundaryParticles[0]);
	}
}

void partioExport()
{
	FluidModel &model = base.getSimulationMethod().model;
	std::string exportPath = FileSystem::normalizePath(base.getExePath() + "/PartioExport");
	FileSystem::makeDirs(exportPath);

	std::string fileName = "ParticleData";
	fileName = fileName + std::to_string(frameCounter) + ".bgeo";
	std::string exportFileName = FileSystem::normalizePath(exportPath + "/" + fileName);

	PartioReaderWriter::writeParticles(exportFileName, model.numActiveParticles(), &model.getPosition(0, 0), &model.getVelocity(0, 0), model.getParticleRadius());
}