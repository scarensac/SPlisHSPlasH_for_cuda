#include "SPlisHSPlasH/Common.h"
#include "GL/glew.h"
#include "Visualization/MiniGL.h"
#include "GL/glut.h"
#include "SPlisHSPlasH/TimeManager.h"
#include <Eigen/Dense>
#include <iostream>
#include "SPlisHSPlasH/Utilities/Timing.h"
#include "Utilities/PartioReaderWriter.h"
#include "PositionBasedDynamicsWrapper/PBDRigidBody.h"
#include "Utilities/OBJLoader.h"
#include "SPlisHSPlasH/Utilities/PoissonDiskSampling.h"
#include "PositionBasedDynamicsWrapper/PBDWrapper.h"
#include "Demos/Common/DemoBase.h"
#include "Utilities/FileSystem.h"
#include "SPlisHSPlasH/DFSPH/DFSPH_CUDA.h"


#define FFMPEG_RENDER
#ifdef FFMPEG_RENDER
FILE* ffmpeg = NULL;
//#define USE_MULTIPLES_SHADER
#endif

// Enable memory leak detection
#ifdef _DEBUG
#ifndef EIGEN_ALIGN
	#define new DEBUG_NEW 
#endif
#endif

using namespace SPH;
using namespace Eigen;
using namespace std;

void timeStep ();
void initBoundaryData();
void render ();
void renderBoundary();
void reset();
void updateBoundaryParticles(const bool forceUpdate);
void updateBoundaryForces();
void simulationMethodChanged();
void partioExport();

DemoBase base;
PBDWrapper pbdWrapper;
Real nextFrameTime = 0.0;
unsigned int frameCounter = 1;

// main 
int main( int argc, char **argv )
{
	REPORT_MEMORY_LEAKS;

	base.init(argc, argv, "DynamicBoundaryDemo");


	//////////////////////////////////////////////////////////////////////////
	// PBD
	//////////////////////////////////////////////////////////////////////////
	pbdWrapper.initShader();
	pbdWrapper.readScene(base.getSceneFile());

	initBoundaryData();
	base.buildModel();
	base.setSimulationMethodChangedFct(simulationMethodChanged);
	pbdWrapper.initGUI();

	pbdWrapper.initModel(TimeManager::getCurrent()->getTimeStepSize());

	MiniGL::setClientIdleFunc(50, timeStep);
	MiniGL::setKeyFunc(0, 'r', reset);
	MiniGL::setClientSceneFunc(render);

	glutMainLoop ();	

	base.cleanup();

	Timing::printAverageTimes();
	Timing::printTimeSums();
	
	return 0;
}

void reset()
{
	Timing::printAverageTimes();
	Timing::reset();

	//////////////////////////////////////////////////////////////////////////
	// PBD
	//////////////////////////////////////////////////////////////////////////
	pbdWrapper.reset();

	updateBoundaryParticles(true);

	base.getSimulationMethod().simulation->reset();
	TimeManager::getCurrent()->setTime(0.0);
	base.getSelectedParticles().clear();

	nextFrameTime = 0.0;
	frameCounter = 1;
}

void timeStep ()
{
	{
		static bool firstTime = false;
		if (firstTime) {
			firstTime = false;

			MiniGL::setViewport(40.0, 0.1f, 500.0, Vector3r(-11.0, 3.0, 0.0), Vector3r(0.0, 2.0, 0.0));
		}
	}

	if ((base.getPauseAt() > 0.0) && (base.getPauseAt() < TimeManager::getCurrent()->getTime()))
		base.setPause(true);

	float pauseRbAfterTime = 0.72;
	if (pauseRbAfterTime > 0) {
		if (TimeManager::getCurrent()->getTime() > pauseRbAfterTime) {
			base.setRbPause(true);
		}
	}

	if (base.getSimulationMethod().simulationMethod == DemoBase::SimulationMethods::DFSPH_CUDA) {
		DFSPHCUDA* sim = dynamic_cast<DFSPHCUDA*>(base.getSimulationMethod().simulation);


		sim->handleDynamicBodiesPause(base.getRbPause());
		

		//save th simulation state if asked
		sim->handleSimulationSave(base.getSaveLiquid()|| base.getSaveSimulation(), base.getSaveSimulation(), base.getSaveSimulation());
		//I'll handle the save as token so I need to consume them
		base.setSaveLiquid(false);
		base.setSaveSimulation(false);



		//load the simulation state if asked
		//sim->handleSimulationLoad(base.getLoadLiquid() || base.getLoadSimulation(), true, base.getLoadSimulation(), true, base.getLoadSimulation(), true);
		sim->handleSimulationLoad(base.getLoadLiquid() || base.getLoadSimulation(), true, false, false, base.getLoadSimulation(), true);
		//I'll handle the save as token so I need to consume them
		base.setLoadLiquid(false);
		base.setLoadSimulation(false);

		if (base.getZeroVelocities()) {
			sim->zeroFluidVelocities();
		}
		base.setZeroVelocities(false);
	}

	if (base.getPause())
		return;



	/*
	static int k = 0;
	k++;
	std::cout << k << std::endl;
	
	if (k == 1) {
		MiniGL::rotateY(0.001*1454);
		MiniGL::rotateX(0.00001*4784);
		MiniGL::move(0.001*1305, 0, 0);
	}
	
	

	//MiniGL::move(0.001, 0, 0);
	//MiniGL::move(0, 0.001, 0);
	//MiniGL::move(0, 0, 0.001);
	//MiniGL::move();
	//MiniGL::rotateX(0.00001);
	//MiniGL::rotateY(-0.0001);
	//*/

	// Simulation code
	for (unsigned int i = 0; i < base.getNumberOfStepsPerRenderUpdate(); i++)
	{

		START_TIMING("SimStep");
		base.getSimulationMethod().simulation->step();
		STOP_TIMING_AVG;

		if (!base.getRbPause()) {
			bool use_external_rb_engine = false;
			if ((use_external_rb_engine)&& (base.getSimulationMethod().simulationMethod == DemoBase::SimulationMethods::DFSPH_CUDA)) {
				DFSPHCUDA* sim = dynamic_cast<DFSPHCUDA*>(base.getSimulationMethod().simulation);
				//send the information to the physics engine
				sim->updateRigidBodiesStateToFile();

				//read the information back (there is a wait in this function)
				sim->updateRigidBodiesStatefromFile();
			}
			else {
				updateBoundaryForces();
		
				bool controlBoat = false;
				if (controlBoat) {
					bool manualBoatVelocityControl = false;
					bool manualBoatOrientationControl = false;

					FluidModel::RigidBodyParticleObject *rbpo = base.getSimulationMethod().model.getRigidBodyParticleObject(1);
					RigidBodyObject *rbo = rbpo->m_rigidBody;
					if (rbo->isDynamic())
					{
						//std::cout << "test nbr particle boat: " << rbpo->numberOfParticles() << std::endl;
						//Vector3r angles= rbo->getRotation().eulerAngles(0, 1, 2);
						//std::cout << "boat euler angles: " << angles[0] << "  " << angles[1] << "  " << angles[2] << std::endl;

						//a system to counter gravity to debug the boat control
						if (false&&rbo->getPosition()[1] < 1.5) {
							Vector3r f_counter_grav(0, 1200, 0);
							rbo->addForce(f_counter_grav);
						}



						//a system that deactivate a boat control key when the oposite key is pressed
						static bool boatForward_old = false;
						static bool boatBackward_old = false;
						static bool boatLeft_old = false;
						static bool boatRight_old = false;

						if (boatForward_old&&base.getBoatBackward()) {
							base.setBoatForward(false);
						}

						if (boatBackward_old&&base.getBoatForward()) {
							base.setBoatBackward(false);
						}

						if (boatLeft_old&&base.getBoatRight()) {
							base.setBoatLeft(false);
						}

						if (boatRight_old&&base.getBoatLeft()) {
							base.setBoatRight(false);
						}

						//and save the keyboard state for next iter
						boatForward_old = base.getBoatForward();
						boatBackward_old = base.getBoatBackward();
						boatLeft_old = base.getBoatLeft();
						boatRight_old = base.getBoatRight();

						//some intermediary boleans to integrate the atomatic boat control easily
						bool boatForward = false;
						bool boatBackward = false;
						bool boatLeft = false;
						bool boatRight = false;

						float controlForceIntensity = base.getBoatForceIntensity();
						if (manualBoatVelocityControl) {
							boatForward = boatForward_old;
							boatBackward = boatBackward_old;
						}
						else {
							//here code the scenarios you want for the automatic control
							controlForceIntensity = 5000; // 800;
							boatForward = true;
						}

						//apply the boat control
						Matrix3r rot_marix=rbo->getRotation();
						//*
						//this should have the boat base direction as a default value
						Vector3r boatControlForce(-1,0,0);
						if (boatForward) {
							boatControlForce *= controlForceIntensity;
						}
						else if (boatBackward) {
							boatControlForce *= -controlForceIntensity;
						}
						else {
							boatControlForce = Vector3r(0, 0, 0);
						}
						if (boatControlForce.norm() > 1e-6) {
							//convert the force to global coordinates
							boatControlForce= rot_marix*boatControlForce;
							
							//in fact I want the force to be in the horizontal plane.
							// I'll go the easy and dirty way
							Real norm_f = boatControlForce.norm();
							boatControlForce[1] = 0;
							boatControlForce *= norm_f / boatControlForce.norm();

							//apply the force
							rbo->addForce(boatControlForce);
						}
						//*/

						//*

						float controlTorqueIntensity = base.getBoatTorqueIntensity();
						if (manualBoatOrientationControl) {
							boatLeft = boatLeft_old;
							boatRight = boatRight_old;
						}
						else {
							//here code the scenarios you want for the automatic control
							Vector3r boatDirLocal(-1, 0, 0);
							Vector3r boatDirGlobal=rbo->getRotation()*boatDirLocal;
							
							//I want the direction in the Horizontal plane 
							boatDirGlobal[1] = 0;
							boatDirGlobal.normalize();

							Vector3r targetDir(-1, 0, 0);

							bool with_animation = true;
							if (with_animation) {
								if (TimeManager::getCurrent()->getTime() > 20) {
									targetDir = Vector3r(-0.25, 0, -0.75);
									targetDir.normalize();
								}else if (TimeManager::getCurrent()->getTime() > 10) {
									targetDir = Vector3r(-0.25, 0, 0.75);
									targetDir.normalize();
								}
							}


							//and now depending on the target direction activatethe left or right control
							//for the intensity I want the actual angle
							//most likely I need a pd-controler for that
							//float angleToTarget=acosf(targetDir.dot(boatDirGlobal));

							//currently using a fixed intesity since it works well
							//controlTorqueIntensity = 50;
							controlTorqueIntensity = 400;

							//and wealso need to know the direction
							float crossProdRes = targetDir.cross(boatDirGlobal).y();
							if (crossProdRes > 0) {
								boatRight = true;
							}
							else {
								boatLeft = true;
							}

							//std::cout << "automatic boat control internal values: " << angleToTarget << "   " << crossProdRes << std::endl;
						}

						bool apply_local_torque = false;
						Vector3r boatControlTorque(0, 1, 0);
						if (apply_local_torque) {
							boatControlTorque= Vector3r(0, 0, -1);
						}
						if (boatLeft) {
							boatControlTorque *= controlTorqueIntensity;
						}
						else if (boatRight) {
							boatControlTorque *= -controlTorqueIntensity;
						}
						else {
							boatControlTorque = Vector3r(0, 0, 0);
						}
						if (boatControlTorque.norm() > 1e-6) {
							//convert the torque to global coordinates
							if (apply_local_torque) {
								boatControlTorque = rbo->getRotation() * boatControlTorque;
							}

							//apply the torque
							rbo->addTorque(boatControlTorque);
						}
						//*/

						//std::cout << "control intensity force/torque: " << controlForceIntensity << "  " << controlTorqueIntensity << std::endl;

					}
				}

				

			//////////////////////////////////////////////////////////////////////////
			// PBD
			//////////////////////////////////////////////////////////////////////////
				START_TIMING("SimStep - PBD");
				pbdWrapper.timeStep();
				STOP_TIMING_AVG;

				updateBoundaryParticles(false);

				bool camera_follow_boat = false;
				if (camera_follow_boat) {
					FluidModel::RigidBodyParticleObject *rbpo = base.getSimulationMethod().model.getRigidBodyParticleObject(1);
					RigidBodyObject *rbo = rbpo->m_rigidBody;
					if (rbo->isDynamic())
					{
						Vector3r newBoatpos;
						newBoatpos = rbo->getPosition();

						//let's use the lookat procedure from the setViewport function since it's easier for now
						Vector3r eyePos = newBoatpos;
						Vector3r lookAt = newBoatpos;
						
						//the type of camera used
						//0: folow small boat
						//1: follow large boat
						int camera_type = 1;
						if (camera_type == 0) {
							eyePos[0] += 0.5;
							eyePos[1] = 3;
							eyePos[2] += 5;
							lookAt[1] = 1;
						}
						else if (camera_type == 1) {
							eyePos[0] += 0.5*1.66;
							eyePos[1] = 2*1.66;
							eyePos[2] += 5*1.66;
							lookAt[1] = 1;
						}
						

						MiniGL::setViewport(40.0, 0.1f, 500.0, eyePos, lookAt);
					}
				}
			}
		}

		

		if (base.getEnablePartioExport())
		{
			if (TimeManager::getCurrent()->getTime() >= nextFrameTime)
			{
				nextFrameTime += 1.0 / base.getFramesPerSecond();
				partioExport();
				frameCounter++;
			}
		}
	}

#ifdef FFMPEG_RENDER
	//the part to save to file
	if (ffmpeg != NULL) {
		if (TimeManager::getCurrent()->getTime()>20) {
			_pclose(ffmpeg);
			exit(0);
		}
	}
#endif
}

void simulationMethodChanged()
{
	pbdWrapper.initGUI();
	reset();
}

void renderBoundary()
{
	DemoBase::SimulationMethod &simulationMethod = base.getSimulationMethod();
	Shader &shader = base.getShader();
	Shader &meshShader = base.getMeshShader();
	SceneLoader::Scene &scene = base.getScene();
	const int renderWalls = base.getRenderWalls();

	float wallColor[4] = { 0.1f, 0.6f, 0.6f, 1.0f };
	if ((renderWalls == 1) || (renderWalls == 2))
	{
		if (MiniGL::checkOpenGLVersion(3, 3))
		{
			shader.begin();
			glUniform3fv(shader.getUniform("color"), 1, &wallColor[0]);


			if (base.isDFSPH()) {
				base.renderBoundariesDFSPH_CUDA(renderWalls == 1);
			}else{
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
}


void render()
{
	MiniGL::coordinateSystem();

	static int width = glutGet(GLUT_WINDOW_WIDTH);
	static int height = glutGet(GLUT_WINDOW_HEIGHT);


#ifdef FFMPEG_RENDER
	static int* buffer = new int[width*height];

	if (!base.getPause()) {
		if (ffmpeg == NULL) {
			int framerate = (1 / TimeManager::getCurrent()->getTimeStepSize());
			std::cout << "video framerate: " << framerate << std::endl;
			std::ostringstream oss;
			// start ffmpeg telling it to expect raw rgba 720p-60hz frames
			// -i - tells it to read frames from stdin
			oss << "D:\\ffmpeg-4.1.3-win64-static\\bin\\ffmpeg " <<
				" -r " << framerate << " -f rawvideo -pix_fmt rgba -s " << width << "x" << height << " -i - " <<
				"-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4";



			// open pipe to ffmpeg's stdin in binary write mode
			ffmpeg = _popen(oss.str().c_str(), "wb");
		}
	}
#endif

	base.renderFluid();



	renderBoundary();

	//////////////////////////////////////////////////////////////////////////
	// PBD
	//////////////////////////////////////////////////////////////////////////

	PBD::SimulationModel &model = pbdWrapper.getSimulationModel();
	PBD::SimulationModel::RigidBodyVector &rb = model.getRigidBodies();

	const int renderWalls = base.getRenderWalls();
	SceneLoader::Scene &scene = base.getScene();
	if ((renderWalls == 3) || (renderWalls == 4))
	{
		for (size_t i = 0; i < rb.size(); i++)
		{
			const PBD::VertexData &vd = rb[i]->getGeometry().getVertexData();
			const PBD::IndexedFaceMesh &mesh = rb[i]->getGeometry().getMesh();
			if ((renderWalls == 3) || (!scene.boundaryModels[i]->isWall))
			{
				float *col = &scene.boundaryModels[i]->color[0];
				if (!scene.boundaryModels[i]->isWall)
				{
					base.meshShaderBegin(col);
					pbdWrapper.drawMesh(vd, mesh, 0, col);
					base.meshShaderEnd();
				}
				else
				{
					base.meshShaderBegin(col);
					pbdWrapper.drawMesh(vd, mesh, 0, col);
					base.meshShaderEnd();
				}
			}
		}
	}

	pbdWrapper.renderTriangleModels();
	pbdWrapper.renderTetModels();
	pbdWrapper.renderConstraints();
	pbdWrapper.renderBVH();


#ifdef FFMPEG_RENDER
	if (ffmpeg != NULL) {
		glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);

		fwrite(buffer, sizeof(int)*width*height, 1, ffmpeg);
	}

#endif

}

void initBoundaryData()
{
	std::string base_path = FileSystem::getFilePath(base.getSceneFile());
	SceneLoader::Scene &scene = base.getScene();
	const bool useCache = base.getUseParticleCaching();

	std::cout << "nbr of boundary models: "<< scene.boundaryModels.size() <<std::endl;

	for (unsigned int i = 0; i < scene.boundaryModels.size(); i++)
	{
		std::vector<Vector3r> boundaryParticles;
		if (scene.boundaryModels[i]->samplesFile != "")
		{
			string particleFileName = base_path + "/" + scene.boundaryModels[i]->samplesFile;
			PartioReaderWriter::readParticles(particleFileName, Vector3r::Zero(), Matrix3r::Identity(), scene.boundaryModels[i]->scale[0], boundaryParticles);
		}

		PBD::SimulationModel &model = pbdWrapper.getSimulationModel();
		PBD::SimulationModel::RigidBodyVector &rigidBodies = model.getRigidBodies();
		PBDRigidBody *rb = new PBDRigidBody(rigidBodies[i]);
		PBD::RigidBodyGeometry &geo = rigidBodies[i]->getGeometry();
		PBD::IndexedFaceMesh &mesh = geo.getMesh();
		PBD::VertexData &vd = geo.getVertexData();

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
				std::cout << "Surface sampling of " << scene.boundaryModels[i]->meshFile << "\n";
				START_TIMING("Poisson disk sampling");
				PoissonDiskSampling sampling;
				sampling.sampleMesh(mesh.numVertices(), &vd.getPosition(0), mesh.numFaces(), mesh.getFaces().data(), scene.particleRadius, 10, 1, boundaryParticles);
				STOP_TIMING_AVG;

				// Cache sampling
				if (useCache && (FileSystem::makeDir(cachePath) == 0))
				{
					std::cout << "Save particle sampling: " << particleFileName << "\n";
					PartioReaderWriter::writeParticles(particleFileName, (unsigned int)boundaryParticles.size(), boundaryParticles.data(), NULL, scene.particleRadius);

					FileSystem::writeMD5File(base.getSceneFile(), md5FileName);
				}
			}
			// transform back to local coordinates
			for (unsigned int j = 0; j < boundaryParticles.size(); j++) {
				boundaryParticles[j] = rb->getRotation().transpose() * (boundaryParticles[j] - rb->getPosition());
			}
		}
		base.getSimulationMethod().model.addRigidBodyObject(rb, static_cast<unsigned int>(boundaryParticles.size()), &boundaryParticles[0]);
	}
	updateBoundaryParticles(true);
}

void updateBoundaryParticles(const bool forceUpdate = false)
{
	SceneLoader::Scene &scene = base.getScene();
	const unsigned int nObjects = base.getSimulationMethod().model.numberOfRigidBodyParticleObjects();	
	for (unsigned int i = 0; i < nObjects; i++)
	{
		FluidModel::RigidBodyParticleObject *rbpo = base.getSimulationMethod().model.getRigidBodyParticleObject(i);
		RigidBodyObject *rbo = rbpo->m_rigidBody;
		if (rbo->isDynamic() || forceUpdate)
		{
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)  
				for (int j = 0; j < (int)rbpo->numberOfParticles(); j++)
				{
					rbpo->m_x[j] = rbo->getRotation() * rbpo->m_x0[j] + rbo->getPosition();
					rbpo->m_v[j] = rbo->getAngularVelocity().cross(rbpo->m_x[j] - rbo->getPosition()) + rbo->getVelocity();
				}
			}
		}
	}
}

void updateBoundaryForces()
{
	Real h = TimeManager::getCurrent()->getTimeStepSize();
	SceneLoader::Scene &scene = base.getScene();
	const unsigned int nObjects = base.getSimulationMethod().model.numberOfRigidBodyParticleObjects();	
	for (unsigned int i = 0; i < nObjects; i++)
	{
		FluidModel::RigidBodyParticleObject *rbpo = base.getSimulationMethod().model.getRigidBodyParticleObject(i);
		RigidBodyObject *rbo = rbpo->m_rigidBody;
		if (rbo->isDynamic())
		{
			((PBDRigidBody*)rbo)->updateTimeStepSize();
			Vector3r force, torque;
			force.setZero();
			torque.setZero();

			for (int j = 0; j < (int)rbpo->numberOfParticles(); j++)
			{
				force += rbpo->m_f[j];
				torque += (rbpo->m_x[j] - rbo->getPosition()).cross(rbpo->m_f[j]);
				rbpo->m_f[j].setZero();
			}
			rbo->addForce(force);
			rbo->addTorque(torque);
		}
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

	PartioReaderWriter::writeParticles(exportFileName, model.numActiveParticles(), &model.getPosition(0, 0), &model.getVelocity(0, 0), 0.0);
}