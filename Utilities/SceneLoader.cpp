#include "SceneLoader.h"
#include <iostream>
#include "extern/json/json.hpp"
#include <fstream>
#include "FileSystem.h"

using namespace SPH;



void SceneLoader::readScene(const char *fileName, Scene &scene)
{
	std::cout << "Load scene file: " << fileName << "\n";

	std::ifstream input_file(fileName);
	if (!input_file.is_open())
	{
		std::cerr << "Cannot open file!\n";
		return;
	}
	nlohmann::json j;
	try
	{
		j << input_file;
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		exit(1);
	}	

	std::string base_path = FileSystem::getFilePath(fileName);

	//////////////////////////////////////////////////////////////////////////
	// read configuration 
	//////////////////////////////////////////////////////////////////////////
	if (j.find("Configuration") != j.end())
	{
		nlohmann::json config = j["Configuration"];

		scene.timeStepSize = 0.001;
		readValue(config["timeStepSize"], scene.timeStepSize);

		scene.particleRadius = 0.025;
		readValue(config["particleRadius"], scene.particleRadius);

		scene.pauseAt = -1.0;
		readValue(config["pauseAt"], scene.pauseAt);

		scene.numberOfStepsPerRenderUpdate = 8;
		readValue(config["numberOfStepsPerRenderUpdate"], scene.numberOfStepsPerRenderUpdate);

		scene.renderMaxVelocity = 25.0;
		readValue(config["renderMaxVelocity"], scene.renderMaxVelocity);

		scene.renderAngularVelocities = false;
		readValue(config["renderAngularVelocities"], scene.renderAngularVelocities);

		scene.cflMethod = 1;
		readValue(config["cflMethod"], scene.cflMethod);

		scene.cflFactor = 0.5;
		readValue(config["cflFactor"], scene.cflFactor);

		scene.cflMaxTimeStepSize = 0.005;
		readValue(config["cflMaxTimeStepSize"], scene.cflMaxTimeStepSize);

		scene.simulationMethod = 4;
		readValue(config["simulationMethod"], scene.simulationMethod);

		scene.maxIterations = 100;
		readValue(config["maxIterations"], scene.maxIterations);

		scene.maxError = 0.01;
		readValue(config["maxError"], scene.maxError);

		scene.maxIterationsV = 100;
		readValue(config["maxIterationsV"], scene.maxIterationsV);

		scene.maxErrorV = 0.1;
		readValue(config["maxErrorV"], scene.maxErrorV);

		scene.viscosity = 0.02;
		readValue(config["viscosity"], scene.viscosity);

		scene.viscoMaxIter = 50;
		readValue(config["viscoMaxIter"], scene.viscoMaxIter);

		scene.viscoMaxError = 0.01;
		readValue(config["viscoMaxError"], scene.viscoMaxError);

		scene.viscosityMethod = 2;
		readValue(config["viscosityMethod"], scene.viscosityMethod);

		scene.surfaceTension = 0.05;
		readValue(config["surfaceTension"], scene.surfaceTension);

		scene.surfaceTensionMethod = 0;
		readValue(config["surfaceTensionMethod"], scene.surfaceTensionMethod);

		scene.vorticityMethod = 0;
		readValue(config["vorticityMethod"], scene.vorticityMethod);

		scene.vorticityCoeff = 0.01;
		readValue(config["vorticityCoeff"], scene.vorticityCoeff);

		scene.viscosityOmega = 0.1;
		readValue(config["viscosityOmega"], scene.viscosityOmega);

		scene.inertiaInverse = 0.5;
		readValue(config["inertiaInverse"], scene.inertiaInverse);

		scene.dragMethod = 0;
		readValue(config["dragMethod"], scene.dragMethod);

		scene.dragCoefficient = 0.01;
		readValue(config["dragCoefficient"], scene.dragCoefficient);

		scene.density0 = 1000.0;
		readValue(config["density0"], scene.density0);

		scene.gravitation = Vector3r(0.0, -9.81, 0.0);
		readVector(config["gravitation"], scene.gravitation);

		scene.velocityUpdateMethod = 0;
		readValue(config["velocityUpdateMethod"], scene.velocityUpdateMethod);

		scene.stiffness = 50000.0;
		readValue(config["stiffness"], scene.stiffness);

		scene.exponent = 7.0;
		readValue(config["exponent"], scene.exponent);

		scene.enableDivergenceSolver = true;
		readValue(config["enableDivergenceSolver"], scene.enableDivergenceSolver);

		scene.maxEmitterParticles = 1000;
		readValue(config["maxEmitterParticles"], scene.maxEmitterParticles);

		scene.enablePartioExport = false;
		readValue(config["enablePartioExport"], scene.enablePartioExport);

		scene.partioFPS = 25;
		readValue(config["partioFPS"], scene.partioFPS);

		// reuse particles if they are outside of a bounding box
		scene.emitterReuseParticles = false;
		readValue(config["emitterReuseParticles"], scene.emitterReuseParticles);

		// boxMin
		scene.emitterBoxMin = Vector3r(-1.0, -1.0, -1.0);
		readVector(config["emitterBoxMin"], scene.emitterBoxMin);

		// boxMax
		scene.emitterBoxMax = Vector3r(1.0, 1.0, 1.0);
		readVector(config["emitterBoxMax"], scene.emitterBoxMax);
	}

	//////////////////////////////////////////////////////////////////////////
	// read boundary models
	//////////////////////////////////////////////////////////////////////////
	if (j.find("RigidBodies") != j.end())
	{
		nlohmann::json boundaryModels = j["RigidBodies"];
		for (auto& boundaryModel : boundaryModels)
		{
			std::string particleFile = "";
			std::string meshFile = "";
			const bool bMesh = readValue<std::string>(boundaryModel["geometryFile"], meshFile);
			const bool bSamples = readValue<std::string>(boundaryModel["particleFile"], particleFile);

			if (bMesh || bSamples)
			{
				BoundaryData *data = new BoundaryData();
				data->meshFile = meshFile;
				data->samplesFile = particleFile;

				// translation
				data->translation = Vector3r::Zero();
				readVector(boundaryModel["translation"], data->translation);

				// rotation axis
				Vector3r axis = Vector3r::Zero();
				Real angle = 0.0;
				data->rotation = Matrix3r::Identity();
				if (readVector(boundaryModel["rotationAxis"], axis) &&
					readValue<Real>(boundaryModel["rotationAngle"], angle))
					data->rotation = AngleAxisr(angle, axis);

				// scale
				data->scale = Vector3r::Ones();
				readVector(boundaryModel["scale"], data->scale);

				data->dynamic = false;
				readValue<bool>(boundaryModel["isDynamic"], data->dynamic);

				data->isWall = false;
				readValue<bool>(boundaryModel["isWall"], data->isWall);

				data->color = Eigen::Vector4f(1.0f, 0.0f, 0.0f, 0.0f);
				readVector(boundaryModel["color"], data->color);

				scene.boundaryModels.push_back(data);
			}
		}
	}


	//////////////////////////////////////////////////////////////////////////
	// read fluid models
	//////////////////////////////////////////////////////////////////////////
	if (j.find("FluidModels") != j.end())
	{
		nlohmann::json fluidModels = j["FluidModels"];
		for (auto& fluidModel : fluidModels)
		{
			std::string particleFile;
			if (readValue<std::string>(fluidModel["particleFile"], particleFile))
			{
				FluidData *data = new FluidData();
				data->samplesFile = particleFile;

				// translation
				data->translation = Vector3r::Zero();
				readVector(fluidModel["translation"], data->translation);

				// rotation axis
				Vector3r axis = Vector3r::Zero();
				Real angle = 0.0;
				data->rotation = Matrix3r::Identity();
				if (readVector(fluidModel["rotationAxis"], axis) &&
					readValue<Real>(fluidModel["rotationAngle"], angle))
					data->rotation = AngleAxisr(angle, axis);

				// scale
				data->scale = 1.0;
				readValue(fluidModel["scale"], data->scale);

				scene.fluidModels.push_back(data);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// read fluid blocks
	//////////////////////////////////////////////////////////////////////////
	if (j.find("FluidBlocks") != j.end())
	{
		nlohmann::json fluidBlocks = j["FluidBlocks"];
		for (auto& fluidBlock : fluidBlocks)
		{
			// translation
			Vector3r translation = Vector3r::Zero();
			readVector(fluidBlock["translation"], translation);

			// scale
			Vector3r scale = Vector3r::Ones();
			readVector(fluidBlock["scale"], scale);

			Vector3r minX, maxX;
			if (readVector(fluidBlock["start"], minX) &&
				readVector(fluidBlock["end"], maxX))
			{
				FluidBlock *block = new FluidBlock();
				block->box.m_minX[0] = scale[0] * minX[0] + translation[0];
				block->box.m_minX[1] = scale[1] * minX[1] + translation[1];
				block->box.m_minX[2] = scale[2] * minX[2] + translation[2];
				block->box.m_maxX[0] = scale[0] * maxX[0] + translation[0];
				block->box.m_maxX[1] = scale[1] * maxX[1] + translation[1];
				block->box.m_maxX[2] = scale[2] * maxX[2] + translation[2];

				readValue(fluidBlock["denseMode"], block->mode);

				// velocity
				block->initialVelocity = Vector3r::Zero();
				readVector(fluidBlock["initialVelocity"], block->initialVelocity);

				scene.fluidBlocks.push_back(block);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// read emitters
	//////////////////////////////////////////////////////////////////////////
	if (j.find("Emitters") != j.end())
	{
		nlohmann::json emitters = j["Emitters"];
		for (auto& emitter : emitters)
		{
			EmitterData *data = new EmitterData();

			// width
			data->width = 5;
			readValue(emitter["width"], data->width);

			// height
			data->height = 5;
			readValue(emitter["height"], data->height);

			// translation
			data->x = Vector3r::Zero();
			readVector(emitter["translation"], data->x);

			// direction
			data->dir = Vector3r(1.0, 0.0, 0.0);
			readVector(emitter["direction"], data->dir);

			// velocity
			data->v = Vector3r(1.0, 0.0, 0.0);
			readVector(emitter["velocity"], data->v);

			// emits per second
			data->emitsPerSecond = 10;
			readValue(emitter["emitsPerSecond"], data->emitsPerSecond);

			// type: 0 = rectangular, 1 = circle
			data->type = 0;
			readValue(emitter["type"], data->type);

			scene.emitters.push_back(data);
		}
	}
}


template <>
bool SceneLoader::readValue(const nlohmann::json &j, bool &v)
{
	if (j.is_null())
		return false;

	if (j.is_number_integer())
	{
		int val = j.get<int>();
		v = val != 0;
	}
	else
		v = j.get<bool>();
	return true;
}
