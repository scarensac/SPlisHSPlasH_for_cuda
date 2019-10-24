#include "SPH_kernels.h"

#ifdef SPLISHSPLASH_FRAMEWORK
#include "SPlisHSPlasH/TimeManager.h"
#include "SimulationDataDFSPH.h"
#endif //SPLISHSPLASH_FRAMEWORK
#include "SPlisHSPlasH/SPHKernels.h"
#include "DFSPH_cuda_basic.h"
#include "DFSPH_define_cuda.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <thread>
#include <sstream>


#include "SPH_memory_storage_precomp_kernels.h"


using namespace SPH;
using namespace std;



void PrecomputedCubicKernelPerso::setRadius(RealCuda val)
{
	m_resolution = PRECOMPUTED_KERNELS_SAMPLE_COUNT;
	m_radius = val;
	m_radius2 = m_radius*m_radius;
	const RealCuda stepSize = m_radius / (RealCuda)m_resolution;
	m_invStepSize = 1.0 / stepSize;




	if (true) {
		RealCuda* W_temp = new RealCuda[m_resolution];
		RealCuda* gradW_temp = new RealCuda[m_resolution + 1];

		/*
		//directly from the existing precomp kernel
		for (unsigned int i = 0; i < m_resolution; ++i) {
			W_temp[i] = FluidModel::PrecomputedCubicKernel::m_W[i];
			gradW_temp[i] = FluidModel::PrecomputedCubicKernel::m_gradW[i];
		}
		//*/


		//init values
		CubicKernelPerso kernel;
		kernel.setRadius(val);

		for (unsigned int i = 0; i < m_resolution; i++)
		{
			const RealCuda posX = stepSize * (RealCuda)i;		// Store kernel values in the middle of an interval
			RealCuda W_val = kernel.W(posX);
			Vector3d grad_W_val;
			if (posX > 1.0e-9)
				grad_W_val = kernel.gradW(Vector3d(posX, 0.0, 0.0)).x / posX;
			else
				grad_W_val = 0.0;


			W_temp[i] = W_val;
			gradW_temp[i] = grad_W_val.x;
			//std::cout << "kernel comparison:  " << W_temp[i] << "  " << gradW_temp[i] << "    //    " << W_val << "  " << grad_W_val.x << std::endl;
			//std::cout << "kernel comparison:  " << W_temp[i] << "  " << gradW_temp[i]  << std::endl;
		}

		//std::cout << "kernel comparison:  " << m_radius << "  " << m_radius2 << "  " << m_invStepSize << "  " << std::endl;


		gradW_temp[m_resolution] = 0.0;
		m_W_zero = kernel.W(0.0);


#ifdef PRECOMPUTED_KERNELS_USE_CONSTANT_MEMORY
		writte_to_precomp_kernel(W_temp,gradW_temp,m_radius,m_radius2, m_invStepSize);
#else
		allocate_precomputed_kernel_managed(*this, true);
		init_precomputed_kernel_from_values(*this, W_temp, gradW_temp);
#endif

		//clean
		delete[] W_temp;
		delete[] gradW_temp;
	}
	else {
		m_W = new RealCuda[m_resolution];
		m_gradW = new RealCuda[m_resolution + 1];
		
		//init values
		CubicKernelPerso kernel;
		kernel.setRadius(val);
		for (unsigned int i = 0; i < m_resolution; i++)
		{
			const RealCuda posX = stepSize * (RealCuda)i;		// Store kernel values in the middle of an interval
			m_W[i] = kernel.W(posX);
			kernel.setRadius(val);
			if (posX > 1.0e-9)
				m_gradW[i] = kernel.gradW(Vector3d(posX, 0.0, 0.0)).x / posX;
			else
				m_gradW[i] = 0.0;
		}
		m_gradW[m_resolution] = 0.0;
		m_W_zero = W(0.0);
	}

}


void PrecomputedCubicKernelPerso::freeMemory(){
    free_precomputed_kernel_managed(*this);
}

