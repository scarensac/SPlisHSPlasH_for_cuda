This project is build from the SPlisHSPlasH framework:
https://github.com/InteractiveComputerGraphics/SPlisHSPlasH

Modification are made to implement the DFSPH algorithm on gpu.

This project requires CUDA in top of the requirements of the original project.

Note:
this project still use the mass based impelmentation and not the volume based implementation currently used in the splishsplash project for the multifluid systems


The GPU implementation in this repository contains multiples optimizations that can be activated/deactivated through the use of the #defines in the files DFSPH_define_c.h and DFSPH_define_cuda.h


This project also contains the following systems that can be activated inside the stap function in the DFSPH_CUDA class:
	- an implemantation of a special initializations system to start a simulation with a fluid at rest for any shape of boundary
		See the RestFluidLoader.h file to explore this system

	- a "simple" open boundary method that can absob any wave generated inside the simulated area
		See the OpenBoundarySimple.h file to explore this system

	- a dynamic window that can be used to move the simulated area to follow a subject of interect (e.g. a boat)
		See the DynamicWindow.h file to explore this system