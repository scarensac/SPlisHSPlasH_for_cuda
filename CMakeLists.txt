cmake_minimum_required(VERSION 3.9)

project(SPlishSPlasH LANGUAGES C CXX CUDA)

## add_definitions(-DUSE_DOUBLE)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
 
set(PROJECT_PATH ${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(${PROJECT_PATH}/CMake/Common.cmake)

string(APPEND CMAKE_CUDA_FLAGS "  -gencode arch=compute_61,code=sm_61")

 
message(STATUS "CUDA compiler: " ${CMAKE_CUDA_COMPILER})

if (WIN32)
  subdirs(
  extern/freeglut 
  extern/AntTweakBar 
  extern/glew 
  extern/zlib 
  extern/partio 
  extern/MD5
  Demos SPlisHSPlasH Utilities Tools)
else()
  subdirs(
	extern/AntTweakBar 
	extern/glew 
	extern/zlib 
	extern/partio 
	extern/md5
	Demos SPlisHSPlasH Utilities Tools)
endif()

add_definitions(-DSPH_DATA_PATH="../data")

message(STATUS "message to display: " ${CMAKE_CUDA_FLAGS})
message(STATUS "message to display2: " ${CMAKE_CUDA_STANDARD})


## PositionBasedDynamics
include(ExternalProject)
set(ExternalInstallDir "${CMAKE_SOURCE_DIR}/extern/install")
ExternalProject_Add(
   ExternalProject_PositionBasedDynamics
   PREFIX "${CMAKE_SOURCE_DIR}/extern/PositionBasedDynamics"
   GIT_REPOSITORY https://github.com/InteractiveComputerGraphics/PositionBasedDynamics.git
   GIT_TAG "39a5099c39df84494df14aaad6fb9af845793ff7"
   INSTALL_DIR ${ExternalInstallDir}/PositionBasedDynamics
   CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ExternalInstallDir}/PositionBasedDynamics -DPBD_NO_DEMOS:BOOL=1		
) 

## CompactNSearch
ExternalProject_Add(
   ExternalProject_CompactNSearch
   PREFIX "${CMAKE_SOURCE_DIR}/extern/CompactNSearch"
   GIT_REPOSITORY https://github.com/InteractiveComputerGraphics/CompactNSearch.git
   GIT_TAG "1d9cf800ab4cbb2af956577b99fdd355c16181ba"
   INSTALL_DIR ${ExternalInstallDir}/CompactNSearch
   CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ExternalInstallDir}/CompactNSearch
) 




