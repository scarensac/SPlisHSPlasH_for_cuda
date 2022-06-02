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

set(CMAKE_CUDA_ARCHITECTURES 75)

#if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL  "9.0.0")
#message(STATUS "using cuda sm61")
#string(APPEND CMAKE_CUDA_FLAGS "  -gencode arch=compute_61,code=sm_61")
#else()
#message(STATUS "using cuda sm30")
#string(APPEND CMAKE_CUDA_FLAGS "  -gencode arch=compute_30,code=sm_30")
#endif()
 
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

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

message(STATUS "cuda version: " ${CMAKE_CUDA_COMPILER_VERSION})
message(STATUS "cuda standart: " ${CMAKE_CUDA_STANDARD})
message(STATUS "cuda flags: " ${CMAKE_CUDA_FLAGS})


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

## Discregrid
ExternalProject_Add(
   Ext_Discregrid
   PREFIX "${CMAKE_SOURCE_DIR}/extern/Discregrid"
   GIT_REPOSITORY https://github.com/InteractiveComputerGraphics/Discregrid.git
   GIT_TAG "c0fb5aeac4c8a83e9f37c720315f13a834409b81"
   INSTALL_DIR ${ExternalInstallDir}/Discregrid
   CMAKE_ARGS -DCMAKE_BUILD_TYPE:STRING=${EXT_CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=${ExternalInstallDir}/Discregrid -DBUILD_CMD_EXECUTABLE:BOOL=0 -DEIGEN3_INCLUDE_DIR:PATH=${EIGEN3_INCLUDE_DIR}
) 


