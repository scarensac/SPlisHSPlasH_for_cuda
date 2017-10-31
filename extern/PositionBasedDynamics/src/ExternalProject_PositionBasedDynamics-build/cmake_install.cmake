# Install script for directory: D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/install/PositionBasedDynamics")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics/./Common" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics/./data" FILES_MATCHING REGEX "/[^/]*\\.glsl$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics-build/extern/freeglut/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics-build/extern/AntTweakBar/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics-build/extern/glew/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics-build/Demos/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics-build/PositionBasedDynamics/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics-build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
