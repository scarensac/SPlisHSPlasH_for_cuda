# Install script for directory: D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/SPlishSPlasH")
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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/build/extern/freeglut/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/build/extern/AntTweakBar/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/build/extern/glew/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/build/extern/zlib/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/build/extern/partio/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/build/extern/MD5/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/build/Demos/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/build/SPlisHSPlasH/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/build/Utilities/cmake_install.cmake")
  include("D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/build/Tools/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
