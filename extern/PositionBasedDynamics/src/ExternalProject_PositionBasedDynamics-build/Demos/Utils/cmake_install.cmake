# Install script for directory: D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics/Demos/Utils

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
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics/lib/Debug/Utils_d.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics/lib/Release/Utils.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics/lib/MinSizeRel/Utils_ms.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics/lib/RelWithDebInfo/Utils_rd.lib")
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Demos/Utils" TYPE DIRECTORY FILES "D:/Users/scarensac/Documents/GitHub/SPlisHSPlasH/extern/PositionBasedDynamics/src/ExternalProject_PositionBasedDynamics/Demos/Utils/." FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

