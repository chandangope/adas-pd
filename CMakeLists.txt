cmake_minimum_required(VERSION 3.5)

#message("Your CMake version supports these C++ features:")
#get_property(known_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
#foreach(i ${known_features})
#  message("${i}")
#endforeach()

project(ivadaspdtest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
option(USE_G "Use -g with gcc" YES)

if (USE_G)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O1 -v")
else (USE_G)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif (USE_G)

MESSAGE("CMAKEPATH= ${CMAKE_MODULE_PATH}")
#find_package( OpenCV REQUIRED )

#MESSAGE("OPENCVPATH= ${OpenCV_INCLUDE_DIRS}")
#MESSAGE("OPENCVVER= ${OpenCV_VERSION}")

include_directories(
	include
	/usr/local/include/opencv 
	/usr/local/include
	)

file(GLOB SOURCES 
	src/*.cpp
	)

MESSAGE("SOURCES = ${SOURCES}")

#For the shared library:
set ( PROJECT_LINK_LIBS libivadaspd.so libtensorflow.so
	libopencv_core.so libopencv_highgui.so libopencv_imgcodecs.so libopencv_imgproc.so libopencv_videoio.so)
link_directories( lib  /usr/local/lib)

add_executable(testivadaspd ${SOURCES})
target_link_libraries(testivadaspd ${PROJECT_LINK_LIBS} )

set_property(TARGET testivadaspd PROPERTY CXX_STANDARD 11)
set_property(TARGET testivadaspd PROPERTY CXX_STANDARD_REQUIRED ON)
