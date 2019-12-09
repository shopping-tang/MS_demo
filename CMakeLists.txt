cmake_minimum_required(VERSION 2.8)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#project(gles)
#set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(#../rk_opengl/native/include
					/usr/lib64/libmali/include
					/usr/include/libdrm
					./
                      /usr/local/include/opencv4
                     /usr/lib64/rknn_api_sdk/rknn_api/include)
#include_directories(/usr/lib64/libmali/include
                     #/usr/local/include/opencv4
                     #${CMAKE_SOURCE_DIR}/rknn_api/include
                     #./)

#link_directories(./native/lib)
#link_directories(/usr/lib)
set(link_libs /usr/lib/aarch64-linux-gnu/libdrm.so
			  /usr/lib/libmali-gbm.so
			  /usr/lib/aarch64-linux-gnu/libpthread.so
			  /usr/local/lib/libopencv_core.so
               /usr/local/lib/libopencv_highgui.so
               /usr/local/lib/libopencv_imgcodecs.so
               /usr/local/lib/libopencv_imgproc.so
		       /usr/local/lib/libopencv_videoio.so
               /usr/lib64/rknn_api_sdk/rknn_api/lib64/librknn_api.so)
add_executable(gles gles_main.cpp gles_base.cpp gles_drm.cpp v4l2.cpp main.cpp)
#add_executable(glsample_wl main_wayland.cpp)

#target_include_directories(gles PUBLIC ./native/include/EGLs/winsys_gbm)
#target_include_directories(glsample_wl PUBLIC ./native/include/EGLs/winsys_wayland)

target_link_libraries(gles ${link_libs})
#target_link_libraries(glsample_wl mali-wayland drm)

