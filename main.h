#ifndef __MAIN_H__
#define __MAIN_H__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <fstream>
#include <atomic>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>
#include <sys/time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <gbm.h>
#include <xf86drmMode.h>
#include <xf86drm.h>

#define GL_GLEXT_PROTOTYPES
#include <GLES/gl.h>
#include <GLES/egl.h>
#include <GLES/glext.h>

#include <GLES3/gl32.h>
#include <GLES3/gl3platform.h>
#include <linux/videodev2.h>

#include "rknn_api.h"
#include "gles_drm.h"
#include "gles_base.h"
//#include "Shader.h"
#include "v4l2.h"

#include <unistd.h>
#include <sys/syscall.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

#define NUM_RESULTS         1917
#define NUM_CLASSES         91

#define Y_SCALE  10.0f
#define X_SCALE  10.0f
#define H_SCALE  5.0f
#define W_SCALE  5.0f

#define  TRUE   1
#define  FALSE  0
#define  IMAGE_WIDTH    640    // input image width
#define  IMAGE_HEIGHT   480    // input image height

#define CHANNEL 3              // input image channel
#define BUFFER_SIZE_src IMAGE_WIDTH*IMAGE_HEIGHT*2
#define BUFFER_SIZE_det IMAGE_WIDTH*IMAGE_HEIGHT*CHANNEL

#define __AVE_TIC__(tag) static int ____##tag##_total_time=0; \
        static int ____##tag##_total_conut=0;\
        timeval ____##tag##_start_time, ____##tag##_end_time;\
        gettimeofday(&____##tag##_start_time, 0);

#define __AVE_TOC__(tag) gettimeofday(&____##tag##_end_time, 0); \
        ____##tag##_total_conut++; \
        ____##tag##_total_time+=((int)____##tag##_end_time.tv_sec-(int)____##tag##_start_time.tv_sec)*1000000+((int)____##tag##_end_time.tv_usec-(int)____##tag##_start_time.tv_usec); \
        fprintf(stderr,  #tag ": %d us\n", ____##tag##_total_time/____##tag##_total_conut);

#define __TIC__(tag) timeval ____##tag##_start_time, ____##tag##_end_time;\
        gettimeofday(&____##tag##_start_time, 0);

#define __TOC__(tag) gettimeofday(&____##tag##_end_time, 0); \
        int ____##tag##_total_time=((int)____##tag##_end_time.tv_sec-(int)____##tag##_start_time.tv_sec)*1000000+((int)____##tag##_end_time.tv_usec-(int)____##tag##_start_time.tv_usec); \
        fprintf(stderr,  #tag ": %d us\n", ____##tag##_total_time);

#ifdef SHOWTIME
#define _T(func)                                                              \
    {                                                                         \
        auto _start = system_clock::now();                                    \
        func;                                                                 \
        auto _end = system_clock::now();                                      \
        auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
        string tmp = #func;                                                   \
        tmp = tmp.substr(0, tmp.find('('));                                   \
        cout << "[TimeTest]" << left << setw(30) << tmp;                      \
        cout << left << setw(10) << duration << "us" << endl;                 \
    }
#else
#define _T(func) func;
#endif

typedef queue<pair<unsigned int, Mat>> p_queue;  // input queue

extern "C" {
    void run_process(int cpuid, unsigned int thread_id, void *pmodel, int model_len, mutex& mtxQueueInput, p_queue& queueInput);
    void get_video_play(int cpuid, int video_index, int crtc_index, int plane_index, uint32_t display_x, uint32_t display_y, uint32_t display_w, uint32_t display_h, mutex& mtxQueueInput, p_queue& queueInput);
}

#endif
