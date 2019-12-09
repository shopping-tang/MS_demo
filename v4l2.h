#ifndef __V4L2_H__
#define __V4L2_H__

#include <sys/mman.h>
#include <linux/types.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <stdlib.h>
#include <linux/videodev2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "main.h"

using namespace cv;
#define  COUNT  4

typedef struct buffer
{
    int fd;
    void * start[COUNT];
} BUF ;

extern "C"
{
    void yuyv2bgr24(unsigned char*yuyv, unsigned char*rgb);
    void yuyv2bgra32(unsigned char*yuyv, unsigned char*bgra);
    void unchar_to_Mat(unsigned char *_buffer, cv::Mat& img);
    int v4l2(char *FILE_VIDEO, BUF *buffers);
    unsigned char *get_img(BUF *buffers, unsigned char *srcBuffer);
    int close_v4l2(BUF *buffers);

}


#endif
