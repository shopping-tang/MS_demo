#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include <queue>
#include <thread>
#include <mutex>

#include "main.h"

using namespace std;

// single camera
int main()
{
  int cpus = 0;
  const char *model_path = "/home/linaro/Videos/mobilenet_ssd.rknn";
  FILE *fp = fopen(model_path, "rb");
  if(fp == NULL) {
    printf("fopen %s fail!\n", model_path);
    return -1;
  }
  fseek(fp, 0, SEEK_END);

  unsigned int model_len = ftell(fp);
  void *model = malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if(model_len != fread(model, 1, model_len, fp)) {
    printf("fread %s fail!\n", model_path);
    free(model);
    return -1;
  }

  cpus = sysconf(_SC_NPROCESSORS_CONF);
  printf("This system has %d processor(s).\n", cpus);

  mutex mtxQueueInput_1, mtxQueueInput_2;               // mutex of input queue
  //mutex mtxQueueShow_1,  mtxQueueShow_2 ;

  p_queue queueInput_1, queueInput_2;  // input queue
  //p_queue queueShow_1 , queueShow_2 ;  // display queue

  const int thread_count = 6;
  array<thread, thread_count> threads;
  threads = {thread(get_video_play, 5, 10, 0, 0, 128, 0, IMAGE_WIDTH*2, IMAGE_HEIGHT*2, std::ref(mtxQueueInput_1), std::ref(queueInput_1)),
             thread(run_process, 1, 0, model, model_len, std::ref(mtxQueueInput_1), std::ref(queueInput_1)),
             thread(run_process, 3, 1, model, model_len, std::ref(mtxQueueInput_1), std::ref(queueInput_1)),

             thread(get_video_play, 4, 12, 0, 2, 128, 1088, IMAGE_WIDTH*2, IMAGE_HEIGHT*2, std::ref(mtxQueueInput_2), std::ref(queueInput_2)),
             thread(run_process, 0, 0, model, model_len, std::ref(mtxQueueInput_2), std::ref(queueInput_2)),
             thread(run_process, 2, 1, model, model_len, std::ref(mtxQueueInput_2), std::ref(queueInput_2)),
             };

  for (int i = 0; i < thread_count; i++){
    threads[i].join();
  }
  return 0;
}

