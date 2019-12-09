#include "main.h"
#include "Shader.h"

Scalar colorArray[10] = {
        Scalar(139,   0,   0, 255),
        Scalar(139,   0, 139, 255),
        Scalar(  0,   0, 139, 255),
        Scalar(  0, 100,   0, 255),
        Scalar(139, 139,   0, 255),
        Scalar(209, 206,   0, 255),
        Scalar(  0, 127, 255, 255),
        Scalar(139,  61,  72, 255),
        Scalar(  0, 255,   0, 255),
        Scalar(255,   0,   0, 255),
};

float MIN_SCORE = 0.4f;

float NMS_THRESHOLD = 0.45f;

int multi_npu_process_initialized[2] = { 0 , 0 };

static int _terminate = 0;

inline pid_t gettid()
{
  return syscall(__NR_gettid);
}

static void sigint_handler(int arg)
{
    _terminate = 1;
}

int loadLabelName(string locationFilename, string* labels) {
    ifstream fin(locationFilename);
    string line;
    int lineNum = 0;
    while(getline(fin, line))
    {
        labels[lineNum] = line;
        lineNum++;
    }
    return 0;
}

int loadCoderOptions(string locationFilename, float (*boxPriors)[NUM_RESULTS])
{
    const char *d = ", ";
    ifstream fin(locationFilename);
    string line;
    int lineNum = 0;
    while(getline(fin, line))
    {
        char *line_str = const_cast<char *>(line.c_str());
        char *p;
        p = strtok(line_str, d);
        int priorIndex = 0;
        while (p) {
            float number = static_cast<float>(atof(p));
            boxPriors[lineNum][priorIndex++] = number;
            p=strtok(nullptr, d);
        }
        if (priorIndex != NUM_RESULTS) {
            return -1;
        }
        lineNum++;
    }
    return 0;

}

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = max(0.f, min(xmax0, xmax1) - max(xmin0, xmin1));
    float h = max(0.f, min(ymax0, ymax1) - max(ymin0, ymin1));
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    return u <= 0.f ? 0.f : (i / u);
}

float expit(float x) {
    return (float) (1.0 / (1.0 + exp(-x)));
}

void decodeCenterSizeBoxes(float* predictions, float (*boxPriors)[NUM_RESULTS]) {

    for (int i = 0; i < NUM_RESULTS; ++i) {
        float ycenter = predictions[i*4+0] / Y_SCALE * boxPriors[2][i] + boxPriors[0][i];
        float xcenter = predictions[i*4+1] / X_SCALE * boxPriors[3][i] + boxPriors[1][i];
        float h = (float) exp(predictions[i*4 + 2] / H_SCALE) * boxPriors[2][i];
        float w = (float) exp(predictions[i*4 + 3] / W_SCALE) * boxPriors[3][i];

        float ymin = ycenter - h / 2.0f;
        float xmin = xcenter - w / 2.0f;
        float ymax = ycenter + h / 2.0f;
        float xmax = xcenter + w / 2.0f;

        predictions[i*4 + 0] = ymin;
        predictions[i*4 + 1] = xmin;
        predictions[i*4 + 2] = ymax;
        predictions[i*4 + 3] = xmax;
    }
}

int scaleToInputSize(float * outputClasses, int (*output)[NUM_RESULTS], int numClasses)
{
    int validCount = 0;
    // Scale them back to the input size.
    for (int i = 0; i < NUM_RESULTS; ++i) {
        float topClassScore = static_cast<float>(-1000.0);
        int topClassScoreIndex = -1;

        // Skip the first catch-all class.
        for (int j = 1; j < numClasses; ++j) {
            float score = expit(outputClasses[i*numClasses+j]);
            if (score > topClassScore) {
                topClassScoreIndex = j;
                topClassScore = score;
            }
        }

        if (topClassScore >= MIN_SCORE) {
            output[0][validCount] = i;
            output[1][validCount] = topClassScoreIndex;
            ++validCount;
        }
    }

    return validCount;
}

int nms(int validCount, float* outputLocations, int (*output)[NUM_RESULTS])
{
    for (int i=0; i < validCount; ++i) {
        if (output[0][i] == -1) {
            continue;
        }
        int n = output[0][i];
        for (int j=i + 1; j<validCount; ++j) {
            int m = output[0][j];
            if (m == -1) {
                continue;
            }
            float xmin0 = outputLocations[n*4 + 1];
            float ymin0 = outputLocations[n*4 + 0];
            float xmax0 = outputLocations[n*4 + 3];
            float ymax0 = outputLocations[n*4 + 2];

            float xmin1 = outputLocations[m*4 + 1];
            float ymin1 = outputLocations[m*4 + 0];
            float xmax1 = outputLocations[m*4 + 3];
            float ymax1 = outputLocations[m*4 + 2];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou >= NMS_THRESHOLD) {
                output[0][j] = -1;
            }
        }
    }

    return 0;
}

void get_video_play(int cpuid, int video_index, int crtc_index, int plane_index, uint32_t display_x, uint32_t display_y, uint32_t display_w, uint32_t display_h, mutex& mtxQueueInput, p_queue& queueInput)
{
  /***************** bind CPU *****************/
  int initialization_finished = 1;
  cpu_set_t mask;

  CPU_ZERO(&mask);
  CPU_SET(cpuid, &mask);

  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    cerr << "set thread affinity failed" << endl;

  printf("Bind get_video_play process to CPU %d\n", cpuid);
  /***************** v4l2 paramters *****************/
  BUF *buffer = (BUF *)malloc(1*(sizeof (BUF)));

  char video[25];
  snprintf(video, sizeof(video), "/dev/video%d", video_index);

  unsigned char *srcBuffer = (unsigned char*)malloc(sizeof(unsigned char) * BUFFER_SIZE_src);

  /***************** base paramters *****************/
  struct kms kms;
  struct display displays ;
  /***************** init egl gbm *****************/
  init_egl_gbm(&kms, &displays);

  /***************** opengles function *****************/
  Shader ourShader(1);

  GLuint VAO = bind_array() ;

  GLuint textures[3];
  //GLuint textures;
  init_texture_1(textures);

  eglSwapBuffers(displays.eglDisplay, displays.eglSurface); // ready to display , whrite image to gbmSurface .
  displays.bo = gbm_surface_lock_front_buffer(displays.gbmSurface); // get buffer_object

  uint32_t handle = gbm_bo_get_handle(displays.bo).u32;  // get handle of buffer_object
  uint32_t stride = gbm_bo_get_stride(displays.bo);
  int width = gbm_bo_get_width(displays.bo);
  int height = gbm_bo_get_height(displays.bo);

  //printf("handle=%d, stride=%d rect=%dx%d\n", handle, stride, width, height); // here is frame's parameters : handle, stride, width, height .

  uint32_t handles[4], strides[4], offsets[4];
  handles[0] = handle;
  strides[0] = stride;
  offsets[0] = 0;
  int ret = drmModeAddFB2(kms.fd, width, height, GBM_FORMAT_RGB888,
                                         handles, strides, offsets,
                                         &kms.fb_id, 0);
  if (ret) {
     LOGE("failed to create fb");
     //goto rm_fb;
     //return -1;
  }

  drmModeCrtcPtr saved_crtc = drmModeGetCrtc(kms.fd, kms.crtcs[crtc_index]);
  if (saved_crtc == NULL) {
      LOGE("failed to crtc: %m");
      //return -1;
  }
  while (true) {
    initialization_finished = 1;

    for (unsigned int i = 0; i < sizeof(multi_npu_process_initialized) / sizeof(int); i++) {
        if (multi_npu_process_initialized[i] == 0) {
            initialization_finished = 0;
        }
    }

    if (initialization_finished){ break; }
    sleep(1);
  }

  ret = v4l2(video, buffer) ;
  if (ret == 0){
     printf("v4l2 run failed .\n");
     //return 0 ;
  }

  unsigned int idxInputImage = 1;
  Mat img( IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC(3),cv::Scalar(0,0,0));
  while (!_terminate) {
      // read function
      srcBuffer = get_img(buffer, srcBuffer);

      struct gbm_bo *next_bo = NULL;

      glClearColor(1.0, 1.0, 1.0, 1);
      glClear(GL_COLOR_BUFFER_BIT);
      ourShader.Use();
      glBindVertexArray(VAO);
      draw_1(ourShader.Program, textures, srcBuffer);
      glBindVertexArray(0);

      eglSwapBuffers(displays.eglDisplay, displays.eglSurface); // ready to display , whrite image to gbmSurface .
      next_bo = gbm_surface_lock_front_buffer(displays.gbmSurface); // get buffer_object

      ret = drmModeSetPlane(kms.fd, kms.planes[plane_index], // display FrameBuffer
          kms.crtcs[crtc_index], kms.fb_id, 0, display_x, display_y, display_w, display_h,
          0, 0, width << 16, height << 16);
      if (ret) {
         printf("failed to set plane 0 %d\n", ret);
         //return 0;
      }

      gbm_surface_release_buffer(displays.gbmSurface, displays.bo);
      displays.bo = next_bo;

      Mat yuyv = Mat( IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC(2), srcBuffer, 0);

      cvtColor(yuyv, img, COLOR_YUV2BGR_YUYV);

      mtxQueueInput.lock();
      queueInput.push(make_pair(idxInputImage, img));
      if (queueInput.size() >= 30) {
         mtxQueueInput.unlock();
         cout << "[Warning]input queue size is " << queueInput.size() << endl;
         sleep(1);
      } else {
         mtxQueueInput.unlock();
      }
      idxInputImage = idxInputImage^1 ;


  }

  ret = drmModeSetCrtc(kms.fd, saved_crtc->crtc_id, saved_crtc->buffer_id,
                         saved_crtc->x, saved_crtc->y,
                         &kms.connector->connector_id, 1, &saved_crtc->mode);
  ourShader.Use_end();
  if (ret) {
     LOGE("failed to restore crtc: %m");
  }

  free(buffer);
  free(srcBuffer);
  close_v4l2(buffer);
}

void run_process(int cpuid, unsigned int thread_id, void *pmodel, int model_len, mutex& mtxQueueInput, p_queue& queueInput)
{
  const char *label_path = "/home/linaro/Videos/coco_labels_list.txt";
  const char *box_priors_path = "/home/linaro/Videos/box_priors.txt";
  const int img_width = 300;
  const int img_height = 300;
  const int img_channels = 3;
  const int input_index = 0;      // node name "reprocessor/sub"

  //const int output_elems1 = NUM_RESULTS * 4;
  //const uint32_t output_size1 = output_elems1 * sizeof(float);
  //const int output_index1 = 0;    // node name "concat"

  //const int output_elems2 = NUM_RESULTS * NUM_CLASSES;
  //const uint32_t output_size2 = output_elems2 * sizeof(float);
  //const int output_index2 = 1;    // node name "concat_1"

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpuid, &mask);

  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    cerr << "set thread affinity failed" << endl;

  printf("Bind NPU process(%d) to CPU %d\n", thread_id, cpuid);

  // Start Inference
  rknn_input inputs[1];
  rknn_output outputs[2];
  rknn_tensor_attr outputs_attr[2];

  int ret = 0;
  rknn_context ctx = 0;

  ret = rknn_init(&ctx, pmodel, model_len, RKNN_FLAG_PRIOR_MEDIUM);
  if(ret < 0) {
    printf("rknn_init fail! ret=%d\n", ret);
    return;
  }

  outputs_attr[0].index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
  if(ret < 0) {
    printf("rknn_query fail! ret=%d\n", ret);
    return;
  }

  outputs_attr[1].index = 1;
  ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[1]), sizeof(outputs_attr[1]));
  if(ret < 0) {
    printf("rknn_query fail! ret=%d\n", ret);
    return;
  }

  if (thread_id > sizeof(multi_npu_process_initialized) / sizeof(int) - 1)
    return;

  multi_npu_process_initialized[thread_id] = 1;
  printf("The initialization of NPU Process %d has been completed.\n", thread_id);

  cv::Mat resimg;
  while (!_terminate) {
    pair<unsigned int, Mat> pairIndexImage;
    mtxQueueInput.lock();
    if (queueInput.empty()) {
      mtxQueueInput.unlock();
      continue;
    } else if (queueInput.front().first == thread_id){
        // Get an image from input queue
        pairIndexImage = queueInput.front();
        queueInput.pop();
        mtxQueueInput.unlock();
        //Mat grey ;
        cv::resize(pairIndexImage.second, resimg, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);

        inputs[0].index = input_index;
        inputs[0].buf = resimg.data;
        inputs[0].size = img_width * img_height * img_channels;
        inputs[0].pass_through = false;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        ret = rknn_inputs_set(ctx, 1, inputs);
        if(ret < 0) {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return;
        }

        ret = rknn_run(ctx, nullptr);
        if(ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return;
        }

        outputs[0].want_float = true;
        outputs[0].is_prealloc = false;
        outputs[1].want_float = true;
        outputs[1].is_prealloc = false;
        ret = rknn_outputs_get(ctx, 2, outputs, nullptr);
        if(ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return;
        }
        if(outputs[0].size == outputs_attr[0].n_elems*sizeof(float) && outputs[1].size == outputs_attr[1].n_elems*sizeof(float))
        {
            float boxPriors[4][NUM_RESULTS];
            string labels[91];

            /* load label and boxPriors */
            loadLabelName(label_path, labels);
            loadCoderOptions(box_priors_path, boxPriors);

            float* predictions = (float*)outputs[0].buf;
            float* outputClasses = (float*)outputs[1].buf;

            int output[2][NUM_RESULTS];

            /* transform */
            decodeCenterSizeBoxes(predictions, boxPriors);

            int validCount = scaleToInputSize(outputClasses, output, NUM_CLASSES);
            //printf("validCount: %d\n", validCount);

            if (validCount < 100) {
                /* detect nest box */
                nms(validCount, predictions, output);

                /* box valid detect target */
                for (int i = 0; i < validCount; ++i) {
                    if (output[0][i] == -1) {
                        continue;
                    }
                    int n = output[0][i];
                    int topClassScoreIndex = output[1][i];

                    int x1 = static_cast<int>(predictions[n * 4 + 1] * IMAGE_WIDTH );
                    int y1 = static_cast<int>(predictions[n * 4 + 0] * IMAGE_HEIGHT);
                    int x2 = static_cast<int>(predictions[n * 4 + 3] * IMAGE_WIDTH );
                    int y2 = static_cast<int>(predictions[n * 4 + 2] * IMAGE_HEIGHT);

                    string label = labels[topClassScoreIndex];

                    //std::cout << label << "\t@ (" << x1 << ", " << y1 << ") (" << x2 << ", " << y2 << ")" << "\n";

                    rectangle(pairIndexImage.second, Point(x1, y1), Point(x2, y2), colorArray[topClassScoreIndex%10], 3);
                    putText(pairIndexImage.second, label, Point(x1, y1 - 12), 1, 2, Scalar(0, 255, 0, 255));
                }

            } else { // validCount_if
                printf("validCount too much!\n");
            }

        }
        else{ // ( outputs[0].size == outputs_attr[0] )_if
             printf("rknn_outputs_get fail! get outputs_size = [%d, %d], but expect [%lu, %lu]!\n",
             outputs[0].size, outputs[1].size, outputs_attr[0].n_elems*sizeof(float), outputs_attr[1].n_elems*sizeof(float));
        }
        rknn_outputs_release(ctx, 2, outputs);
    }else{
        mtxQueueInput.unlock();
        continue;
    }

    //mtxQueueShow.lock();
    // Put the processed iamge to show queue
    //queueShow.push(pairIndexImage);
    //mtxQueueShow.unlock();
  }
}
