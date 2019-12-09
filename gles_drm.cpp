#include <gles_drm.h>


int setup_kms(struct kms *kms)
{
    drmModeRes *resources;
    drmModeConnector *connector;
    drmModePlaneRes *plane_res;
    int i;
    resources = drmModeGetResources(kms->fd); // drmModeRes *resources: �� �� �� �� �� �� �� �� �� �� �� �� Ϣ�� connector�� encoder�� crtc�� modes��  , get their count .
    if (!resources) {
        fprintf(stderr, "drmModeGetResources failed\n");
        return 0;
    }

    drmSetClientCap(kms->fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);

    for (i = 0; i < resources->count_connectors; i++) {
        connector = drmModeGetConnector(kms->fd, resources->connectors[i]); // get drm_mode �� �� ȡ  connecotr �� �� �� �� Դ  .
        if (connector == NULL)
            continue;

        if (connector->connection == DRM_MODE_CONNECTED &&  // find connected connector ,if find it ,we break loop
            connector->count_modes > 0)
            break;

        drmModeFreeConnector(connector); // if all conditions don't be accepted , connector will be free
    }

    if (i == resources->count_connectors) {
        fprintf(stderr, "No currently active connector found.\n"); // if code in L59 don't run ,then i will equal to count_connectors .
        return 0;
    }

    plane_res = drmModeGetPlaneResources(kms->fd);

    kms->connector = connector;
    kms->mode = connector->modes[0]; // display's basic data
    kms->planes = plane_res->planes;
    kms->crtcs = resources->crtcs ;

    return 1;
}

int init_egl_gbm(struct kms *kms, struct display *displays)
{
    // 1. Get Display
    EGLConfig eglConfig;
    EGLContext eglContext;

    EGLint verMajor, verMinor, n;
    EGLint numConfigs;

    struct gbm_device *gbm;

    static const char drm_device_name[] = "/dev/dri/card0";

    kms->fd = open(drm_device_name, O_RDWR);
    if (kms->fd < 0) {
        /* Probably permissions error */
        LOGE("couldn't open %s, skipping", drm_device_name);
        return -1;
    }

    gbm = gbm_create_device(kms->fd); // Create EGL Context using GBM ? create native_display
    if (gbm == NULL) {
        LOGE("couldn't create gbm device");
        close(kms->fd);
        return -1;
    }

    displays->eglDisplay = eglGetDisplay(gbm); // not EGL_DEFAULT_DISPLAY ! �� �� һ �� EGL�� ʾ �� �� ��  .
    if( displays->eglDisplay == EGL_NO_DISPLAY || eglGetError() != EGL_SUCCESS ) {
        LOGE("getdisplay error !");
        return eglGetError();
    }

    // 2. Initialize EGL . verMajor : EGL main version ; verMinor : EGL minor version
    if ( eglInitialize(displays->eglDisplay, &verMajor, &verMinor) == EGL_FALSE || eglGetError() != EGL_SUCCESS ) { //�� ʼ ��  EGL �� �� �� �� , �� ��  EGL �� �� �� �� �� .
        LOGE("egl init error ! %d", eglGetError());
        return eglGetError();
    }

    const char *ver = eglQueryString(displays->eglDisplay, EGL_VERSION); // query and get EGL_version

    if ( eglGetConfigs( displays->eglDisplay, NULL, 0, &numConfigs) == EGL_FALSE || eglGetError() != EGL_SUCCESS ) {  // �� ѯ �� ȡ �� �� �� �� �� Ϣ  .
        std::cerr << "getdisplay error !" << std::endl;
        return eglGetError();
    }

    LOGD("* EGL_VERSION = %s (have %d configs)", ver, numConfigs);

    if (!setup_kms(kms)) {
        LOGE("setup kms failed !");
        return -1;
    }
    eglBindAPI(EGL_OPENGL_ES_API); // �� �� �� ǰ �� Ⱦ  API  Specifies the client API to bind, one of EGL_OPENGL_API, EGL_OPENGL_ES_API, or EGL_OPENVG_API.

    // 3. Choose Config ,Configʵ �� ָ �� ��  FrameBuffer �� �� ��  .
    if (!eglChooseConfig(displays->eglDisplay, attribs, &eglConfig, 1, &n) || n != 1) { // ѡ �� �� ƥ �� Ҫ �� �� �� ��. .
        LOGE("failed to choose argb config");
        return eglGetError();
    }

    // 4. Create GBM Surface = native_window .
    displays->gbmSurface = gbm_surface_create(gbm, kms->mode.hdisplay, kms->mode.vdisplay,
                            GBM_FORMAT_RGB888,
                            GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
    fprintf(stderr, "%s %d gbmSurface=%p\n",__func__,__LINE__, displays->gbmSurface);
    displays->eglSurface = eglCreateWindowSurface(displays->eglDisplay, eglConfig, displays->gbmSurface, NULL); // �� �� һ �� �� ʵ �� �� ʾ ��  EGL Surface, Surface ʵ �� �� �� �� һ ��  FrameBuffer��Ҳ �� �� �� Ⱦ Ŀ �� �� .


    // 5. Create Context
    static const EGLint context_attribs[] = {
            EGL_CONTEXT_CLIENT_VERSION, 2,
            EGL_NONE
    };
    eglContext = eglCreateContext(displays->eglDisplay, eglConfig, EGL_NO_CONTEXT, context_attribs); // Context�� �� �� �� �� ״ ̬ �� , �� �� ǰ �� �� ɫ ���� �� �� �� ���� �� �� �� ��Ѥ Ⱦ ģ ʽ �� һ �� �� ״ ̬  .
    if (eglContext == NULL) {
        LOGE("failed to create context");
        return eglGetError();
    }
    // //eglMakeCurrent�� �� �� ��  surface �� �� �� �� ��  opengl �� ͼ ��  .
    if (!eglMakeCurrent(displays->eglDisplay, displays->eglSurface, displays->eglSurface, eglContext)) { // ָ �� ĳ ��  eglContext Ϊ �� ǰ �� �� �� ���� �� �� �� ��  eglContext ��  eglSurface.
        LOGE("failed to make context current");
        return eglGetError();
    }
    return 1 ;
}

