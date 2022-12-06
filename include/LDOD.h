#ifndef LDOD_H
#define LDOD_H
#include <iostream>
#include <python3.6/Python.h>
// #include "numpy/ndarrayobject.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class LDOD
{
    public:
    PyObject *py_LODO_module, *py_mask_func, *py_setparam_func;
    int image_height;
    int image_width;
    LDOD();
    LDOD(int image_height, int image_width);
    Mat GetLeftMask(Mat g_leftImage);

};
#endif