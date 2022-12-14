#include <iostream>
#include <python3.6/Python.h>
#include "LDOD.h"
#include <opencv2/opencv.hpp>
#include "numpy/ndarrayobject.h"
//아래의 키워드를 주니 정상적으로 작동
#undef NUMPY_IMPORT_ARRAY_RETVAL
#define NUMPY_IMPORT_ARRAY_RETVAL
using namespace std;
using namespace cv;

LDOD::LDOD()
{
    setenv("PYTHONPATH", "/home/jeon/orbprob_map_viewer/src/python", 1);//before initialize
    Py_Initialize();
    this->py_LODO_module = PyImport_ImportModule("runner");

    if(this->py_LODO_module == NULL)
    {
        cout<<"모듈 초기화에 실패하였습니다."<<endl;
        exit(0);
    }
    this->py_mask_func = PyObject_GetAttrString(this->py_LODO_module,"get_mask");
    if(this->py_mask_func == NULL)
    {
        cout<<"함수 초기화에 실패하였습니다."<<endl;
        exit(0);
    }
    // import_array1(-1);
    
    
}

LDOD::LDOD(int image_height, int image_width)
{
    setenv("PYTHONPATH","./src/python/",1);
    Py_Initialize();
    if(PyArray_API == NULL)
    {
        import_array(); 
    }
    // import_array1(-1);
    this->py_LODO_module = PyImport_ImportModule("runner");
    if(this->py_LODO_module == NULL)
    {
        cout<<"모듈 초기화에 실패하였습니다."<<endl;
        exit(0);
    }
    this->py_setparam_func = PyObject_GetAttrString(this->py_LODO_module,"set_param");
    this->py_mask_func = PyObject_GetAttrString(this->py_LODO_module,"get_mask");
    this->image_height = image_height;
    this->image_width = image_width;
    //이미지 차원을 등록합니다.
    PyObject *pArgs;
    pArgs = PyTuple_New(2);//인자들을 모아서 저장할 변수
    PyObject *py_width, *py_height;
    py_width = PyLong_FromLong(this->image_width);
    py_height = PyLong_FromLong(this->image_height);
    PyTuple_SetItem(pArgs,0,py_height);//0번째로 인자숫자 1을 넣어줍니다.
    PyTuple_SetItem(pArgs,1,py_width);//0번째로 인자숫자 1을 넣어줍니다.
    PyObject_CallObject(py_setparam_func,pArgs);

    if(this->py_mask_func == NULL)
    {
        cout<<"함수 초기화에 실패하였습니다."<<endl;
        exit(0);
    }
}

Mat LDOD::GetLeftMask(Mat g_leftImage)
{
    const unsigned int nElem = this->image_height*this->image_width*3;//요소의 개수
    uchar* m = new uchar[nElem];//적절한 형태로 저장할 matrix를 만듭니다.
    std::memcpy(m, g_leftImage.data, nElem * sizeof(uchar));
    npy_intp mdim[] = { this->image_height, this->image_width,3 };//matrix차원
    PyObject* gray_mat = PyArray_SimpleNewFromData(3, mdim, NPY_UINT8, (void*) m);//차원갯수, 각 원소수, 타입(uint8), 넣을 데이터 
    //결과 마스크를 획득합니다.
    PyObject *pArgs;
    pArgs = PyTuple_New(1);//인자들을 모아서 저장할 변수
    PyTuple_SetItem(pArgs,0,gray_mat);//0번째로 인자숫자 1을 넣어줍니다.
    PyObject* mask_result = PyObject_CallObject(this->py_mask_func,pArgs);//결과를얻습니다.
    Mat return_gray(this->image_height, this->image_width, CV_8UC1, PyArray_DATA(mask_result));
    // imshow("mask",return_gray);
    // waitKey(1);
    return return_gray;
}
