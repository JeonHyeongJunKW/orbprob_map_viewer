import numpy as np
import cv2 
import torch 

class Model_returner:
    def __init__(self):
        self.initial_set = False
        self.height =0
        self.width =0
        self.resnet_model = None
        self.
    def setparam(self,height,width):
        self.height = height
        self.width = width
        self.initial_set = True
        self.init_model()
        
    def init_model(self):
        self.resnet_model =torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        self.resnet_model.eval()
    
    def return_model(self,gray_image):
        return gray_image

model = Model_returner()
def set_param(arg_height,arg_width):
    global model
    model.setparam(arg_height,arg_width)
    
    

def get_mask(gray_image):
    if model.initial_set:
        return model.return_model(gray_image)
        
    else:
        return gray_image
