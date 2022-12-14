import numpy as np
import cv2 
import torch 
import torchvision.transforms as transforms
import sys
import model



# print(sys.version)
class Model_returner:
    def __init__(self):
        self.initial_set = False
        self.height =0
        self.width =0
        self.LDLO = None
        self.transform = None
    def setparam(self,height,width):
        self.height = height
        self.width = width
        self.initial_set = True
        self.init_model()
        
    def init_model(self):
        #수정할 부분들
        global weight
        self.LDLO = model.LoopNet_model(1240,360,16,4)
        self.LDLO.load_state_dict(torch.load('./src/python/main.pth'))
        # self.
        self.LDLO.eval()
        self.LDLO = self.LDLO.cuda()
        self.transform = transforms.Compose(
        [transforms.ToTensor(),\
        transforms.Resize((360,1240)),\
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
        )
    
    def return_model(self,gray_image):
        #수정할 부분들
        torch_image = self.transform(gray_image)
        torch_image = torch_image.unsqueeze(0).cuda()
        with torch.no_grad():
            _, output= self.LDLO.main_forward(torch_image)

        output_image = (output[0].cpu().permute(1,2,0).squeeze(2).detach().numpy()*255).astype(np.uint8)
        output_image = cv2.resize(output_image,(self.width,self.height))
        # cv2.imshow("return",output_image)
        return output_image

LDLO = model.LoopNet_model(1240,360,16,4)
cmodel = Model_returner()

def set_param(arg_height,arg_width):
    global cmodel
    cmodel.setparam(arg_height,arg_width)
    
    

def get_mask(gray_image):
    if cmodel.initial_set:
        return cmodel.return_model(gray_image)
        
    else:
        return gray_image


# set_param(1240,360)