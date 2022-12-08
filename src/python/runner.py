import numpy as np
import cv2 
import torch 
import torchvision.transforms as transforms
class Model_returner:
    def __init__(self):
        self.initial_set = False
        self.height =0
        self.width =0
        self.resnet_model = None
        self.transform = None
    def setparam(self,height,width):
        self.height = height
        self.width = width
        self.initial_set = True
        self.init_model()
        
    def init_model(self):
        #수정할 부분들
        self.resnet_model =torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        self.resnet_model.eval()
        self.transform = transforms.Compose(
        [transforms.ToTensor(),\
        transforms.Resize((self.height,self.width)),\
        transforms.Normalize(0.5, 0.5) ]
        )
    
    def return_model(self,gray_image):
        #수정할 부분들
        torch_image = self.transform(gray_image)
        torch_image = torch_image.unsqueeze(0).expand(1,3,self.height,self.width)
        # print(torch_image.shape)
        with torch.no_grad():
            output = self.resnet_model(torch_image)
            output = output['out'][0]
        output_predictions = output.argmax(0)#(21, H, W)
        output_predictions = output_predictions.cpu().detach().numpy()
        return_image = np.zeros((gray_image.shape[0],gray_image.shape[1]),dtype=np.uint8)
        for i in range(21):
            if i== 0 : 
                return_image[output_predictions==i] = 255
            if i== 9 : 
                return_image[output_predictions==i] = 255
            if i== 11 : 
                return_image[output_predictions==i] = 255
            if i== 16 : 
                return_image[output_predictions==i] = 255
            if i== 18 : 
                return_image[output_predictions==i] = 255
            if i== 20 : 
                return_image[output_predictions==i] = 255
        return return_image
        # return gray_image

model = Model_returner()
def set_param(arg_height,arg_width):
    global model
    model.setparam(arg_height,arg_width)
    
    

def get_mask(gray_image):
    if model.initial_set:
        return model.return_model(gray_image)
        
    else:
        return gray_image
