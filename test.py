import torch
from torchvision import transforms, datasets, models
from model import mymodel
from torch.autograd import Variable
import os
from PIL import Image
import time

data_dir = './data/test_imgs' #change the dir path to your images
weights_path ='./data/weights'
class_path = './data/pytorch'

classes = os.listdir(class_path)
classes.sort()
use_gpu = torch.cuda.is_available()

transforms_op = [
    transforms.Resize((400,200), interpolation=3),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
data_transforms = transforms.Compose(transforms_op)

model = mymodel(len(classes))
model.load_state_dict(torch.load(os.path.join(weights_path,'net_23.pth')))
if use_gpu:
    model.cuda()
model.train(False)

for name in os.listdir(data_dir):
    img = Image.open(os.path.join(data_dir,name))
    img = img.convert('RGB')
    img = data_transforms(img)
    img = img.unsqueeze(0)
    if use_gpu:
        img = Variable(img.cuda())
    else:
        img = Variable(img)
    output = model(img)
    _, preds = torch.max(output.data, 1)
    print('The pic {} is {}.'.format(name, classes[preds.data[0]][10:]))

