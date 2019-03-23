import torch
from torchvision import transforms, datasets, models
from model import mymodel
import torch.optim as optim
from torch.autograd import Variable
import os
import time

data_dir = './data/pytorch' #change the dir path to your images
# data_dir = './data/test_imgs' #change the dir path to your images
restore_net = True # if u want to restore the net for continuing training,set to true
weights_path ='./data/weights'

transforms_op = [
    transforms.Resize((400,200), interpolation=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

data_transforms = transforms.Compose(transforms_op)

image_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=0)
inputs, classes = next(iter(dataloader))
classes_num = len(image_dataset.classes)
dataset_size = len(image_dataset)

use_gpu = torch.cuda.is_available()
# print(use_gpu)

# model = mymodel(classes_num)
model = mymodel(120)
if use_gpu:
    model.cuda()
if restore_net:
    model.load_state_dict(torch.load(os.path.join(weights_path,'net_23.pth')))

criterion = torch.nn.CrossEntropyLoss()
lr = 0.1
optimizer = optim.SGD(model.parameters(),lr)

def adjust_learning_rate(optimizer, lr, epoch):

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = lr * (0.1 ** (epoch // 10))

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr

def save_network(network, weights_path, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(weights_path,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()

def train(model,dataloader, criterion, optimizer, epochs,weights_path,lr):
    start_time = time.time()
    for epoch in range(epochs):
        since = time.time()
        adjust_learning_rate(optimizer,lr, epoch)
        model.train(True)
        print("Epoch {}:".format(epoch))
        running_loss = 0.0
        running_corrects = 0.0
        for data in dataloader:
            print('__________________________________')
            img,label = data
            now_batch_size, c, h, w = inputs.shape
            if use_gpu:
                img,label = Variable(img.cuda()),Variable(label.cuda())
            else:
                img, label = Variable(img), Variable(label)
            output = model(img)
            loss = criterion(output,label)
            _, preds = torch.max(output.data, 1)
            print('preds:')
            print(preds)
            print('label:')
            print(label.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * now_batch_size
            running_corrects += float(torch.sum(preds == label.data))

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects / dataset_size
        end =time.time()-since
        print("loss:{}.acc:{},time:{:.0f}m {:.0f}s".format(epoch_loss, epoch_acc, end//60, end%60))

        if epoch%10==0:
            save_network(model,weights_path, epoch)
        if epoch == epochs-1:
            save_network(model, weights_path, epoch)
            finish =time.time()-start_time
            print("training completed in {}m,{}s".format(finish//60, finish%60))

train(model,dataloader,criterion,optimizer,24,weights_path,lr)


