import time
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import numpy as np
import glob
import os
from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

# ------- 1. define loss function --------
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
bce_loss = nn.BCELoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, weight=None):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)
    if weight == None:
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    else:
        loss = weight[0]*loss0 + weight[1]*loss1 + weight[2]*loss2 + weight[3]*loss3 \
           + weight[4]*loss4 + weight[5]*loss5 + weight[6]*loss6
    return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2net_weighted'  # 'u2netp'
tra_image_dir = "/home/dp/HDDdisk/lqr/datasets/SOD/DUTS/DUTS-TR/DUTS-TR-Image/"
tra_label_dir = "/home/dp/HDDdisk/lqr/datasets/SOD/DUTS/DUTS-TR/DUTS-TR-Mask/"

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)
print(len(tra_img_name_list))
tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)
batch_size_train = 12
salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

# ------- 3. define model --------
# define the net
if model_name == 'u2net' or model_name == 'u2net_weighted':
    net = U2NET(3, 1)
elif model_name == 'u2netp':
    net = U2NETP(3, 1)

if torch.cuda.is_available():
    net.cuda()
    print("cuda ok!")
# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
epoch_num = 10000  # 100000
batch_size_val = 4
val_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 1000  # save the model every 2000 iterations
print_frq = 50
ite_num = 0
for epoch in range(0, epoch_num):
    start_time = time.time()
    net.train()
    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1
        inputs, labels = data['image'], data['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
        # y zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        weight = [0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1]
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, weight)
        loss.backward()
        optimizer.step()
        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()
        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
        if ite_num % print_frq == 0:
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                    epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                    running_tar_loss / ite_num4val))
        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), model_dir + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
            ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0
    epoch_time = time.time()-start_time
    print(f'[log] roughly {(epoch_num - epoch)/3600.*epoch_time:.2f} h left\n')
