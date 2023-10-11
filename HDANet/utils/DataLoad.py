import sys
sys.path.append('..')
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop
import os
import re
import numpy as np
import cv2
from tqdm import tqdm
import torch.nn as nn
from captum.attr import GuidedGradCam, NoiseTunnel


class SAR_VGG16(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(SAR_VGG16, self).__init__()
        cfgs = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        def make_features(cfg: list):
            layers = []
            in_channels = 3
            for v in cfg:
                if v == "M":
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, nn.ReLU()]
                    in_channels = v
            return nn.Sequential(*layers)
        self.features = make_features(cfgs['vgg16'])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(64, 64),
            nn.Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512 * 1 * 1)
        out = self.classifier(x)
        return out


def sailency(data, id=0, target_number=10):
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(id)
    device = torch.device("cuda" if use_cuda else "cpu")
    data_set = torch.utils.data.TensorDataset(torch.from_numpy(data).float().type(torch.FloatTensor))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=8, shuffle=False)
    sa_model = SAR_VGG16().to(device)
    sa_model.load_state_dict(torch.load('./Model/VGG16.pth', map_location=device), False)
    sa_model.eval()
    way = GuidedGradCam(sa_model, sa_model.features[29])
    way = NoiseTunnel(way)
    for i, data in enumerate(tqdm(data_loader)):
        a = data[0]
        a0 = np.zeros(a.shape)
        for j in range(target_number):
            a0 += way.attribute(a.to(device), target=torch.tensor(j).to(device), nt_type='smoothgrad_sq', nt_samples=5, stdevs=0.05)\
                        .abs().cpu().detach().numpy()
        for j in range(a0.shape[0]):
            temp = a0[j].squeeze()
            max_val = np.nanpercentile(temp.flatten(), 100)
            image = np.array((temp/max_val), dtype='float')
            image[image < 0.1] = 0
            a0[j, 0, :, :] = image

        if i == 0:
            bb = a0
        else:
            bb = np.concatenate((bb, a0), axis=0)
    return bb


def crop_transform(picture_size):
    return Compose([
        # Resize(picture_size),
        CenterCrop(picture_size), ])


def load_data(file_dir, id=0, picture_size=128):
    data_name = re.split('[/\\\]', file_dir)[-2]
    if data_name == 'SOC':
        label_name = {'BMP2': 0, 'BTR70': 1, 'T72': 2, 'BTR_60': 3, '2S1': 4, 'BRDM_2': 5, 'D7': 6, 'T62': 7,
                      'ZIL131': 8, 'ZSU_23_4': 9}
    elif data_name == 'EOC-Depression':
        label_name = {'2S1': 0, 'BRDM_2': 1, 'ZSU_23_4': 2, 'T72': 3}
    elif data_name == 'EOC-Scene':
        label_name = {'BRDM_2': 0, 'ZSU_23_4': 1, 'T72': 2}
    elif data_name == 'EOC-Configuration-Version':
        label_name = {'T72': 0, 'BMP2': 1, 'BRDM_2': 2, 'BTR70': 3}

    path_list = []
    jpeg_list = []
    label_list = []
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file in files:
            if os.path.splitext(file)[1] == '.jpeg':
                path_list.append(os.path.join(root, file))
    for jpeg_path in path_list[0:20]:
        jpeg = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
        pic = crop_transform(picture_size)(torch.from_numpy(jpeg))
        jpeg_list.append(np.array(pic.div(pic.max())))
        label_list.append(label_name[re.split('[/\\\]', jpeg_path)[5]])

    jpeg_list = np.array(jpeg_list)
    data = np.expand_dims(jpeg_list, axis=1)
    mask = sailency(data=data, id=id)
    label = np.array(label_list)
    data_set = torch.utils.data.TensorDataset(torch.from_numpy(data).type(torch.FloatTensor),
                                              torch.from_numpy(mask).type(torch.FloatTensor),
                                              torch.from_numpy(label).type(torch.LongTensor))
    return data_set, label_name


def load_test(file_dir, picture_size=128):
    data_name = re.split('[/\\\]', file_dir)[-2]
    if data_name == 'SOC':
        label_name = {'BMP2': 0, 'BTR70': 1, 'T72': 2, 'BTR_60': 3, '2S1': 4, 'BRDM_2': 5, 'D7': 6, 'T62': 7,
                      'ZIL131': 8, 'ZSU_23_4': 9}
    elif data_name == 'EOC-Depression':
        label_name = {'2S1': 0, 'BRDM_2': 1, 'ZSU_23_4': 2, 'T72': 3}
    elif data_name == 'EOC-Scene':
        label_name = {'BRDM_2': 0, 'ZSU_23_4': 1, 'T72': 2}
    elif data_name == 'EOC-Configuration-Version':
        label_name = {'T72': 0, 'BMP2': 1, 'BRDM_2': 2, 'BTR70': 3}
    path_list = []
    jpeg_list = []
    label_list = []
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file in files:
            if os.path.splitext(file)[1] == '.jpeg':
                path_list.append(os.path.join(root, file))
    for jpeg_path in path_list:
        jpeg = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
        pic = crop_transform(picture_size)(torch.from_numpy(jpeg).div(jpeg.max()))
        jpeg_list.append(np.array(pic))
        label_list.append(label_name[re.split('[/\\\]', jpeg_path)[5]])

    jpeg_list = np.array(jpeg_list)
    data = np.expand_dims(jpeg_list, axis=1)
    label = np.array(label_list)
    data_set = torch.utils.data.TensorDataset(torch.from_numpy(data).type(torch.FloatTensor),
                                              torch.from_numpy(label).type(torch.LongTensor))
    return data_set, label_name
