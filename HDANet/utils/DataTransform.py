import sys
sys.path.append('..')
import torch
import numpy as np
from torch import nn
from torchvision import transforms as T
import random
import torchvision.transforms.functional as TF


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        y = torch.zeros(x.shape)
        for i in range(x.shape[0]):
            if random.random() < self.p:
                y[i, 0, :, :] = self.fn(x[i, 0].clone().detach())
            else:
                y[i, 0, :, :] = x[i, 0].clone().detach()
        return y


def move(img, mask, p=0.3, angle=5):
    img_ = img.clone().detach()
    mask_ = mask.clone().detach()
    for i in range(img_.shape[0]):
        if np.random.rand() < p:
            ang = np.random.randint(-angle, angle)
            img_[i, :, :, :] = TF.rotate(img[i, :, :, :].squeeze().unsqueeze(0).unsqueeze(0).clone().detach(), ang)
            mask_[i, :, :, :] = TF.rotate(mask[i, :, :, :].squeeze().unsqueeze(0).unsqueeze(0).clone().detach(), ang)

    return img_, mask_


class AddPepperNoise(object):

    def __init__(self, snr=0.95):
        self.snr = snr

    def __call__(self, pic):
        img = pic.clone().detach()
        signal_pct = np.random.uniform(self.snr, 1, 1).squeeze()
        noise_pct = (1 - signal_pct)
        mask = np.random.choice((0, 1), size=img.shape, p=[signal_pct, noise_pct])
        mask = torch.from_numpy(mask)
        noise = torch.rand(size=img.shape)
        img[mask == 1] = noise[mask==1]

        return img


class AddGaussianNoise(object):
    def __init__(self, mean=0.1, variance=0.1, amplitude=0.5):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
    def __call__(self, pic):
        img = pic.clone().detach()
        N = (self.amplitude + np.random.rand()) * torch.normal(mean=self.mean, std=self.variance, size=img.shape)
        out = N + img
        out[out < 0] = 0
        return out.div(out.max())



class Cuda(object):
    def __init__(self):
        self.p = 1
    def __call__(self, img):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return img.to(device)


augment_fn = T.Compose([
    RandomApply(AddGaussianNoise(), p=0.2),
    RandomApply(AddPepperNoise(), p=0.2),
    Cuda()],
)

class MyRotateTransform(object):
    def __init__(self, angle=5):
        self.angle = angle

    def __call__(self, pic):
        img = pic.unsqueeze(0).unsqueeze(0)
        ang = np.random.randint(-self.angle, self.angle)
        out = TF.rotate(img, ang).squeeze()
        return out


other_augment_fn = T.Compose([
    RandomApply(MyRotateTransform(), p=0.3),
    RandomApply(AddGaussianNoise(), p=0.2),
    RandomApply(AddPepperNoise(), p=0.2),
    Cuda()],
)




