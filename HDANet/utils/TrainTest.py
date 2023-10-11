import torch
import numpy as np
import sys
sys.path.append('..')
from torch import nn
import torch.nn.functional as F
import random
from HDANet.utils.DataTransform import augment_fn, move, AddPepperNoise, RandomApply, Cuda
from typing import Iterable, Callable


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self.features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, input, output):
            self.features[layer_id] = output
        return fn

    def forward(self, x):
        out = self.model(x)
        return out, self.features


def loss_fn(x, y):
    y = y.detach()
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 1/2 - (x * y).sum(dim=-1).mean()/2


class CapsuleLoss(nn.Module):
    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda

    def forward(self, logits, labels):
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        labels = torch.zeros(logits.shape).cuda().scatter_(1, labels.unsqueeze(1), 1)
        margin_loss = (labels * left).sum(-1).mean() + self.lmda * ((1 - labels) * right).sum(-1).mean()
        return margin_loss


def model_train(model, data_loader, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train_acc = 0
    cr1 = CapsuleLoss()
    cr2 = loss_fn
    cr3 = nn.BCELoss()
    cr4 = nn.L1Loss()
    train_loss, train_loss1, train_loss2, train_loss3, train_loss4 = 0, 0, 0, 0, 0
    for i, data in enumerate(data_loader):
        x, mask, y = data
        [x_one, mask_one], [x_two, mask_two] = move(x, mask), move(x, mask)
        image_one, image_two = augment_fn(x_one), augment_fn(x_two)

        median_extractor = FeatureExtractor(model, layers=['MLP', 'Project', 'conv10', 'mask'])

        out_one, median_one = median_extractor(image_one)
        predic_one, project_one, saliency_one = median_one['MLP'], median_one['Project'], median_one['conv10']
        med_mask1 = median_one['mask']
        out_two, median_two = median_extractor(image_two)
        predic_two, project_two, saliency_two = median_two['MLP'], median_two['Project'], median_two['conv10']
        med_mask2 = median_two['mask']

        pred = out_one.max(1, keepdim=True)[1]
        train_acc += pred.eq(y.to(device).view_as(pred)).sum().item()
        pred = out_two.max(1, keepdim=True)[1]
        train_acc += pred.eq(y.to(device).view_as(pred)).sum().item()

        loss1 = cr1(out_one, y.to(device)) + cr1(out_two, y.to(device))
        loss2 = cr2(predic_one, project_two) + cr2(predic_two, project_one)
        loss3 = cr3(saliency_one, mask_one.to(device)) + cr3(saliency_two, mask_two.to(device))
        loss4 = cr4(med_mask1, torch.zeros(med_mask1.shape).to(device)) + cr4(med_mask2, torch.zeros(med_mask1.shape).to(device))

        loss = loss1 + loss2 + 1e-1*loss3 + 1e-2*loss4
        train_loss, train_loss1, train_loss2, train_loss3, train_loss4 = \
            train_loss+loss.item(), train_loss1+loss1.item(), train_loss2+loss2.item(), train_loss3+loss3.item(), train_loss4+loss4.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # print("Train loss is:{:.8f}, loss1 is:{:.8f}, loss2 is:{:.8f}, loss3 is:{:.8f}, loss4 is:{:.8f}"
    #       .format(train_loss / len(data_loader), train_loss1 / len(data_loader), train_loss2 / len(data_loader),
    #               train_loss3 / len(data_loader), train_loss4 / len(data_loader)))
    # print("Train accuracy is:{:.2f} % ".format(train_acc / 2/ len(data_loader.dataset) * 100.))
    return train_loss/len(data_loader)


def model_test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)



def model_train_eoc4G(model, data_loader, opt):
    class AddGaussianNoise(object):
        def __init__(self, mean=0.0, variance=0.8, amplitude=0.0):
            self.mean = mean
            self.variance = variance
            self.amplitude = amplitude

        def __call__(self, pic):
            img = pic.clone().detach()
            N = (self.amplitude + np.random.rand()) * torch.normal(mean=self.mean, std=self.variance, size=img.shape)
            out = N + img
            return (out - out.min()).div(out.max() - out.min())

    augment_fn_4R = T.Compose([
        RandomApply(AddGaussianNoise(), p=0.2),
        RandomApply(AddPepperNoise(), p=0.2),
        Cuda()],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train_acc = 0
    cr1 = CapsuleLoss()
    cr2 = loss_fn
    cr3 = nn.BCELoss()
    cr4 = nn.L1Loss()
    train_loss, train_loss1, train_loss2, train_loss3, train_loss4 = 0, 0, 0, 0, 0
    for i, data in enumerate(data_loader):
        x, mask, y = data
        [x_one, mask_one], [x_two, mask_two] = move(x, mask), move(x, mask)
        image_one, image_two = augment_fn_4R(x_one), augment_fn_4R(x_two)

        median_extractor = FeatureExtractor(model, layers=['MLP', 'Project', 'conv10', 'mask'])

        out_one, median_one = median_extractor(image_one)
        predic_one, project_one, saliency_one = median_one['MLP'], median_one['Project'], median_one['conv10']
        med_mask1 = median_one['mask']
        out_two, median_two = median_extractor(image_two)
        predic_two, project_two, saliency_two = median_two['MLP'], median_two['Project'], median_two['conv10']
        med_mask2 = median_two['mask']

        pred = out_one.max(1, keepdim=True)[1]
        train_acc += pred.eq(y.to(device).view_as(pred)).sum().item()
        pred = out_two.max(1, keepdim=True)[1]
        train_acc += pred.eq(y.to(device).view_as(pred)).sum().item()

        loss1 = cr1(out_one, y.to(device)) + cr1(out_two, y.to(device))
        loss2 = cr2(predic_one, project_two) + cr2(predic_two, project_one)
        loss3 = cr3(saliency_one, mask_one.to(device)) + cr3(saliency_two, mask_two.to(device))
        loss4 = cr4(med_mask1, torch.zeros(med_mask1.shape).to(device)) + cr4(med_mask2, torch.zeros(med_mask1.shape).to(device))

        loss = loss1 + loss2 + 1e-1*loss3 + 1e-2*loss4
        train_loss, train_loss1, train_loss2, train_loss3, train_loss4 = \
            train_loss+loss.item(), train_loss1+loss1.item(), train_loss2+loss2.item(), train_loss3+loss3.item(), train_loss4+loss4.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # print("Train loss is:{:.8f}, loss1 is:{:.8f}, loss2 is:{:.8f}, loss3 is:{:.8f}, loss4 is:{:.8f}"
    #       .format(train_loss / len(data_loader), train_loss1 / len(data_loader), train_loss2 / len(data_loader),
    #               train_loss3 / len(data_loader), train_loss4 / len(data_loader)))
    # print("Train accuracy is:{:.2f} % ".format(train_acc / 2/ len(data_loader.dataset) * 100.))
    return train_loss/len(data_loader)
from torchvision import transforms as T

class TEST_AddGussain(object):
    def __init__(self, snr):
        self.snr = snr
    def __call__(self, pic):
        img = pic.clone().detach()
        snr = 10 ** (self.snr / 10.0)
        xpower = torch.sum(img.flatten() ** 2)/img.shape[0]/img.shape[1]
        npower = xpower / snr
        noise = torch.normal(mean=0, std=np.sqrt(npower), size=pic.shape)
        out = img + noise
        return (out-out.min()).div(out.max()-out.min())

def model_test_eoc4G(model, test_loader, SNR):
    EOC4G = T.Compose([
        RandomApply(TEST_AddGussain(SNR), p=1),
        Cuda()],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device)
            data = EOC4G(data)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

class TEST_AddPepperNoise(object):
    def __init__(self, snr):
        self.snr = snr
    def __call__(self, pic):
        img = pic.clone().detach()
        signal_pct = self.snr
        noise_pct = (1 - signal_pct)
        mask = np.random.choice((0, 1), size=img.shape, p=[signal_pct, noise_pct])
        mask = torch.from_numpy(mask)
        noise = torch.rand(size=img.shape)
        img[mask == 1] = noise[mask==1]
        return img

def model_test_eoc4R(model, test_loader, SNR):
    EOC4R = T.Compose([
        RandomApply(TEST_AddPepperNoise(SNR), p=1),
        Cuda()],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device)
            data = EOC4R(data)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

import math
class TEST_Ersion(object):

    def __init__(self, SNR):
        self.SNR = SNR

    def __call__(self, pic):
        # 把img转化成ndarry的形式.
        img = pic.clone().detach()
        scl = self.SNR

        # print(img.shape)

        # area = img.shape[0]*img.shape[1]
        # target_area = scl*area
        target_area = self.SNR
        aspect_ratio = random.uniform(0.5, 2)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(int(img.shape[0]/2)-32, int(img.shape[0]/2)+32 - h)
            y1 = random.randint(int(img.shape[0]/2)-32, int(img.shape[0]/2)+32 - w)

            img[x1:x1 + h, y1:y1 + w] = 0
            # plt.imshow(img)
            # plt.show()
            return img

        print('Not Ersion')
        return img

def model_test_eoc5(model, test_loader, SNR):
    EOC5_fn = T.Compose([
        RandomApply(TEST_Ersion(SNR), p=1),
        Cuda()],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss = 0
    correct = 0
    pred_all = np.array([[]]).reshape((0, 1))
    real_all = np.array([[]]).reshape((0, 1))
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device)
            data = EOC5_fn(data)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)