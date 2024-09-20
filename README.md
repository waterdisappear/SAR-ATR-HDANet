<h1 align="center"> HDANet: Hierarchical Disentanglement-Alignment Network for Robust SAR Vehicle Recognition </h1> 

<h5 align="center"><em> Weijie Li (李玮杰), Wei Yang (杨威), Wenpeng Zhang (张文鹏), Tianpeng Liu (刘天鹏), Yongxiang (刘永祥), and Li Liu (刘丽), and  </em></h5>

<p align="center">
<a href="https://arxiv.org/abs/2304.03550"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>
<a href="https://ieeexplore.ieee.org/document/10283916"><img src="https://img.shields.io/badge/Paper-IEEE%20IEEE J-STARS-red"></a>
</p>

## Introduction

This paper proposes a novel domain alignment framework, Hierarchical Disentanglement-Alignment Network (HDANet), to enhance SAR ATR features' causality and robustness. 

本文提出了一种新颖的域对齐框架，分层解耦对齐网络（HDANet），以增强SAR目标识别特征的因果性和鲁棒性。

**Abstract:** Vehicle recognition is a fundamentale problem in synthetic aperture radar (SAR) image interpretation. However, robustly recognizing vehicle targets is a challenging task in SAR due to the large intraclass variations and small interclass variations. In addition, the lack of large datasets further complicates the task. Inspired by the analysis of target signature variations and deep learning explainability, this article proposes a novel domain alignment framework, named the hierarchical disentanglement-alignment network (HDANet), to achieve robustness under various operating conditions. Concisely, HDANet integrates feature disentanglement and alignment into a unified framework with three modules: domain data generation; multitask-assisted mask disentanglement; and the domain alignment of target features. The first module generates diverse data for alignment, and three simple but effective data augmentation methods are designed to simulate target signature variations. The second module disentangles the target features from background clutter using the multitask-assisted mask to prevent clutter from interfering with subsequent alignment. The third module employs a contrastive loss for domain alignment to extract robust target features from generated diverse data and disentangled features. Finally, the proposed method demonstrates impressive robustness across nine operating conditions in the MSTAR dataset, and extensive qualitative and quantitative analyses validate the effectiveness of our framework. 

**摘要:** 车辆识别是合成孔径雷达（SAR）图像解译中的一个基本问题。然而，由于类内差异大、类间差异小，在SAR中鲁棒性地识别车辆目标是一项具有挑战性的任务。此外，缺乏大型数据集也使这项任务变得更加复杂。受目标特征变化分析和深度学习可解释性的启发，本文提出了一种新颖的域对齐框架，命名为分层解耦对齐网络（HDANet），以实现各种操作条件下的鲁棒性。简而言之，HDANet 将特征解耦和对齐集成到一个统一的框架中，包括三个模块：域数据生成、多任务辅助掩码解耦和目标特征的域对齐。第一个模块生成用于域对齐的各种域数据，设计了三种简单而有效的数据增强方法来模拟目标特征的变化。第二个模块使用多任务辅助掩码将目标特征从背景杂波中分离出来，以防止杂波干扰后续对齐。第三个模块采用对比损失进行域对齐，从生成的多样化数据和解耦特征中提取稳健的目标特征。最后，所提出的方法在 MSTAR 数据集中的九种操作条件下表现出了令人印象深刻的鲁棒性，大量的定性和定量分析验证了我们框架的有效性。

<p align="center">
  <img src="https://github.com/waterdisappear/SAR-ATR-HDANet/blob/main/fig_framework.png" width="720">
</p>

<p align="center">
  <img src="https://github.com/waterdisappear/SAR-ATR-HDANet/blob/main/fig_radarmap.png" width="480">
</p>

## Data
The folder includes MSTAR images under SOC and EOCs and detailed information can be found in our paper. (JPEG-E)

## HDANet
Requirements
- Python 
- PyTorch 
- Numpy
- Captum (We used captum to generate pseudo-labels.)

A simple demo.
```python
from HDANet.utils.DataLoad import load_data, load_test
from HDANet.utils.TrainTest import model_train, model_test
from HDANet.Model.HDANet import HDANet

train_all, label_name = load_data(arg.data_path + 'TRAIN', id=arg.GPU_ids)
test_set, _ = load_test(arg.data_path + 'TEST')
train_loader = torch.utils.data.DataLoader(train_all, batch_size=arg.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)

model = HDANet(num_classes=len(label_name))
opt = torch.optim.NAdam(model.parameters(), lr=arg.lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
best_test_accuracy = 0

for epoch in range(1, arg.epochs + 1):
    # print("##### " + str(k + 1) + " EPOCH " + str(epoch) + "#####")
    model_train(model=model, data_loader=train_loader, opt=opt)
    scheduler.step()

acc = model_test(model, test_loader)
```



## Contact us
If you have any questions, please contact us at lwj2150508321@sina.com

```
@ARTICLE{10283916,
  author={Li, Weijie and Yang, Wei and Zhang, Wenpeng and Liu, Tianpeng and Liu, Yongxiang and Liu, Li},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Hierarchical Disentanglement-Alignment Network for Robust SAR Vehicle Recognition}, 
  year={2023},
  volume={16},
  number={},
  pages={9661-9679},
  doi={10.1109/JSTARS.2023.3324182}}
```
