# Hierarchical Disentanglement-Alignment Network for Robust SAR Vehicle Recognition

This paper proposes a novel domain alignment framework, <a href="https://ieeexplore.ieee.org/document/10283916">Hierarchical Disentanglement-Alignment Network (HDANet)</a>, to enhance features' causality and robustness.

<p align="center">
  <img src="https://github.com/waterdisappear/SAR-ATR-HDANet/blob/main/fig_framework.png" width="960">
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
