import sys
sys.path.append('..')
import torch
import numpy as np
from tqdm import tqdm
import argparse
import collections
from HDANet.utils.DataLoad import load_data, load_test
from HDANet.utils.TrainTest import model_train, model_test
from HDANet.Model.HDANet import HDANet

def parameter_setting():
    # argparse settings
    parser = argparse.ArgumentParser(description='Origin Input')
    parser.add_argument('--data_path', type=str, default="../Data/MSTAR_JPEG_E/EOC-Configuration-Version/",
                        help='where data is stored')
    parser.add_argument('--GPU_ids', type=int, default=0,
                        help='GPU ids')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--fold', type=int, default=5,
                        help='K-fold')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    arg = parameter_setting()
    torch.cuda.set_device(arg.GPU_ids)
    # torch.manual_seed(arg.seed)
    # torch.cuda.manual_seed(arg.seed)
    history = collections.defaultdict(list)  # 记录每一折的各种指标

    train_all, label_name = load_data(arg.data_path + 'TRAIN', arg.GPU_ids)
    test_set_1, _ = load_test(arg.data_path + 'TEST_Configuration')
    test_set_2, _ = load_test(arg.data_path + 'TEST_Version')
    for k in tqdm(range(arg.fold)):
        train_loader = torch.utils.data.DataLoader(train_all, batch_size=arg.batch_size, shuffle=True)
        test_loader_C = torch.utils.data.DataLoader(test_set_1, batch_size=arg.batch_size, shuffle=False)
        test_loader_V = torch.utils.data.DataLoader(test_set_2, batch_size=arg.batch_size, shuffle=False)

        model = HDANet(num_classes=len(label_name))
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        opt = torch.optim.NAdam(model.parameters(), lr=arg.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
        best_test_accuracy = 0
        for epoch in range(1, arg.epochs + 1):
            # print("##### " + str(k + 1) + " EPOCH " + str(epoch) + "#####")
            model_train(model=model, data_loader=train_loader, opt=opt)
            scheduler.step()

        acc_C = model_test(model, test_loader_C)
        acc_V = model_test(model, test_loader_V)
        print('test accuracy is {}/ {}'.
              format(acc_C, acc_V))

        history['accuracy_30'].append(acc_C)
        history['accuracy_45'].append(acc_V)


    print('OA is {}, STD is {}'.format(np.mean(history['accuracy_30']), np.std(history['accuracy_30'])))
    print(history['accuracy_30'])
    print('OA is {}, STD is {}'.format(np.mean(history['accuracy_45']), np.std(history['accuracy_45'])))
    print(history['accuracy_45'])


