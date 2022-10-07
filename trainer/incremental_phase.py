# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: blacklancer
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Copyright (c) 2022
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 记录传递过程中的feature
cur_features = []
ref_features = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def incremental_phase(args, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, is_start_iteration, lamda,
            fix_bn=False, weight_per_class=None, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
    num_classes = tg_model.fc.out_features
    if not is_start_iteration:
        ref_model.eval()
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
        num_old_classes = ref_model.fc.out_features

    epochs = args.epochs
    T = args.T
    beta = args.beta
    for epoch in range(epochs):
        #train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr())


        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = torch.tensor(targets, dtype=torch.long)
            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)

            if is_start_iteration:
                loss1 = 0
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else:
                ref_outputs = ref_model(inputs)

                loss11 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), \
                    torch.ones(inputs.shape[0]).to(device)) * lamda

                loss12 = nn.KLDivLoss()(F.log_softmax(outputs[:, :num_old_classes] / T, dim=1), \
                                       F.softmax(ref_outputs.detach() / T, dim=1)) * T * T * beta * num_old_classes
                loss1 = loss11 + loss12

                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)


            loss = loss1 + loss2
            loss.backward()
            tg_optimizer.step()

            # 记录各个损失值
            train_loss += loss.item()
            if not is_start_iteration:
                train_loss1 += loss1.item()
            else:
                train_loss1 += loss1
            train_loss2 += loss2.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f},Train Loss: {:.4f},Acc: {:.4f}'.format(len(trainloader), \
            train_loss1/(batch_idx+1), train_loss2/(batch_idx+1),train_loss/(batch_idx+1), 100.*correct/total))

        #eval
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = torch.tensor(targets, dtype=torch.long)
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    handle_cur_features.remove()
    if not is_start_iteration:
        # print("Removing register_forward_hook")
        handle_ref_features.remove()

    return tg_model