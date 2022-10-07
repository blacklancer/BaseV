#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: blacklancer
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Copyright (c) 2022
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Class-incremental learning base trainer. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
import os.path as osp
import copy
try:
    import cPickle as pickle
except:
    import pickle
import math
import utils.misc
import models.modified_linear as modified_linear
import models.resnet_cifar as resnet_cifar
import models.resnet_imagenet as resnet_imagenet
from utils.imagenet.utils_dataset import *
from utils.compute_features import compute_features
from utils.compute_accuracy import compute_accuracy
import warnings
warnings.filterwarnings('ignore')


class BaseTrainer(object):
    """The class that contains the code for base trainer class.
    This file only contains the related functions used in the training process.
    If you hope to view the overall training process, you may find it in the file
    named trainer.py in the same folder.
    """

    def __init__(self, the_args):
        """The function to initialize this class.
        Args:
          the_args: all inputted parameter.
        """
        self.args = the_args
        self.set_save_path()
        self.set_cuda_device()
        self.set_dataset_variables()


    def set_save_path(self):
        """The function to set the saving path."""
        self.log_dir = './logs/'
        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.save_path = self.log_dir + self.args.dataset + \
                         '_nfg' + str(self.args.nb_cl_fg) + \
                         '_ncls' + str(self.args.nb_cl) + \
                         '_nproto' + str(self.args.nb_protos) + \
                         '_seed' + str(self.args.random_seed)
        if self.args.fix_budget:
            self.save_path += '_fix'
        else:
            self.save_path += '_dynamic'
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)


    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def set_dataset_variables(self):
        """The function to set the dataset parameters."""
        if self.args.dataset == 'cifar100':
            # Set CIFAR-100
            # Set the pre-processing steps for training set
            self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5071, 0.4866, 0.4409),
                                                                            (0.2009, 0.1984, 0.2023)),])
            # Set the pre-processing steps for test set
            self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.5071, 0.4866, 0.4409),
                                                                           (0.2009, 0.1984, 0.2023)),])
            # Initial the dataloader
            self.trainset = torchvision.datasets.CIFAR100(root='./data',
                                                          train=True,
                                                          download=True,
                                                          transform=self.transform_train)
            self.testset = torchvision.datasets.CIFAR100(root='./data',
                                                         train=False,
                                                         download=True,
                                                         transform=self.transform_test)
            self.evalset = torchvision.datasets.CIFAR100(root='./data',
                                                         train=False,
                                                         download=False,
                                                         transform=self.transform_test)

            # Set the network architecture
            self.network = resnet_cifar.resnet32
            # Set the learning rate decay parameters
            self.lr_start = [int(self.args.epochs * 0.5), int(self.args.epochs * 0.75)]
            # Set the dictionary size
            self.dictionary_size = 500
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            # Set imagenet-subset and imagenet
            # Set the data directories
            traindir = os.path.join(self.args.data_dir, 'train')
            valdir = os.path.join(self.args.data_dir, 'val')
            # Set the dataloaders
            train_transforms = [transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=63 / 255)]
            test_transforms = [transforms.Resize(256),
                               transforms.CenterCrop(224)]
            common_transforms = [transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])]
            train_trsf = transforms.Compose([*train_transforms, *common_transforms])
            test_trsf = transforms.Compose([*test_transforms, *common_transforms])

            # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            self.trainset = datasets.ImageFolder(traindir, train_trsf)
            self.testset = datasets.ImageFolder(valdir, test_trsf)
            self.evalset = datasets.ImageFolder(valdir, test_trsf)

            # Set the network architecture
            if self.args.dataset == 'imagenet':
                self.network = resnet_imagenet.resnet18
            elif self.args.dataset == 'imagenet_sub':
                self.network = resnet_imagenet.resnet34
            else:
                raise ValueError('Please set the correct backbone.')
            # Set the learning rate decay parameters
            self.lr_start = [int(self.args.epochs * 0.333), int(self.args.epochs * 0.667)]
            # Set the dictionary size
            self.dictionary_size = 1500
        else:
            raise ValueError('Please set the correct dataset.')


    def map_labels(self, order_list, Y_set):
        """The function to map the labels according to the class order list.
        Args:
          order_list: the class order list.
          Y_set: the target labels before mapping
        Return:
          map_Y: the mapped target labels
        """
        map_Y = []
        for idx in Y_set:
            map_Y.append(order_list.index(idx))
        map_Y = np.array(map_Y)
        return map_Y
    
    
    def set_dataset(self):
        """The function to set the datasets.
        Returns:
          X_train_total: an array that contains all training samples
          Y_train_total: an array that contains all training labels
          X_valid_total: an array that contains all validation samples
          Y_valid_total: an array that contains all validation labels
        """
        if self.args.dataset == 'cifar100':
            X_train_total = np.array(self.trainset.data)
            Y_train_total = np.array(self.trainset.targets)
            X_valid_total = np.array(self.testset.data)
            Y_valid_total = np.array(self.testset.targets)
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            X_train_total, Y_train_total = split_images_labels(self.trainset.imgs)
            X_valid_total, Y_valid_total = split_images_labels(self.testset.imgs)
        else:
            raise ValueError('Please set the correct dataset.')

        return X_train_total, Y_train_total, X_valid_total, Y_valid_total


    def init_class_order(self):
        """The function to initialize the class order.
        Returns:
          order: an array for the class order
          order_list: a list for the class order
        """
        # Set the random seed according to the config
        np.random.seed(self.args.random_seed)
        # Set the name for the class order file
        order_name = osp.join(self.save_path, "seed_{}_order.pkl".format(self.args.random_seed))
        # Print the name for the class order file
        print("Order name:{}".format(order_name))

        if osp.exists(order_name):
            # If we have already generated the class order file, load it
            print("Loading the saved class order")
            order = utils.misc.unpickle(order_name)
        else:
            # If we don't have the class order file, generate a new one
            print("Generating a new class order")
            order = np.arange(self.args.num_classes)
            np.random.shuffle(order)
            utils.misc.savepickle(order, order_name)
        # Transfer the array to a list
        order_list = list(order)
        # Print the class order
        print(order_list)
        return order, order_list
    

    def init_prototypes(self, dictionary_size, order, X_train_total, Y_train_total):
        """The function to intialize the prototypes.
           Please note that the prototypes here contains all training samples.
           alpha_dr_herding contains the indexes for the selected exemplars
           because we want to compare our method with the theoretical case where all the training samples are stored
        Args:
          dictionary_size: the dictionary size, i.e., the maximum number of samples for each class
          order: the class order
          X_train_total: an array that contains all training samples
          Y_train_total: an array that contains all training labels
        Returns:
          alpha_dr_herding: an empty array to store the indexes for the exemplars
          prototypes: an array contains all training samples for all phases
        """
        # Set an empty to store the indexes for the selected exemplars
        alpha_dr_herding  = np.zeros((int(self.args.num_classes/self.args.nb_cl),
                                      dictionary_size, self.args.nb_cl), np.float32)
        if self.args.dataset == 'cifar100':
            # CIFAR-100, directly load the tensors for the training samples
            prototypes = np.zeros((self.args.num_classes, dictionary_size,
                                   X_train_total.shape[1], X_train_total.shape[2], X_train_total.shape[3]))
            for orde in range(self.args.num_classes):
                prototypes[orde,:,:,:,:] = X_train_total[np.where(Y_train_total==order[orde])]
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            # ImageNet, save the paths for the training samples if an array
            prototypes = [[] for _ in range(self.args.num_classes)]
            for orde in range(self.args.num_classes):
                prototypes[orde] = X_train_total[np.where(Y_train_total==order[orde])]
            prototypes = np.array(prototypes)
        else:
            raise ValueError('Please set correct dataset.')
        return alpha_dr_herding, prototypes
    
    
    def init_current_phase_model(self, iteration, start_iter, tg_model):
        """The function to intialize the models for the current phase 
        Args:
          iteration: the iteration index 
          start_iter: the iteration index for the 0th phase
          tg_model: the model from last phase
        Returns:
          tg_model: the model from the current phase
          ref_model: the reference model from last phase (frozen, not trainable)
          last_iter: the iteration index for last phase
        """
        lamda_mult = 0
        if iteration == start_iter:
            ############################################################
            last_iter = 0
            ############################################################
            tg_model = self.network(num_classes=self.args.nb_cl_fg)
            # Get the information about the input and output features from the network
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features)
            # The reference model are not used, set them to None
            ref_model = None
        elif iteration == start_iter + 1:
            ############################################################
            last_iter = iteration
            ############################################################
            #increment classes
            ref_model = copy.deepcopy(tg_model)
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            # Set the final FC layer for classification
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features, self.args.nb_cl)
            new_fc.fc1.weight.data = tg_model.fc.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc
            print("in_features:", tg_model.fc.in_features, "out_features:", tg_model.fc.out_features)
            # Update the lambda parameter for the current phase
            lamda_mult = tg_model.fc.out_features * 1.0 / self.args.nb_cl
        else:
            ############################################################
            last_iter = iteration
            ############################################################
            #increment classes, copy model to reference
            ref_model = copy.deepcopy(tg_model)
            in_features = tg_model.fc.in_features
            out_features1 = tg_model.fc.fc1.out_features
            out_features2 = tg_model.fc.fc2.out_features
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features1 + out_features2, self.args.nb_cl)
            new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc
            print("in_features:", tg_model.fc.in_features, "out_features:", tg_model.fc.out_features)
            # Update the lambda parameter for the current phase
            lamda_mult = tg_model.fc.out_features * 1.0 / self.args.nb_cl

        # Update the current lambda value for the current phase
        if iteration > start_iter:
            cur_the_lambda = self.args.the_lambda * math.sqrt(lamda_mult)
        else:
            cur_the_lambda = self.args.the_lambda

        return tg_model, ref_model, last_iter, lamda_mult, cur_the_lambda


    def init_current_phase_dataset(self, iteration, start_iter, last_iter, order, order_list, \
                                   X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
                                   X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, \
                                   X_protoset_cumuls, Y_protoset_cumuls):
        """The function to intialize the dataset for the current phase
        Args:
          iteration: the iteration index
          start_iter: the iteration index for the 0th phase
          last_iter: the iteration index for last phase
          order: the array for the class order
          order_list: the list for the class order
          X_train_total: the array that contains all training samples
          Y_train_total: the array that contains all training labels
          X_valid_total: the array that contains all validation samples
          Y_valid_total: the array that contains all validation labels
          X_train_cumuls: the array that contains old training samples
          Y_train_cumuls: the array that contains old training labels
          X_valid_cumuls: the array that contains old validation samples
          Y_valid_cumuls: the array that contains old validation labels
          X_protoset_cumuls: the array that contains old exemplar samples
          Y_protoset_cumuls: the array that contains old exemplar labels
        Returns:
          indices_train: the indexes of new-class samples
          X_train_cumuls: an array that contains old training samples, updated
          Y_train_cumuls: an array that contains old training labels, updated
          X_valid_cumuls: an array that contains old validation samples, updated
          Y_valid_cumuls: an array that contains old validation labels, updated
          X_protoset_cumuls: an array that contains old exemplar samples, updated
          Y_protoset_cumuls: an array that contains old exemplar labels, updated
          X_train: current-phase training samples, including new-class samples and old-class exemplars
          map_Y_train: mapped labels for X_train
          map_Y_valid_cumul: mapped labels for X_valid_cumuls
          X_valid_ori: an array that contains the 0th-phase validation samples, updated
          Y_valid_ori: an array that contains the 0th-phase validation labels, updated
          X_protoset: an array that contains the exemplar samples
          Y_protoset: an array that contains the exemplar labels
        """
        # Get the indexes of new-class samples (including training and test)
        indices_train = np.array(
            [i in order[range(last_iter * self.args.nb_cl, (iteration + 1) * self.args.nb_cl)] for i in Y_train_total])
        indices_test = np.array(
            [i in order[range(last_iter * self.args.nb_cl, (iteration + 1) * self.args.nb_cl)] for i in Y_valid_total])

        # Get the samples according to the indexes
        X_train = X_train_total[indices_train]
        X_valid = X_valid_total[indices_test]

        # Add the new-class samples to the cumulative X array
        # 这里的X_train_cumuls是不经过筛选的，所有的data都存了起来,后续增量训练的时候用的不是这个
        X_train_cumuls.append(X_train)
        X_valid_cumuls.append(X_valid)
        X_train_cumul = np.concatenate(X_train_cumuls)
        X_valid_cumul = np.concatenate(X_valid_cumuls)

        # Get the labels according to the indexes, and add them to the cumulative Y array
        Y_train = Y_train_total[indices_train]
        Y_valid = Y_valid_total[indices_test]
        Y_train_cumuls.append(Y_train)
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumul = np.concatenate(Y_train_cumuls)
        Y_valid_cumul = np.concatenate(Y_valid_cumuls)

        if iteration == start_iter:
            # Save the 0th-phase validation samples and labels
            # X_valid_ori and Y_valid_ori will no longer change
            # they just keep the 0th-phase data for test
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:
            # Update the exemplar set
            X_protoset = np.concatenate(X_protoset_cumuls)
            Y_protoset = np.concatenate(Y_protoset_cumuls)
            # Create the training samples/labels for the current phase training
            X_train = np.concatenate((X_train, X_protoset), axis=0)
            Y_train = np.concatenate((Y_train, Y_protoset))

        # Generate the mapped labels, according the order list
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        # print('Max and Min of valid labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))

        # Return different variables for different phases
        if iteration == start_iter:
            return indices_train, X_train_cumul, Y_train_cumul, X_valid_cumul, Y_valid_cumul,\
                   X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls,\
                   X_train, map_Y_train, map_Y_valid_cumul, X_protoset_cumuls, Y_protoset_cumuls, X_valid_ori, Y_valid_ori
        else:
            return indices_train, X_valid_cumul, X_train_cumul, Y_valid_cumul, Y_train_cumul, \
                   X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, \
                   X_train, map_Y_train, map_Y_valid_cumul, X_protoset_cumuls, Y_protoset_cumuls, X_protoset, Y_protoset


    def imprint_weights(self, tg_model, iteration, X_train, map_Y_train, dictionary_size):
        """The function to imprint FC classifier's weights 
        Args:
          tg_model: the model from last phase
          iteration: the iteration index
          X_train: current-phase training samples, including new-class samples and old-class exemplars
          map_Y_train: mapped labels for X_train
          dictionary_size: the dictionary size, i.e., the maximum number of samples for each class
        Returns:
          tg_model: the model from the current phase, the FC classifier is updated
          Avoid newly added class classifier weights being randomly initialized to affect performance
        """
        if self.args.dataset == 'cifar100':
            # Load previous FC weights, transfer them from GPU to CPU
            old_embedding_norm = tg_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
            # tg_feature_model is tg_model without the FC layer
            tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
            # Get the shape of the feature inputted to the FC layers, i.e., the shape for the final feature maps
            num_features = tg_model.fc.in_features
            # Intialize the new FC weights with zeros
            novel_embedding = torch.zeros((self.args.nb_cl, num_features))
            for cls_idx in range(iteration*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Get the indexes of samples for one class
                cls_indices = np.array([i == cls_idx  for i in map_Y_train])
                # Check the number of samples in this class
                assert(len(np.where(cls_indices==1)[0])==dictionary_size)
                # Set a temporary dataloader for the current class
                self.evalset.data = X_train[cls_indices].astype('uint8')
                self.evalset.targets = np.zeros(self.evalset.data.shape[0])
                evalloader = torch.utils.data.DataLoader(self.evalset,
                                                         batch_size=self.args.eval_batch_size,
                                                         shuffle=False,
                                                         num_workers=self.args.num_workers)
                num_samples = self.evalset.data.shape[0]
                # Compute the feature maps using the current model
                cls_features = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                # Compute the normalized feature maps 
                norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                # Update the FC weights using the imprint weights, i.e., the normalized averged feature maps 
                cls_embedding = torch.mean(norm_features, dim=0)
                novel_embedding[cls_idx-iteration*self.args.nb_cl] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
            # Transfer all weights of the model to GPU
            tg_model.to(self.device)
            tg_model.fc.fc2.weight.data = novel_embedding.to(self.device)
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            # Load previous FC weights, transfer them from GPU to CPU
            old_embedding_norm = tg_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
            # tg_feature_model is b1_model without the FC layer
            tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
            # Get the shape of the feature inputted to the FC layers, i.e., the shape for the final feature maps
            num_features = tg_model.fc.in_features
            # Intialize the new FC weights with zeros
            novel_embedding = torch.zeros((self.args.nb_cl, num_features))
            for cls_idx in range(iteration*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Get the indexes of samples for one class
                cls_indices = np.array([i == cls_idx  for i in map_Y_train])
                # Check the number of samples in this class
                assert(len(np.where(cls_indices==1)[0])<=dictionary_size)
                # Set a temporary dataloader for the current class
                current_eval_set = merge_images_labels(X_train[cls_indices], np.zeros(len(X_train[cls_indices])))
                self.evalset.imgs = self.evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(self.evalset,
                                                         batch_size=self.args.eval_batch_size,
                                                         shuffle=False,
                                                         num_workers=2)
                num_samples = len(X_train[cls_indices])
                # Compute the feature maps using the current model
                cls_features = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                # Compute the normalized feature maps 
                norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                # Update the FC weights using the imprint weights, i.e., the normalized averged feature maps
                cls_embedding = torch.mean(norm_features, dim=0)
                novel_embedding[cls_idx-iteration*self.args.nb_cl] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
            # Transfer all weights of the model to GPU
            tg_model.to(self.device)
            tg_model.fc.fc2.weight.data = novel_embedding.to(self.device)
        else:
            raise ValueError('Please set correct dataset.')
        return tg_model


    def update_train_and_valid_loader(self, X_train, map_Y_train, X_valid_cumul, map_Y_valid_cumul):
        """The function to update the dataloaders
        Args:
          X_train: current-phase training samples, including new-class samples and old-class exemplars
          map_Y_train: mapped labels for X_train
          X_valid_cumul: an array that contains old validation samples
          map_Y_valid_cumul: mapped labels for X_valid_cumuls
        Returns:
          trainloader: the training dataloader
          testloader: the test dataloader
        """
        print('Setting the dataloaders ...')
        if self.args.dataset == 'cifar100':
            # Set the training dataloader
            self.trainset.data = X_train.astype('uint8')
            self.trainset.targets = map_Y_train
            trainloader = torch.utils.data.DataLoader(self.trainset,
                                                      batch_size=self.args.train_batch_size,
                                                      shuffle=True,
                                                      num_workers=self.args.num_workers)
            # Set the test dataloader
            self.testset.data = X_valid_cumul.astype('uint8')
            self.testset.targets = map_Y_valid_cumul
            testloader = torch.utils.data.DataLoader(self.testset,
                                                     batch_size=self.args.test_batch_size,
                                                     shuffle=False,
                                                     num_workers=self.args.num_workers)
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            # Set the training dataloader
            current_train_imgs = merge_images_labels(X_train, map_Y_train)
            self.trainset.imgs = self.trainset.samples = current_train_imgs
            trainloader = torch.utils.data.DataLoader(self.trainset, 
                                                      batch_size=self.args.train_batch_size,
                                                      shuffle=True, 
                                                      num_workers=self.args.num_workers, 
                                                      pin_memory=True)
            # Set the test dataloader
            current_test_imgs = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
            self.testset.imgs = self.testset.samples = current_test_imgs
            testloader = torch.utils.data.DataLoader(self.testset, 
                                                     batch_size=self.args.test_batch_size,
                                                     shuffle=False, 
                                                     num_workers=self.args.num_workers)
        else:
            raise ValueError('Please set the correct dataset.')
        return trainloader, testloader
    

    def set_optimizer(self, iteration, start_iter, tg_model):
        """The function to set the optimizers for the current phase 
        Args:
          tg_model: the model from the current phase
        Returns:
          tg_optimizer: the optimizer for tg_model
          tg_lr_scheduler: the learning rate decay scheduler for tg_model
        """
        if iteration > start_iter and self.args.fix_old:
            # fix the embedding of old classes
            ignored_params = list(map(id, tg_model.fc.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, \
                                 tg_model.parameters())
            tg_params = [{'params': base_params, 'lr': self.args.base_lr, 'weight_decay': self.args.custom_weight_decay}, \
                         {'params': tg_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
        else:
            tg_params = tg_model.parameters()
        tg_optimizer = optim.SGD(tg_params,
                                 lr=self.args.base_lr,
                                 momentum=self.args.custom_momentum,
                                 weight_decay=self.args.custom_weight_decay)
        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer,
                                                   milestones=self.lr_start,
                                                   gamma=self.args.lr_factor)

        return tg_optimizer, tg_lr_scheduler


    def set_exemplar_set(self, tg_model, iteration, last_iter, order, alpha_dr_herding, prototypes):
        """The function to select the exemplars
        Args:
          tg_model: the model from the current phase
          iteration: the iteration index
          last_iter: the iteration index for last phase
          order: the array for the class order
          alpha_dr_herding: the empty array to store the indexes for the exemplars
          prototypes: the array contains all training samples for all phases
        Returns:
          # 这里可以看出来下面俩才是存储真正的proto的,而prototypes实际上算是所有数据的暂存器
          X_protoset_cumuls: an array that contains old exemplar samples
          Y_protoset_cumuls: an array that contains old exemplar labels
          class_means: the mean values for each class
          alpha_dr_herding: the empty array to store the indexes for the exemplars, updated
        """
        # Use the dictionary size defined in this class-incremental learning class
        dictionary_size = self.dictionary_size
        if self.args.fix_budget:
            # Using fixed exemplar budget. The total memory size is unchanged
            nb_protos_cl = int(np.ceil(self.args.nb_protos * 100. / self.args.nb_cl / (iteration + 1)))
        else:
            # Using dynamic exemplar budget, i.e., 20 exemplars each class.
            # In this setting, the total memory budget is increasing
            nb_protos_cl = self.args.nb_protos
        # print("nb-proto", nb_protos_cl)

        # Get tg_feature_model, which is a model copied from b1_model, without the FC layer
        # tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        tg_feature_model = nn.Sequential(tg_model.conv1, tg_model.bn1, tg_model.relu,
                                         tg_model.layer1, tg_model.layer2, tg_model.layer3,
                                         tg_model.avgpool, tg_model.flatten)
        # Get the shape for the feature maps
        num_features = tg_model.fc.in_features

        # Herding
        if self.args.dataset == 'cifar100':
            for iter_dico in range(last_iter*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Set a temporary dataloader for the current class
                self.evalset.data = prototypes[iter_dico].astype('uint8')
                self.evalset.targets = np.zeros(self.evalset.data.shape[0])
                evalloader = torch.utils.data.DataLoader(self.evalset,
                                                         batch_size=self.args.eval_batch_size,
                                                         shuffle=False,
                                                         num_workers=self.args.num_workers)
                num_samples = self.evalset.data.shape[0]
                # Compute the features for the current class
                mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                # Herding algorithm
                D = mapped_prototypes.T
                D = D/np.linalg.norm(D, axis=0)
                mu  = np.mean(D, axis=1)
                index1 = int(iter_dico/self.args.nb_cl)
                index2 = iter_dico % self.args.nb_cl
                alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
                w_t = mu
                iter_herding     = 0
                iter_herding_eff = 0
                while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl,500)) and iter_herding_eff<1000:
                    tmp_t   = np.dot(w_t, D)
                    ind_max = np.argmax(tmp_t)
                    iter_herding_eff += 1
                    if alpha_dr_herding[index1,ind_max,index2] == 0:
                        alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
                        iter_herding += 1
                    w_t = w_t + mu - D[:,ind_max]
                # print("iter_herding=", iter_herding)
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            for iter_dico in range(last_iter*self.args.nb_cl, (iteration+1)*self.args.nb_cl):
                # Set a temporary dataloader for the current class
                current_eval_set = merge_images_labels(prototypes[iter_dico], np.zeros(len(prototypes[iter_dico])))
                self.evalset.imgs = self.evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(self.evalset,
                                                         batch_size=self.args.eval_batch_size,
                                                         shuffle=False,
                                                         num_workers=self.args.num_workers,
                                                         pin_memory=True)
                num_samples = len(prototypes[iter_dico])
                # Compute the features for the current class
                mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                # Herding algorithm
                D = mapped_prototypes.T
                D = D/np.linalg.norm(D,axis=0)
                mu  = np.mean(D,axis=1)
                index1 = int(iter_dico/self.args.nb_cl)
                index2 = iter_dico % self.args.nb_cl
                alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
                w_t = mu
                iter_herding     = 0
                iter_herding_eff = 0
                while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl, 500)) and iter_herding_eff<1000:
                    tmp_t   = np.dot(w_t,D)
                    ind_max = np.argmax(tmp_t)
                    iter_herding_eff += 1
                    if alpha_dr_herding[index1,ind_max,index2] == 0:
                        alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
                        iter_herding += 1
                    w_t = w_t+mu-D[:,ind_max]
        else:
            raise ValueError('Please set the correct dataset.')

        # Set two empty lists for the exemplars and the labels
        X_protoset_cumuls = []
        Y_protoset_cumuls = []
        if self.args.dataset == 'cifar100':
            class_means = np.zeros((num_features, self.args.num_classes, 2))
            for iteration2 in range(iteration+1):
                for iter_dico in range(self.args.nb_cl):
                    # Compute the D and D2 matrizes, which are used to compute the class mean values
                    current_cl = order[range(iteration2*self.args.nb_cl, (iteration2+1)*self.args.nb_cl)]
                    self.evalset.data = prototypes[iteration2*self.args.nb_cl+iter_dico].astype('uint8')
                    self.evalset.targets = np.zeros(self.evalset.data.shape[0])
                    evalloader = torch.utils.data.DataLoader(self.evalset,
                                                             batch_size=self.args.eval_batch_size,
                                                             shuffle=False,
                                                             num_workers=self.args.num_workers)
                    num_samples = self.evalset.data.shape[0]
                    mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                    D = mapped_prototypes.T
                    D = D/np.linalg.norm(D, axis=0)

                    self.evalset.data = prototypes[iteration2*self.args.nb_cl+iter_dico][:, :, :, ::-1].astype('uint8')
                    evalloader = torch.utils.data.DataLoader(self.evalset,
                                                             batch_size=self.args.eval_batch_size,
                                                             shuffle=False,
                                                             num_workers=self.args.num_workers)
                    mapped_prototypes2 = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                    D2 = mapped_prototypes2.T
                    D2 = D2/np.linalg.norm(D2,axis=0)
                    # Using the indexes selected by herding
                    # select for NME
                    alph = alpha_dr_herding[iteration2, :, iter_dico]
                    alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                    # Add the exemplars and the labels to the lists
                    X_protoset_cumuls.append(prototypes[iteration2*self.args.nb_cl+iter_dico,np.where(alph==1)[0]])
                    Y_protoset_cumuls.append(order[iteration2*self.args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
                    alph = alph / np.sum(alph)
                    class_means[:, current_cl[iter_dico], 0] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
                    class_means[:, current_cl[iter_dico], 0] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 0])
                    # select for NMS
                    alph = np.ones(dictionary_size) / dictionary_size
                    class_means[:, current_cl[iter_dico], 1] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
                    class_means[:, current_cl[iter_dico], 1] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 1])
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':
            class_means = np.zeros((num_features, self.args.num_classes, 2))
            for iteration2 in range(iteration+1):
                for iter_dico in range(self.args.nb_cl):
                    # Compute the D and D2 matrizes, which are used to compute the class mean values
                    current_cl = order[range(iteration2*self.args.nb_cl,(iteration2+1)*self.args.nb_cl)]
                    current_eval_set = merge_images_labels(prototypes[iteration2*self.args.nb_cl+iter_dico], \
                        np.zeros(len(prototypes[iteration2*self.args.nb_cl+iter_dico])))
                    self.evalset.imgs = self.evalset.samples = current_eval_set
                    evalloader = torch.utils.data.DataLoader(self.evalset, 
                                                             batch_size=self.args.eval_batch_size,
                                                             shuffle=False, 
                                                             num_workers=self.args.num_workers, 
                                                             pin_memory=True)
                    num_samples = len(prototypes[iteration2*self.args.nb_cl+iter_dico])
                    mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                    D = mapped_prototypes.T
                    D = D/np.linalg.norm(D,axis=0)
                    D2 = D
                    # Using the indexes selected by herding
                    alph = alpha_dr_herding[iteration2,:,iter_dico]
                    assert((alph[num_samples:]==0).all())
                    alph = alph[:num_samples]
                    alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                    # Add the exemplars and the labels to the lists
                    X_protoset_cumuls.append(prototypes[iteration2*self.args.nb_cl+iter_dico][np.where(alph==1)[0]])
                    Y_protoset_cumuls.append(order[iteration2*self.args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
                    # Compute the class mean values
                    alph = alph/np.sum(alph)
                    class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])
                    alph = np.ones(num_samples)/num_samples
                    class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                    class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])
        else:
            raise ValueError('Please set the correct dataset.')

        # Save the class mean values
        # torch.save(class_means, osp.join(self.save_path, 'iter_{}_class_means.pth'.format(iteration)))
        return X_protoset_cumuls, Y_protoset_cumuls, class_means, alpha_dr_herding


    def compute_acc(self, class_means, order, order_list, tg_model, X_valid_ori, Y_valid_ori,
                    X_valid_cumul, Y_valid_cumul, iteration, top1_acc_list_ori, top1_acc_list_cumul):
        """The function to compute the accuracy
        Args:
          class_means: the mean values for each class
          order: the array for the class order
          order_list: the list for the class order
          tg_model: the model from the current phase
          X_valid_ori: the array that contains the 0th-phase validation samples, updated
          Y_valid_ori: the array that contains the 0th-phase validation labels, updated
          X_valid_cumul: the array that contains old validation samples
          Y_valid_cumul: the array that contains old validation labels
          iteration: the iteration index
          top1_acc_list_ori: the list to store the results for the 0th classes
          top1_acc_list_cumul: the list to store the results for the current phase
        Returns:
          top1_acc_list_ori: the list to store the results for the 0th classes, updated
          top1_acc_list_cumul: the list to store the results for the current phase, updated
        """
        # Get tg_feature_model, which is a model copied from b1_model, without the FC layer
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        # Get the class mean values for all seen classes
        current_means = class_means[:, order[range(0,(iteration+1)*self.args.nb_cl)]]

        # Get mapped labels for the 0-th phase data, according the the order list
        map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
        print('Computing accuracy on the 0-th phase classes...')
        # Set a temporary dataloader for the 0-th phase data
        if self.args.dataset == 'cifar100':
            self.evalset.data = X_valid_ori.astype('uint8')
            self.evalset.targets = map_Y_valid_ori
            pin_memory = False
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':   
            current_eval_set = merge_images_labels(X_valid_ori, map_Y_valid_ori)
            self.evalset.imgs = self.evalset.samples = current_eval_set
            pin_memory = True
        else:
            raise ValueError('Please set the correct dataset.')

        evalloader = torch.utils.data.DataLoader(self.evalset,
                                                 batch_size=self.args.eval_batch_size,
                                                 shuffle=False,
                                                 num_workers=self.args.num_workers,
                                                 pin_memory=pin_memory)
        # Compute the accuracies for the 0-th phase test data
        ori_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader)
        # Add the results to the array, which stores all previous results
        top1_acc_list_ori[iteration, :, 0] = np.array(ori_acc).T

        # Get mapped labels for the current-phase data, according the the order list
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        # Set a temporary dataloader for the current-phase data
        print('Computing cumulative accuracy...')
        if self.args.dataset == 'cifar100':
            self.evalset.data = X_valid_cumul.astype('uint8')
            self.evalset.targets = map_Y_valid_cumul
        elif self.args.dataset == 'imagenet_sub' or self.args.dataset == 'imagenet':  
            current_eval_set = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
            self.evalset.imgs = self.evalset.samples = current_eval_set
        else:
            raise ValueError('Please set the correct dataset.')

        evalloader = torch.utils.data.DataLoader(self.evalset,
                                                 batch_size=self.args.eval_batch_size,
                                                 shuffle=False,
                                                 num_workers=self.args.num_workers,
                                                 pin_memory=pin_memory)
        # Compute the accuracies for the current-phase data    
        cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader)
        # Add the results to the array, which stores all previous results
        top1_acc_list_cumul[iteration, :, 0] = np.array(cumul_acc).T

        return top1_acc_list_ori, top1_acc_list_cumul