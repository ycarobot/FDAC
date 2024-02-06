import torch
import numpy as np
import os
import csv
from easydict import EasyDict
import random
from baselines import SourceOnly, ViT_CL
from utils import data_path, data_utils


def get_dataloaders(opt, dataset, dt, bs, baseline=None):
    train_loaders, test_loaders, source_domains = [], [], []
    if dataset == 'Office31':
        epoch_samples = 5000
        total_epochs = 100
        source_domains = ['amazon', 'webcam', 'dslr']
        root_path = data_path.Office31_path
        opt.n_classes = 31

    elif dataset == 'OfficeHome':
        epoch_samples = 5000
        total_epochs = 100
        source_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
        root_path = data_path.OfficeHome_path
        opt.n_classes = 65

    elif dataset == 'OfficeCaltech':
        epoch_samples = 5000
        total_epochs = 100
        source_domains = ['amazon', 'caltech', 'dslr', 'webcam']
        root_path = data_path.OfficeClatech_path
        opt.n_classes = 10


    elif dataset == 'DigitFive':
        epoch_samples = 5000
        total_epochs = 100
        source_domains = ['amazon', 'caltech', 'dslr', 'webcam']
        root_path = data_path.OfficeClatech_path
        opt.n_classes = 10

    elif dataset == 'DomainNet':
        # The total data numbers we use in each epoch
        epoch_samples = 30000
        total_epochs = 80
        source_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        root_path = data_path.DomainNet_path
        opt.n_classes = 345


    # load dataloaders
    train_loader_dt, test_loader_dt = data_utils.get_dataloaders_train_test(
        root_path=root_path, domain=dt, bs=bs, dataset=dataset, baseline=baseline)
    train_loaders.append(train_loader_dt)
    test_loaders.append(test_loader_dt)
    source_domains.remove(dt)
    for ds in source_domains:
        train_loader_ds, test_loader_ds = data_utils.get_dataloaders_train_test(
        root_path=root_path, domain=ds, bs=bs, dataset=dataset, baseline=baseline)
        train_loaders.append(train_loader_ds)
        test_loaders.append(test_loader_ds)
    return train_loaders, test_loaders, source_domains


def training(opt):
    train_loaders, test_loaders, source_domains = get_dataloaders(
         opt=opt, dt=opt.dt, dataset=opt.dataset, bs=opt.bs, baseline=opt.baseline)
    opt.source_domains = source_domains  # Convenient for the later experiments about malicious domains 

    net_dict = {
        'SourceOnly': [SourceOnly.SourceOnly_net, ViT_CL.domain_adaptation],
        'FDAC': [ViT_CL.ViT_CL_base, ViT_CL.domain_adaptation]
    }

    # Federated domain adaptation
    net_list = []
    for _ in range(len(train_loaders)):
        net_list.append(net_dict[opt.baseline][0](opt.n_classes).cuda())
    result = net_dict[opt.baseline][1](train_loaders, test_loaders, net_list, opt)
    return result


def train_to_csv(baseline, dataset, dt, domain_name, seed=0, **kwargs):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    opt = EasyDict()
    opt.dataset = dataset
    opt.baseline = baseline
    opt.domain_name = domain_name
    opt.dt = dt  # n_ds -> dt
    opt.bs = 45
    opt.lr = 1e-3
    opt.threshold = 0.85
    opt.train_epochs = 70
    opt.communication_cost = 1
    opt.local_epochs = 100
    # fine-grained setting

    for mixup in [False]:
        opt.mixup_p = mixup
        opt.tb_name = '{}'.format(opt.baseline)
        result_path = r'./{}/{}'.format(opt.dataset, opt.tb_name)
        os.makedirs(result_path, exist_ok=True)
        for local_epochs in [10, 20, 50, 100, 200]:
            # opt.train_epochs = int(100 / local_epochs)
            opt.local_epochs = local_epochs
            for iter in range(1):
                opt.iter = iter
                result = training(opt)
                with open('./{}/{}_LE_{}.csv'.format(result_path, opt.domain_name, opt.local_epochs), 'a+',
                          newline='') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow([result])


if __name__ == '__main__':
    train_to_csv('FDAC', dt='RealWorld', dataset='OfficeHome', domain_name='R')