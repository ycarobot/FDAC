from tensorboardX import SummaryWriter
from timm.utils import NativeScaler
import time
import torch
from torch.cuda import amp
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import eval_utils, federated_aggregation_utils
from baselines import SourceOnly


def train_ds_normal(train_dloader, net, opt, loss_scaler):
    criterion_c = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
    net.train()
    for (x, y, index) in train_dloader:
        if x.shape[0] == 1:
            continue
        x, y = x.cuda().float(), y.long().cuda()
        with amp.autocast(): # Automatic mixed precision
            z, _, _ = net(x)
            loss_total = criterion_c(z, y)
        optimizer.zero_grad()
        loss_scaler(loss_total, optimizer, clip_grad=None, parameters=net.parameters())

def train_dt(epoch_out, train_loader, net_list, optimizer, loss_scaler, criterion, test_loader=None, **kwargs):
    for epoch_in, (x, _, target_indices) in enumerate(train_loader):
        net_list[0].train()
        x = x.cuda()
        with amp.autocast():
            loss_total = criterion.forward(
                x=x, net_list=net_list, global_epoch=epoch_out, local_epoch=epoch_in, current_index=target_indices,
                dataloader=test_loader)
        optimizer.zero_grad()
        loss_scaler(loss_total, optimizer, clip_grad=None, parameters=net_list[0].parameters())

def domain_adaptation(train_loaders, test_loaders, net_list, opt):
    # target domain training method
    criterion_dt = train_dict[opt.baseline].Training(n_classes=opt.n_classes)
    optimizer = optim.SGD(net_list[0].parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    best_result = -99
    loss_scaler = NativeScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

    # In FDA, only one complete cycle(all dataloader) can be calculated
    for epoch in range(opt.train_epochs):
        # start = time.time()
        # ------------------------------ Step 1: Train for Ds
        for f in range(1):
            for train_dloader, net in zip(train_loaders[1:], net_list[1:]):
                train_ds_normal(train_dloader, net, opt, loss_scaler)

        # ------------------------------Step 2: Train for Dt
        train_dt(
            epoch, train_loaders[0], net_list, optimizer=optimizer, loss_scaler=loss_scaler,
            prototype_weight_list=None, test_loader=test_loaders[0], criterion=criterion_dt)
        # ------------------------------Step 3: Federated Aggregation
        federated_aggregation_utils.federated_update_parameters_Blocks(
            net_list, block_index_list=None, domain_weight=None)

        if (epoch + 1) % 10 == 0:
            result = eval_utils.eval_dt_multi_classifier(net_list[1:], test_loaders[0])
            best_result = result if result > best_result else best_result
            # scheduler.step()
    return best_result
