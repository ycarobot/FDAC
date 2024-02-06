import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
from ViT_network import ViT_Cross
from tensorboardX import SummaryWriter
from timm.utils import NativeScaler
import time
from torch.cuda import amp
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import eval_utils, federated_aggregation_utils


def to_one_hot(label, n_classes):
    y_onehot = torch.FloatTensor(label.shape[0], n_classes).to(label.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, label.unsqueeze(1), 1)
    return y_onehot


def knowledge_vote(knowledge_list, confidence_gate, num_classes):
    """
    :param torch.tensor knowledge_list : recording the knowledge from each source domain Model
    :param float confidence_gate: the confidence gate to judge which sample to use
    :return: consensus_confidence,consensus_knowledge,consensus_knowledge_weight
    """

    max_p, max_p_class = knowledge_list.max(2)  # In Ds,  [bs, n_ds]   k_list: [bs, n_ds]
    max_conf, _ = max_p.max(1)  #  
    max_p_mask = (max_p > confidence_gate).float().cuda()  # To judge
    consensus_knowledge = torch.zeros(knowledge_list.size(0), knowledge_list.size(2)).cuda()  # [bs, n_classes]
    for batch_idx, (p, p_class, p_mask) in enumerate(zip(max_p, max_p_class, max_p_mask)):  # batch times in total
        # to solve the [0,0,0] situation
        if torch.sum(p_mask) > 0:
            p = p * p_mask
        for source_idx, source_class in enumerate(p_class):  # p_class: [1, n_ds]
            #  
            #   p[source_idx] represents the probability of this category
            consensus_knowledge[batch_idx, source_class] += p[source_idx]  # 0.9 + 0.9 =1.8  [bs, n_classes]
    consensus_knowledge_conf, consensus_knowledge = consensus_knowledge.max(1)  # [bs, n_c] and get pseudo-labels
    # only consider samples which has high probability output from ds. do not consider which class and which domain. So this is sample-level!
    consensus_knowledge_mask = (max_conf > confidence_gate).float().cuda()  #  
    consensus_knowledge = torch.zeros(consensus_knowledge.size(0), num_classes).cuda().scatter_(
        1, consensus_knowledge.view(-1, 1), 1)  # to one hot
    return consensus_knowledge_conf, consensus_knowledge, consensus_knowledge_mask


def get_consensus_knowledge(x, net_list, n_classes):
    with torch.no_grad():
        knowledge_list = [torch.softmax(net_list[i+1].predict(x), dim=1).unsqueeze(1) for i in range(len(net_list[1:]))]
        knowledge_list = torch.cat(knowledge_list, 1)  # [bs, n_ds, n_classes]
        _, consensus_knowledge, consensus_weight = knowledge_vote(
            knowledge_list, confidence_gate=0.85, num_classes=n_classes)  # c_k: [bs, n_c]  one-hot
    return consensus_knowledge, consensus_weight


def feature_mixup(x, y_onehot=None):
    device = x.device
    lam = np.random.beta(2, 2, x.shape[0])
    index = torch.randperm(x.shape[0]).to(device)
    x0 = torch.cat(([x[i].unsqueeze(0) * lam[i] for i in range(x.shape[0])]))
    x1 = torch.cat(([x[index][i].unsqueeze(0) * (1 - lam[i]) for i in range(x.shape[0])]))
    x_mixed = x0 + x1
    if y_onehot is not None:
        y0 = torch.cat(([y_onehot[i].unsqueeze(0) * lam[i] for i in range(x.shape[0])]))
        y1 = torch.cat(([y_onehot[index][i].unsqueeze(0) * (1 - lam[i]) for i in range(x.shape[0])]))
        y_onehot_mixed = y0 + y1
        return x_mixed, y_onehot_mixed
    else: return x_mixed


def SupConLoss_normal(features, labels, class_weight=None):
    if len(labels.shape) > 1:
        labels = labels.max(1)[1].unsqueeze(1)
    # features.shape = [bs, feas_dim]
    device = features.device
    temperature, base_temperature = 0.07, 0.07
    # compute logits
    anchor_feature = features
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, features.T), temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    # tile mask
    mask = torch.eq(labels, labels.T).float().to(device)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask), 1, torch.arange(features.shape[0]).view(-1, 1).to(device), 0)
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.mean()
    return loss


class MultiLevel_CL:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.n_blks = 12  # vit-base
        self.blk_indices = [5, 6, 7, 8, 9, 10, 11]
        self.loss_weights = [0.05, 0.001, 0.1, 1.0]  # block, affinity, prototype, pl
        self.matching_loss = nn.MSELoss(reduction='sum')

    def forward_ConsensusKnowledge(self, x, net_list):
        y_t_onehot, consensus_weight = get_consensus_knowledge(x, net_list, self.n_classes) # get pseudo-labels
        # MixUp
        x_mixed, y_onehot_mixed = feature_mixup(x, y_onehot=y_t_onehot)
        z_t, latent_tokens, f_t = net_list[0](x_mixed)  # get output
        z_t_log = torch.log_softmax(z_t, dim=1)
        loss_c = torch.mean(consensus_weight * torch.sum(-1 * y_onehot_mixed * z_t_log, dim=1))
        self.loss_analysis.append(['Loss_base/loss_c', float(loss_c)])
        return loss_c, z_t, latent_tokens, f_t, y_t_onehot, y_onehot_mixed

    def forward_BlockLevel(self, x, net_list):
        device = x.device
        x_mixed = feature_mixup(x)
        _, blocks_t, _ = net_list[0](x_mixed)
        blocks_s_list = []
        for net_ds in net_list[1:]:
            _, blocks_s, _ = net_ds(x_mixed, return_latent_output=True)  # update
            blocks_s_list.append(blocks_s)
        # Block-level alignment
        # generate contrastive labels
        contrastive_labels = torch.arange(x.shape[0]).repeat(len(net_list), 1).to(device)
        contrastive_labels_1hot = to_one_hot(contrastive_labels.flatten(), x.shape[0])
        loss = 0
        for blk in range(self.n_blks):
            if blk in self.blk_indices:
                features_ds = torch.cat([blocks_s_list[i][blk] for i in range(len(blocks_s_list))], 0)
                # do not use cls_tokens
                features_dt, features_ds = blocks_t[blk][:, 1:, :].mean(1), features_ds[:, 1:, :].mean(1)
                contrastive_features = torch.cat((features_dt, features_ds), 0)
                contrastive_features = net_list[0].contrastor(contrastive_features)
                # loss_cl = contrastive_loss.focal_SupConLoss_PrototypeWeight(
                #     contrastive_features, self.n_classes, contrastive_labels_1hot, mixup=False, prototype_weight=None)
                loss_cl = SupConLoss_normal(contrastive_features, contrastive_labels_1hot)
                loss += loss_cl
                self.loss_analysis.append(['BlockLevel/loss_cl', float(loss_cl)])
        # loss = loss / len(self.blk_indices)
        loss = loss
        return loss

    def update_prototypes(self, net_list, m=0.85):
        prototype_dt = net_list[0].classifier.net.weight  # [n_classes, feas_dim]
        for i in range(len(net_list[1:])):
            net_list[i+1].classifier.net.weight.data = m * net_list[i+1].classifier.net.weight + (1 - m) * prototype_dt

    def forward_GlobalLevel(self, tokens_fea, y_onehot, net_list, AffinityMatrix=None, mixup_p=True):
        device = tokens_fea.device
        # cls_token = tokens_fea[:, 0]
        cls_token = tokens_fea[:, 1:, :].mean(1)
        loss_cl = 0
        contrastive_features = cls_token
        contrastive_labels = y_onehot
        # the same category nodes
        diag_idx = np.diag_indices(self.n_classes)  # tuple (x:[0, 1,2,3,4,...], y:[0, 1,2,3,4,...])
        for i, net_ds in enumerate(net_list[1:]):
            # idx = [diag_idx[0], diag_idx[1] + self.n_classes * i]
            prototypes = net_ds.classifier.net.weight  # [n_classes, feas_dim]
            prototypes_y = torch.arange(prototypes.shape[0]).long().to(device)
            prototypes_y_onehot = to_one_hot(prototypes_y, self.n_classes)
            contrastive_features = torch.cat((contrastive_features, prototypes), 0)
            contrastive_labels = torch.cat((contrastive_labels, prototypes_y_onehot), 0)
        contrastive_features = net_list[0].contrastor(contrastive_features)
        # loss_cl = contrastive_loss.focal_SupConLoss_PrototypeWeight(
        #     contrastive_features, n_classes=self.n_classes, ont_hot_labels=contrastive_labels, mixup=False)
        loss_cl = SupConLoss_normal(contrastive_features, contrastive_labels)
        self.loss_analysis.append(['Loss_base/loss_cl', float(loss_cl)])
        return loss_cl

    def forward(self, x, net_list):
        loss_total, self.loss_analysis = 0, []
        loss_c, z_t, latent_tokens, tokens_fea, y_t_onehot, y_onehot_mix = self.forward_ConsensusKnowledge(x, net_list)
        loss_blk = self.forward_BlockLevel(x, net_list)  # local-level
        loss_global = self.forward_GlobalLevel(
            tokens_fea=tokens_fea, y_onehot=y_onehot_mix, net_list=net_list, AffinityMatrix=None, mixup_p=False)
        loss_total = self.loss_weights[0] * loss_blk + \
                     self.loss_weights[1]  * loss_global + self.loss_weights[2]  * loss_c
        # self.update_prototypes(net_list)
        # print(loss_blk, loss_aff, loss_c, loss_global)
        return loss_total, self.loss_analysis

class Classifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super(Classifier, self).__init__()
        self.net = weightNorm(nn.Linear(in_dim, n_classes, bias=False), name='weight')  # class prototype
        # self.net = nn.Linear(in_dim, n_classes, bias=False)
        # self.weight = self.net.weight  # do not use this! wrong!!! make device(gpu-to-cpu)

    def forward(self, x):
        return self.net(x)

class Contrastor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Contrastor, self).__init__()
        self.input = nn.Linear(in_dim, in_dim)
        self.output = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.output(x)
        return F.normalize(x)

class Affinity(nn.Module):
    # M = X * A * Y^T
    def __init__(self, in_dim, ):
        super(Affinity, self).__init__()
        self.in_dim = in_dim
        self.h_dim = in_dim * 2
        self.fc_M = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim), nn.ReLU(), nn.Linear(self.h_dim, 1))
        self.project_sr = nn.Linear(self.in_dim, self.in_dim, bias=False)
        self.project_tg = nn.Linear(self.in_dim, self.in_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for i in self.fc_M:
            if isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, std=0.01)
                nn.init.constant_(i.bias, 0)
        nn.init.normal_(self.project_sr.weight, std=0.01)
        nn.init.normal_(self.project_tg.weight, std=0.01)

    def forward(self, X, Y):
        X, Y = self.project_sr(X), self.project_tg(Y)
        N1, C = X.size()
        N2, C = Y.size()
        X_k = X.unsqueeze(1).expand(N1, N2, C)
        Y_k = Y.unsqueeze(0).expand(N1, N2, C)
        M = torch.cat([X_k, Y_k], dim=-1)
        M = self.fc_M(M).squeeze()
        return M

class ViT_CL_base(nn.Module):
    def __init__(self, n_classes, model_path=r'/home/ubuntu/sda_fast/Dataset/Pth/ViT/vit_small_distilled.pth', temp=0.05):
        super(ViT_CL_base, self).__init__()
        self.T = temp
        self.backbone = ViT_Cross()
        self.classifier = Classifier(self.backbone.embed_dim, n_classes)
        # Contrastive Learning
        self.contrastor = Contrastor(self.backbone.embed_dim, self.backbone.embed_dim)
        # Re-weight class semantic information
        self.Affinity = Affinity(self.backbone.embed_dim)
        self.InstNorm_layer = nn.InstanceNorm2d(1)
        self.parameter_list = [
            {"params": self.backbone.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {"params": self.classifier.parameters(), 'lr_mult': 10, 'decay_mult': 2},
            {"params": self.contrastor.parameters(), 'lr_mult': 10, 'decay_mult': 2},
            {"params": self.Affinity.parameters(), 'lr_mult': 10, 'decay_mult': 2},
            {"params": self.InstNorm_layer.parameters(), 'lr_mult': 1, 'decay_mult': 2},
        ]
        # if not use this, the prototype.device is cpu, why?
        # self.classifier = nn.Linear(self.backbone.embed_dim, n_classes, bias=False)
        if model_path is not None:
            self.backbone.load_param(model_path)

    def forward(self, x, return_latent_output=False):
        tokens, latent_tokens = self.backbone.get_latent_output(x)
        cls_token = tokens[:, 0]
        cls_token = F.normalize(cls_token, p=2.0, dim=1)
        outputs_c = self.classifier(cls_token) / self.T
        if self.training or return_latent_output:  #  
            return outputs_c, latent_tokens, tokens
        else:
            return outputs_c

    def predict(self, x):
        tokens, latent_tokens = self.backbone.get_latent_output(x)
        outputs_c = self.classifier(tokens[:, 0])
        return outputs_c


def inv_lr_scheduler(optimizer, iter_num, gamma=0.001, power=0.75, lr=1e-3, weight_decay=0.0005):
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i+=1
    return optimizer

def train_ds_normal(train_dloader, net, opt, loss_scaler):
    criterion_c = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
    # iter_d = Continuous_Dataloader(train_dloader)
    # for _ in range(opt.train_epochs):
    net.train()
    for le, (x, y, index) in enumerate(train_dloader):
        if le > opt.local_epochs:
            return
        if x.shape[0] == 1:
            continue
        x, y = x.cuda().float(), y.long().cuda()
        with amp.autocast(): # Automatic mixed precision
            z, _, _ = net(x)
            loss_c_ds = criterion_c(z, y)
        # loss_c_ds = losses.CrossEntropyLabelSmooth(opt.n_classes, 0.1)(z, y)
        optimizer.zero_grad()
        loss_scaler(loss_c_ds, optimizer, clip_grad=None, parameters=net.parameters())
        # break

def train_dt(epoch_out, train_loader, net_list, opt, optimizer, loss_scaler, writer=None, **kwargs):
    length = len(train_loader)
    criterion = MultiLevel_CL(opt.n_classes)
    for epoch_in, (x, _, __) in enumerate(train_loader):
        if epoch_in > opt.local_epochs:
            return
        net_list[0].train()
        x = x.cuda()
        with amp.autocast():
            loss_total, loss_analysis = criterion.forward(x, net_list)
        optimizer.zero_grad()
        loss_scaler(loss_total, optimizer, clip_grad=None, parameters=net_list[0].parameters())

        if writer:
            for loss_item in loss_analysis:
                writer.add_scalar(loss_item[0], loss_item[1], epoch_out*length + epoch_in)

def domain_adaptation(train_loaders, test_loaders, net_list, opt):
    best_result = -99
    loss_scaler = NativeScaler()
    optimizer = optim.SGD(net_list[0].parameter_list, lr=1e-3, momentum=0.9, weight_decay=1e-4, nesterov=True)
    for epoch in range(opt.train_epochs):
        # start = time.time()
        # ------------------------------Step 1: Train for Ds
        for f in range(opt.communication_cost):
            for train_dloader, net in zip(train_loaders[1:], net_list[1:]):
                train_ds_normal(train_dloader, net, opt, loss_scaler)
                # pass
        prototype_weight_list = None
        # ------------------------------Step 2: Train for Dt
        optimizer = inv_lr_scheduler(optimizer, epoch)
        train_dt(
            epoch, train_loaders[0], net_list, opt, optimizer=optimizer, loss_scaler=loss_scaler, writer=None,
            prototype_weight_list=prototype_weight_list)
        # ------------------------------Step 3: Federated Aggregation
        federated_aggregation_utils.federated_update_parameters_Blocks(
            net_list, block_index_list=None, domain_weight=None)

        if (epoch + 1) % 5 == 0 or epoch == (opt.train_epochs - 1):
            # start = time.time()
            result = eval_utils.evaluate(net_list[0], test_loaders[0])
            best_result = result if result > best_result else best_result

    return best_result
