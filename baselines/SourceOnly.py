import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
from baselines.ViT_network import ViT_Cross


class Classifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super(Classifier, self).__init__()
        self.net = weightNorm(nn.Linear(in_dim, n_classes, bias=False), name='weight')
        
    def forward(self, x):
        return  self.net(x)

class SourceOnly_net(nn.Module):
    def __init__(self, n_classes, model_path=r'/root/autodl-tmp/Dataset/Pth/ViT/vit_small_distilled.pth'):
        # '/home/ubuntu/sda_fast/Dataset/Pth/ViT/vit_small_distilled.pth'
        #/root/autodl-tmp/Dataset/Pth
        super(SourceOnly_net, self).__init__()
        self.backbone = ViT_Cross()
        self.classifier = Classifier(self.backbone.embed_dim, n_classes)
        # if not use this, the prototype.device is cpu, why?
        # self.classifier = nn.Linear(self.backbone.embed_dim, n_classes, bias=False)
        # AutoDL:/root/autodl-tmp/Dataset/Image_DA/Pth
        if model_path is not None:
            self.backbone.load_param(model_path)

    def forward(self, x, return_latent_output=False):
        tokens, latent_tokens = self.backbone.get_latent_output(x)
        outputs_c = self.classifier(tokens[:, 0])
        if self.training or return_latent_output:  #  
            return outputs_c, latent_tokens, tokens
        else:
            return outputs_c

    def predict(self, x):
        tokens, latent_tokens = self.backbone.get_latent_output(x)
        outputs_c = self.classifier(tokens[:, 0])
        return outputs_c

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs).cuda()
        #  
        targets = torch.zeros(log_probs.size()).cuda().scatter_(1, targets.unsqueeze(1), 1)
        targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss

# class Training():
#     use Pl
#     def __init__(self, n_classes):
#         self.n_classes = n_classes
#         self.criterion_c = nn.CrossEntropyLoss()
#
#     def forward_classifier(self, x, y):
#         # loss = self.criterion_c(x, y)
#         loss = F.cross_entropy(x, y)
#         return loss
#
#     def get_pseudo_labels(self, x, net_list):
#         # why? if put outside will raise error: no inf checks were recorede for this optimizer
#         o_c_list = torch.tensor([]).to(x.device)
#         for net_ds in net_list[1:]:
#             outputs_c, _, __ = net_ds(x, return_latent_output=True)
#             o_c_list = torch.cat((o_c_list, outputs_c.unsqueeze(0)), 0)
#         o_c_list = o_c_list.mean(0)
#         y = o_c_list.max(1)[1]
#         return y
#
#     def forward(self, x, net_list, **kwargs):
#         outputs_c, latent_tokens, tokens = net_list[0](x)
#         # get pseudo-labels
#         y = self.get_pseudo_labels(x, net_list)
#         # loss = self.forward_classifier(outputs_c, y)
#         loss = F.cross_entropy(outputs_c, y.detach())
#         return loss
#

class Training():
    # Do Not use PL, only source training and predict
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.criterion_c = nn.CrossEntropyLoss()

    def forward_classifier(self, x, y):
        # loss = self.criterion_c(x, y)
        loss = F.cross_entropy(x, y)
        return loss

    def get_pseudo_labels(self, x, net_list):
        # why? if put outside will raise error: no inf checks were recorede for this optimizer
        o_c_list = torch.tensor([]).to(x.device)
        for net_ds in net_list[1:]:
            outputs_c, _, __ = net_ds(x, return_latent_output=True)
            o_c_list = torch.cat((o_c_list, outputs_c.unsqueeze(0)), 0)
        o_c_list = o_c_list.mean(0)
        y = o_c_list.max(1)[1]
        return y

    def forward(self, x, net_list, **kwargs):
        outputs_c, latent_tokens, tokens = net_list[0](x)
        # get pseudo-labels
        y = self.get_pseudo_labels(x, net_list)
        # loss = self.forward_classifier(outputs_c, y)
        loss = F.cross_entropy(outputs_c, y.detach())
        return loss