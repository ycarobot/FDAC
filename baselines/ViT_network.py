
import copy
import math
from itertools import repeat
import torch
from torch import nn
import torch.nn.functional as F
# from torch._six import container_abcs
import collections.abc as container_abcs
from functools import partial

# From PyTorch internals
# Generate the iterators
def _ntuple(n):  #  
    # to_2tuple = _ntuple(2) print(to_2tuple(4))
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)


# --------- 
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
     patch_size: 3-channel 
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        # print('using stride: {}, and part number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y  # Muliplication 
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # inchans:3,  (out_channels)embed_dim=384 or 368，  kernel_size and stride = [16, 16]
        # out = [(in + 2*padding - (16-1) - 1)] / 16 + 1
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = x.flatten(2).transpose(1, 2)
        return x

class HybridEmbed(nn.Module):  # Use different kinds of features
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    # Simply Attention 
    #  
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn = None
    def forward(self, x, return_ds_info=None):
        B, N, C = x.shape
        # reshape. Compute the attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # q, k shape: [B, num_heads, N, feas_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]   
        # softmax  
        attn = attn.softmax(dim=-1)
        self.attn = attn  # [bs, n_heads, n_tokens, n_tokens]
        attn = self.attn_drop(attn)
        # [B, num_heads, N, N] * [B, num_heads, N, feas_dim] = [B, num_heads, N, N] * [B, num_heads, N, feas_dim]
        # transpose: [B, N, num_heads, feas_dim]  reshape  C = num_heads * feas_dim
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Matrix Multiplication
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_ds_info == 'attn':
            current_info = attn.detach()  #  
            return x, current_info
        elif return_ds_info == 'query':
            current_info = q.detach()
            return x, current_info
        else:
            # print(attn[0, 0, :, :].sum(-1))
            return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_ds_info=False):
        if return_ds_info:
            x, ds_info = self.attn(self.norm1(x), return_ds_info=return_ds_info)
            x = x + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, ds_info
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def resize_pos_embed(posem_ori, posemb_new, hight, width):
    # Only change the size of patch
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posem_ori[:, :1], posem_ori[0, 1:]  # [1, 1, 384], [196, 384]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posem_ori[:, :0], posem_ori[0]
    gs_old = int(math.sqrt(len(posemb_grid)))  # n_patches

    # print('Position embedding resize to height:{} width: {}'.format(hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    # 参考 https://blog.csdn.net/qq_50001789/article/details/120297401
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear', align_corners=True)
    # posemb_grid = F.interpolate(posemb_grid, size=(width, hight), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posem_ori = torch.cat([posemb_tok, posemb_grid], dim=1)  # tok is cls_token
    return posem_ori

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight_bias, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight_bias, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight_bias, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight_bias, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

# network
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class ViT_base(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=384,
                 depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), distilled=None):
        super().__init__()
        self.distilled = distilled
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other_baselines models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        # This architecture will make information transfer between patch tokens and class tokens
        self.dist_token = 0
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # Classifier head
        # self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)  # 0.01 better
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        # # random offset setting
        # self.sr_alpha = 1
        # self.sr_layers = list(range(depth)) #  
        # self.sr_alpha_adap = 0.2 # 0.3

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # 0.01 bette # 0.01 betterr
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']

        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']

        for k, v in param_dict.items():
            if 'head' in k:
                continue
            if 'dist' in k and not self.distilled:
                continue

            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight_bias.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using Model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
                # self.state_dict()[k].copy_(revise)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(
                    k, v.shape, self.state_dict()[k].shape))

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  #  
        # patch后 [bs, , dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # [bs, num_x * num_y + 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)  # drop out
        for blk in self.blocks:  # Encoder computation
            x = blk(x)
        x = self.norm(x)  # [bs, 197, 384 or 786]
        return x[:, 0]  # [bs, 684 or 786]  class_token

# -----Cross Domain Adaptation
class ViT_Cross(ViT_base):
    def __init__(self):
        super(ViT_Cross, self).__init__()

    def get_latent_output(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        # distilled_token 
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        latent_output = []
        for layer, blk in enumerate(self.blocks):
            x = blk(x)
            # token = x[:, 1:].mean(1)
            token = x  # just for experiment, save all tokens
            latent_output.append(token)

        x = self.norm(x)
        return x, latent_output
