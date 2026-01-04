import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from .attmil import AttentionGated
from .hagmil import IAMBlock
from .transmil import TransMIL
from .admil import CenterLoss
import time
from .HP_aggregator import AggregationBlock, AggregationBlockv2, AggregationBlockv3, AggregationBlockv4, \
    AB_MIL, DSMIL
       
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, add_cls_head=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        self.add_cls_head = add_cls_head
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        if add_cls_head:
            self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
            self.softmax = nn.Softmax(dim=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x_feature = self.forward_features(x)
        # print(x_feature.shape)
        if self.add_cls_head:
            x_logits = self.head(x_feature)
            x_softmax = self.softmax(x_logits)

            return x_feature, x_logits, x_softmax

        return x_feature
    

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


# @register_model
def pvt_v2_b0(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


# @register_model
def pvt_v2_b1(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


# @register_model
def pvt_v2_b2(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


# @register_model
def pvt_v2_b3(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


# @register_model
def pvt_v2_b4(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


# @register_model
def pvt_v2_b5(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


# @register_model
def pvt_v2_b2_li(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    model.default_cfg = _cfg()

    return model


class MILModel(nn.Module):
    def __init__(self, is_center_loss=False, is_img_center_loss=False, is_img_loss=False, feature_dim=512, hidden_dim=512, output_dim=2, num_layers=4, instance_batch_size=20, pvt_add_cls_head=False):
        super(MILModel, self).__init__()
        self.instance_batch_size = instance_batch_size
        self.pvt_add_cls_head = pvt_add_cls_head
        self.feature_extractor = pvt_v2_b2(num_classes=2, add_cls_head=pvt_add_cls_head)
        # self.aggregate = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        # self.aggregate = TransMIL(embed_dim=512,dropout=False,act='relu')
        self.aggregate = IAMBlock(in_dim=feature_dim, out_dim=hidden_dim, final_dim=hidden_dim, size=[feature_dim,512,256], dropout=0.25, gate=True) # 1xfinal_dim
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.is_center_loss = is_center_loss
        self.is_img_loss = is_img_loss
        self.is_img_center_loss = is_img_center_loss
        if is_img_loss:
            self.img_classifier = nn.Linear(hidden_dim, output_dim)
        if is_center_loss:
            self.center_loss = CenterLoss(num_classes=2, feat_dim=512)
        if is_img_center_loss:
            self.center_loss_img = CenterLoss(num_classes=2, feat_dim=512)
        self.img_criterion = nn.CrossEntropyLoss()

    def forward(self, bags, labels=None, labels_img=None, topk=7):
        # t1 = time.time()
        _, max_instances, _, _, _ = bags.size()
        bag_img_feature = []
        for instance in range(0, max_instances, self.instance_batch_size):
            # 分块处理并保留梯度
            split_images = bags[:, instance:instance+self.instance_batch_size].cuda().contiguous()
            
            # 前向传播获取特征（保留梯度）
            img_features = self.forward_features(split_images) 
            img_features = img_features.contiguous()
            
            del split_images 
            # 保留特征但管理内存
            bag_img_feature.append(img_features)
            # if instance % 2 == 0:  # 每处理2个实例块后执行清理
            #     torch.cuda.empty_cache()

        # 拼接特征时保持梯度
        bag_img_feature = torch.cat(bag_img_feature, dim=1).contiguous()
        # t2 = time.time()
        # print('2- encoder time: ', t2-t1)
        b,k,e = bag_img_feature.shape

        # shuffle and choose 20 instances
        # x_feature = x_feature[:, torch.randperm(x_feature.size(1)), :]
        # x_feature = x_feature[:, :self.instance_batch_size, :]
        
        ## aggregation
        # aggregation_feature = self.aggregate(x_feature)[0][:, -1, :]
        aggregation_feature, importance_scores = self.aggregate(bag_img_feature)

        ## classification
        out_aggregation = self.fc(aggregation_feature)
        topk_features = None
        out = {'bag_out': out_aggregation, 'importance_scores': importance_scores, 'aggregation_feature': aggregation_feature}

        if self.is_center_loss and labels is not None:
            out['center_loss'] = self.center_loss(F.normalize(aggregation_feature,dim=-1), labels)
        
        if self.is_img_loss or self.is_img_center_loss:
            topk_idxs = torch.topk(importance_scores, topk, dim=1)[1]
            topk_features = bag_img_feature[torch.arange(bag_img_feature.size(0)).unsqueeze(1), topk_idxs]
            topk_features = topk_features.view(-1, topk_features.size(2)).contiguous()
        if self.is_img_center_loss and labels_img is not None:
            out['center_loss_img'] = self.center_loss_img(F.normalize(topk_features,dim=-1), labels_img) 
        if self.is_img_loss and labels_img is not None:
            logits = self.img_classifier(topk_features)
            logits = F.softmax(logits, dim=1)
            # print(logits.shape, img_label.shape)
            out['img_loss'] = self.img_criterion(logits, labels_img)

        return out
        # return out_aggregation, importance_scores, aggregation_feature #,x_feature,x_logits,x_softmax        


    def forward_features(self, x):
        b,k,c,h,w = x.shape
        x = x.view(b*k, c, h, w).contiguous()
        if self.pvt_add_cls_head:
            x_features, x_logits, x_softmax = self.feature_extractor(x)
            x_logits = x_logits.view(b, k, -1).contiguous()
            x_softmax = x_softmax.view(b, k, -1).contiguous()
            x_features = x_features.view(b, k, -1).contiguous()
            return x_features, x_logits, x_softmax

        else:
            x_features = self.feature_extractor(x)
            x_features = x_features.view(b, k, -1).contiguous()
        
        return x_features
    
    # def forward_center_loss(self, x_feature, labels):
    #     return self.center_loss(x_feature, labels)

    # def forward_center_loss_img(self, x_feature, labels):
    #     return self.center_loss_img(x_feature, labels)

class MILModel_NeiborAgg(nn.Module):
    def __init__(self, is_center_loss=False, is_img_center_loss=False, is_img_loss=False, is_center_loss_prot=False, attention_constrain=False, feature_dim=512, hidden_dim=512, output_dim=2, num_layers=4, instance_batch_size=20, pvt_add_cls_head=False):
        super(MILModel_NeiborAgg, self).__init__()
        self.instance_batch_size = instance_batch_size
        self.pvt_add_cls_head = pvt_add_cls_head
        self.feature_extractor = pvt_v2_b2(num_classes=2, add_cls_head=pvt_add_cls_head)
        # self.aggregate = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        # self.aggregate = TransMIL(embed_dim=512,dropout=False,act='relu')
        self.aggregate = AggregationBlockv3(in_dim=feature_dim, mid_dim=hidden_dim, final_dim=hidden_dim, size=[feature_dim,512,256], dropout=0.25, gate=True, HPmlp=True, HPhead=True) # 1xfinal_dim
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.is_center_loss = is_center_loss
        self.is_img_loss = is_img_loss
        self.is_img_center_loss = is_img_center_loss
        if is_img_loss:
            self.img_classifier = nn.Linear(hidden_dim, output_dim)
        if is_center_loss:
            self.center_loss = CenterLoss(num_classes=2, feat_dim=512)
        if is_img_center_loss:
            self.center_loss_img = CenterLoss(num_classes=2, feat_dim=512)
        self.img_criterion = nn.CrossEntropyLoss()
        self.is_center_loss_prot = is_center_loss_prot
        self.attention_constrain = attention_constrain


    def forward(self, bags, labels=None, labels_img=None, topk=7, train=True):
        # t1 = time.time()
        _, max_instances, _, _, _ = bags.size()
        bag_img_feature = []
        for instance in range(0, max_instances, self.instance_batch_size):
            # 分块处理并保留梯度
            split_images = bags[:, instance:instance+self.instance_batch_size].cuda().contiguous()
            
            # 前向传播获取特征（保留梯度）
            img_features = self.forward_features(split_images) 
            img_features = img_features.contiguous()
            
            del split_images 
            # 保留特征但管理内存
            bag_img_feature.append(img_features)
            # if instance % 2 == 0:  # 每处理2个实例块后执行清理
            #     torch.cuda.empty_cache()

        # 拼接特征时保持梯度
        bag_img_feature = torch.cat(bag_img_feature, dim=1).contiguous()
        # t2 = time.time()
        # print('2- encoder time: ', t2-t1)
        b,k,e = bag_img_feature.shape

        # shuffle and choose 20 instances
        # x_feature = x_feature[:, torch.randperm(x_feature.size(1)), :]
        # x_feature = x_feature[:, :self.instance_batch_size, :]
        
        ## aggregation
        # aggregation_feature = self.aggregate(x_feature)[0][:, -1, :]
        aggregation_feature, importance_scores, importance_scores_raw, logits_pre, z_norm = self.aggregate(bag_img_feature)

        ## classification
        out_aggregation = self.fc(aggregation_feature)
        topk_features = None
        out = {'bag_out': out_aggregation, 'importance_scores': importance_scores, 'aggregation_feature': aggregation_feature}

        if train:
            if self.is_center_loss and labels is not None:
                out['center_loss'] = self.center_loss(F.normalize(aggregation_feature,dim=-1), labels)
            
            if self.is_img_loss or self.is_img_center_loss:
                topk_idxs = torch.topk(importance_scores, topk, dim=1)[1]
                topk_features = bag_img_feature[torch.arange(bag_img_feature.size(0)).unsqueeze(1), topk_idxs]
                topk_features = topk_features.view(-1, topk_features.size(2)).contiguous()
            if self.is_img_center_loss and labels_img is not None:
                out['center_loss_img'] = self.center_loss_img(F.normalize(topk_features,dim=-1), labels_img) 
            if self.is_img_loss and labels_img is not None:
                logits = self.img_classifier(topk_features)
                logits = F.softmax(logits, dim=1)
                # print(logits.shape, img_label.shape)
                out['img_loss'] = self.img_criterion(logits, labels_img)
            if self.is_center_loss_prot:
                # print('1', z_norm.shape) #[B,K,E]
                topk_idxs = torch.topk(importance_scores, topk, dim=1)[1]
                mean_features = z_norm[torch.arange(z_norm.size(0)).unsqueeze(1), topk_idxs] # [B,K,E]
                # print('2', mean_features.shape, labels.shape)
                mean_features = mean_features.mean(dim=1) #[1,e]\
                # print('3', mean_features.shape, labels.shape)
                out['center_loss_prot'] = self.center_loss_prot(mean_features, labels)
            if self.attention_constrain:
                # logits_pre_b = (logits_pre>0.7).long() # [B,K,C]
                # labels: [B]; importance_scores_raw: [B,K]
                loss_attention_constrain = self.attention_constrain_loss(logits_pre, labels, importance_scores_raw)
                out['loss_attention_constrain'] = loss_attention_constrain


        return out
        # return out_aggregation, importance_scores, aggregation_feature #,x_feature,x_logits,x_softmax        
    def attention_constrain_loss(self, logits_pre, labels, importance_scores_raw):
        # 提取分类结果
        logits_pre_neg_b = (torch.max(logits_pre[..., :2], dim=-1)[0] > 0.6).long().detach()  # [B, K]
        logits_pre_pos_b = (torch.max(logits_pre[..., 2:], dim=-1)[0] > 0.6).long().detach() # [B, K]

        # 根据 bag 标签生成条件掩码
        is_negative_bag = labels == 0  # [B]

        # 扩展维度以便广播
        is_negative_bag_expanded = is_negative_bag.unsqueeze(1)  # [B, 1]

        # 定义条件逻辑（通过广播）
        related_mask = (
            (~is_negative_bag_expanded) * logits_pre_pos_b + 
            is_negative_bag_expanded * logits_pre_neg_b
        ).bool()  # [B, K]

        unknown_mask = (
            (~is_negative_bag_expanded) * (logits_pre_neg_b.bool() & ~logits_pre_pos_b.bool()) +
            is_negative_bag_expanded * (logits_pre_pos_b.bool() & ~logits_pre_neg_b.bool())
        ).bool()

        unrelated_mask = (~logits_pre_neg_b.bool() & ~logits_pre_pos_b.bool())  # [B, K]

        if not related_mask.any():
            return torch.tensor(0.0)

        logits = importance_scores_raw[related_mask | unrelated_mask] # [L]
        logits = logits - logits.min()  # 避免负数（可选）
        logits_log_softmax = F.log_softmax(logits, dim=0)

        sum_related = related_mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B, 1]
        # 构造目标分布：related_mask → 1/sum_related，unrelated_mask → 0
        labels = (related_mask.float() / sum_related)[related_mask | unrelated_mask].float() # [L]

        # 计算 KL 散度损失
        loss = F.kl_div(logits_log_softmax, labels, reduction='batchmean')

        return loss
        


    def forward_features(self, x):
        b,k,c,h,w = x.shape
        x = x.view(b*k, c, h, w).contiguous()
        if self.pvt_add_cls_head:
            x_features, x_logits, x_softmax = self.feature_extractor(x)
            x_logits = x_logits.view(b, k, -1).contiguous()
            x_softmax = x_softmax.view(b, k, -1).contiguous()
            x_features = x_features.view(b, k, -1).contiguous()
            return x_features, x_logits, x_softmax

        else:
            x_features = self.feature_extractor(x)
            x_features = x_features.view(b, k, -1).contiguous()
        
        return x_features
    
    def set_bag_prototypes(self, bag_prototypes):
        self.bag_prototypes = bag_prototypes
        self.is_center_loss_prot = True
        self.center_loss_prot = CenterLoss(num_classes=bag_prototypes.shape[0], feat_dim=bag_prototypes.shape[1], prototypes=bag_prototypes)



class MILModel_NeiborAggv2(nn.Module):
    def __init__(self, is_center_loss=False, is_img_center_loss=False, is_img_loss=False, is_center_loss_prot=False, attention_constrain=False, feature_dim=512, hidden_dim=512, output_dim=2, num_layers=4, instance_batch_size=20, pvt_add_cls_head=False):
        super(MILModel_NeiborAggv2, self).__init__()
        self.instance_batch_size = instance_batch_size
        self.pvt_add_cls_head = pvt_add_cls_head
        self.feature_extractor = pvt_v2_b2(num_classes=2, add_cls_head=pvt_add_cls_head)
        # self.aggregate = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        # self.aggregate = TransMIL(embed_dim=512,dropout=False,act='relu')
        self.aggregate = AggregationBlockv2(in_dim=feature_dim, mid_dim=hidden_dim, final_dim=hidden_dim, size=[feature_dim,512,256], dropout=0.25, gate=True, HPmlp=True, HPhead=True) # 1xfinal_dim
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.is_center_loss = is_center_loss
        self.is_img_loss = is_img_loss
        self.is_img_center_loss = is_img_center_loss
        if is_img_loss:
            self.img_classifier = nn.Linear(hidden_dim, output_dim)
        if is_center_loss:
            self.center_loss = CenterLoss(num_classes=2, feat_dim=512)
        if is_img_center_loss:
            self.center_loss_img = CenterLoss(num_classes=2, feat_dim=512)
        self.img_criterion = nn.CrossEntropyLoss()
        self.is_center_loss_prot = is_center_loss_prot
        self.attention_constrain = attention_constrain


    def forward(self, bags, labels=None, labels_img=None, topk=7, train=True):
        # t1 = time.time()
        _, max_instances, _, _, _ = bags.size()
        bag_img_feature = []
        for instance in range(0, max_instances, self.instance_batch_size):
            # 分块处理并保留梯度
            split_images = bags[:, instance:instance+self.instance_batch_size].cuda().contiguous()
            
            # 前向传播获取特征（保留梯度）
            img_features = self.forward_features(split_images) 
            img_features = img_features.contiguous()
            
            del split_images 
            # 保留特征但管理内存
            bag_img_feature.append(img_features)
            # if instance % 2 == 0:  # 每处理2个实例块后执行清理
            #     torch.cuda.empty_cache()

        # 拼接特征时保持梯度
        bag_img_feature = torch.cat(bag_img_feature, dim=1).contiguous()
        # t2 = time.time()
        # print('2- encoder time: ', t2-t1)
        b,k,e = bag_img_feature.shape

        # shuffle and choose 20 instances
        # x_feature = x_feature[:, torch.randperm(x_feature.size(1)), :]
        # x_feature = x_feature[:, :self.instance_batch_size, :]
        
        ## aggregation
        # aggregation_feature = self.aggregate(x_feature)[0][:, -1, :]
        aggregation_feature, importance_scores, importance_scores_raw, logits_pre, z_norm = self.aggregate(bag_img_feature)

        ## classification
        out_aggregation = self.fc(aggregation_feature)
        topk_features = None
        out = {'bag_out': out_aggregation, 'importance_scores': importance_scores, 'aggregation_feature': aggregation_feature}

        if train:
            if self.is_center_loss and labels is not None:
                out['center_loss'] = self.center_loss(F.normalize(aggregation_feature,dim=-1), labels)
            
            if self.is_img_loss or self.is_img_center_loss:
                topk_idxs = torch.topk(importance_scores, topk, dim=1)[1]
                topk_features = bag_img_feature[torch.arange(bag_img_feature.size(0)).unsqueeze(1), topk_idxs]
                topk_features = topk_features.view(-1, topk_features.size(2)).contiguous()
            if self.is_img_center_loss and labels_img is not None:
                out['center_loss_img'] = self.center_loss_img(F.normalize(topk_features,dim=-1), labels_img) 
            if self.is_img_loss and labels_img is not None:
                logits = self.img_classifier(topk_features)
                logits = F.softmax(logits, dim=1)
                # print(logits.shape, img_label.shape)
                out['img_loss'] = self.img_criterion(logits, labels_img)
            if self.is_center_loss_prot:
                # print('1', z_norm.shape) #[B,K,E]
                topk_idxs = torch.topk(importance_scores, topk, dim=1)[1]
                mean_features = z_norm[torch.arange(z_norm.size(0)).unsqueeze(1), topk_idxs] # [B,K,E]
                # print('2', mean_features.shape, labels.shape)
                mean_features = mean_features.mean(dim=1) #[1,e]\
                # print('3', mean_features.shape, labels.shape)
                out['center_loss_prot'] = self.center_loss_prot(mean_features, labels)
            if self.attention_constrain:
                # logits_pre_b = (logits_pre>0.7).long() # [B,K,C]
                # labels: [B]; importance_scores_raw: [B,K]
                loss_attention_constrain = self.attention_constrain_loss(logits_pre, labels, importance_scores_raw)
                out['loss_attention_constrain'] = loss_attention_constrain


        return out
        # return out_aggregation, importance_scores, aggregation_feature #,x_feature,x_logits,x_softmax        
    def attention_constrain_loss(self, logits_pre, labels, importance_scores_raw):
        # 提取分类结果
        logits_pre_neg_b = (torch.max(logits_pre[..., :2], dim=-1)[0] > 0.6).long().detach()  # [B, K]
        logits_pre_pos_b = (torch.max(logits_pre[..., 2:], dim=-1)[0] > 0.6).long().detach() # [B, K]

        # 根据 bag 标签生成条件掩码
        is_negative_bag = labels == 0  # [B]

        # 扩展维度以便广播
        is_negative_bag_expanded = is_negative_bag.unsqueeze(1)  # [B, 1]

        # 定义条件逻辑（通过广播）
        related_mask = (
            (~is_negative_bag_expanded) * logits_pre_pos_b + 
            is_negative_bag_expanded * logits_pre_neg_b
        ).bool()  # [B, K]

        unknown_mask = (
            (~is_negative_bag_expanded) * (logits_pre_neg_b.bool() & ~logits_pre_pos_b.bool()) +
            is_negative_bag_expanded * (logits_pre_pos_b.bool() & ~logits_pre_neg_b.bool())
        ).bool()

        unrelated_mask = (~logits_pre_neg_b.bool() & ~logits_pre_pos_b.bool())  # [B, K]

        if not related_mask.any():
            return torch.tensor(0.0)

        logits = importance_scores_raw[related_mask | unrelated_mask] # [L]
        logits = logits - logits.min()  # 避免负数（可选）
        logits_log_softmax = F.log_softmax(logits, dim=0)

        sum_related = related_mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B, 1]
        # 构造目标分布：related_mask → 1/sum_related，unrelated_mask → 0
        labels = (related_mask.float() / sum_related)[related_mask | unrelated_mask].float() # [L]

        # 计算 KL 散度损失
        loss = F.kl_div(logits_log_softmax, labels, reduction='batchmean')

        return loss
        
    def forward_features(self, x):
        b,k,c,h,w = x.shape
        x = x.view(b*k, c, h, w).contiguous()
        if self.pvt_add_cls_head:
            x_features, x_logits, x_softmax = self.feature_extractor(x)
            x_logits = x_logits.view(b, k, -1).contiguous()
            x_softmax = x_softmax.view(b, k, -1).contiguous()
            x_features = x_features.view(b, k, -1).contiguous()
            return x_features, x_logits, x_softmax

        else:
            x_features = self.feature_extractor(x)
            x_features = x_features.view(b, k, -1).contiguous()
        
        return x_features
    
    def set_bag_prototypes(self, bag_prototypes):
        self.bag_prototypes = bag_prototypes
        self.is_center_loss_prot = True
        self.center_loss_prot = CenterLoss(num_classes=bag_prototypes.shape[0], feat_dim=bag_prototypes.shape[1], prototypes=bag_prototypes)

class MILModel_combine(nn.Module):
    def __init__(self, opt=None, is_center_loss=False, is_img_center_loss=False, is_img_loss=False, is_center_loss_prot=False, attention_constrain=False, \
        feature_dim=512, hidden_dim=512, output_dim=2, num_layers=4, instance_batch_size=20, \
        pvt_add_cls_head=False, aggregator='Mean',joint_learning=False, MaskDrop_threshold=0.7):
        super(MILModel_combine, self).__init__()
        self.instance_batch_size = instance_batch_size
        self.pvt_add_cls_head = pvt_add_cls_head
        self.attention_constrain = attention_constrain
        self.feature_extractor = pvt_v2_b2(num_classes=2, add_cls_head=pvt_add_cls_head)
        self.aggregator = aggregator
        if aggregator == 'AB':
            self.aggregate = AB_MIL(mid_dim=feature_dim, size=[feature_dim,512,256])
        elif aggregator == 'Trans':
            self.aggregate = TransMIL(embed_dim=feature_dim,dropout=0.2,act='relu')
        elif aggregator == 'IAM':
            self.aggregate = IAMBlock(in_dim=feature_dim, out_dim=feature_dim, final_dim=feature_dim, size=[feature_dim,512,256], dropout=0.25, gate=True) # 1xfinal_dim
        elif aggregator == 'LSTM':
            self.aggregate = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        elif aggregator == 'DSMIL':
            self.aggregate = DSMIL(mid_dim=feature_dim, cls_num=output_dim)
        elif aggregator == 'HPMIL':
            self.aggregate = AggregationBlockv2(in_dim=feature_dim, mid_dim=hidden_dim, final_dim=hidden_dim, \
                size=[feature_dim,512,256], dropout=0.25, gate=True, HPmlp=True, HPhead=True, MaskDrop_threshold=MaskDrop_threshold) # 1xfinal_dim

        self.joint_learning = joint_learning
        self.fc = nn.Linear(feature_dim, output_dim)
        self.is_center_loss = is_center_loss
        self.is_img_loss = is_img_loss
        self.is_img_center_loss = is_img_center_loss
        if is_img_loss:
            self.img_classifier = nn.Linear(hidden_dim, output_dim)
        if is_center_loss:
            self.center_loss = CenterLoss(num_classes=2, feat_dim=512)
        if is_img_center_loss:
            self.center_loss_img = CenterLoss(num_classes=2, feat_dim=512)
        if self.joint_learning:
            self.instance_classifier = nn.Linear(hidden_dim, 8)
            self.instance_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.img_criterion = nn.CrossEntropyLoss()

    def forward(self, bags, labels=None, labels_img=None, labels_8class=None, topk=7):
        # t1 = time.time()
        out = {}
        _, max_instances, _, _, _ = bags.size()
        bag_img_feature = []
        for instance in range(0, max_instances, self.instance_batch_size):
            # 分块处理并保留梯度
            split_images = bags[:, instance:instance+self.instance_batch_size].cuda().contiguous()
            
            # 前向传播获取特征（保留梯度）
            img_features = self.forward_features(split_images) 
            img_features = img_features.contiguous()
            
            del split_images 
            # 保留特征但管理内存
            bag_img_feature.append(img_features)
            # if instance % 2 == 0:  # 每处理2个实例块后执行清理
            #     torch.cuda.empty_cache()

        # 拼接特征时保持梯度
        bag_img_feature = torch.cat(bag_img_feature, dim=1).contiguous()
        # t2 = time.time()
        # print('2- encoder time: ', t2-t1)
        b,k,e = bag_img_feature.shape

        # shuffle and choose 20 instances
        # x_feature = x_feature[:, torch.randperm(x_feature.size(1)), :]
        # x_feature = x_feature[:, :self.instance_batch_size, :]
        
        ## aggregation
        # aggregation_feature = self.aggregate(x_feature)[0][:, -1, :]
        if self.aggregator == 'DSMIL':
            out_aggregation, aggregation_feature, instance_pred = self.aggregate(bag_img_feature)
        else:
            if self.aggregator == 'Mean':
                aggregation_feature = bag_img_feature.mean(dim=1) # [b,k,e] -> [b,e]
            elif self.aggregator == 'Max':
                aggregation_feature = bag_img_feature.max(dim=1)[0] # [b,k,e] -> [b,e]
            elif self.aggregator == 'Trans':
                aggregation_feature = self.aggregate(bag_img_feature)
            elif self.aggregator == 'IAM' or self.aggregator == 'AB':
                aggregation_feature, importance_scores = self.aggregate(bag_img_feature)
                out['importance_scores'] = importance_scores
            elif self.aggregator == 'LSTM':
                h_lstm, _ = self.aggregate(bag_img_feature)
                aggregation_feature = h_lstm[:, -1, :]
            elif self.aggregator == 'HPMIL':
                aggregation_feature, importance_scores, importance_scores_raw, logits_pre, z_norm = self.aggregate(bag_img_feature)
                out['importance_scores'] = importance_scores
                out['importance_scores_raw'] = importance_scores_raw
                out['phenotype_logits'] = logits_pre
                out['latent_features'] = z_norm
            ## classification
            out_aggregation = self.fc(aggregation_feature)
            topk_features = None
            # out = {'bag_out': out_aggregation, 'importance_scores': importance_scores, 'aggregation_feature': aggregation_feature}
        out['bag_out'] = out_aggregation
        out['aggregation_feature'] = aggregation_feature

        if self.is_center_loss and labels is not None:
            out['center_loss'] = self.center_loss(F.normalize(aggregation_feature,dim=-1), labels)
        if self.is_img_loss or self.is_img_center_loss and 'importance_scores' in out:
            topk_idxs = torch.topk(importance_scores, topk, dim=1)[1]
            topk_features = bag_img_feature[torch.arange(bag_img_feature.size(0)).unsqueeze(1), topk_idxs]
            topk_features = topk_features.view(-1, topk_features.size(2)).contiguous()
        if self.is_img_center_loss and labels_img is not None:
            out['center_loss_img'] = self.center_loss_img(F.normalize(topk_features,dim=-1), labels_img) 
        if self.is_img_loss and labels_img is not None:
            logits = self.img_classifier(topk_features)
            logits = F.softmax(logits, dim=1)
            # print(logits.shape, img_label.shape)
            out['img_loss'] = self.img_criterion(logits, labels_img)
        if self.attention_constrain and 'importance_scores_raw' in out and labels is not None:
            # logits_pre_b = (logits_pre>0.7).long() # [B,K,C]
            # labels: [B]; importance_scores_raw: [B,K]
            # print(logits_pre.shape, labels.shape, importance_scores_raw.shape)
            loss_attention_constrain = self.attention_constrain_loss(logits_pre, labels, importance_scores_raw)
            out['loss_attention_constrain'] = loss_attention_constrain
        if self.joint_learning and labels_8class is not None:
            labels_8class = labels_8class.view(-1, labels_8class.size(-1)) # [B*k, 8]
            instance_out = self.instance_classifier(bag_img_feature.view(-1, bag_img_feature.size(-1))) # [B*k, 8]
            mask_labeled = (labels_8class != -1).float()
            joint_loss = self.instance_criterion(instance_out, labels_8class.float()) 
            out['joint_learning_loss'] = (joint_loss * mask_labeled).sum(dim=1).mean()

        return out
        # return out_aggregation, importance_scores, aggregation_feature #,x_feature,x_logits,x_softmax        
    
    def attention_constrain_loss(self, logits_pre, labels, importance_scores_raw):
        # 提取分类结果
        logits_pre_neg_b = (torch.max(logits_pre[..., :2], dim=-1)[0] > 0.6).long().detach()  # [B, K]
        logits_pre_pos_b = (torch.max(logits_pre[..., 2:], dim=-1)[0] > 0.6).long().detach() # [B, K]

        # 根据 bag 标签生成条件掩码
        is_negative_bag = labels == 0  # [B]

        # 扩展维度以便广播
        is_negative_bag_expanded = is_negative_bag.unsqueeze(1)  # [B, 1]

        # 定义条件逻辑（通过广播）
        related_mask = (
            (~is_negative_bag_expanded) * logits_pre_pos_b + 
            is_negative_bag_expanded * logits_pre_neg_b
        ).bool()  # [B, K]

        unknown_mask = (
            (~is_negative_bag_expanded) * (logits_pre_neg_b.bool() & ~logits_pre_pos_b.bool()) +
            is_negative_bag_expanded * (logits_pre_pos_b.bool() & ~logits_pre_neg_b.bool())
        ).bool()

        unrelated_mask = (~logits_pre_neg_b.bool() & ~logits_pre_pos_b.bool())  # [B, K]

        if not related_mask.any():
            return torch.tensor(0.0)

        logits = importance_scores_raw[related_mask | unrelated_mask] # [L]
        logits = logits - logits.min()  # 避免负数（可选）
        logits_log_softmax = F.log_softmax(logits, dim=0)

        sum_related = related_mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B, 1]
        # 构造目标分布：related_mask → 1/sum_related，unrelated_mask → 0
        labels = (related_mask.float() / sum_related)[related_mask | unrelated_mask].float() # [L]

        # 计算 KL 散度损失
        loss = F.kl_div(logits_log_softmax, labels, reduction='batchmean')

        return loss

    def forward_features(self, x):
        b,k,c,h,w = x.shape
        x = x.view(b*k, c, h, w).contiguous()
        if self.pvt_add_cls_head:
            x_features, x_logits, x_softmax = self.feature_extractor(x)
            x_logits = x_logits.view(b, k, -1).contiguous()
            x_softmax = x_softmax.view(b, k, -1).contiguous()
            x_features = x_features.view(b, k, -1).contiguous()
            return x_features, x_logits, x_softmax

        else:
            x_features = self.feature_extractor(x)
            x_features = x_features.view(b, k, -1).contiguous()
        
        return x_features

if __name__ == '__main__':
    model = MILModel_combine(aggregator='Mean')
    inputs = torch.randn(1, 7, 3, 224, 224)
    out = model(inputs)
    print(out['bag_out'].shape, out['aggregation_feature'].shape)