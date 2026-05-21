import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .s2block import S2Block
# from .bayes_loss import *
import numpy as np
import settings
#from hamburger import ConvBNReLU, get_hamburger
from hamburger import get_hamburger

L1Loss = torch.nn.L1Loss()
def L1_Loss(y1, y2):
    # dis = torch.abs(y1 - y2)
    dis = L1Loss(y1, y2)
    # dis = dis * Weight
    # return torch.mean(dis)
    return dis

def L_AU_loss(y1, y2, AU):
    dis = torch.pow(y1 - y2, 2)
    l = 0.5 * torch.exp(-1.0 * AU) * dis + 0.5 * AU
    # AU =AU + 1e-8
    # AU = 2 * torch.log(AU)
    # l = torch.exp(-1.0 * AU) * dis + 2 * AU
    return torch.mean(l)

'''------------------------------------------------
conv1x1函数： 定义1×1卷积层
输入：输入通道数、输出通道数
输出：设置好的卷积层
------------------------------------------------'''
def conv1x1(in_channels, out_channels, stride=1):
    #return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

'''------------------------------------------------
conv3x3函数： 定义3×3卷积层
输入：输入通道数、输出通道数
输出：设置好的卷积层
------------------------------------------------'''
def conv3x3(in_channels, out_channels, stride=1):
    #return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

'''------------------------------------------------
ResBlock函数： 定义残差块网络
输入：输入通道数、输出通道数
输出：两层3×3卷积、ReLU激活的残差块网络
------------------------------------------------'''
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

'''========================================================================
Feature_extraction模块： 浅特征提取，卷积+残差+卷积 用于通道调整
输入：原始尺寸的HSI/pan
输出：256通道
========================================================================'''
class Feature_extraction(nn.Module):
    def __init__(self, in_feats, num_res_blocks, n_feats, res_scale):
        super(Feature_extraction, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(in_feats, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale))
        self.conv_tail = conv3x3(n_feats, n_feats)
        
        # 批归一化和层归一化
        self.OutBN = nn.BatchNorm2d(num_features=n_feats)  
        
    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        #x = x + x1
        
        #x = self.OutBN(x)
        #x = F.relu(x)
        return x

'''========================================================================
Feature_adjustment模块：特征调整模块，卷积+残差+卷积
输入：原始尺寸的HSI
输出：HSI特征图
========================================================================'''
class Feature_adjustment(nn.Module):
    def __init__(self, in_feats, num_res_blocks, n_feats, res_scale):
        super(Feature_adjustment, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(in_feats, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, 
                res_scale=res_scale))
        self.conv_tail = conv3x3(n_feats, n_feats)
        
    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        #x = F.relu(self.conv_tail(x))
        x = self.conv_tail(x)
        #x = x + x1
        return x

'''===========================================================================
晶格模块中PAN的空间注意力模块(a part of CBAM)：
输入：通道调整后的PAN
输出：空间增强后的权重A
===========================================================================''' 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)      #self.sigmoid(x) * x

'''===========================================================================
晶格模块中HSI的通道注意力模块(a part of CBAM)：
输入：通道调整后的HSI
输出：通道增强后权重B
===========================================================================''' 
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享权重的MLP
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)        #self.sigmoid(out) * x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, 2, 2, 0),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                                 nn.LeakyReLU())

    def forward(self, x1):
        return self.up1(x1)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                                  nn.LeakyReLU())

    def forward(self, x):
        return self.conv(x)

class Conv_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, inplace=False):
        super(Conv_BN_ReLU, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

class BasicConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, bn=True):
        super(BasicConv2d, self).__init__()
        if bn:
            self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))


# channels：输入特征图的通道数。T：用于估计 Epistemic 不确定性的采样次数。q：用于随机掩码的丢弃概率。
class UncertaintyEstimation(nn.Module):
    def __init__(self, channels, GT_channel, T, q):
        super(UncertaintyEstimation, self).__init__()
        self.T = T
        self.q = q
        self.conv = nn.Sequential(nn.Conv2d(channels, GT_channel, 3, 1, 1, bias=True))
        self.out = nn.Sequential(nn.Conv2d(GT_channel, GT_channel, 1, 1, 0), nn.Tanh())
        self.aue = nn.Sequential(nn.Conv2d(GT_channel*(self.T+1), GT_channel, 3, 1, 1), nn.Sigmoid())

    def channels_random_mask(self, x, q):
        mask = np.random.binomial(n=1, p=1 - q, size=x.shape[1])  # 生成一个二项分布的随机掩码，丢弃概率为 q。
        mask = torch.tensor(mask).cuda()  # 将掩码转换为 PyTorch 张量并移动到 GPU。
        mask = rearrange(mask, "C -> 1 C 1 1")  # 调整掩码的形状，使其与输入特征图的形状匹配。
        return x * mask

    def spatial_random_mask(self, x, q):
        mask = np.random.binomial(n=1, p=1 - q, size=(x.shape[2], x.shape[3]))  # 生成一个二项分布的随机掩码，丢弃概率为 q。
        mask = torch.tensor(mask).cuda()  # 将掩码转换为 PyTorch 张量并移动到 GPU。
        mask = rearrange(mask, "H W -> 1 1 H W")  # 调整掩码的形状，使其与输入特征图的形状匹配。
        return x * mask

    def EU_AU(self, x):
        mean = 0  # 初始化均值为 0。
        xs_e = []
        xs_a = []
        # 进行 T 次采样。
        for i in range(self.T):
            x_cur = self.channels_random_mask(x, self.q)     # channels mask 应用随机掩码，并通过 out 层生成当前采样结果，
            x_cur = self.out(self.spatial_random_mask(x_cur, self.q))  # spatial mask
            xs_a.append(x_cur)
            x_cur = rearrange(x_cur, "B C H W -> 1 B C H W")  # 调整结果的形状。
            xs_e.append(x_cur)  # 将当前采样结果添加到列表中。

        xs_eu = torch.cat(xs_e, dim=0)  # 将所有采样结果沿第 0 维拼接。
        # print("xs.shape:------------>", xs.shape)
        EU, mean = torch.var_mean(input=xs_eu, dim=0, unbiased=True)  # 计算所有采样结果的方差和均值，方差作为 Epistemic 不确定性。
        # print("EU.shape:------------>", EU.shape)
        xs_au = torch.cat(xs_a, dim=1)
        # print("xs_au.shape:------------>", xs_au.shape)
        x_au = torch.cat((xs_au, x), dim=1)
        AU = self.aue(x_au)

        return AU, EU, mean

    def forward(self, x, lms):
        x = self.conv(x)
        AU, EU, mean = self.EU_AU(x)
        return AU, EU, mean

class EU_Confidence(nn.Module):
    def __init__(self, channels, out_channels, ratio, kernel_size):
        super(EU_Confidence, self).__init__()
        self.channels = channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_att = ChannelAttention(self.channels, self.ratio)
        self.spatial_att = SpatialAttention(self.kernel_size)
        self.conv_cha = conv1x1(channels, out_channels)
        self.conv_spa = conv1x1(channels, out_channels)
        self.conv = conv1x1(out_channels, out_channels)

    def channel_confidence(self, U):
        exp_U = torch.exp(U)
        sum_exp_U = torch.sum(exp_U, dim=1, keepdim=True)  # shape [B, 1, H, W]
        normalized_exp_U = exp_U / sum_exp_U  # shape [B, C, H, W]
        channel_con = 1 - normalized_exp_U  # shape [B, C, H, W]
        channel_con = self.conv_cha(channel_con)
        # print("channel_con.shape:------>", channel_con.shape)
        return channel_con

    def spatial_confidence(self, U):
        exp_U = torch.exp(U)
        sum_exp_U = torch.sum(exp_U, dim=[2, 3], keepdim=True)  # shape [B, C, 1, 1]
        normalized_exp_U = exp_U / sum_exp_U  # shape [B, C, H, W]
        spatial_con = 1 - normalized_exp_U  # shape [B, C, H, W]
        spatial_con = self.conv_spa(spatial_con)
        return spatial_con

    def forward(self, x, y):
        cha_con = self.channel_confidence(x) * y
        spa_con = self.spatial_confidence(x) * y
        out_con = self.conv(cha_con + spa_con)
        return out_con



class DSRU(nn.Module):
    def __init__(self, in_channels, out_channels, group_channels, MD_S, MD_D, MD_R):
        super(DSRU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_channels = group_channels
        self.MD_S = MD_S
        self.MD_D = MD_D
        self.MD_R = MD_R
        self.hs_conv = BasicConv2d(self.in_channels, self.out_channels * 4)    # 通道数增加
        self.pan_conv = BasicConv2d(self.in_channels, self.out_channels * 4)    # 通道数增加

        self.hs_convbnrelu = Conv_BN_ReLU(self.in_channels, self.in_channels)  # 用在forward()开始。
        self.pan_convbnrelu = Conv_BN_ReLU(self.in_channels, self.in_channels)  # 用在forward()开始。
        # self.S2B_c = S2Block(dim=16, heads=1, dim_head=16//(4*self.num_head), mlp_dim=int(16*0.5))

        Hamburger = get_hamburger(settings.VERSION)
        self.hamburger = Hamburger(self.group_channels, settings, self.MD_S, self.MD_D, self.MD_R)
        # nn.LeakyReLU()
        self.c_t_channel_transform = nn.Sequential(nn.Conv2d(group_channels*2, group_channels*2, 3, 1, 1),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(group_channels*2, group_channels, 1, 1, 0),
                                     nn.LeakyReLU())

    # 计算 SRU 单元的输出。y_h_t:历史信息;  C_t:状态;  y_c_t: ; F_t:遗忘门; R_t:重置门.
    def SRUconvLayer(self, y_h_t_hs, C_t_hs, y_c_t_hs, F_t_hs, R_t_hs, y_h_t_pan, C_t_pan, y_c_t_pan, F_t_pan, R_t_pan):
        if C_t_hs is None:
            C_t_hs = 1 - F_t_hs
            C_t_pan = 1 - F_t_pan
            C_t_ = torch.cat((C_t_hs, C_t_pan), dim=1)  # [B, 2C, H, W]
            C_t_ = self.c_t_channel_transform(C_t_)
            F_MD = self.hamburger(C_t_)
            C_t_hs, C_t_pan = F_MD["x_01"], F_MD["x_01"]
        else:
            C_t_hs = F_t_hs * C_t_hs + (1 - F_t_hs) * y_c_t_hs
            C_t_pan = F_t_pan * C_t_pan + (1 - F_t_pan) * y_c_t_pan

            C_t_ = torch.cat((C_t_hs, C_t_pan), dim=1)  # [B, 2C, H, W]
            C_t_ = self.c_t_channel_transform(C_t_)
            F_MD = self.hamburger(C_t_)
            C_t_hs, C_t_pan = F_MD["x_01"], F_MD["x_01"]

        h_t_hs = R_t_hs * C_t_hs + (1 - R_t_hs) * y_h_t_hs
        h_t_pan = R_t_pan * C_t_pan + (1 - R_t_pan) * y_h_t_pan
        return C_t_hs, h_t_hs, C_t_pan, h_t_pan

    # 计算门控函数的结果，并对其中的部分进行激活函数处理。这是初始化操作，将需要的门控内容用 X 初始化出来
    def SRUconvGates(self, hsi, pan):
        gates_hs = self.hs_conv(hsi)    # 扩充通道数，使其通道数扩充为原来的四倍 Conv + BN
        gates_pan = self.pan_conv(pan)

        # gates按照第一个维度（即通道数）进行切分，得到了四个子张量：Wx、ft、rt和X。这些子张量分别表示权重、遗忘门、重置门和输出。
        y_c_t_hs, F_t_hs, R_t_hs, y_h_t_hs = gates_hs.split(split_size=self.out_channels, dim=1)
        y_c_t_pan, F_t_pan, R_t_pan, y_h_t_pan = gates_pan.split(split_size=self.out_channels, dim=1)

        y_c_t_hs = y_c_t_hs.tanh()
        F_t_hs = F_t_hs.sigmoid()
        R_t_hs = R_t_hs.sigmoid()
        y_h_t_hs = y_h_t_hs.tanh()

        y_c_t_pan = y_c_t_pan.tanh()
        F_t_pan = F_t_pan.sigmoid()
        R_t_pan = R_t_pan.sigmoid()
        y_h_t_pan = y_h_t_pan.tanh()

        return y_c_t_hs, F_t_hs, R_t_hs, y_h_t_hs, y_c_t_pan, F_t_pan, R_t_pan, y_h_t_pan

    # x:hsi, y:pan
    def forward(self, x, y, reverse=False):
        x = self.hs_convbnrelu(x)  # conv + BN + ReLU
        y = self.pan_convbnrelu(y)  # conv + BN + ReLU
        # y_c_t_hs, F_t_hs, R_t_hs, y_h_t_hs,y_c_t_pan, F_t_pan, R_t_pan, y_h_t_pan = self.SRUconvGates(x, y)
        y_c_t_hs, F_t_hs, R_t_hs, y_h_t_hs = x, x, x, x
        y_c_t_pan, F_t_pan, R_t_pan, y_h_t_pan = y, y, y, y

        # 按照第二个维度进行切分
        y_c_ts_hs = [t.tanh() for t in y_c_t_hs.split(self.group_channels, dim=1)]
        F_ts_hs = [t.sigmoid() for t in F_t_hs.split(self.group_channels, dim=1)]
        R_ts_hs = [t.sigmoid() for t in R_t_hs.split(self.group_channels, dim=1)]
        y_h_ts_hs = [t.tanh() for t in y_h_t_hs.split(self.group_channels, dim=1)]

        #pan
        y_c_ts_pan = [t.tanh() for t in y_c_t_pan.split(self.group_channels, dim=1)]
        F_ts_pan = [t.sigmoid() for t in F_t_pan.split(self.group_channels, dim=1)]
        R_ts_pan = [t.sigmoid() for t in R_t_pan.split(self.group_channels, dim=1)]
        y_h_ts_pan = [t.tanh() for t in y_h_t_pan.split(self.group_channels, dim=1)]

        C_t_hs = torch.zeros(y_c_ts_hs[0].size(0), y_c_ts_hs[0].size(1), y_c_ts_hs[0].size(2), y_c_ts_hs[0].size(3)).cuda()
        C_t_pan = torch.zeros(y_c_ts_pan[0].size(0), y_c_ts_pan[0].size(1), y_c_ts_pan[0].size(2), y_c_ts_pan[0].size(3)).cuda()
        h_t_hs = None
        h_t_pan = None
        htl_hs = []
        htl_pan = []
        # 遍历xs、Wxs、fts和rts，并调用名为SRUconvLayer的函数进行处理，得到新的Ct和ht。
        for time, (y_h_t_hs_, y_c_t_hs_, F_t_hs_, R_t_hs_, y_h_t_pan_, y_c_t_pan_, F_t_pan_, R_t_pan_) in enumerate(zip(y_h_ts_hs, y_c_ts_hs, F_ts_hs, R_ts_hs, y_h_ts_pan, y_c_ts_pan, F_ts_pan, R_ts_pan)):
            C_t_hs, h_t_hs, C_t_pan, h_t_pan = self.SRUconvLayer(y_h_t_hs_, C_t_hs.data, y_c_t_hs_, F_t_hs_, R_t_hs_, y_h_t_pan_, C_t_pan.data, y_c_t_pan_, F_t_pan_, R_t_pan_)
            htl_hs.append(h_t_hs)
            htl_pan.append(h_t_pan)
        htl_hs = torch.cat(htl_hs, dim=1)
        htl_pan = torch.cat(htl_pan, dim=1)


        C_t_hs = torch.zeros(y_c_ts_hs[0].size(0), y_c_ts_hs[0].size(1), y_c_ts_hs[0].size(2), y_c_ts_hs[0].size(3)).cuda()
        C_t_pan = torch.zeros(y_c_ts_pan[0].size(0), y_c_ts_pan[0].size(1), y_c_ts_pan[0].size(2), y_c_ts_pan[0].size(3)).cuda()
        h_t_hs = None
        h_t_pan = None
        htr_hs = []
        htr_pan = []
        # 遍历xs、Wxs、fts和rts，并调用名为SRUconvLayer的函数进行处理，得到新的Ct和ht。
        for time, (y_h_t_hs_, y_c_t_hs_, F_t_hs_, R_t_hs_, y_h_t_pan_, y_c_t_pan_, F_t_pan_, R_t_pan_) in enumerate(zip(reversed(y_h_ts_hs), reversed(y_c_ts_hs), reversed(F_ts_hs), reversed(R_ts_hs), reversed(y_h_ts_pan), reversed(y_c_ts_pan), reversed(F_ts_pan), reversed(R_ts_pan))):
            C_t_hs, h_t_hs, C_t_pan, h_t_pan = self.SRUconvLayer(y_h_t_hs_, C_t_hs.data, y_c_t_hs_, F_t_hs_, R_t_hs_, y_h_t_pan_, C_t_pan.data, y_c_t_pan_, F_t_pan_, R_t_pan_)
            htr_hs.insert(0, h_t_hs)
            htr_pan.insert(0, h_t_pan)
        htr_hs = torch.cat(htr_hs, dim=1)
        htr_pan = torch.cat(htr_pan, dim=1)

        ht_hs = htl_hs + htr_hs
        ht_pan = htl_pan + htr_pan

        return {"ht_hs": ht_hs, "ht_pan": ht_pan}
        # return {"ht_hs":htl_hs, "ht_pan":htl_pan}

'''================================================
HFRUNetpre 模块：总启动程序 
================================================'''
class HFRUNetpre(nn.Module):
    def __init__(self, config):
        super(HFRUNetpre, self).__init__()
        self.num_head = config["N_modules"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]  # 光谱通道数
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]  # 尺寸缩放比例

        self.unet_res_blocks = [1, 1, 1, 1, 1]
        self.res_scale = 1
        self.feature_chanels = [128, 256, 512, 256, 128]
        # self.feature_chanels = [64, 128, 256, 128, 64]

        ''' LR-HSI和PAN的特征通道调整(Feature_extraction模块) '''
        self.hs_compress = conv3x3(self.in_channels, self.feature_chanels[0])
        self.pan_compress = conv3x3(1, self.feature_chanels[0])

        ''' ======================== 降采样操作 通道数保持不变 ========================== '''
        self.hs_Down00 = Down(self.feature_chanels[0], self.feature_chanels[1])
        self.hs_Down01 = Down(self.feature_chanels[1], self.feature_chanels[2])

        ''' ======================== 上采样操作 通道数保持不变 ========================== '''
        self.hs_Up01 = Up(self.feature_chanels[2], self.feature_chanels[1])
        self.hs_Up00 = Up(self.feature_chanels[1], self.feature_chanels[0])

        ''' ====================== 残差块  ============================ '''
        # 00 [B, C, H, W]
        self.hs_res_block00 = nn.ModuleList()
        for i in range(self.unet_res_blocks[0]):
            self.hs_res_block00.append(
                ResBlock(in_channels=self.feature_chanels[0], out_channels=self.feature_chanels[0], res_scale=1))
        self.hs_res_block00_conv = conv3x3(self.feature_chanels[0], self.feature_chanels[0])

        # 01 [B, C, H/2, W/2]
        self.hs_res_block01 = nn.ModuleList()
        for i in range(self.unet_res_blocks[1]):
            self.hs_res_block01.append(
                ResBlock(in_channels=self.feature_chanels[1], out_channels=self.feature_chanels[1], res_scale=1))
        self.hs_res_block01_conv = conv3x3(self.feature_chanels[1], self.feature_chanels[1])

        # 02 [B, C, H/4, W/4]
        self.hs_res_block02 = nn.ModuleList()
        for i in range(self.unet_res_blocks[2]):
            self.hs_res_block02.append(
                ResBlock(in_channels=self.feature_chanels[2], out_channels=self.feature_chanels[2], res_scale=1))
        self.hs_res_block02_conv = conv3x3(self.feature_chanels[2], self.feature_chanels[2])

        # 03 [B, C, H/2, W/2]
        self.hs_res_block03 = nn.ModuleList()
        for i in range(self.unet_res_blocks[3]):
            self.hs_res_block03.append(
                ResBlock(in_channels=self.feature_chanels[3], out_channels=self.feature_chanels[3], res_scale=1))
        self.hs_res_block03_conv = conv3x3(self.feature_chanels[3], self.feature_chanels[3])

        # 04 [B, C, H, W]
        self.hs_res_block04 = nn.ModuleList()
        for i in range(self.unet_res_blocks[4]):
            self.hs_res_block04.append(
                ResBlock(in_channels=self.feature_chanels[3], out_channels=self.feature_chanels[3], res_scale=1))
        self.hs_res_block04_conv = conv3x3(self.feature_chanels[3], self.feature_chanels[3])

        # 双向循环网络
        self.RNN_encoder00 = DSRU(in_channels=self.feature_chanels[0], out_channels=self.feature_chanels[0], group_channels =8, MD_S=1, MD_D=8, MD_R=2)
        self.RNN_encoder01 = DSRU(in_channels=self.feature_chanels[1], out_channels=self.feature_chanels[1], group_channels =16, MD_S=1, MD_D=16, MD_R=4 )
        self.RNN_encoder02 = DSRU(in_channels=self.feature_chanels[2], out_channels=self.feature_chanels[2], group_channels =32, MD_S=1, MD_D=32, MD_R=8 )

        self.RNN_decoder01 = DSRU(in_channels=self.feature_chanels[3], out_channels=self.feature_chanels[3], group_channels =16, MD_S=1, MD_D=16, MD_R=4 )
        self.RNN_decoder00 = DSRU(in_channels=self.feature_chanels[4], out_channels=self.feature_chanels[4], group_channels =8, MD_S=1, MD_D=8, MD_R=2 )

        ''' 最终融合结果特征调整(Feature_adjustment模块) '''
        # self.feature_adjustment = Feature_adjustment(self.feature_chanels[0], self.num_res_blocks[0], self.out_channels, self.res_scale)
        self.compress = nn.Conv2d(self.feature_chanels[3], self.out_channels, kernel_size=1, padding=0, bias=True)
        # AU
        # no use Linear,  spa:[B,C,H,W], SPE:[B,C,1,1]
        self.spa_var = nn.Sequential(nn.Conv2d(self.feature_chanels[3], self.feature_chanels[4], kernel_size=3, padding=1, bias=True),
                                      nn.ELU(),
                                      nn.Conv2d(self.feature_chanels[4], self.out_channels, kernel_size=3, padding=1, bias=True),
                                      nn.ELU())
        self.spe_var = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.feature_chanels[3], self.feature_chanels[4], kernel_size=1, padding=0, bias=True),
                                      nn.ELU(),
                                      nn.Conv2d(self.feature_chanels[4], self.out_channels, kernel_size=1, padding=0, bias=True),
                                      nn.ELU())

    def forward(self, LR_HSI, PAN, GT_D0):
        # 调整PAN尺寸为[batch_size, 1，HR_size, HR_size]
        PAN = PAN.unsqueeze(dim=1)
        UP_LR_HS = F.interpolate(LR_HSI, scale_factor=(self.factor, self.factor), mode ='bicubic')   # 上采样的LR-HSI，用于晶格结构

        ''' 初始特征通道调整 '''
        # F_pan_00 = self.pan_feature_extraction(PAN)  # # PAN调整通道
        F_pan_00 = self.pan_compress(PAN)
        # F_up_lrhs_00 = self.up_lrhs_feature_extraction(UP_LR_HS)  # HSI调整通道
        F_up_lrhs_00 = self.hs_compress(UP_LR_HS)

        ''' -------------------------- 编码第一尺度 ------------------------- '''
        # 融合 ----> RBs ----> 留下跳转连接变量 ----> 下采样 ----> （下一尺度融合）
        F_MD_00 = self.RNN_encoder00(F_up_lrhs_00, F_pan_00)
        F_encoder_00 = F_up_lrhs_00 + F_MD_00["ht_pan"]

        # RBs
        for i in range(self.unet_res_blocks[0]):
            F_encoder_00 = self.hs_res_block00[i](F_encoder_00)
            F_pan_00 = self.hs_res_block00[i](F_pan_00)
        F_encoder_00 = self.hs_res_block00_conv(F_encoder_00)
        F_pan_00 = self.hs_res_block00_conv(F_pan_00)

        # 留下跳转连接变量
        F_encoder_00_skip = F_encoder_00
        F_pan_00_skip = F_pan_00

        F_encoder_01 = self.hs_Down00(F_encoder_00)
        F_pan_01 = self.hs_Down00(F_pan_00)

        ''' -------------------------- 编码第二尺度 ------------------------- '''
        F_MD_01 = self.RNN_encoder01(F_encoder_01, F_pan_01)
        F_encoder_01 = F_encoder_01 + F_MD_01["ht_pan"]

        # RBs
        for i in range(self.unet_res_blocks[1]):
            F_encoder_01 = self.hs_res_block01[i](F_encoder_01)
            F_pan_01 = self.hs_res_block01[i](F_pan_01)
        F_encoder_01 = self.hs_res_block01_conv(F_encoder_01)
        F_pan_01 = self.hs_res_block01_conv(F_pan_01)


        # 留下跳转连接变量
        F_encoder_01_skip = F_encoder_01
        F_pan_01_skip = F_pan_01

        F_encoder_02 = self.hs_Down01(F_encoder_01)
        F_pan_02 = self.hs_Down01(F_pan_01)

        ''' -------------- 编码第三尺度 不再下采样，准备上采样 ---------------- '''
        F_MD_02 = self.RNN_encoder02(F_encoder_02, F_pan_02)
        F_encoder_02 = F_encoder_02 + F_MD_02["ht_pan"]

        # RBs
        for i in range(self.unet_res_blocks[2]):
            F_encoder_02 = self.hs_res_block02[i](F_encoder_02)
            F_pan_02 = self.hs_res_block02[i](F_pan_02)
        F_encoder_02 = self.hs_res_block02_conv(F_encoder_02)
        F_pan_02 = self.hs_res_block02_conv(F_pan_02)

        ''' ------------------------ 解码第二尺度  ------------------------ '''
        # 上采样 ----> 跳转连接相加 ----> 融合 ----> 残差 ----> （下一尺度上采样）
        F_decoder_01 = self.hs_Up01(F_encoder_02)  # 上采样
        F_pan_de01 = self.hs_Up01(F_pan_02)  # 上采样

        F_decoder_01 = F_decoder_01 + F_encoder_01_skip  # 跳转连接相加
        F_pan_de01 = F_pan_de01 + F_pan_01_skip  # 跳转连接相加

        F_MD_de01 = self.RNN_decoder01(F_decoder_01, F_pan_de01)
        F_decoder_01 = F_decoder_01 + F_MD_de01["ht_pan"]

        # RBs
        for i in range(self.unet_res_blocks[3]):
            F_decoder_01 = self.hs_res_block03[i](F_decoder_01)
            F_pan_de01 = self.hs_res_block03[i](F_pan_de01)
        F_decoder_01 = self.hs_res_block03_conv(F_decoder_01)
        F_pan_de01 = self.hs_res_block03_conv(F_pan_de01)

        ''' ------------------------ 解码第一尺度  ------------------------ '''

        F_decoder_00 = self.hs_Up00(F_decoder_01)  # 上采样
        F_pan_de00 = self.hs_Up00(F_pan_de01)  # 上采样

        F_decoder_00 = F_decoder_00 + F_encoder_00_skip  # 跳转连接相加
        F_pan_de00 = F_pan_de00 + F_pan_00_skip  # 跳转连接相加

        # F_decoder_00 = F_decoder_00 + F_pan_de00
        F_decoder_00 = torch.cat((F_decoder_00, F_pan_de00), dim=1)  # [B, 2C, H, W]

        # RBs
        for i in range(self.unet_res_blocks[4]):
            F_decoder_00 = self.hs_res_block04[i](F_decoder_00)
        F_decoder_00 = self.hs_res_block04_conv(F_decoder_00)

        ''' ------------------------ 最后特征调整输出融合结果 ---------------------'''
        # HR_HSI = self.feature_adjustment(F_decoder_00)
        HR_HSI = self.compress(F_decoder_00)

        ''' ------------------------ AU(spatial & spectral) --------------------- '''
        s_spa = self.spa_var(F_decoder_00)    # uncertainty spatial [B, 1, H, W]
        s_spe = self.spe_var(F_decoder_00)  # uncertainty spectral  [B, C, 1, 1]

        output = {"pred": HR_HSI, "s_spa": s_spa, "s_spe": s_spe}

        return output


'''================================================
HFRUNet 模块：总启动程序 
================================================'''
class HFRUNet(nn.Module):
    def __init__(self, config):
        super(HFRUNet, self).__init__()
        self.num_head = config["N_modules"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]  # 光谱通道数
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]  # 尺寸缩放比例

        self.num_res_blocks = [2, 2]  # Feature_extraction模块与Feature_adjustment模块中所用的残差块数量
        self.unet_res_blocks = [1, 1, 1, 1, 1]
        self.res_scale = 1
        # self.feature_chanels = [128, 256, 512, 256, 128]
        self.feature_chanels = [64, 128, 256, 128, 64]

        ''' LR-HSI和PAN的特征通道调整(Feature_extraction模块) '''
        # self.up_lrhs_feature_extraction = Feature_extraction(self.in_channels, self.num_res_blocks[1], self.feature_chanels[0], self.res_scale)
        # self.pan_feature_extraction = Feature_extraction(1, self.num_res_blocks[1], self.feature_chanels[0], self.res_scale)
        self.hs_compress = conv3x3(self.in_channels, self.feature_chanels[0])
        self.pan_compress = conv3x3(1, self.feature_chanels[0])

        ''' ======================== 降采样操作 通道数保持不变 ========================== '''
        self.hs_Down00 = Down(self.feature_chanels[0], self.feature_chanels[1])
        self.hs_Down01 = Down(self.feature_chanels[1], self.feature_chanels[2])

        ''' ======================== 上采样操作 通道数保持不变 ========================== '''
        self.hs_Up01 = Up(self.feature_chanels[2], self.feature_chanels[1])
        self.hs_Up00 = Up(self.feature_chanels[1], self.feature_chanels[0])

        ''' ====================== 残差块  ============================ '''
        # 00 [B, C, H, W]
        self.hs_res_block00 = nn.ModuleList()
        for i in range(self.unet_res_blocks[0]):
            self.hs_res_block00.append(
                ResBlock(in_channels=self.feature_chanels[0], out_channels=self.feature_chanels[0], res_scale=1))
        self.hs_res_block00_conv = conv3x3(self.feature_chanels[0], self.feature_chanels[0])

        # 01 [B, C, H/2, W/2]
        self.hs_res_block01 = nn.ModuleList()
        for i in range(self.unet_res_blocks[1]):
            self.hs_res_block01.append(
                ResBlock(in_channels=self.feature_chanels[1], out_channels=self.feature_chanels[1], res_scale=1))
        self.hs_res_block01_conv = conv3x3(self.feature_chanels[1], self.feature_chanels[1])

        # 02 [B, C, H/4, W/4]
        self.hs_res_block02 = nn.ModuleList()
        for i in range(self.unet_res_blocks[2]):
            self.hs_res_block02.append(
                ResBlock(in_channels=self.feature_chanels[2], out_channels=self.feature_chanels[2], res_scale=1))
        self.hs_res_block02_conv = conv3x3(self.feature_chanels[2], self.feature_chanels[2])

        # 03 [B, C, H/2, W/2]
        self.hs_res_block03 = nn.ModuleList()
        for i in range(self.unet_res_blocks[3]):
            self.hs_res_block03.append(
                ResBlock(in_channels=self.feature_chanels[3], out_channels=self.feature_chanels[3], res_scale=1))
        self.hs_res_block03_conv = conv3x3(self.feature_chanels[3], self.feature_chanels[3])

        # 04 [B, C, H, W]
        self.hs_res_block04 = nn.ModuleList()
        for i in range(self.unet_res_blocks[4]):
            self.hs_res_block04.append(
                ResBlock(in_channels=self.feature_chanels[3], out_channels=self.feature_chanels[3], res_scale=1))
        self.hs_res_block04_conv = conv3x3(self.feature_chanels[3], self.feature_chanels[3])

        # 双向循环网络
        self.RNN_encoder00 = DSRU(in_channels=self.feature_chanels[0], out_channels=self.feature_chanels[0], group_channels =8, MD_S=1, MD_D=8, MD_R=2)
        self.RNN_encoder01 = DSRU(in_channels=self.feature_chanels[1], out_channels=self.feature_chanels[1], group_channels =16, MD_S=1, MD_D=16, MD_R=4 )
        self.RNN_encoder02 = DSRU(in_channels=self.feature_chanels[2], out_channels=self.feature_chanels[2], group_channels =32, MD_S=1, MD_D=32, MD_R=8 )

        self.RNN_decoder01 = DSRU(in_channels=self.feature_chanels[3], out_channels=self.feature_chanels[3], group_channels =16, MD_S=1, MD_D=16, MD_R=4 )
        self.RNN_decoder00 = DSRU(in_channels=self.feature_chanels[4], out_channels=self.feature_chanels[4], group_channels =8, MD_S=1, MD_D=8, MD_R=2 )

        ''' 最终融合结果特征调整(Feature_adjustment模块) '''
        # self.feature_adjustment = Feature_adjustment(self.feature_chanels[0], self.num_res_blocks[0], self.out_channels, self.res_scale)
        self.compress = nn.Conv2d(self.feature_chanels[3], self.out_channels, kernel_size=1, padding=0, bias=True)
        # AU
        # no use Linear,  spa:[B,C,H,W], SPE:[B,C,1,1]
        self.spa_var = nn.Sequential(nn.Conv2d(self.feature_chanels[3], self.feature_chanels[4], kernel_size=3, padding=1, bias=True),
                                      nn.ELU(),
                                      nn.Conv2d(self.feature_chanels[4], self.out_channels, kernel_size=3, padding=1, bias=True),
                                      nn.ELU())
        self.spe_var = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.feature_chanels[3], self.feature_chanels[4], kernel_size=1, padding=0, bias=True),
                                      nn.ELU(),
                                      nn.Conv2d(self.feature_chanels[4], self.out_channels, kernel_size=1, padding=0, bias=True),
                                      nn.ELU())

    def forward(self, LR_HSI, PAN, GT_D0):
        # 调整PAN尺寸为[batch_size, 1，HR_size, HR_size]
        PAN = PAN.unsqueeze(dim=1)
        UP_LR_HS = F.interpolate(LR_HSI, scale_factor=(self.factor, self.factor), mode ='bicubic')   # 上采样的LR-HSI，用于晶格结构

        ''' 初始特征通道调整 '''
        # F_pan_00 = self.pan_feature_extraction(PAN)  # # PAN调整通道
        F_pan_00 = self.pan_compress(PAN)
        # F_up_lrhs_00 = self.up_lrhs_feature_extraction(UP_LR_HS)  # HSI调整通道
        F_up_lrhs_00 = self.hs_compress(UP_LR_HS)

        ''' -------------------------- 编码第一尺度 ------------------------- '''
        # 融合 ----> RBs ----> 留下跳转连接变量 ----> 下采样 ----> （下一尺度融合）
        F_MD_00 = self.RNN_encoder00(F_up_lrhs_00, F_pan_00)
        F_encoder_00 = F_up_lrhs_00 + F_MD_00["ht_pan"]

        # RBs
        for i in range(self.unet_res_blocks[0]):
            F_encoder_00 = self.hs_res_block00[i](F_encoder_00)
            F_pan_00 = self.hs_res_block00[i](F_pan_00)
        F_encoder_00 = self.hs_res_block00_conv(F_encoder_00)
        F_pan_00 = self.hs_res_block00_conv(F_pan_00)

        # 留下跳转连接变量
        F_encoder_00_skip = F_encoder_00
        F_pan_00_skip = F_pan_00

        F_encoder_01 = self.hs_Down00(F_encoder_00)
        F_pan_01 = self.hs_Down00(F_pan_00)

        ''' -------------------------- 编码第二尺度 ------------------------- '''
        F_MD_01 = self.RNN_encoder01(F_encoder_01, F_pan_01)
        F_encoder_01 = F_encoder_01 + F_MD_01["ht_pan"]

        # RBs
        for i in range(self.unet_res_blocks[1]):
            F_encoder_01 = self.hs_res_block01[i](F_encoder_01)
            F_pan_01 = self.hs_res_block01[i](F_pan_01)
        F_encoder_01 = self.hs_res_block01_conv(F_encoder_01)
        F_pan_01 = self.hs_res_block01_conv(F_pan_01)


        # 留下跳转连接变量
        F_encoder_01_skip = F_encoder_01
        F_pan_01_skip = F_pan_01

        F_encoder_02 = self.hs_Down01(F_encoder_01)
        F_pan_02 = self.hs_Down01(F_pan_01)

        ''' -------------- 编码第三尺度 不再下采样，准备上采样 ---------------- '''
        F_MD_02 = self.RNN_encoder02(F_encoder_02, F_pan_02)
        F_encoder_02 = F_encoder_02 + F_MD_02["ht_pan"]

        # RBs
        for i in range(self.unet_res_blocks[2]):
            F_encoder_02 = self.hs_res_block02[i](F_encoder_02)
            F_pan_02 = self.hs_res_block02[i](F_pan_02)
        F_encoder_02 = self.hs_res_block02_conv(F_encoder_02)
        F_pan_02 = self.hs_res_block02_conv(F_pan_02)

        ''' ------------------------ 解码第二尺度  ------------------------ '''
        # 上采样 ----> 跳转连接相加 ----> 融合 ----> 残差 ----> （下一尺度上采样）
        F_decoder_01 = self.hs_Up01(F_encoder_02)  # 上采样
        F_pan_de01 = self.hs_Up01(F_pan_02)  # 上采样

        F_decoder_01 = F_decoder_01 + F_encoder_01_skip  # 跳转连接相加
        F_pan_de01 = F_pan_de01 + F_pan_01_skip  # 跳转连接相加

        F_MD_de01 = self.RNN_decoder01(F_decoder_01, F_pan_de01)
        F_decoder_01 = F_decoder_01 + F_MD_de01["ht_pan"]

        # RBs
        for i in range(self.unet_res_blocks[3]):
            F_decoder_01 = self.hs_res_block03[i](F_decoder_01)
            F_pan_de01 = self.hs_res_block03[i](F_pan_de01)
        F_decoder_01 = self.hs_res_block03_conv(F_decoder_01)
        F_pan_de01 = self.hs_res_block03_conv(F_pan_de01)

        ''' ------------------------ 解码第一尺度  ------------------------ '''

        F_decoder_00 = self.hs_Up00(F_decoder_01)  # 上采样
        F_pan_de00 = self.hs_Up00(F_pan_de01)  # 上采样

        F_decoder_00 = F_decoder_00 + F_encoder_00_skip  # 跳转连接相加
        F_pan_de00 = F_pan_de00 + F_pan_00_skip  # 跳转连接相加

        # F_decoder_00 = F_decoder_00 + F_pan_de00
        F_decoder_00 = torch.cat((F_decoder_00, F_pan_de00), dim=1)  # [B, 2C, H, W]

        # RBs
        for i in range(self.unet_res_blocks[4]):
            F_decoder_00 = self.hs_res_block04[i](F_decoder_00)
        F_decoder_00 = self.hs_res_block04_conv(F_decoder_00)

        ''' ------------------------ 最后特征调整输出融合结果 ---------------------'''
        # HR_HSI = self.feature_adjustment(F_decoder_00)
        HR_HSI = self.compress(F_decoder_00)

        ''' ------------------------ AU(spatial & spectral) --------------------- '''
        with torch.no_grad():
            s_spa = self.spa_var(F_decoder_00)    # uncertainty spatial [B, 1, H, W]
            s_spe = self.spe_var(F_decoder_00)  # uncertainty spectral  [B, C, 1, 1]
            # s_spe = self.spe_var(F_decoder_00).unsqueeze(-1).unsqueeze(-1)  #if use Linear,  uncertainty spectral  [B, C, 1, 1]

            # s_spa = self.spa_var(HR_HSI)    # uncertainty spatial [B,1,H,W]
            # s_spe = self.spe_var(HR_HSI).unsqueeze(-1).unsqueeze(-1)  # uncertainty spectral  [B, C, 1, 1]

        # print("s_spa.shape, s_spe.shape", s_spa.shape, s_spe.shape)
        output = {"pred": HR_HSI, "s_spa": s_spa, "s_spe": s_spe}

        return output
    




