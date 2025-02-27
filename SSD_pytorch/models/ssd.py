import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import *
from utils.config import opt
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    SSD模型由去掉全连接层的vgg网络为基础组成。在之后添加了多盒转化层。
    每个多盒层分支是：
        1）conv2d 获取分类置信度
        2）conv2d进行坐标位置预测
        3）相关层去产生特定于该层特征图大小的默认的预测框bounding  boxes



    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size  输入的图像尺寸
        base: VGG16 layers for input, size of either 300 or 500   经过修改的vgg网络
        extras: extra layers that feed to multibox loc and conf layers
                提供多盒定位的格外层  和 分类置信层（vgg网络后面新增的额外层）
        head: "multibox head" consists of loc and conf conv layers
                由定位和分类卷积层组成的multibox head
                (loc_layers, conf_layers)     vgg与extras中进行分类和回归的层
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = opt.voc

        # 新定义一个类，该类的功能：对于每个feature map，生成预测框（中心坐标及偏移量）
        self.priorbox = PriorBox(self.cfg)

        # 调用forward，返回生成的预测框结果
        # 对于所有预测的feature map，存储着生成的不同长宽比的默认框（可以理解为anchor）
        # Tensor不能反向传播，Variable才可以反向传播
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())

        self.size = size  # 300

        # SSD network范围
        # 经过修改的vgg网络
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        # Layer层从conv4_3学习去缩放l2正则化特征
        # 论文中conv4_3 相比较于其他的layers，有着不同的 feature scale，我们使用 ParseNet 中的 L2 normalization 技术
        # 将conv4_3 feature map 中每一个位置的 feature norm scale 到 20，并且在 back-propagation 中学习这个 scale
        self.L2Norm = L2Norm(512, 20)
        # vgg网络后面新增的额外层
        self.extras = nn.ModuleList(extras)
        # vgg与extras中进行分类和回归的层
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        # 如果网络用于测试，则加入softmax和检测
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        前向传播

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test测试集:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train训练集:
                list of concat outputs from:
                    1: 分类层confidence layers, Shape: [batch*num_priors,num_classes]
                    2: 回归定位层localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        # sources保存 网络生成的不同层feature map结果，以便使用这些feature map来进行分类与回归
        sources = list()
        # 保存预测层不同feature map通过回归和分类网络的输出结果
        loc = list()
        conf = list()

        # 原论文中vgg的conv4_3，relu之后加入L2 Normalization正则化，然后保存feature map
        # apply vgg up to conv4_3 relu
        # 将vgg层的feature map保存
        for k in range(23):
            x = self.vgg[k](x)
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7，即将原fc7层更改为卷积层输出的结果，经过relu之后保存结果
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        # 将新增层的feature map保存
        for k, v in enumerate(self.extras):
            # 每经过一个conv卷积，都relu一下
            x = F.relu(v(x), inplace=True)
            # 论文中隔一个conv保存一个结果
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        # permute  将tensor的维度换位  参数为换位顺序
        # contiguous 返回一个内存连续的有相同数据的tensor

        # source保存的是每个预测层的网络输出,即feature maps
        # loc 通过使用feature map去预测回归
        # conf通过使用feature map去预测分类
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # 在给定维度上对输入的张量序列seq 进行连接操作    dimension=1表示在列上连接
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # 测试集上的输出
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds  定位的预测
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds  分类的预测
                self.priors.type(type(x.data))  # default boxes  预测框
            )
        else:
            # 训练集上的输出
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds [32,8732,4] 通过网络输出的定位的预测
                conf.view(conf.size(0), -1, self.num_classes),  # conf preds [32,8732,21]  通过网络输出的分类的预测
                self.priors  # 不同feature map根据公式生成的锚结果 [8732,4]   内容为 中心点坐标和宽高
            )
        return output

    def saveSSD(self, name=None):
        '''
        保存模型
        '''
        # 保存训练集最后一次的模型
        if name is None:
            prefix = opt.checkpoint_root + 'last_time_SSD'
            name = prefix + '.pth'
        # 每隔一段时间保存一次模型
        else:
            prefix = opt.checkpoint_root + 'SSD_iter' + name
            name = prefix + '.pth'
        torch.save(self.state_dict(), name)
        return name


# This function is derived from torchvision VGG make_layers()
# 此方法源自torchvision VGG make_layers（）
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    '''
    vgg的结构
    cfg:  vgg的结构
     '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    i: 3   输入图像通道数
    batch_norm    为False。若为True，则网络中加入batch_norm

    返回没有全连接层的vgg网络
    '''
    # 保存vgg所有层
    layers = []
    # 输入图像通道数
    in_channels = i
    for v in cfg:  # M与C会导致生成的feature map大小出现变化
        # ceil与floor分别为向下取整与向上取整
        if v == 'M':  # 最大池化层   默认floor模式
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':  # 最大池化层   ceil模式   两种不同的maxpool方式    参考https://blog.csdn.net/GZHermit/article/details/79351803
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            # 卷积
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # 论文将 Pool5 layer 的参数，从 卷积核2×2步长为2  转变成 卷积核3×3 步长为1 外加一个 pad
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # 论文中将VGG的FC6 layer、FC7 layer 转成为 卷积层conv6,conv7 并从模型的FC6、FC7 上的参数，进行采样得到这两个卷积层的 参数
    # 输入通道512  输出通道为1024  卷积核为3  padding为6    dilation为卷积核中元素之间的空洞大小
    # 修改Pool5 layer参数，导致感受野大小改变。所以conv6采用 atrous 算法，即孔填充算法。
    # 孔填充算法将卷积 weights 膨胀扩大，即原来卷积核是 3x3，膨胀后，可能变成 7x7 了，这样 receptive field 变大了，而 score map 也很大，即输出变成 dense
    # 这么做的好处是，输出的 score map 变大了，即是 dense 的输出了，而且 receptive field 不会变小，而且可以变大。这对做分割、检测等工作非常重要。
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # 输入通道512  输出通道为1024  卷积核为3
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    # 将 修改的层也加入到vgg网络中
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    '''
    vgg网络后面新增的额外层
    :param cfg:  '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    :param i:    1024  输入通道数
    :param batch_norm:  flase
    :return:
    '''
    # 添加到VGG的额外图层用于特征缩放
    layers = []
    # 1024  输入通道数
    in_channels = i
    # 控制卷积核尺寸，一维数组选前一个数还是后一个数。在每次循环时flag都改变，导致网络的卷积核尺寸为1,3,1,3交替
    # False 为1，True为3
    # SSD网络图中s1指步长为1，s2指步长为2
    # 在该代码中，S代表步长为2，无S代表默认，即步长为1，所以cfg与论文网络结构完全匹配
    flag = False
    # enumerate枚举   k为下标   v为值
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    '''
    :param vgg: 经过修改后的vgg网络（去掉全连接层，修改pool5参数并添加新层）
    :param extra_layers: vgg网络后面新增的额外层
    :param cfg: '300': [4, 6, 6, 6, 4, 4],  不同部分的feature map上一个网格预测多少框
    :param num_classes: 20分类+1背景，共21类
    :return:
    '''

    # 保存所有参与预测的网络层
    loc_layers = []
    conf_layers = []
    # 传入的修改过的vgg网络用于预测的网络是21层以及 倒数第二层
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        # 4是回归的坐标参数  cfg代表该层feature map上一个网格预测多少框
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        # num_classes是类别数 cfg代表该层feature map上一个网格预测多少框
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]
    # [x::y] 从下标x开始，每隔y取值
    # 论文中新增层也是每隔一个层添加一个预测层
    # 将新增的额外层中的预测层也添加上   start=2：下标起始位置
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    # 数字为每层feature map的层数  M代表最大池化层（默认floor模式）    C代表最大池化层（ceil模式）  (去掉vgg16的最后的 maxpool、fc、fc、fc、softmax)
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    #
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # 不同部分的feature map上一个网格预测多少框
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    """
    新建SSD模型
    """

    # 训练或测试
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    # 当前SSD300只支持大小300×300的数据集训练
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return

    # base_： 经过修改后的vgg网络（去掉全连接层，修改pool5参数并添加新层）
    # extras_：  vgg网络后面新增的额外层
    # head_ :    (loc_layers, conf_layers)   vgg与extras中进行分类和回归的层
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),  # vgg方法返回 经过修改后的vgg网络（去掉全连接层，修改pool5参数并添加新层）
                                     add_extras(extras[str(size)], 1024),  # vgg网络后面新增的额外层
                                     mbox[str(size)],  # mbox指不同部分的feature map上一个网格预测多少框
                                     num_classes)
    # phase：'train'    size：300    num_classes： 21 类别数（20类+1背景）
    return SSD(phase, size, base_, extras_, head_, num_classes)
