import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import os
import torch.distributed as dist

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.loss_det = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        #do relu here

        self.block1 = Block(64,
                            128,
                            2,
                            2,
                            start_with_relu=False,
                            grow_first=True)
        self.block2 = Block(128,
                            256,
                            2,
                            2,
                            start_with_relu=True,
                            grow_first=True)
        self.block3 = Block(256,
                            728,
                            2,
                            2,
                            start_with_relu=True,
                            grow_first=True)

        self.block4 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block5 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block6 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block7 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)

        self.block8 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block9 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block10 = Block(728,
                             728,
                             3,
                             1,
                             start_with_relu=True,
                             grow_first=True)
        self.block11 = Block(728,
                             728,
                             3,
                             1,
                             start_with_relu=True,
                             grow_first=True)

        self.block12 = Block(728,
                             1024,
                             2,
                             2,
                             start_with_relu=True,
                             grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        #do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x_fc = self.fc(x)
        x_softmax = self.softmax(x_fc)
        
        return x_fc, x_softmax

    def forward(self, x):
        x = self.features(x)
        x_fc, x_softmax = self.logits(x)

        return x_fc, x_softmax

        # loss_det = self.loss_det(out_det, y_det)
        # return {'pred_det': out_det, 'loss': loss_det}

class MILModel(nn.Module):
    def __init__(self, num_classes, k, split_size):
        super(MILModel, self).__init__()
        self.cnn = Xception(num_classes)
        self.k = k
        self.split_size = split_size
        # self.classifier = nn.Linear(1, num_classes)
    
    def forward(self, sample):
        x = sample["image"]
        y = sample["label"]
        batch_size, max_instances, _, _, _ = x.size()
        # print(int(os.environ["LOCAL_RANK"]), "forward", batch_size, max_instances)
        
        # 分批加载和处理图片
        all_features = []
        for j in range(batch_size):
            batch_features = []
            for i in range(0, max_instances, self.split_size):
                # if int(os.environ["LOCAL_RANK"]) <=0:
                #     print(int(os.environ["LOCAL_RANK"]), j, i)
                split_images = x[j, i:i+self.split_size, :, :, :]
                split_images = split_images.contiguous().view(-1, 3, 299, 299)
                # if int(os.environ["LOCAL_RANK"]) <=0:
                #     print("success", split_images.size(), split_images.size(0))
                
                if split_images.size(0) == 0:
                    continue

                mini_split_features = self.cnn(split_images)
                batch_features.append(mini_split_features.squeeze(-1))
                # if int(os.environ["LOCAL_RANK"]) <=0:
                #     print("success")
            
            batch_features = torch.cat(batch_features, dim=0)
            all_features.append(batch_features)
        
        instance_features = torch.stack(all_features, dim=0)  # (batch_size * max_instances, 1)

        probs_class_1 = instance_features[:, :, 1]
        topk_values, topk_indices = torch.topk(probs_class_1, self.k)

        batch_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, self.k)
        topk_samples = instance_features[batch_indices, topk_indices]
        topk_vals = torch.mean(topk_samples, dim=1)

        output = {"top_images": x[batch_indices, topk_indices], "top_fc_out": topk_samples,
                  "top_softmax": topk_samples, "top_mean_softmax": topk_vals}
        # print(int(os.environ["LOCAL_RANK"]), "return")
        return output




        # topk_vals = torch.mean(topk_vals, dim=1)
        # if int(os.environ["LOCAL_RANK"]) > 0:
        #     print(topk_vals)
        # logits = self.classifier(topk_values.unsqueeze(-1))
        # return logits.squeeze(-1)