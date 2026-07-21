import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import time


class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3, groups=in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )
    def forward(self, x):
        return self.dwconv(x)


class CBA1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.cba(x)


class CBA3x3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.cba(x)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ECA(nn.Module):
    def __init__(self, kernal=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernal, padding=(kernal - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class task_interaction_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sem2Change = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction='mean')

    def forward(self, old_sem1, old_sem2, old_change, change_result):
        sem_max_out, _ = torch.max(torch.abs(old_sem1 - old_sem2), dim=1, keepdim=True)
        sem_avg_out = torch.mean(torch.abs(old_sem1 - old_sem2), dim=1, keepdim=True)
        sem_out = self.sigmoid(self.Sem2Change(torch.cat([sem_max_out, sem_avg_out], dim=1)))
        new_change = old_change * sem_out

        b, c, h, w = old_sem1.size()
        fea_sem1 = torch.reshape(old_sem1.permute(0,2,3,1), [b*h*w, c])
        fea_sem2 = torch.reshape(old_sem2.permute(0,2,3,1), [b*h*w, c])

        change_mask = torch.argmax(change_result, dim=1)
        unchange_mask = ~change_mask.bool()
        target = unchange_mask.float()
        target = target - change_mask.float()
        target = torch.reshape(target, [b * h * w])
        similarity_loss = self.loss_f(fea_sem1, fea_sem2, target)
        return new_change, similarity_loss


class decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        #upsample
        self.upconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.catconv = CBA3x3(out_channel * 2, out_channel)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, up, skip):
        #upsample
        up = self.upconv(up)
        up = torch.cat([up, skip], dim=1)
        up = self.catconv(up)
        return up

# BCFE
class Change_Specific_Transfer(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv = CBA1x1(in_channel*2, in_channel)
        self.eca = ECA()
        self.resblock = self._make_layer(ResBlock, 256, 128, 6, stride=1)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        xc1 = self.conv(torch.cat([x1, x2], dim=1))
        xc2 = self.conv(torch.cat([x2, x1], dim=1))
        change = self.eca(xc1 + xc2)
        diff = torch.abs(x1 - x2)
        change = torch.cat([change, diff], dim=1)
        change = self.resblock(change)
        return change


class Multi_Level_Feature_Aggreagation(nn.Module):

    def __init__(self,):
        super(Multi_Level_Feature_Aggreagation, self).__init__()

        self.proj1 = DWConv(512, 128)
        self.proj2 = DWConv(256, 128)

        self.cat_conv = CBA1x1(384, 128)


    def forward(self, x1, x2, x3):
        x3 = self.proj1(x3)
        x2 = self.proj2(x2)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.cat_conv(x)
        return x


class Boundary_Decoder(nn.Module):
    def __init__(self):
        super(Boundary_Decoder, self).__init__()
        self.sobel_x, self.sobel_y = get_sobel(64, 1)
        self.conv = nn.Conv2d(64, 64, 1, 1, 0)

    def forward(self, x, size):
        x = F.upsample(x, size, mode='bilinear')
        x = run_sobel(self.sobel_x, self.sobel_y, x)
        x = self.conv(x)
        return x


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y



class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.mlfa = Multi_Level_Feature_Aggreagation()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)  # size:1/2
        x = self.maxpool(x)  # size:1/4
        x_low = self.layer1(x)  # size:1/4
        x1 = self.layer2(x_low)  # size:1/8
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        x = self.mlfa(x1, x2, x3)
        return x, x_low


class BTSCD(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(BTSCD, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)

        self.change_specific_transfer = Change_Specific_Transfer(128) # BCFE

        self.DecCD = decoder(128, 64)
        self.Dec1 = decoder(128, 64)
        self.Dec2 = decoder(128, 64)

        self.task_interaction = task_interaction_module()

        self.classifierSem1 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierSem2 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierCD = nn.Conv2d(64, 2, 1, 1, 0, bias=False)

        self.boundary_decoder = Boundary_Decoder()
        self.eca = ECA()
        self.boundary_classifier = nn.Sequential(
            CBA3x3(64, 32),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.Sigmoid()
        )


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x_size = x1.size()

        x1, x1_low = self.FCN(x1)
        x2, x2_low = self.FCN(x2)

        xc = self.change_specific_transfer(x1, x2)

        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)

        xc_low = torch.abs(x1 - x2)
        xc = self.DecCD(xc, xc_low)

        #Classifier
        change = self.classifierCD(xc)
        new_xc, pixel_sim_loss = self.task_interaction(x1, x2, xc, change)

        out1 = self.classifierSem1(x1)
        out2 = self.classifierSem2(x2)
        new_change = self.classifierCD(new_xc)

        out1 = F.upsample(out1, x_size[2:], mode='bilinear')
        out2 = F.upsample(out2, x_size[2:], mode='bilinear')
        change_out = F.upsample(new_change, x_size[2:], mode='bilinear')

        boundary_x1 = self.boundary_decoder(x1, x_size[2:])
        boundary_x2 = self.boundary_decoder(x2, x_size[2:])
        boundary_change = self.boundary_decoder(new_xc, x_size[2:])

        boundary_sem = self.eca(boundary_x1 + boundary_x2)
        boundary_sem = self.boundary_classifier(boundary_sem)
        boundary_change = self.boundary_classifier(boundary_change)

        return change_out, out1, out2, pixel_sim_loss, boundary_sem, boundary_change


# if __name__ == '__main__':
#     x1 = torch.randn(1, 3, 512, 512).cuda().float()
#     x2 = torch.randn(1, 3, 512, 512).cuda().float()

#     model = BTSCD(3, num_classes=7).cuda()
#     model.eval()  
#     from fvcore.nn import FlopCountAnalysis
#     flops = FlopCountAnalysis(model, (x1, x2))
#     total = sum([param.nelement() for param in model.parameters()])
#     print("Params_Num: %.2fM" % (total/1e6))
#     print("FLOPs: %.2fG" % (flops.total()/1e9))

#     with torch.no_grad():
#         for _ in range(10):
#             _ = model(x1, x2)


#     start_time = time.time()
#     with torch.no_grad():
#         output = model(x1, x2)
#     end_time = time.time()

#     inference_time = end_time - start_time
#     print(f"Inference time: {inference_time * 1000:.2f} ms")
