import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.fpn import FPN
from blocks.adapter import DINOV3Wrapper, DenseAdapterLite
from backbone.resnet50 import resnet50

# 0.TBTT：双向统一时序交互模块
class TBTT(nn.Module):
    def __init__(self, dim, head=4):
        super().__init__()
        self.head = head
        self.dim = dim

        # T1 -> T2
        self.q1 = nn.Conv2d(dim, dim, 1)
        self.k1 = nn.Conv2d(dim, dim, 1)
        self.v1 = nn.Conv2d(dim, dim, 1)

        # T2 -> T1
        self.q2 = nn.Conv2d(dim, dim, 1)
        self.k2 = nn.Conv2d(dim, dim, 1)
        self.v2 = nn.Conv2d(dim, dim, 1)

        # 可学习门控
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))

    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        scale = (C // self.head) ** 0.5

        # ---------------- T1 -> T2 ----------------
        q1 = self.q1(f1).view(B, self.head, C//self.head, H*W).permute(0,1,3,2).contiguous()
        k1 = self.k1(f2).view(B, self.head, C//self.head, H*W).contiguous()
        v1 = self.v1(f2).view(B, self.head, C//self.head, H*W).permute(0,1,3,2).contiguous()
        att1 = torch.softmax(q1 @ k1 / scale, dim=-1)
        o1 = (att1 @ v1).permute(0,1,3,2).reshape(B,C,H,W).contiguous()

        # ---------------- T2 -> T1 ----------------
        q2 = self.q2(f2).view(B, self.head, C//self.head, H*W).permute(0,1,3,2).contiguous()
        k2 = self.k2(f1).view(B, self.head, C//self.head, H*W).contiguous()
        v2 = self.v2(f1).view(B, self.head, C//self.head, H*W).permute(0,1,3,2).contiguous()
        att2 = torch.softmax(q2 @ k2 / scale, dim=-1)
        o2 = (att2 @ v2).permute(0,1,3,2).reshape(B,C,H,W).contiguous()

        f1_o = f1 + self.g1 * o1
        f2_o = f2 + self.g2 * o2
        return f1_o, f2_o


# trick/ SCP 语义净化模块
class SCP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.change_guess = nn.Conv2d(dim, 1, 1, bias=False)
        self.gate_t1 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim))
        self.gate_t2 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim))
        self.clean_t1 = nn.Sequential(nn.Conv2d(dim, dim, 3,1,1), nn.BatchNorm2d(dim), nn.ReLU(True))
        self.clean_t2 = nn.Sequential(nn.Conv2d(dim, dim, 3,1,1), nn.BatchNorm2d(dim), nn.ReLU(True))

    def forward(self, f1, f2, diff):
        mask = torch.sigmoid(self.change_guess(diff))
        no_change = 1 - mask

        f1m = f1 * no_change
        f2m = f2 * no_change

        g1 = torch.sigmoid(self.gate_t1(f1m))
        g2 = torch.sigmoid(self.gate_t2(f2m))

        f1c = self.clean_t1(f1 + f2m * g1)
        f2c = self.clean_t2(f2 + f1m * g2)
        return f1c, f2c

# 1.MCE/多尺度变化增强
class MCE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scales = nn.ModuleList([nn.Identity(), nn.AvgPool2d(2), nn.AvgPool2d(4)])
        self.att = nn.ModuleList([nn.Conv2d(dim, 1, 1, bias=False) for _ in range(3)])
        self.f1 = nn.Sequential(nn.Conv2d(dim, dim, 3,1,1), nn.BatchNorm2d(dim), nn.ReLU(True))
        self.f2 = nn.Sequential(nn.Conv2d(dim, dim, 3,1,1), nn.BatchNorm2d(dim), nn.ReLU(True))
        self.out = nn.Sequential(nn.Conv2d(dim, dim, 3,1,1), nn.BatchNorm2d(dim), nn.ReLU(True))

    def forward(self, d, f1, f2):
        dif = torch.abs(f1 - f2)
        ms = []
        for s, a in zip(self.scales, self.att):
            sd, sf = s(dif), s(d)
            ms.append(sf * torch.sigmoid(a(sd)))

        x = F.interpolate(ms[2], ms[1].shape[2:], mode='bilinear', align_corners=True)
        x = self.f1(x + ms[1])
        x = F.interpolate(x, ms[0].shape[2:], mode='bilinear', align_corners=True)
        x = self.f2(x + ms[0])
        return self.out(x + d)

# 2.BCE/双向变化增强
class BiChangeEnhance(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dc = nn.Conv2d(dim, dim, 3, padding=1)
        self.sg = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.cg = nn.Conv2d(dim, dim, 1)

    def forward(self, f1, f2):
        d = self.dc(torch.abs(f1-f2))
        return d * self.sg(f1+f2) + self.cg(d)

# 门控融合 CNN + DINO
class GatedFusion(nn.Module):
    def __init__(self, id, od):
        super().__init__()
        self.p = nn.Conv2d(id, od, 1)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(od,od,1), nn.SiLU(True), nn.Conv2d(od,od,1,bias=False), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False), nn.Sigmoid())

    def forward(self, c, d):
        x = self.p(torch.cat([c,d],1))
        x = x * self.ca(x)
        s = torch.cat([x.mean(1,keepdim=True), x.amax(1,keepdim=True)], 1)
        x = x * self.sa(s)
        return c + x

# 金字塔Fusion
class PyramidFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.ModuleList([GatedFusion(dim+256, dim) for _ in range(4)])

    def forward(self, c_feats, d_feats):
        out = []
        for i in range(4):
            out.append(self.layers[i](c_feats[i], d_feats[i]))
        return out

# Encoder
class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.fpn = FPN([256,512,1024,2048], dim)
        self.dino = DINOV3Wrapper()
        self.adapter = DenseAdapterLite(1024, 256)
        self.pff = PyramidFusion(dim)

    def forward(self, x):
        cnn_feat = self.fpn(self.backbone(x))
        dino_feat = self.adapter(self.dino(x))
        return self.pff(cnn_feat, dino_feat)

# MTBTT 多尺度TBTT（保留完整 T1 + T2 所有层）

class MultiScaleTBTT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tbtt0 = TBTT(dim)
        self.tbtt1 = TBTT(dim)
        self.tbtt2 = TBTT(dim)
        self.tbtt3 = TBTT(dim)

    def forward(self, f1, f2):
        o0, t2_0 = self.tbtt0(f1[0], f2[0])
        o1, t2_1 = self.tbtt1(f1[1], f2[1])
        o2, t2_2 = self.tbtt2(f1[2], f2[2])
        o3, t2_3 = self.tbtt3(f1[3], f2[3])
        return [o0, o1, o2, o3], [t2_0, t2_1, t2_2, t2_3]


# Feature Fusion/ 特征融合（T1 + T2 双时相完整输入）
class ChangeFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fuse = nn.Conv2d(dim * 7, dim, 1)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1), nn.SiLU(True),
            nn.Conv2d(dim, dim, 1), nn.Sigmoid()
        )

    def forward(self, f1s, f2s, diff):
        shape = f1s[0].shape[2:]
        feats = []
        for f in f1s: feats.append(F.interpolate(f, size=shape, mode='bilinear', align_corners=True))
        for f in f2s: feats.append(F.interpolate(f, size=shape, mode='bilinear', align_corners=True))
        feats.append(F.interpolate(diff, size=shape, mode='bilinear', align_corners=True))
        x = self.fuse(torch.cat(feats, dim=1))
        return x * self.att(x)


# Decoder Head/ 解耦头
class SCDHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.h_cd = nn.Sequential(nn.Conv2d(dim,dim,3,1,1), nn.BatchNorm2d(dim), nn.ReLU(True))
        self.h_s1 = nn.Sequential(nn.Conv2d(dim,dim,3,1,1), nn.BatchNorm2d(dim), nn.ReLU(True))
        self.h_s2 = nn.Sequential(nn.Conv2d(dim,dim,3,1,1), nn.BatchNorm2d(dim), nn.ReLU(True))
        self.h_eg = nn.Sequential(nn.Conv2d(dim,dim,3,1,1), nn.BatchNorm2d(dim), nn.ReLU(True))

        self.cd = nn.Conv2d(dim, 1, 1)
        self.s1 = nn.Conv2d(dim, num_classes, 1)
        self.s2 = nn.Conv2d(dim, num_classes, 1)
        self.edge = nn.Conv2d(dim, 1, 1)

    def forward(self, x):
        cd = self.cd(self.h_cd(x))
        s1 = self.s1(self.h_s1(x))
        s2 = self.s2(self.h_s2(x))
        eg = self.edge(self.h_eg(x))
        return cd, s1, s2, eg

# SemDINO
class SemDINO(nn.Module):
    def __init__(self, num_classes=6, dim=128):
        super().__init__()
        self.encoder = Encoder(dim)
        self.mtbtt = MultiScaleTBTT(dim)
        self.bce = BiChangeEnhance(dim)
        self.scp = SCP(dim)
        self.mce = MCE(dim)
        self.fusion = ChangeFusion(dim)
        self.head = SCDHead(dim, num_classes)

    def forward(self, img1, img2):
        # 双时相编码
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)

        # 双向时序对齐（保留全部 T1 + T2 特征）
        f1_align, f2_align = self.mtbtt(f1, f2)

        # 变化增强与净化
        diff = self.bce(f1_align[3], f2_align[3])
        f1c, f2c = self.scp(f1_align[3], f2_align[3], diff)
        diff = self.mce(diff, f1c, f2c)

        fused = self.fusion(f1_align[:3], f2_align[:3], diff)

        # 多任务输出
        out_cd, out_s1, out_s2, out_edge = self.head(fused)

        # 上采样到原图尺寸
        out_cd = F.interpolate(out_cd, img1.shape[2:], mode='bilinear', align_corners=True)
        out_s1 = F.interpolate(out_s1, img1.shape[2:], mode='bilinear', align_corners=True)
        out_s2 = F.interpolate(out_s2, img1.shape[2:], mode='bilinear', align_corners=True)
        out_edge = F.interpolate(out_edge, img1.shape[2:], mode='bilinear', align_corners=True)

        return out_cd, out_s1, out_s2, out_edge
