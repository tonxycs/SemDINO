# import os
# import time
# import argparse
# import numpy as np
# import torch
# from skimage import io, exposure
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# #################################
# from datasets import RS_Landsat as RS
# # from datasets import RS_SECOND as RS
# # from datasets import RS_HRSCD as RS
# from SemDINO import SemDINO as Net
# DATA_NAME = 'Landsat'
# DATA_NAME = 'SECOND'
# DATA_NAME = 'HRSCD'
# #################################

# class PredOptions():
#     def __init__(self):
#         self.initialized = False

#     def initialize(self, parser):
#         working_path = os.path.dirname(os.path.abspath(__file__))
#         parser.add_argument('--pred_batch_size', required=False, default=1, help='prediction batch size')
#         parser.add_argument('--test_dir', required=False, default='/root/autodl-fs/AAAI-SemDINO/LandsatSCD/test', help='directory to test images')
#         parser.add_argument('--pred_dir', required=False, default='/root/autodl-fs/visminimum', help='directory to output masks')
#         parser.add_argument('--chkpt_path', required=False, default='/root/autodl-fs/SCanNet_49e_mIoU88.58_Sek59.12_Fscd88.73_OA96.20.pth')
#         self.initialized = True
#         return parser
        
#     def gather_options(self):
#         if not self.initialized:
#             parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#             parser = self.initialize(parser)
#         self.parser = parser
#         return parser.parse_args()

#     def parse(self):
#         self.opt = self.gather_options()
#         return self.opt
        
# def compare_models(model_1, model_2):
#     models_differ = 0
#     for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
#         if torch.equal(key_item_1[1], key_item_2[1]):
#             pass
#         else:
#             models_differ += 1
#             if (key_item_1[0] == key_item_2[0]):
#                 print('Mismtach found at', key_item_1[0])
#             else:
#                 raise Exception
#     if models_differ == 0:
#         print('Models match perfectly! :)')        

# def main():
#     begin_time = time.time()
#     opt = PredOptions().parse()
#     net = Net(num_classes=RS.num_classes, dim=128).cuda()
#     # 加载权重
#     ckpt = torch.load(opt.chkpt_path, map_location='cpu')
#     new_ckpt = {}
#     for k, v in ckpt.items():
#         if k.startswith('module.'):
#             new_ckpt[k[7:]] = v
#         else:
#             new_ckpt[k] = v
#     net.load_state_dict(new_ckpt)
#     net.eval()
    
#     test_set = RS.Data_test(opt.test_dir)
#     test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
#     # 修复：intermediate=True 生成完整全图语义图（论文要用）
#     predict(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True, intermediate=True)    
#     time_use = time.time() - begin_time
#     print('Total time: %.2fs'%time_use)

# #For models with 4 outputs: change + semA + semB + edge
# def predict(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False, intermediate=False):
#     pred_A_dir_rgb = os.path.join(pred_dir, 'im1_rgb')
#     pred_B_dir_rgb = os.path.join(pred_dir, 'im2_rgb')
#     if not os.path.exists(pred_A_dir_rgb): os.makedirs(pred_A_dir_rgb)
#     if not os.path.exists(pred_B_dir_rgb): os.makedirs(pred_B_dir_rgb)
#     if index_map:
#         pred_A_dir = os.path.join(pred_dir, 'im1')
#         pred_B_dir = os.path.join(pred_dir, 'im2')
#         if not os.path.exists(pred_A_dir): os.makedirs(pred_A_dir)
#         if not os.path.exists(pred_B_dir): os.makedirs(pred_B_dir)
#     if intermediate:
#         pred_mA_dir = os.path.join(pred_dir, 'im1_semantic')
#         pred_mB_dir = os.path.join(pred_dir, 'im2_semantic')
#         pred_change_dir = os.path.join(pred_dir, 'change')
#         if not os.path.exists(pred_mA_dir): os.makedirs(pred_mA_dir)
#         if not os.path.exists(pred_mB_dir): os.makedirs(pred_mB_dir)
#         if not os.path.exists(pred_change_dir): os.makedirs(pred_change_dir)
    
#     for vi, data in enumerate(pred_loader):
#         imgs_A, imgs_B = data
#         imgs_A = imgs_A.cuda().float()
#         imgs_B = imgs_B.cuda().float()
#         mask_name = pred_set.get_mask_name(vi)
#         with torch.no_grad(): 
#             # 修复：丢弃第四个edge输出，匹配4输出模型
#             out_change, outputs_A, outputs_B, _ = net(imgs_A, imgs_B)
#             out_change = F.sigmoid(out_change)
#         if flip:
#             outputs_A = F.softmax(outputs_A, dim=1)
#             outputs_B = F.softmax(outputs_B, dim=1)
            
#             imgs_A_v = torch.flip(imgs_A, [2])
#             imgs_B_v = torch.flip(imgs_B, [2])
#             out_change_v, outputs_A_v, outputs_B_v, _ = net(imgs_A_v, imgs_B_v)
#             outputs_A_v = torch.flip(outputs_A_v, [2])
#             outputs_B_v = torch.flip(outputs_B_v, [2])
#             out_change_v = torch.flip(F.sigmoid(out_change_v), [2])
#             outputs_A += F.softmax(outputs_A_v, dim=1)
#             outputs_B += F.softmax(outputs_B_v, dim=1)
#             out_change += out_change_v
            
#             imgs_A_h = torch.flip(imgs_A, [3])
#             imgs_B_h = torch.flip(imgs_B, [3])
#             out_change_h, outputs_A_h, outputs_B_h, _ = net(imgs_A_h, imgs_B_h)
#             outputs_A_h = torch.flip(outputs_A_h, [3])
#             outputs_B_h = torch.flip(outputs_B_h, [3])
#             out_change_h = torch.flip(F.sigmoid(out_change_h), [3])
#             outputs_A += F.softmax(outputs_A_h, dim=1)
#             outputs_B += F.softmax(outputs_B_h, dim=1)
#             out_change += out_change_h
            
#             imgs_A_hv = torch.flip(imgs_A, [2,3])
#             imgs_B_hv = torch.flip(imgs_B, [2,3])
#             out_change_hv, outputs_A_hv, outputs_B_hv, _ = net(imgs_A_hv, imgs_B_hv)
#             outputs_A_hv = torch.flip(outputs_A_hv, [2,3])
#             outputs_B_hv = torch.flip(outputs_B_hv, [2,3])
#             out_change_hv = torch.flip(F.sigmoid(out_change_hv), [2,3])
#             outputs_A += F.softmax(outputs_A_hv, dim=1)
#             outputs_B += F.softmax(outputs_B_hv, dim=1)
#             out_change += out_change_hv
#             out_change = out_change/4

#         outputs_A = outputs_A.cpu().detach()
#         outputs_B = outputs_B.cpu().detach()
#         change_mask = out_change.cpu().detach()>0.5
#         change_mask = change_mask.squeeze()
#         pred_A = torch.argmax(outputs_A, dim=1).squeeze()
#         pred_B = torch.argmax(outputs_B, dim=1).squeeze()
        
#         if intermediate:
#             pred_A_path = os.path.join(pred_mA_dir, mask_name)
#             pred_B_path = os.path.join(pred_mB_dir, mask_name)
#             pred_change_path = os.path.join(pred_change_dir, mask_name)
#             # 无掩码完整全图，论文标准图
#             io.imsave(pred_A_path, RS.Index2Color(pred_A.numpy()))
#             io.imsave(pred_B_path, RS.Index2Color(pred_B.numpy()))
#             change_map = exposure.rescale_intensity(change_mask.numpy(), 'image', 'dtype')
#             io.imsave(pred_change_path, change_map)
#         # 乘掩码，仅变化区域显色（空白多）
#         pred_A = (pred_A*change_mask.long()).numpy()
#         pred_B = (pred_B*change_mask.long()).numpy()      
#         pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
#         pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)  
#         io.imsave(pred_A_path, RS.Index2Color(pred_A))
#         io.imsave(pred_B_path, RS.Index2Color(pred_B))   
#         print(pred_A_path)
#         if index_map:
#             pred_A_path = os.path.join(pred_A_dir, mask_name)
#             pred_B_path = os.path.join(pred_B_dir, mask_name)
#             io.imsave(pred_A_path, pred_A.astype(np.uint8))
#             io.imsave(pred_B_path, pred_B.astype(np.uint8))  

# def predict_direct(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False,):
#     pred_A_dir_rgb = os.path.join(pred_dir, 'im1_rgb')
#     pred_B_dir_rgb = os.path.join(pred_dir, 'im2_rgb')
#     if not os.path.exists(pred_A_dir_rgb): os.makedirs(pred_A_dir_rgb)
#     if not os.path.exists(pred_B_dir_rgb): os.makedirs(pred_B_dir_rgb)
#     if index_map:
#         pred_A_dir = os.path.join(pred_dir, 'im1')
#         pred_B_dir = os.path.join(pred_dir, 'im2')
#         if not os.path.exists(pred_A_dir): os.makedirs(pred_A_dir)
#         if not os.path.exists(pred_B_dir): os.makedirs(pred_B_dir)
    
#     for vi, data in enumerate(pred_loader):
#         imgs_A, imgs_B = data
#         imgs_A = imgs_A.cuda().float()
#         imgs_B = imgs_B.cuda().float()
#         mask_name = pred_set.get_mask_name(vi)
#         with torch.no_grad(): 
#             outputs_A, outputs_B = net(imgs_A, imgs_B)
#         if flip:
#             outputs_A = F.softmax(outputs_A, dim=1)
#             outputs_B = F.softmax(outputs_B, dim=1)
            
#             imgs_A_v = torch.flip(imgs_A, [2])
#             imgs_B_v = torch.flip(imgs_B, [2])
#             outputs_A_v, outputs_B_v = net(imgs_A_v, imgs_B_v)
#             outputs_A_v = torch.flip(outputs_A_v, [2])
#             outputs_B_v = torch.flip(outputs_B_v, [2])
#             outputs_A += F.softmax(outputs_A_v, dim=1)
#             outputs_B += F.softmax(outputs_B_v, dim=1)
            
#             imgs_A_h = torch.flip(imgs_A, [3])
#             imgs_B_h = torch.flip(imgs_B, [3])
#             outputs_A_h, outputs_B_h = net(imgs_A_h, imgs_B_h)
#             outputs_A_h = torch.flip(outputs_A_h, [3])
#             outputs_B_h = torch.flip(outputs_B_h, [3])
#             outputs_A += F.softmax(outputs_A_h, dim=1)
#             outputs_B += F.softmax(outputs_B_h, dim=1)
            
#             imgs_A_hv = torch.flip(imgs_A, [2,3])
#             imgs_B_hv = torch.flip(imgs_B, [2,3])
#             outputs_A_hv, outputs_B_hv = net(imgs_A_hv, imgs_B_hv)
#             outputs_A_hv = torch.flip(outputs_A_hv, [2,3])
#             outputs_B_hv = torch.flip(outputs_B_hv, [2,3])
#             outputs_A += F.softmax(outputs_A_hv, dim=1)
#             outputs_B += F.softmax(outputs_B_hv, dim=1)
            
#         outputs_A = outputs_A.cpu().detach()
#         outputs_B = outputs_B.cpu().detach()
#         pred_A = torch.argmax(outputs_A, dim=1)
#         pred_B = torch.argmax(outputs_B, dim=1)
#         pred_A = pred_A.squeeze().numpy().astype(np.uint8)
#         pred_B = pred_B.squeeze().numpy().astype(np.uint8)
        
#         pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
#         pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)
#         io.imsave(pred_A_path, RS.Index2Color(pred_A))
#         print(pred_A_path)
#         if index_map:
#             pred_A_path = os.path.join(pred_A_dir, mask_name)
#             pred_B_path = os.path.join(pred_B_dir, mask_name)
#             io.imsave(pred_A_path, pred_A.astype(np.uint8))
#             io.imsave(pred_B_path, pred_B.astype(np.uint8))

# if __name__ == '__main__':
#     main()
    
    
    
    
    
'''
else vis
'''

import os
import time
import argparse
import numpy as np
import torch
from skimage import io, exposure
from torch.nn import functional as F
from torch.utils.data import DataLoader
#################################
from datasets import RS_Landsat as RS
from SemDINO import SemDINO
from models.TBFFNet import TBFFNet
from models.BiSRNet import BiSRNet
from models.TED import TED
from models.SSCDl import SSCDl
from models.SCanNet import SCanNet as Net
from models.HRSCD4 import HRSCD4
from SCGNet import SCGNet
DATA_NAME = 'Landsat'
#################################

class PredOptions():
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', required=False, default=1, help='prediction batch size')
        parser.add_argument('--test_dir', required=False, default='/root/autodl-fs/AAAI-SemDINO/LandsatSCD/test', help='directory to test images')
        parser.add_argument('--pred_dir', required=False, default='/root/autodl-fs/visvisvisLandsat', help='directory to output masks')
        parser.add_argument('--chkpt_path', required=False, default='/root/autodl-fs/AAAI-SemDINO/checkpoints2/Landsat/SCGNet/SCGNet_98e_mIoU85.87_Sek51.44_Fscd85.46_OA95.12.pth')
        self.initialized = True
        return parser
        
    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt
        
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')        

# def main():
#     begin_time = time.time()
#     opt = PredOptions().parse()
#     net = SCGNet(3, RS.num_classes).cuda()
#     net.load_state_dict( torch.load(opt.chkpt_path) )
#     net.eval()
    
#     test_set = RS.Data_test(opt.test_dir)
#     test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
#     predict(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True, intermediate=False)    
#     #predict_direct(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True)
#     time_use = time.time() - begin_time
#     print('Total time: %.2fs'%time_use)

def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = SCGNet(3, RS.num_classes).cuda()
    # 处理多卡module前缀
    ckpt = torch.load(opt.chkpt_path)
    new_state = {}
    for key, val in ckpt.items():
        if key.startswith("module."):
            new_state[key[7:]] = val
        else:
            new_state[key] = val
    net.load_state_dict(new_state)
    net.eval()
    
    test_set = RS.Data_test(opt.test_dir)
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    predict(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True, intermediate=False)    
    time_use = time.time() - begin_time
    print('Total time: %.2fs'%time_use)

#For models with 3 outputs: 1 change map + 2 semantic maps.
#Parameters: flip->test time augmentation     index_map->"False" means rgb results      intermediate->whether to outputs the intermediate maps
def predict(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False, intermediate=False):
    pred_A_dir_rgb = os.path.join(pred_dir, 'im1_rgb')
    pred_B_dir_rgb = os.path.join(pred_dir, 'im2_rgb')
    if not os.path.exists(pred_A_dir_rgb): os.makedirs(pred_A_dir_rgb)
    if not os.path.exists(pred_B_dir_rgb): os.makedirs(pred_B_dir_rgb)
    if index_map:
        pred_A_dir = os.path.join(pred_dir, 'im1')
        pred_B_dir = os.path.join(pred_dir, 'im2')
        if not os.path.exists(pred_A_dir): os.makedirs(pred_A_dir)
        if not os.path.exists(pred_B_dir): os.makedirs(pred_B_dir)
    if intermediate:
        pred_mA_dir = os.path.join(pred_dir, 'im1_semantic')
        pred_mB_dir = os.path.join(pred_dir, 'im2_semantic')
        pred_change_dir = os.path.join(pred_dir, 'change')
        if not os.path.exists(pred_mA_dir): os.makedirs(pred_mA_dir)
        if not os.path.exists(pred_mB_dir): os.makedirs(pred_mB_dir)
        if not os.path.exists(pred_change_dir): os.makedirs(pred_change_dir)
    
    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B = data
        #imgs = torch.cat([imgs_A, imgs_B], 1)
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        mask_name = pred_set.get_mask_name(vi)
        with torch.no_grad(): 
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)#,aux
            out_change = F.sigmoid(out_change)
        if flip:
            outputs_A = F.softmax(outputs_A, dim=1)
            outputs_B = F.softmax(outputs_B, dim=1)
            
            imgs_A_v = torch.flip(imgs_A, [2])
            imgs_B_v = torch.flip(imgs_B, [2])
            out_change_v, outputs_A_v, outputs_B_v = net(imgs_A_v, imgs_B_v)
            outputs_A_v = torch.flip(outputs_A_v, [2])
            outputs_B_v = torch.flip(outputs_B_v, [2])
            out_change_v = torch.flip(out_change_v, [2])
            outputs_A += F.softmax(outputs_A_v, dim=1)
            outputs_B += F.softmax(outputs_B_v, dim=1)
            out_change += F.sigmoid(out_change_v)
            
            imgs_A_h = torch.flip(imgs_A, [3])
            imgs_B_h = torch.flip(imgs_B, [3])
            out_change_h, outputs_A_h, outputs_B_h = net(imgs_A_h, imgs_B_h)
            outputs_A_h = torch.flip(outputs_A_h, [3])
            outputs_B_h = torch.flip(outputs_B_h, [3])
            out_change_h = torch.flip(out_change_h, [3])
            outputs_A += F.softmax(outputs_A_h, dim=1)
            outputs_B += F.softmax(outputs_B_h, dim=1)
            out_change += F.sigmoid(out_change_h)
            
            imgs_A_hv = torch.flip(imgs_A, [2,3])
            imgs_B_hv = torch.flip(imgs_B, [2,3])
            out_change_hv, outputs_A_hv, outputs_B_hv = net(imgs_A_hv, imgs_B_hv)
            outputs_A_hv = torch.flip(outputs_A_hv, [2,3])
            outputs_B_hv = torch.flip(outputs_B_hv, [2,3])
            out_change_hv = torch.flip(out_change_hv, [2,3])
            outputs_A += F.softmax(outputs_A_hv, dim=1)
            outputs_B += F.softmax(outputs_B_hv, dim=1)
            out_change += F.sigmoid(out_change_hv)
            out_change = out_change/4
                        
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = out_change.cpu().detach()>0.5
        change_mask = change_mask.squeeze()
        pred_A = torch.argmax(outputs_A, dim=1).squeeze()
        pred_B = torch.argmax(outputs_B, dim=1).squeeze()
        
        if intermediate:
            pred_A_path = os.path.join(pred_mA_dir, mask_name)
            pred_B_path = os.path.join(pred_mB_dir, mask_name)
            pred_change_path = os.path.join(pred_change_dir, mask_name)
            io.imsave(pred_A_path, RS.Index2Color(pred_A.numpy()))
            io.imsave(pred_B_path, RS.Index2Color(pred_B.numpy()))
            change_map = exposure.rescale_intensity(change_mask.numpy(), 'image', 'dtype')
            io.imsave(pred_change_path, change_map)
        pred_A = (pred_A*change_mask.long()).numpy()
        pred_B = (pred_B*change_mask.long()).numpy()      
        pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
        pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)  
        io.imsave(pred_A_path, RS.Index2Color(pred_A))
        io.imsave(pred_B_path, RS.Index2Color(pred_B))   
        print(pred_A_path)
        if index_map:
            pred_A_path = os.path.join(pred_A_dir, mask_name)
            pred_B_path = os.path.join(pred_B_dir, mask_name)
            io.imsave(pred_A_path, pred_A.astype(np.uint8))
            io.imsave(pred_B_path, pred_B.astype(np.uint8))  

#For models that directly produce 2 SCD maps.
#Parameters: flip->test time augmentation     index_map->"False" means rgb results
def predict_direct(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False,):
    pred_A_dir_rgb = os.path.join(pred_dir, 'im1_rgb')
    pred_B_dir_rgb = os.path.join(pred_dir, 'im2_rgb')
    if not os.path.exists(pred_A_dir_rgb): os.makedirs(pred_A_dir_rgb)
    if not os.path.exists(pred_B_dir_rgb): os.makedirs(pred_B_dir_rgb)
    if index_map:
        pred_A_dir = os.path.join(pred_dir, 'im1')
        pred_B_dir = os.path.join(pred_dir, 'im2')
        if not os.path.exists(pred_A_dir): os.makedirs(pred_A_dir)
        if not os.path.exists(pred_B_dir): os.makedirs(pred_B_dir)
    
    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B = data
        #imgs = torch.cat([imgs_A, imgs_B], 1)
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        mask_name = pred_set.get_mask_name(vi)
        with torch.no_grad(): 
            outputs_A, outputs_B = net(imgs_A, imgs_B)#,aux
        if flip:
            outputs_A = F.softmax(outputs_A, dim=1)
            outputs_B = F.softmax(outputs_B, dim=1)
            
            imgs_A_v = torch.flip(imgs_A, [2])
            imgs_B_v = torch.flip(imgs_B, [2])
            outputs_A_v, outputs_B_v = net(imgs_A_v, imgs_B_v)
            outputs_A_v = torch.flip(outputs_A_v, [2])
            outputs_B_v = torch.flip(outputs_B_v, [2])
            outputs_A += F.softmax(outputs_A_v, dim=1)
            outputs_B += F.softmax(outputs_B_v, dim=1)
            
            imgs_A_h = torch.flip(imgs_A, [3])
            imgs_B_h = torch.flip(imgs_B, [3])
            outputs_A_h, outputs_B_h = net(imgs_A_h, imgs_B_h)
            outputs_A_h = torch.flip(outputs_A_h, [3])
            outputs_B_h = torch.flip(outputs_B_h, [3])
            outputs_A += F.softmax(outputs_A_h, dim=1)
            outputs_B += F.softmax(outputs_B_h, dim=1)
            
            imgs_A_hv = torch.flip(imgs_A, [2,3])
            imgs_B_hv = torch.flip(imgs_B, [2,3])
            outputs_A_hv, outputs_B_hv = net(imgs_A_hv, imgs_B_hv)
            outputs_A_hv = torch.flip(outputs_A_hv, [2,3])
            outputs_B_hv = torch.flip(outputs_B_hv, [2,3])
            outputs_A += F.softmax(outputs_A_hv, dim=1)
            outputs_B += F.softmax(outputs_B_hv, dim=1)
            
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        pred_A = torch.argmax(outputs_A, dim=1)
        pred_B = torch.argmax(outputs_B, dim=1)
        pred_A = pred_A.squeeze().numpy().astype(np.uint8)
        pred_B = pred_B.squeeze().numpy().astype(np.uint8)
        
        pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
        pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)
        io.imsave(pred_A_path, RS.Index2Color(pred_A))
        io.imsave(pred_B_path, RS.Index2Color(pred_B))
        print(pred_A_path)
        if index_map:
            pred_A_path = os.path.join(pred_A_dir, mask_name)
            pred_B_path = os.path.join(pred_B_dir, mask_name)
            io.imsave(pred_A_path, pred_A.astype(np.uint8))
            io.imsave(pred_B_path, pred_B.astype(np.uint8))
        '''
        change_path = os.path.join(pred_dir, 'change', mask_name)
        io.imsave(change_path, (change_mask*255).astype(np.uint8))'''

if __name__ == '__main__':
    main()
