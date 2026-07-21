import os
import time
import argparse
import numpy as np
import torch
import torch.autograd
from skimage import io
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import accuracy, SCDD_eval_all, AverageMeter

#################################
from datasets import RS_Landsat as RS
from SemDINO import SemDINO as Net
DATA_NAME = 'Landsat'
#################################

class PredOptions():
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', required=False, default=2, help='prediction batch size')
        parser.add_argument('--test_dir', required=False, default='~/LandsatSCD/test')
        parser.add_argument('--chkpt_path', required=False, default='~.pth')
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

def main():
    begin_time = time.time()
    opt = PredOptions().parse()

    net = Net(num_classes=RS.num_classes, dim=128).cuda()
    ckpt = torch.load(opt.chkpt_path, map_location='cpu')

    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            new_ckpt[k[7:]] = v
        else:
            new_ckpt[k] = v

    net.load_state_dict(new_ckpt)
    net.eval()
    
    test_set = RS.Data(opt.test_dir, random_flip=False)
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    validate(test_loader, net)
    
    print('Total time: %.2fs' % (time.time() - begin_time))


def validate(val_loader, net):
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()
    acc_meter = AverageMeter()
    preds_all = []
    labels_all = []

    for vi, data in enumerate(tqdm(val_loader)):
        imgs_A, imgs_B, labels_A, labels_B = data
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        labels_A = labels_A.cuda().long()
        labels_B = labels_B.cuda().long()

        with torch.no_grad():

            out_change, outputs_A, outputs_B, _ = net(imgs_A, imgs_B)
            
       
            imgA_flip = torch.flip(imgs_A, dims=[3])
            imgB_flip = torch.flip(imgs_B, dims=[3])
            _, outA_flip, outB_flip, _ = net(imgA_flip, imgB_flip)
            

            outA_flip = torch.flip(outA_flip, dims=[3])
            outB_flip = torch.flip(outB_flip, dims=[3])

            outputs_A = (outputs_A + outA_flip) / 2.0
            outputs_B = (outputs_B + outB_flip) / 2.0

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        change_mask = torch.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1).cpu().numpy()
        preds_B = torch.argmax(outputs_B, dim=1).cpu().numpy()
        preds_A = (preds_A * change_mask.squeeze().long().numpy())
        preds_B = (preds_B * change_mask.squeeze().long().numpy())

        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, _ = accuracy(pred_A, label_A)
            acc_B, _ = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc_meter.update((acc_A + acc_B) * 0.5)
    
    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)
    print('%.1fs  Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f'
          % (time.time() - start, Fscd*100, IoU_mean*100, Sek*100, acc_meter.average()*100))
    return Fscd, IoU_mean, Sek, acc_meter.avg

if __name__ == '__main__':
    main()
