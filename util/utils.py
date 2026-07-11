import numpy as np
import random
import torch
import torch.nn.functional as F
from common import utils
from common.logger import  AverageMeter
from common.evaluation import Evaluator
import numpy as np

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

def count_trainable_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params / 1e6

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def Compute_iou(model, dataloader, nshot):
    utils.fix_randseed(0)
    device = next(model.parameters()).device
    average_meter = AverageMeter(dataloader.dataset, device)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch, device=device)

        sup_rgb = batch['support_imgs'][0][0].unsqueeze(0)     # [1, 3, H, W]
        sup_msk = batch['support_masks'][0][0].unsqueeze(0)    # [1, H, W]
        qry_rgb = batch['query_img'][0].unsqueeze(0)           # [1, 3, H, W]
        qry_msk = batch['query_mask'][0].unsqueeze(0)          # [1, H, W]

        pred = model([sup_rgb], [sup_msk], qry_rgb, qry_msk)[0]  # [1, 2, H, W]
        pred_mask = torch.argmax(pred, dim=1)  # [1, H, W]

        assert pred_mask.size() == qry_msk.size()

        # Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou

class mIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return np.nanmean(iu[1:])