import zetane
import glob
from PIL import Image
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import save_image
import zetane as ztn
import zetane.utils as zutils
import zetane.ZetaneViz as ZetaneViz

def main():
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

    testset = datasets.CocoDetection(root="./cocoapi/images", annFile="./cocoapi/annotations/instances_val2017.json", transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

    resnet = models.resnet18(pretrained=True)

    criterion = nn.CrossEntropyLoss()

    validate(testloader, resnet, criterion)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    viz = ZetaneViz()

    # switch to evaluate mode
    model.eval()
    viz.create_model(model, torch.rand((1,3,224,224)))

    for i, (inputs, target) in enumerate(val_loader):
        target = target[0]['category_id']

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        # viz
        viz.show_input(inputs)
        viz.model_inference(inputs)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
