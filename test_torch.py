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
    viz.create_torch_model(model, torch.rand((1,3,224,224)))

    for i, (inputs, target) in enumerate(val_loader):
        target = target[0]['category_id']
        end = time.time()

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

class ZetaneViz(object):
    def __init__(self):
        self.context = ztn.Context()
        self.iimage = self.context.image().position(-10.0, 2.5, 0.0).scale(0.25, 0.25, 0.0)
        self.vmodel = self.context.model()
        self.context.clear_universe()
    def show_input(self, inputs):
        # move NCWH to WHC
        np_image = np.moveaxis(inputs[0,:,:,:].detach().cpu().numpy(), 0, -1)
        remapped = zutils.remap(np_image)
        self.iimage.position(-10.0, 2.5, 0.0).scale(0.25, 0.25, 0.0).update(data=remapped)
    def create_torch_model(self, model, inputs):
        import torch
        self.vmodel.torch(model, inputs).update()
    def model_inference(self, inputs):
        self.vmodel.inputs(inputs.detach().cpu().numpy()).update()
        time.sleep(2)
    def model_debug(self, inputs):
        self.vmodel.inputs(inputs.detach().cpu().numpy()).debug()

if __name__ == '__main__':
    main()
