'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter','test2','test2mn','test2100']
if torch.cuda.is_available():
    print("cuda")
    device = "cuda"
    torch.backends.cudnn.benckmark = True
else:
    device = "cpu"

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)


    _, pred = output.topk(maxk, 1, True, True)


    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)

        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test2(model, test_loader,args):
    with torch.no_grad():
        model.eval()
        correct = 0.
        tot = 0.

        numofWU=[]
        targetofWU=[]
        for i in range(6):
            numofWU.append(0)
            targetofWU.append(0)
        wu_weight=[]
        for i in range(6):
            wu_weight.append(0)
        for i, data in enumerate(test_loader):
            #(_,_,images), labels,_ = data
            images, labels= data
            # images=pre_process2(images)

            

            images = images.to(device).float()
            labels = labels.to(device).long()

            

            _,out = model(images)
            #out = model(images)
            pred_label = out.max(1)[1]
            correct += (pred_label == labels).float().sum()
            tot += pred_label.size(0)


            

            labels=labels.to(device).long()
            index=labels.view(-1,1)
            #print(index.shape)
            for i in range(len(index)):
                m=labels[i]
                numofWU[m]=numofWU[m]+1
                if m==pred_label[i]:
                    targetofWU[m]=targetofWU[m]+1
            
            for i in range(len(numofWU)):
                if numofWU[i]>0:
                    wu_weight[i]=targetofWU[i]/numofWU[i]
                


            
        acc = correct / tot
        
        return acc,numofWU,wu_weight


def test2mn(model, test_loader, args):
    with torch.no_grad():
        model.eval()
        correct = 0.
        tot = 0.

        numofWU = []
        targetofWU = []
        for i in range(6):
            numofWU.append(0)
            targetofWU.append(0)
        wu_weight = []
        for i in range(6):
            wu_weight.append(0)
        for i, data in enumerate(test_loader):
            (_, _, images), labels = data
            # images, labels= data
            # images=pre_process2(images)

            images = images.to(device).float()
            labels = labels.to(device).long()


            out = model(images)
            pred_label = out.max(1)[1]
            correct += (pred_label == labels).float().sum()
            tot += pred_label.size(0)

            labels = labels.to(device).long()
            index = labels.view(-1, 1)
            # print(index.shape)
            for i in range(len(index)):
                m = labels[i]
                numofWU[m] = numofWU[m] + 1
                if m == pred_label[i]:
                    targetofWU[m] = targetofWU[m] + 1

            for i in range(len(numofWU)):
                if numofWU[i] > 0:
                    wu_weight[i] = targetofWU[i] / numofWU[i]

        acc = correct / tot

        return acc, numofWU, wu_weight


def test2100(model, test_loader):
    with torch.no_grad():
        model.eval()
        correct = 0.
        tot = 0.

        numofWU = []
        targetofWU = []
        for i in range(60):
            numofWU.append(0)
            targetofWU.append(0)
        wu_weight = []
        for i in range(60):
            wu_weight.append(0)
        for i, data in enumerate(test_loader):
            images, labels = data
            # images=pre_process2(images)

            images = images.to(device).float()
            labels = labels.to(device).long()

            # print("C(G(image))")

            out = model(images)
            pred_label = out.max(1)[1]
            correct += (pred_label == labels).float().sum()
            tot += pred_label.size(0)

            labels = labels.to(device).long()
            index = labels.view(-1, 1)
            # print(index.shape)
            for i in range(len(index)):
                m = labels[i]
                numofWU[m] = numofWU[m] + 1
                if m == pred_label[i]:
                    targetofWU[m] = targetofWU[m] + 1

            for i in range(len(numofWU)):
                if numofWU[i] > 0:
                    wu_weight[i] = targetofWU[i] / numofWU[i]

        acc = correct / tot

        return acc, numofWU, wu_weight


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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
