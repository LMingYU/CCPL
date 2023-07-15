from collections import defaultdict
import torch


class FeatureQueue_ori:

    def __init__(self, opt, classwise_max_size=None, bal_queue=False):
        self.num_classes = opt.num_classes
        self.feat_dim = opt.feat_num
        self.max_size = opt.max_size

        device = 'cuda'
        self._bank = defaultdict(lambda: torch.empty(0, self.feat_dim).to(device))
        self.prototypes = torch.zeros(self.num_classes, self.feat_dim).to(device)

        self.classwise_max_size = classwise_max_size
        self.bal_queue = bal_queue

    def enqueue(self, features: torch.Tensor, labels: torch.Tensor):
        for idx in range(self.num_classes):
            # per class max size
            max_size = (
                self.classwise_max_size[idx] * 5  # 5x samples
            ) if self.classwise_max_size is not None else self.max_size
            if self.bal_queue:
                max_size = self.max_size
            # select features by label
            cls_inds = torch.where(labels == idx)[0]
            if len(cls_inds):
                with torch.no_grad():
                    # push to the memory bank
                    feats_selected = features[cls_inds]
                    self._bank[idx] = torch.cat([self._bank[idx], feats_selected], 0)

                    # fixed size
                    current_size = len(self._bank[idx])
                    if current_size > max_size:
                        self._bank[idx] = self._bank[idx][current_size - max_size:]

                    # update prototypes
                    self.prototypes[idx, :] = self._bank[idx].mean(0)


class FeatureQueue:

    def __init__(self, opt, classwise_max_size=None, bal_queue=False):
        self.num_classes = opt.num_classes
        self.feat_dim = opt.feat_num
        self.max_size = opt.max_size

        device = 'cuda'
        self._bank = defaultdict(lambda: torch.empty(0, self.feat_dim).to(device))
        self.prototypes = torch.zeros(self.num_classes, self.feat_dim).to(device)
        self.meansim = torch.zeros(self.num_classes).to(device)

        self.classwise_max_size = classwise_max_size
        self.bal_queue = bal_queue
        self.Tp=opt.Tp

    def enqueue(self, features: torch.Tensor, labels: torch.Tensor):
        for idx in range(self.num_classes):
            # per class max size
            max_size = (
                self.classwise_max_size[idx] * 5  # 5x samples
            ) if self.classwise_max_size is not None else self.max_size
            if self.bal_queue:
                max_size = self.max_size
            # select features by label
            cls_inds = torch.where(labels == idx)[0]
            if len(cls_inds):
                with torch.no_grad():
                    # push to the memory bank
                    feats_selected = features[cls_inds]
                    self._bank[idx] = torch.cat([self._bank[idx], feats_selected], 0)

                    # fixed size
                    current_size = len(self._bank[idx])
                    if current_size > max_size:
                        self._bank[idx] = self._bank[idx][current_size - max_size:]

                    # update prototypes
                    self.prototypes[idx, :] = self._bank[idx].mean(0)

                    sim = torch.div(
                        torch.matmul(self._bank[idx], self.prototypes[idx]),
                        self.Tp)
                    sim=torch.exp(sim)

                    sim=sim.mean(0)
                    #print("sim1",sim.shape)

                    self.meansim[idx]=sim


