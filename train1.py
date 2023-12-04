import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from cifar import DATASET_GETTERS
from wrn_d_SVHN import WRN_C


from utils import AverageMeter, accuracy,test2
from feature_queue import FeatureQueue

import time
import torch.optim.lr_scheduler as lr_scheduler

import warnings
warnings.filterwarnings("ignore", "(?s).*MATPLOTLIBDATA.*", category=UserWarning)
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--root', default='./data', type=str,
                        help='path to data directory')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--dataset', default='SVHN', type=str,
                        choices=['cifar10', 'cifar100','SVHN','mnist'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=100,
                        help='number of labeled data')
    parser.add_argument("--n_unlabels", "-u", default=20000, type=int, help="the number of unlabeled data")
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=100, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.09, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema_decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--num_classes', type=int, default=6,
                        help='id for recording multiple runs')
    parser.add_argument('--feat_num', type=str, default=64,
                        help='id for recording multiple runs')
    parser.add_argument('--max_size', type=str, default=500,
                        help='id for recording multiple runs')
    parser.add_argument('--Tp', type=str, default=0.95,
                        help='id for recording multiple runs')
    '''parser.add_argument('--num-labeled', type=int, default=400,  # 4000
                        help='number of labeled data')'''

    args = parser.parse_args()
    global best_acc

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnetptl(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        elif args.arch == 'CNN':
                from wideresnet import WideResNet, CNN, WNet
                model=CNN(n_out=args.num_classes).to(device)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model



    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)

    if args.dataset == 'cifar10':
        args.num_classes = 6
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    elif args.dataset == 'SVHN':
        args.num_classes = 6
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    elif args.dataset == 'mnist':
        args.num_classes = 6
        args.arch='CNN'

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')
    logger.info(f"labeled: {len(labeled_dataset)}, "
        f"unlabeled: {len(unlabeled_dataset)}",)
    

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    labeled_loader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=2400,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)
    a = sum([param.nelement() for param in model.parameters()])
    print("a", a)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov, weight_decay=0.01)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.9,patience=1)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    queue = FeatureQueue(args, classwise_max_size=None, bal_queue=True)

    pretrain(args, labeled_trainloader,labeled_loader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler,queue)
    
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)

def losscon3(args,f1,f2,proto):
    device='cuda'
    anchor_dot_contrast = torch.div(
        torch.matmul(f1, f2.T),
        args.Tp)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.clone().detach()

    mask = torch.eye(f1.shape[0], dtype=torch.float32).to(device)


    logits=logits*mask
    logits=logits.sum(1,keepdim=True)

    proto=proto.clone().detach()
    sim=torch.div(
        torch.matmul(f1, proto.T),
        args.Tp)
    sim_max,a=torch.max(sim,dim=1,keepdim=True)
    logit_sim=torch.exp(sim-sim_max.clone().detach())
    zero_sim=torch.zeros_like(logit_sim)
    zero=zero_sim.scatter_(1,a,1)
    logit_simmax=zero*logit_sim

    
    logit_sum=torch.sum(logit_sim-logit_simmax,dim=1,keepdim=True)

   

    logits1 = -torch.log(torch.exp(logits) / ( logit_sum))
    logits2=logits1.mean()

    return logits2


def lossconl3(args,f1,tar,proto):
    device='cuda'
    torch.set_grad_enabled(True)
    proto=proto.clone().detach()
    sim=torch.div(
        torch.matmul(f1, proto.T),
        args.Tp)
    sim_max,a=torch.max(sim,dim=1,keepdim=True)
    logit_sim=torch.exp(sim-sim_max.clone().detach())
    zero_sim=torch.zeros_like(logit_sim)
    
    tar=tar.reshape(-1,1)
    zero=zero_sim.scatter_(1,tar,1)
    logit_true=torch.sum(zero*logit_sim,dim=1,keepdim=True)
    logit_sum=torch.sum(logit_sim-zero*logit_sim,dim=1,keepdim=True)
    

    logits1 = -torch.log(logit_true)+ torch.log(logit_sum)
    logits2=logits1.mean()


    return logits2


def losspri3(args,logits_x,tl):
    device='cuda'

    logits_x=torch.softmax(logits_x/2, dim=-1)
    sim_max,a=torch.max(logits_x,dim=1,keepdim=True)
    tl_max,_=torch.max(tl,dim=0,keepdim=True)
    tl=tl/tl_max
    tl_w=1-tl
    tl_w=tl_w.clone().detach()
    Lpri=0.0

    for i in range(len(sim_max)):
        if sim_max[i]==1:
            sim_max[i]=sim_max[i]-(1e-3)
        if sim_max[i]>1:
            sim_max[i]=1-(1e-3)
        Lpri+=tl_w[a[i].data.cpu()]*torch.log(1-sim_max[i])
    Lpri=-1*Lpri/(len(sim_max))



    return Lpri


def pretrain(args, labeled_trainloader, labeled_loader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler,queue):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    
    model.zero_grad
    model.train()
    if args.use_ema:
        print("ema")
        test_model = ema_model.ema
    else:
        test_model = model
    test_loss, test_acc = test(args, test_loader, test_model, 0)
    print("test/1.test_acc:{}".format(test_acc))
    iteration=0
    for epoch in range(args.start_epoch, args.epochs):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_pri=AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            aa = time.time()

            try:
                inputs_x, targets_x,_ = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x,_ = labeled_iter.next()

            if args.dataset == 'mnist':
                inputs_x = inputs_x.unsqueeze(1)
            try:
                (inputs_u_w, inputs_u_s,_), _,_ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s,_), _ ,_= unlabeled_iter.next()
            if args.dataset == 'mnist':
                inputs_u_w = inputs_u_w.unsqueeze(1)
                inputs_u_s=inputs_u_s.unsqueeze(1)
            
            coef =  math.exp(-6 * (1 - min(iteration/200000,1)) ** 2)
            coef2 =  math.exp(-6 * (1 - min(iteration/10000,1)) ** 2)
            coef3 =  math.exp(-6 * (1 - min(iteration/2000,1)) ** 2)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device).long()
            feat, logits = model(inputs.float())
            feat_ema, logits_ema = ema_model.ema(inputs.float())
            featw_ema = feat_ema[len(inputs_x):len(inputs_x) + len(inputs_u_w)]
            feats_ema = feat_ema[len(inputs_x) + len(inputs_u_w):]
            featll_ema= feat_ema[:len(inputs_x)]
            featl, logitl = ema_model.ema(inputs_x.to(args.device,torch.float))
            feat=feat
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits,logitl

            featll_ema=featll_ema.clone().detach()
            featw_ema=featw_ema.clone().detach()
            queue.enqueue(featll_ema.clone().detach(), targets_x.clone().detach())
            del featl
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()


            Lu0 = (F.cross_entropy(logits_u_s, targets_u,reduction='none') * mask).mean()
            featw = feat[len(inputs_x):len(inputs_x) + len(inputs_u_w)]
            feats = feat[len(inputs_x) + len(inputs_u_w):]
            featll= feat[:len(inputs_x)]
            del feat

            simtem = torch.div(
                torch.matmul(featw, queue.prototypes.T),
                args.Tp)
            simtem=torch.exp(simtem)
            mean=queue.meansim
            simmax, a = torch.max(simtem, dim=1, keepdim=True)
            a=a.reshape(-1)
            
            Lssl = (F.cross_entropy(logits_u_s, a.clone().detach(),reduction = 'none'))
            Lu= 0
            for i in range(len(Lssl)):
                tem=simmax[i]/mean[a[i].data.cpu()]
                tem=tem.clone().detach()
                Lu=Lu+Lssl[i]*tem
            Lu=Lu/len(Lssl)

            if args.use_ema:
                test_model = ema_model.ema
            else:
                test_model = model
            Lcon=0
            Lconl=0
            Lpri=0
            

            

            if epoch>-1:
                simtem = torch.div(
                    torch.matmul(featw, queue.prototypes.T),
                    args.Tp)
                simtem=torch.exp(simtem)
                simmax, a = torch.max(simtem, dim=1, keepdim=True)



                Lcon = losscon3(args,featw, feats, queue.prototypes)
                Lconl=lossconl3(args,featll,targets_x,queue.prototypes)
                Lpri = losspri3(args,logits_u_w, queue.meansim)
                Lsslw=Lu0+coef2*Lu

                if epoch>-1:   
                   loss =Lx +Lsslw+ Lconl+coef*Lpri+coef3*Lcon
                   if epoch>5:
                      chosenul = []
                      chosentg = []
                      for i in range(len(featw_ema)):
                        if simmax[i] > queue.meansim[a[i].data.cpu()]:
                            chosenul.append(featw_ema[i])
                            chosentg.append(a[i].data.cpu())
                      if len(chosenul)>0:
                        chosenul = torch.tensor([item.cpu().detach().numpy() for item in chosenul]).cuda()
                        chosentg = torch.tensor([item.cpu().detach().numpy() for item in chosentg]).cuda()
                        chosentg=chosentg.reshape(-1)
                        queue.enqueue(chosenul.clone().detach(), chosentg.clone().detach())
                else:
                    loss =Lx+Lconl

            if args.amp:
                print("amp")
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            

            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            iteration=iteration+1

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. loss_con: {losscon:.4f}. loss_conl: {lossconl:.4f}. loss_pri: {losspri:.4f}.  Mask: {mask:.2f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=optimizer.state_dict()['param_groups'][0]['lr'],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=loss.item(),
                    loss_x=Lx.item(),
                    loss_u=Lu.item(),
                    losscon=Lcon,
                    lossconl=Lconl,
                    losspri=Lpri.item(),
                    mask=mask_probs.avg))
                p_bar.update()

            bb=time.time()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            print("epoch:{}".format(epoch))
            print("train/1.train_loss:{}".format(losses.avg))
            print("train/2.train_loss_x:{}".format(losses_x.avg))
            print("train/3.train_loss_u:{}".format(losses_u.avg))
            print("train/4.mask:{}".format(mask_probs.avg))
            print("test/1.test_acc:{}".format(test_acc))
            print("test/2.test_loss:{}".format(test_loss))
            scheduler.step(test_acc)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            num, t_sim, o_sim, ood_num, ood_t, ood_aver, mini = validate_l(labeled_loader, test_model, args,
                                                                     queue)
            print(num)
            t_sim = t_sim / num
            o_sim = o_sim / num
            print("tl", t_sim)
            print("ol", o_sim)
            del num, t_sim, o_sim, ood_num, ood_t, ood_aver

            num, t_sim, o_sim, ood_num, ood_t, ood_aver, select,oodselect = validate_ul(unlabeled_trainloader, test_model, args,
                                                                      queue,queue.meansim)
            t_sim = t_sim / num
            o_sim = o_sim / num
            print("t", t_sim)
            print("o", o_sim)
            print("ood_t", ood_t / ood_num)
            print("ood_aver", ood_aver / ood_num)
            print("minsimOfLabeled",queue.meansim)
            print("chosen_num",select)
            print("ood_chosennum",oodselect)
            del num, t_sim, o_sim, ood_num, ood_t, ood_aver

            model.zero_grad()

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))




def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            if args.dataset == 'mnist':
              inputs = inputs.unsqueeze(1)
            targets = targets.to(args.device).long()
            feat,outputs = model(inputs)
            
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

    a,num,acc=test2(model,test_loader,args)

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    
    logger.info(" acc: {:.2f}".format(a))
    print(num)
    print(acc)
    return losses.avg, top1.avg


def pre_process(x):
    l_images = np.array(l_images, dtype=np.float)
    l_images -= np.mean(l_images)
    l_images /= np.std(l_images)
    l_images = torch.from_numpy(l_images)
    return l_images

if __name__ == '__main__':
    main()
