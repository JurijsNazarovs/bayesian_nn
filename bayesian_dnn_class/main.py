import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import csv

import sys
import importlib
import subprocess  # gpu memory measure

sys.path.append("../")
import vi_posteriors as vi
importlib.reload(vi)

import uncertainty_estimate as unest
importlib.reload(unest)

import models.densenet as dn
import models.resnet as rn
import models.vgg as vgg
importlib.reload(dn)
importlib.reload(rn)
importlib.reload(vgg)

import models.FCNet as fcn
importlib.reload(fcn)
import models.AlexNet as an
importlib.reload(an)

import numpy as np
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')

parser.add_argument('--net',
                    default='PreActResNet18',
                    type=str,
                    choices=[
                        'PreActResNet18', 'PreActResNet34', 'PreActResNet50',
                        'PreActResNet101', 'PreActResNet152', 'DenseNet100_12',
                        'densenet121', 'densenet161', 'densenet169',
                        'densenet201', 'vgg11', 'vgg11_bn', 'vgg13',
                        'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
                        'FCNet', 'AlexNet'
                    ],
                    help='name of the network')
parser.add_argument('--dataset',
                    default="mnist",
                    choices=["cifar10", "cifar100", "svhn", "mnist"],
                    type=str,
                    help='dataset')
# Training
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epochs',
                    default=40,
                    type=int,
                    help='number of total epochs to run')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=10**-3,
                    type=float,
                    help='initial learning rate')
parser.add_argument(
    '--no-augment',
    dest='augment',
    action='store_false',
    help='whether to use standard augmentation (default: True)')
parser.add_argument('--devices',
                    default='cuda:0',
                    type=str,
                    help='gpuid in str')
parser.add_argument('--print-freq',
                    '-p',
                    default=100,
                    type=int,
                    help='print frequency in iterations (default: 100)')

parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint_path')
parser.add_argument('--checkpoint_path',
                    default="",
                    type=str,
                    help='path to resume model')
parser.add_argument('--checkpoint_dir',
                    default="models_saved",
                    type=str,
                    help='directory to save models')
parser.add_argument('--logs_dir',
                    default="logs_deb",
                    type=str,
                    help='directory to save logs')

# Bayesian arguments
parser.add_argument('--beta_type',
                    default='graves',
                    type=str,
                    help='beta for vi')
parser.add_argument('--approx_post',
                    default='Radial',
                    type=str,
                    help='Approximate posterior: Gaus, Radial')
parser.add_argument('--kl_method',
                    default='repar',
                    type=str,
                    help='method to compute KL: repar, direct, closed')
parser.add_argument('--n_mc_iter',
                    default=1,
                    type=int,
                    help='number of mc iterations to approximate kl')
parser.add_argument('--n_test_iter',
                    default=1,
                    type=int,
                    help='number of test iterations to estimate uncertainty')
parser.add_argument('--n_var_iter',
                    default=1,
                    type=int,
                    help='number of iterations for local repar trick')
parser.add_argument('--note',
                    default='',
                    type=str,
                    help='additional note for an experiment')

args = parser.parse_args()
args.augment = False

model_name = "%s-%s-%s-%s-%d-%d-%d-%f%s" % (
    args.dataset, args.net, args.approx_post, args.kl_method, args.n_mc_iter,
    args.n_test_iter, args.batch_size, args.lr, args.note)
logs_dir = "%s/%s" % (args.logs_dir, model_name)
os.makedirs(logs_dir, exist_ok=True)
unest_dir = "%s/uncert_est/" % logs_dir
os.makedirs(unest_dir, exist_ok=True)

stat_file_path = "%s/stat.csv" % logs_dir
model_summary_file_path = "%s/model_summary.txt" % logs_dir

# Summary variables
best_prec1 = 0


def main():
    global best_prec1

    # Initial checking
    if args.dataset == "mnist" and ("PreActResNet" not in args.net
                                    and "FCNet" not in args.net):
        raise NotImplementedError(
            "For MNIST dataset the only available class of " +
            "networks are PreActResNet, FCNet")

    # Data loading
    if args.dataset == "cifar10":
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
        rand_crop = 32
    if args.dataset == "cifar100":
        normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409),
                                         std=(0.2009, 0.1984, 0.2023))
        rand_crop = 32
    if args.dataset == "svhn":
        normalize = transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
        rand_crop = 32
    if args.dataset == "mnist":
        normalize = transforms.Normalize(
            mean=(0, ), std=(1, ))  # mean=(0.1307, ), std=(0.3081, ))
        rand_crop = 28
    if args.dataset == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(rand_crop, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize
        ])
    else:
        print("No augment for %s" % args.dataset)
        transform_train = transforms.Compose(
            [transforms.ToTensor(), normalize])

    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    kwargs = {
        'num_workers': 1,
        'pin_memory': False,
        'batch_size': args.batch_size
    }

    if args.dataset == "imagenet":
        dataset_ = datasets.ImageFolder(root="./data/imagenet-data/train",
                                        transform=transform_train)

        train_loader = torch.utils.data.DataLoader(dataset_,
                                                   **kwargs,
                                                   shuffle=False)

        dataset_ = datasets.ImageFolder(root="./data/imagenet-data/val",
                                        transform=transform_test)
        val_loader = torch.utils.data.DataLoader(dataset_,
                                                 **kwargs,
                                                 shuffle=False)
    else:
        dataset_ = eval("datasets.%s" % args.dataset.upper())
        if args.dataset == "svhn":
            train_loader = torch.utils.data.DataLoader(dataset_(
                './data',
                split='train',
                download=True,
                transform=transform_train),
                                                       **kwargs,
                                                       shuffle=False)  # True
            val_loader = torch.utils.data.DataLoader(dataset_(
                './data',
                split='test',
                download=True,
                transform=transform_test),
                                                     **kwargs,
                                                     shuffle=False)
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset_('./data',
                         train=True,
                         download=True,
                         transform=transform_train),
                **kwargs,
                shuffle=False)  # True
            val_loader = torch.utils.data.DataLoader(dataset_(
                './data', train=False, transform=transform_test),
                                                     **kwargs,
                                                     shuffle=False)

    if args.dataset == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    # Model definition
    if args.dataset == "imagenet":
        n_classes = 200
    elif args.dataset == "cifar100":
        n_classes = 100
    else:
        n_classes = 10

    if "PreActResNet" in args.net:
        net = "rn.%s" % args.net
    elif "densenet" in args.net:
        net = "dn.%s" % args.net
    elif "vgg" in args.net:
        net = "vgg.%s" % args.net
    elif "FCNet" in args.net:
        net = "fcn.%s" % args.net
    elif "AlexNet" in args.net:
        net = "an.%s" % args.net
    else:
        raise NotImplementedError("Network is not found: %s" % args.net)

    model = eval(net)
    model = model(num_classes=n_classes,
                  in_channels=in_channels,
                  approx_post=args.approx_post,
                  kl_method=args.kl_method,
                  n_mc_iter=args.n_mc_iter)

    if args.devices != 'cpu':
        device_ids = args.devices.split(":")
        if len(device_ids) > 1:
            device_ids = [int(d) for d in device_ids[1].split(",")]
            model = nn.DataParallel(model, device_ids)
        else:
            # Use all available devices:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(args.devices)
        cudnn.benchmark = True

    print(str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Summary of model
    model_summary = vi.Utils.get_summary(model)
    with open(model_summary_file_path, "w+") as summary_file:
        for key, val in model_summary.items():
            summary_file.writelines("%s; %s\n" % (key, val))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.checkpoint_path):
            print("=> loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path,
                                    map_location=args.devices)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.checkpoint_path, checkpoint['epoch']))

            del checkpoint
            torch.cuda.empty_cache()
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    args.checkpoint_path)
    else:
        start_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=5e-4)
    stat_fieldnames = [
        'epoch', 'train_loss', 'train_acc', 'train_avg_batch_time',
        'train_epoch_time', 'n_mc_iter', 'val_loss', 'val_acc',
        'val_avg_batch_time', 'val_epoch_time', 'n_test_iter', 'occupied_mb',
        'train_kl'
    ]

    if os.path.isfile(stat_file_path) and args.resume:
        write_mode = 'a+'
    else:
        write_mode = 'w+'

    with open(stat_file_path, write_mode) as stat_file:
        writer = csv.DictWriter(stat_file, fieldnames=stat_fieldnames)
        if write_mode == 'w+':
            writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)
        start_time = time.time()
        train_acc, train_loss, train_batch_time, occupied_mb, train_kl = train(
            train_loader, model, optimizer, epoch)
        train_time = time.time() - start_time
        start_time = time.time()

        import pdb
        pdb.set_trace()

        with torch.no_grad():
            prec1, val_loss, val_batch_time = validate(val_loader, model,
                                                       epoch, n_classes)
        #prec1, val_loss, val_batch_time = 1, 1, 1
        val_time = time.time() - start_time

        print(
            '\n===> Epoch {}: train_loss: {}, train_acc: {} ---- val_loss: {}, val_acc: {}\n'
            .format(epoch, train_loss, train_acc, val_loss, prec1))

        with open(stat_file_path, 'a+') as stat_file:
            writer = csv.DictWriter(stat_file, fieldnames=stat_fieldnames)
            writer.writerow({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_avg_batch_time': train_batch_time,
                'train_epoch_time': train_time,
                'n_mc_iter': args.n_mc_iter,
                'val_loss': val_loss,
                'val_acc': prec1,
                'val_avg_batch_time': val_batch_time,
                'val_epoch_time': val_time,
                'n_test_iter': args.n_test_iter,
                'occupied_mb': occupied_mb,
                'train_kl': train_kl
            })

        # Summary of the weigts
        if epoch == 0 and not args.resume:
            write_mode = "w+"
        else:
            write_mode = "a+"
        unest.summary_parameters(model,
                                 out_dir="%s/weights/" % unest_dir,
                                 write_mode=write_mode)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
        print('Best accuracy: ', best_prec1)


def train(train_loader, model, optimizer, epoch):
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    kls = AverageMeter()

    beta = vi.Utils.get_beta(args.beta_type, 1, 1)
    if beta == 0:
        vi.Utils.set_compute_kl(model, False)
    else:
        vi.Utils.set_compute_kl(model, True)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # if batch_idx > 1:
        #     break
        inputs, targets = inputs.to(args.devices), targets.to(args.devices)
        optimizer.zero_grad()
        beta = vi.Utils.get_beta(args.beta_type, len(train_loader), batch_idx)

        end = time.time()
        for _ in range(args.n_var_iter):
            output, kl = model(inputs)
            kl = kl.mean()  # if several gpus are used to split minibatch

            loss_, _ = vi.Utils.get_loss_categorical(
                kl, output, targets, beta=beta,
                N=1)  # len(train_loader))  # *args.batch_size)
            loss = loss_ / args.n_var_iter
            loss.backward()

        optimizer.step()
        batch_time.update(time.time() - end)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, targets, topk=(1, ))[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        kls.update(kl.item(), inputs.size(0))

        # measure elapsed time

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'KL {kl.val:.4f} ({kl.avg:.4f})'.format(
                      epoch,
                      batch_idx,
                      len(train_loader),
                      batch_time=batch_time,
                      loss=losses,
                      top1=top1,
                      kl=kls))

    occupied_mb = 0
    for gpu_id in model.device_ids:
        occupied_mb += get_gpu_memory_map()[gpu_id]

    return top1.avg, losses.avg, batch_time.avg, occupied_mb, kls.avg


def validate(val_loader, model, epoch, n_classes):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    model.eval()
    vi.Utils.set_compute_kl(model, False)  # turn off compute kl

    end = time.time()
    test_loss = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # if batch_idx > 20:
        #    break
        inputs, targets = inputs.to(args.devices), targets.to(args.devices)

        n_iter = args.n_test_iter
        outputs = []
        for _ in range(n_iter):
            outputs_, _ = model(inputs)
            loss = criterion(outputs_, targets)
            test_loss += loss.item() / n_iter
            outputs.append(outputs_.max(1)[1])

        outputs = torch.stack(outputs)

        if batch_idx < 30:
            if epoch == 0 and not args.resume:
                write_mode = "w+"
            else:
                write_mode = "a+"

            predicted = unest.summary_class(outputs,
                                            n_classes=n_classes,
                                            out_dir="%s/output/%d" %
                                            (unest_dir, batch_idx),
                                            write_mode=write_mode,
                                            targets=targets)
        else:
            predicted = unest.get_prediction_class(outputs, n_classes)

        # measure accuracy and record loss
        # prec1 = accuracy(outputs_.data, targets, topk=(1, ))[0]
        prec1 = predicted.eq(
            targets.cpu()).sum().item() / targets.shape[0] * 100
        top1.update(prec1, inputs.size(0))
        losses.update(test_loss, inputs.size(0))

        # summary
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx + 1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      batch_idx,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # import pdb
    # pdb.set_trace()

    return top1.avg, losses.avg, batch_time.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    checkpoint_dir = "%s/%s/" % (args.checkpoint_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    filename = checkpoint_dir + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/best.pth.tar' % checkpoint_dir)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1**(epoch // 150)) * (0.1**(epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
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


def get_gpu_memory_map(device_ids=None):
    """Get the current gpu usage.
    device_ids can be a list of specified ids to return

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ],
                                     encoding='utf-8')

    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        vis_ids = [
            int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        ]
        gpu_memory = [gpu_memory[i] for i in vis_ids]

    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    if device_ids is not None:
        gpu_memory_map = {i: gpu_memory_map[i] for i in device_ids}

    return gpu_memory_map


if __name__ == '__main__':
    main()
