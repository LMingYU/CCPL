import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from randaugment import RandAugmentMC,RandAugmentMC_mnist
from imagenet32 import IMAGENET32

rng = np.random.RandomState(seed=1)

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mnist_mean = (0.1307, 0.1307, 0.1307)
mnist_std = (0.3081, 0.3081, 0.3081)
#cifar10_mean = (0.5, 0.5, 0.5)
#cifar10_std = (0.5, 0.5, 0.5)

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

imagenet32_mean = (0.485, 0.456, 0.406)
imagenet32_std = (0.229, 0.224, 0.225)

def split_test(test_set, tot_class=6):
    images = test_set.data
    labels = test_set.targets
    # index = np.arange(len(labels))
    classes = np.unique(labels)
    l_images = []
    l_labels = []
    # idxs = []
    for c in classes[:tot_class]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        # c_idxs = index[cls_mask]
        l_images += [c_images[:]]
        l_labels += [c_labels[:]]


    #indices = rng.permutation(len(test_set["images"]))
    indices = rng.permutation(len(np.concatenate(l_images, 0)))
    #test_set["images"] = test_set["images"][indices]
    #test_set["labels"] = test_set["labels"][indices]
    test_set.data=np.concatenate(l_images, 0)[indices]
    test_set.targets=np.concatenate(l_labels,0)[indices]
    # test_set["index"] = test_set["index"][indices]
    return test_set


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)
    print("bd",base_dataset.data.shape)
    base_dataset.targets = np.array(base_dataset.targets)
    base_dataset.targets -= 2
    base_dataset.targets[np.where(base_dataset.targets == -2)[0]] = 8
    base_dataset.targets[np.where(base_dataset.targets == -1)[0]] = 9

    #train_labeled_idxs, train_unlabeled_idxs = x_u_split(
       # args, base_dataset.targets)
    train_labeled_idxs, train_unlabeled_idxs = x_u_split2(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)
    

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    train_labeled_dataset.targets -= 2
    train_unlabeled_dataset.targets -= 2
    #val_dataset.targets -= 2
    test_dataset.targets = np.array(test_dataset.targets)
    test_dataset.targets -= 2
    test_dataset.targets[np.where(test_dataset.targets == -2)[0]] = 8
    test_dataset.targets[np.where(test_dataset.targets == -1)[0]] = 9
    test_dataset = split_test(test_dataset, tot_class=6)



    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    train_unlabeled_dataset2 = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True)

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_svhn(args, norm=True):
    root = args.root
    name = args.dataset
    if name == "SVHN":
        data_folder = datasets.SVHN
        data_folder_main = SVHNSSL
        mean = cifar10_mean
        std = cifar10_std
        num_class = 10
    
    assert num_class > args.num_classes

    if name == "SVHN":
        base_dataset = data_folder(root, split='train', download=True)
        args.num_classes = 6

    base_dataset.labels = np.array(base_dataset.labels)


    train_labeled_idxs, train_unlabeled_idxs, val_idxs = \
        x_u_split_svhn(args, base_dataset.labels)

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    


    norm_func = TransformFixMatch(mean=mean, std=std)
    if norm:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    if name == 'SVHN':
        train_labeled_dataset = data_folder_main(
            root, train_labeled_idxs, split='train',
            transform=transform_labeled)
        train_unlabeled_dataset = data_folder_main(
            root, train_unlabeled_idxs, split='train',
            transform=norm_func, return_idx=False)
        val_dataset = data_folder_main(
            root, val_idxs, split='train',
            transform=transform_val)



    if name == 'SVHN':
        test_dataset = data_folder(
            root, split='test', transform=transform_val, download=False)




    #print(np.unique(test_dataset.labels))

    target_ind = np.where(test_dataset.labels < args.num_classes)[0]
    test_dataset.labels=test_dataset.labels[target_ind]
    test_dataset.data=test_dataset.data[target_ind]
    #print(np.unique(test_dataset.labels))

    unique_labeled = np.unique(train_labeled_idxs)
    val_labeled = np.unique(val_idxs)

    return train_labeled_dataset, train_unlabeled_dataset, \
        test_dataset

def x_u_split_svhn(args, labels):
    label_per_class = args.num_labeled  # // args.num_classes
    val_per_class = 50  # // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    val_idx = []
    unlabeled_idx = []
    classes = np.unique(labels)
    #print(classes)

    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx2 = np.where(labels == i+5)[0]
        unlabeled_idx.extend(idx2[label_per_class+val_per_class:label_per_class+val_per_class+3332])
        idx = np.random.choice(idx, label_per_class+val_per_class, False)
        labeled_idx.extend(idx[:label_per_class])
        val_idx.extend(idx[label_per_class:])
    for i in classes[args.num_classes+5
    : ]:
        idx = np.where(labels == i)[0]
        unlabeled_idx.extend(idx[:3332])

    labeled_idx = np.array(labeled_idx)

    assert len(labeled_idx) == args.num_labeled * args.num_classes
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)


    return labeled_idx, unlabeled_idx, val_idx

class SVHNSSL(datasets.SVHN):
    def __init__(self, root, indexs, split='train',#‘train’, ‘test’, ‘extra’
                 transform=None, target_transform=None,
                 download=False, return_idx=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
        self.return_idx = return_idx
        self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.labels_index = self.labels[indexes]
        else:
            self.data_index = self.data
            self.labels_index = self.labels

    def init_index(self):
        self.data_index = self.data
        self.labels_index = self.labels

    def __getitem__(self, index):
        img, target = self.data_index[index], self.labels_index[index]
        #print("img:{}".format(img.shape))
        img=img.transpose(1,2,0)
        #print("img2:{}".format(img.shape))
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)
            #print("img3:{}".format(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_idx:
            return img, target,index
        else:
            return img, target, index

    def __len__(self):
        return len(self.data_index)

def x_u_split(args, labels):
    label_per_class = args.num_labeled #// args.num_classes
    labels = np.array(labels)
    labeled_idx = []

    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled*args.num_classes

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    unlabeled_idx = [idx for idx in unlabeled_idx if idx not in labeled_idx]
    return labeled_idx, unlabeled_idx

def x_u_split2(args, labels):
    label_per_class = args.num_labeled #// args.num_classes
    n_unlabels_per_cls = int(args.n_unlabels) // 6
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []

    for i in range(args.num_classes):

        idx = np.where(labels == i)[0]
        idx_u=np.where(labels == i+5)[0]
        #idx = np.random.choice(idx, label_per_class, False)
        idx = idx[:label_per_class]
        idx_u=idx_u[label_per_class:label_per_class + n_unlabels_per_cls]
        labeled_idx.extend(idx)
        unlabeled_idx.extend(idx_u)
    for i in range(args.num_classes,10):

        
        idx_u=np.where(labels == i+5)[0]
        #idx = np.random.choice(idx, label_per_class, False)
        
        idx_u=idx_u[label_per_class:label_per_class + n_unlabels_per_cls]
        #labeled_idx.extend(idx)
        unlabeled_idx.extend(idx_u)
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(labeled_idx) == args.num_labeled*args.num_classes


    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    #unlabeled_idx = [idx for idx in unlabeled_idx if idx not in labeled_idx]
    return labeled_idx, unlabeled_idx


def get_mnist(args, norm=True):
    root = args.root
    name = args.dataset
    if name == "mnist":
        data_folder = datasets.MNIST
        data_folder_main = MNISTSSL
        mean = mnist_mean
        std = mnist_std
        num_class = 10

    assert num_class > args.num_classes

    if name == "mnist":
        base_dataset = data_folder(root, train=True, download=True)
        args.num_classes = 6

    base_dataset.targets = np.array(base_dataset.targets)

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = \
        x_u_split_svhn(args, base_dataset.targets)

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                              padding=int(28 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor()
        #transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])


    norm_func = TransformFixMatch_mnist(mean=mean, std=std)
    if norm:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    if name == 'mnist':
        train_labeled_dataset = data_folder_main(
            root, train_labeled_idxs, train=True,
            transform=norm_func)
        train_unlabeled_dataset = data_folder_main(
            root, train_unlabeled_idxs, train=True,
            transform=norm_func, return_idx=False)
        val_dataset = data_folder_main(
            root, val_idxs, train=True,
            transform=transform_val)

    if name == 'mnist':
        test_dataset = data_folder(
            root, train=False, transform=norm_func, download=False)

    target_ind = np.where(test_dataset.targets < args.num_classes)[0]
    test_dataset.targets = test_dataset.targets[target_ind]
    test_dataset.data = test_dataset.data[target_ind]

    unique_labeled = np.unique(train_labeled_idxs)
    val_labeled = np.unique(val_idxs)
    logger.info("Dataset: %s" % name)
    logger.info(f"Labeled examples: {len(unique_labeled)}"
                f"Unlabeled examples: {len(train_unlabeled_idxs)}"
                f"Valdation samples: {len(val_labeled)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset
           #test_dataset, \



class MNISTSSL(datasets.MNIST):
    def __init__(self, root, indexs, train=True,#‘train’, ‘test’, ‘extra’
                 transform=None, target_transform=None,
                 download=True, return_idx=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.return_idx = return_idx
        self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.labels_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.labels_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.labels_index = self.targets

    def __getitem__(self, index):
        img, target = self.data_index[index], self.labels_index[index]

        img = Image.fromarray(img.numpy())


        if self.transform is not None:
            img = self.transform(img)
            #print("img3:{}".format(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.data_index)


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)

        return self.normalize(weak), self.normalize(strong),self.normalize(x)


class TransformFixMatch_mnist(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(28 * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(28 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)
            #RandAugmentMC_mnist(n=2, m=10)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(), ])

    def __call__(self, x):
        x = x.convert('RGB')
        weak = self.weak(x)
        strong = self.strong(x)

        return self.normalize(weak), self.normalize(strong), self.normalize(x)


#new added in 2022
def gcn(images, multiplier=55, eps=1e-10):
    #global contrast normalization
    images = images.astype(np.float)
    images -= images.mean(axis=(1,2,3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
    per_image_norm[per_image_norm < eps] = 1
    images = multiplier * images / per_image_norm
    return images

def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, height*width*channels)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp

def zca_normalization(images, mean, decomp):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, -1)
    images = np.dot((images - mean), decomp)
    return images.reshape(n_data, height, width, channels)

class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        

        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        

        return img, target#,index


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_imagenet32(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet32_mean, std=imagenet32_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet32_mean, std=imagenet32_std)])

    base_dataset = IMAGENET32(
        root, train=True, download=True)
    test_dataset=IMAGENET32(
        root, train=False, download=True)

    base_dataset.targets=np.array(base_dataset.targets)
    test_dataset.targets = np.array(test_dataset.targets)

    fimnet32 = 'mapcifar2imnet.txt'
    mapcls = np.loadtxt(fimnet32, dtype=int)
    classes = np.unique(base_dataset.targets)
    ood_classes = list(set(classes) - set(mapcls[:, 1]))[:40]
    class_list = np.concatenate([mapcls[:, 1], ood_classes], 0)
    label_map = {}
    for i in range(len(class_list)):
        label_map[class_list[i]] = i
    train_set = reduce_classes(base_dataset, class_list)
    test_set = reduce_classes(test_dataset, class_list)

    for i in range(len(train_set.targets)):
        train_set.targets[i] = label_map[train_set.targets[i]]
    for i in range(len(test_set.targets)):
        test_set.targets[i] = label_map[test_set.targets[i]]

    train_labeled_idxs, train_unlabeled_idxs = x_u_split100(
        args, train_set.targets)

    train_labeled_dataset = IMAGENET32(
        root, train_set, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = IMAGENET32(
        root, train_set, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=imagenet32_mean, std=imagenet32_std))

    train_unlabeled_dataset2 = IMAGENET32(
        root, train_set, train_unlabeled_idxs, train=True)

    test_dataset = IMAGENET32(
        root, test_set,train=False, transform=transform_val, download=False)
    test_dataset = split_test100(test_dataset, tot_class=60)

    print("labeled",np.unique(train_labeled_dataset.targets))
    print("unlabeled", np.unique(train_unlabeled_dataset.targets))
    print("test", np.unique(test_dataset.targets))

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def reduce_classes(dataset, class_list):
    images = dataset.data
    labels = dataset.targets
    classes = np.unique(labels)
    l_images = []
    l_labels = []
    for c in class_list:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:]]
        l_labels += [c_labels[:]]
    dt = {"images" : np.concatenate(l_images, 0), "labels" : np.concatenate(l_labels,0)}

    indices = rng.permutation(len(dt["images"]))
    dataset.data = dt["images"][indices]
    dataset.targets = dt["labels"][indices]
    return dataset


def x_u_split100(args, labels):
    label_per_class = 100 #// args.num_classes
    n_unlabels_per_cls = 3332
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    classes = np.unique(labels)

    for i in range(args.num_classes):

        idx = np.where(labels == i)[0]
        #print(i,len(idx))
        idx_u=np.where(labels == i+30)[0]
        #idx = np.random.choice(idx, label_per_class, False)
        idx = idx[:label_per_class]
        #print(len(idx))
        idx_u=idx_u[label_per_class:label_per_class + n_unlabels_per_cls]
        #print(len(idx_u))
        labeled_idx.extend(idx)
        unlabeled_idx.extend(idx_u)
    for i in classes[args.num_classes+30:]:

        idx_u=np.where(labels == i)[0]
        #idx = np.random.choice(idx, label_per_class, False)

        idx_u=idx_u[:n_unlabels_per_cls]
        #labeled_idx.extend(idx)
        unlabeled_idx.extend(idx_u)
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(labeled_idx) == args.num_labeled*args.num_classes

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    #unlabeled_idx = [idx for idx in unlabeled_idx if idx not in labeled_idx]
    return labeled_idx, unlabeled_idx

def split_test100(test_set, tot_class=6):
    images = test_set.data
    labels = test_set.targets
    labels=np.array(labels)
    # index = np.arange(len(labels))
    classes = np.unique(labels)
    l_images = []
    l_labels = []
    # idxs = []
    for c in classes[:tot_class]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        # c_idxs = index[cls_mask]
        l_images += [c_images[:]]
        l_labels += [c_labels[:]]


    indices = rng.permutation(len(np.concatenate(l_images, 0)))
    #test_set["images"] = test_set["images"][indices]
    #test_set["labels"] = test_set["labels"][indices]
    test_set.data=np.concatenate(l_images, 0)[indices]
    test_set.targets=np.concatenate(l_labels,0)[indices]
    # test_set["index"] = test_set["index"][indices]
    return test_set

DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'SVHN': get_svhn,
                   'mnist':get_mnist, 
                   'imagenet32':get_imagenet32}
