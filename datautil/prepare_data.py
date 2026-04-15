import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from datautil.datasplit import getdataloader
from datautil.fault_preprocess import remap_targets_to_contiguous
from PIL import ImageFile
from util.config import normalize_dataset_name

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_data(data_name):
    """Return the algorithm class with the given name."""
    data_name = normalize_dataset_name(data_name)
    datalist = {'officehome': 'img_union', 'pacs': 'img_union', 'vlcs': 'img_union', 'medmnist': 'medmnist',
                'medmnistA': 'medmnist', 'medmnistC': 'medmnist', 'office-caltech': 'img_union', 'pamap': 'pamap', 'covid': 'covid',
                'cwru': 'cwru', 'seu': 'seu'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]


def gettransforms():
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    return transform_train, transform_test


class mydataset(object):
    def __init__(self, args):
        self.x = None
        self.targets = None
        self.dataset = None
        self.transform = None
        self.target_transform = None
        self.loader = None
        self.args = args

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        x = self.input_trans(self.loader(self.x[index]))
        ctarget = self.target_trans(self.targets[index])
        return x, ctarget

    def __len__(self):
        return len(self.targets)


class ImageDataset(mydataset):
    def __init__(self, args, dataset, root_dir, domain_name, transform=None, target_transform=None):
        super(ImageDataset, self).__init__(args)
        self.imgs = ImageFolder(root_dir+domain_name).imgs
        self.domain_num = 0
        self.dataset = dataset
        imgs = [item[0] for item in self.imgs]
        labels = [item[1] for item in self.imgs]
        self.targets = np.array(labels)
        default_transform, _ = gettransforms()
        self.transform = transform if transform is not None else default_transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.pathx = imgs
        self.x = self.pathx


class MedMnistDataset(Dataset):
    def __init__(self, filename='', transform=None):
        self.data = np.load(filename+'xdata.npy')
        self.targets = np.load(filename+'ydata.npy')
        self.targets = np.squeeze(self.targets)
        self.transform = transform

        self.data = torch.Tensor(self.data)
        self.data = torch.unsqueeze(self.data, dim=1)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class PartitionTransformDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target


class TensorZScoreTransform(object):
    def __init__(self, mean, std):
        self.mean = mean.detach().clone().float()
        self.std = std.detach().clone().float()

    def __call__(self, tensor):
        tensor = tensor.float()
        return (tensor - self.mean) / self.std


def build_train_dataloader(dataset, batch_size):
    dataset_size = len(dataset)
    if dataset_size < 2:
        return None

    effective_batch_size = min(batch_size, dataset_size)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=True,
    )


def compute_partition_channel_stats(partition, eps=1e-6):
    base_dataset = partition.data
    if not hasattr(base_dataset, 'data'):
        raise ValueError('Partition dataset does not expose raw tensor data for normalization.')

    indices = torch.as_tensor(partition.indices, dtype=torch.long)
    samples = base_dataset.data[indices]
    if not torch.is_tensor(samples):
        samples = torch.as_tensor(samples)
    samples = samples.float()

    if samples.ndim < 2:
        raise ValueError('Expected partition samples to have at least 2 dimensions.')

    reduce_dims = (0,) + tuple(range(2, samples.ndim))
    mean = samples.mean(dim=reduce_dims, keepdim=True)
    std = samples.std(dim=reduce_dims, keepdim=True, unbiased=False)
    std = torch.where(std < eps, torch.ones_like(std), std)
    return mean, std


def wrap_client_partitions_with_train_stats(train_partition, val_partition, test_partition):
    mean, std = compute_partition_channel_stats(train_partition)
    transform = TensorZScoreTransform(mean, std)
    return (
        PartitionTransformDataset(train_partition, transform=transform),
        PartitionTransformDataset(val_partition, transform=transform),
        PartitionTransformDataset(test_partition, transform=transform),
    )


class PamapDataset(Dataset):
    def __init__(self, filename='../data/pamap/', transform=None):
        self.data = np.load(filename+'x.npy')
        self.targets = np.load(filename+'y.npy')
        self.select_class()
        self.transform = transform
        self.data = torch.unsqueeze(torch.Tensor(self.data), dim=1)
        self.data = torch.einsum('bxyz->bzxy', self.data)

    def select_class(self):
        xiaochuclass = [0, 5, 12]
        index = []
        for ic in xiaochuclass:
            index.append(np.where(self.targets == ic)[0])
        index = np.hstack(index)
        allindex = np.arange(len(self.targets))
        allindex = np.delete(allindex, index)
        self.targets = self.targets[allindex]
        self.data = self.data[allindex]
        ry = np.unique(self.targets)
        ry2 = {}
        for i in range(len(ry)):
            ry2[ry[i]] = i
        for i in range(len(self.targets)):
            self.targets[i] = ry2[self.targets[i]]

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class CovidDataset(Dataset):
    def __init__(self, filename='../data/covid19/', transform=None):
        self.data = np.load(filename+'xdata.npy')
        self.targets = np.load(filename+'ydata.npy')
        self.targets = np.squeeze(self.targets)
        self.transform = transform
        self.data = torch.Tensor(self.data)
        self.data = torch.einsum('bxyz->bzxy', self.data)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class CWRUDataset(Dataset):
    def __init__(self, filename='../data/cwru/', transform=None):
        self.data = np.load(filename+'x.npy').astype(np.float32)
        self.data = np.nan_to_num(self.data, nan=0.0, posinf=0.0, neginf=0.0)
        raw_targets = np.load(filename+'y.npy')
        self.targets, self.target_mapping = remap_targets_to_contiguous(raw_targets)
        self.transform = transform
        self.data = torch.Tensor(self.data)
        self.data = torch.unsqueeze(self.data, dim=1)  # 添加通道维度
        self.data = torch.unsqueeze(self.data, dim=2)  # 添加高度维度，形状变为 (batch, channel, height, width) = (batch, 1, 1, 1024)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SEUDataset(Dataset):
    def __init__(self, filename='../data/seu/', transform=None):
        self.data = np.load(filename+'x.npy').astype(np.float32)
        self.data = np.nan_to_num(self.data, nan=0.0, posinf=0.0, neginf=0.0)
        raw_targets = np.load(filename+'y.npy')
        self.targets, self.target_mapping = remap_targets_to_contiguous(raw_targets)
        self.transform = transform
        self.data = torch.Tensor(self.data)
        self.data = torch.unsqueeze(self.data, dim=1)  # 添加通道维度
        self.data = torch.unsqueeze(self.data, dim=2)  # 添加高度维度，形状变为 (batch, channel, height, width) = (batch, 1, 1, 1024)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


STANDARDIZED_TENSOR_DATASETS = (
    MedMnistDataset,
    PamapDataset,
    CovidDataset,
    CWRUDataset,
    SEUDataset,
)


def getfeadataloader(args):
    trl, val, tel = [], [], []
    trd, vad, ted = [], [], []
    transform_train, transform_test = gettransforms()
    for item in args.domains:
        train_data = ImageDataset(
            args, args.dataset, args.root_dir+args.dataset+'/', item, transform=transform_train)
        val_data = ImageDataset(
            args, args.dataset, args.root_dir+args.dataset+'/', item, transform=transform_test)
        test_data = ImageDataset(
            args, args.dataset, args.root_dir+args.dataset+'/', item, transform=transform_test)
        l = len(train_data)
        index = np.arange(l)
        np.random.seed(args.seed)
        np.random.shuffle(index)
        l1, l2, l3 = int(l*args.datapercent), int(l *
                                                  args.datapercent), int(l*0.2)
        trl.append(torch.utils.data.Subset(train_data, index[:l1]))
        val.append(torch.utils.data.Subset(val_data, index[l1:l1+l2]))
        tel.append(torch.utils.data.Subset(test_data, index[l1+l2:l1+l2+l3]))
        if len(val[-1]) == 0 or len(tel[-1]) == 0:
            continue
        train_loader = build_train_dataloader(trl[-1], args.batch)
        if train_loader is None:
            continue
        trd.append(train_loader)
        vad.append(torch.utils.data.DataLoader(
            val[-1], batch_size=args.batch, shuffle=False))
        ted.append(torch.utils.data.DataLoader(
            tel[-1], batch_size=args.batch, shuffle=False))
    return trd, vad, ted


def img_union(args):
    return getfeadataloader(args)


def getlabeldataloader(args, data):
    trl, val, tel = getdataloader(args, data)
    trd, vad, ted = [], [], []
    for i in range(len(trl)):
        train_dataset = trl[i]
        val_dataset = val[i]
        test_dataset = tel[i]
        if len(train_dataset) < 2:
            continue
        if len(val_dataset) == 0 or len(test_dataset) == 0:
            continue

        if isinstance(data, STANDARDIZED_TENSOR_DATASETS):
            train_dataset, val_dataset, test_dataset = wrap_client_partitions_with_train_stats(
                trl[i], val[i], tel[i]
            )

        train_loader = build_train_dataloader(train_dataset, args.batch)
        if train_loader is None:
            continue
        trd.append(train_loader)
        vad.append(torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch, shuffle=False))
        ted.append(torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch, shuffle=False))
    return trd, vad, ted


def medmnist(args):
    data = MedMnistDataset(args.root_dir+args.dataset+'/')
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 11
    return trd, vad, ted


def pamap(args):
    data = PamapDataset(args.root_dir+'pamap/')
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 10
    return trd, vad, ted


def covid(args):
    data = CovidDataset(args.root_dir+'covid19/')
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 4
    return trd, vad, ted


def cwru(args):
    data = CWRUDataset(args.root_dir+'cwru/')
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = len(np.unique(data.targets))
    return trd, vad, ted


def seu(args):
    data = SEUDataset(args.root_dir+'seu/')
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = len(np.unique(data.targets))
    return trd, vad, ted


class combinedataset(mydataset):
    def __init__(self, datal, args):
        super(combinedataset, self).__init__(args)

        self.x = np.hstack([np.array(item.x) for item in datal])
        self.targets = np.hstack([item.targets for item in datal])
        s = ''
        for item in datal:
            s += item.dataset+'-'
        s = s[:-1]
        self.dataset = s
        self.transform = datal[0].transform
        self.target_transform = datal[0].target_transform
        self.loader = datal[0].loader


def getwholedataset(args):
    datal = []
    for item in args.domains:
        datal.append(ImageDataset(args, args.dataset,
                     args.root_dir+args.dataset+'/', item))
    # data=torch.utils.data.ConcatDataset(datal)
    data = combinedataset(datal, args)
    return data


def img_union_w(args):
    return getwholedataset(args)


def medmnist_w(args):
    data = MedMnistDataset(args.root_dir+args.dataset+'/')
    args.num_classes = 11
    return data


def pamap_w(args):
    data = PamapDataset(args.root_dir+'pamap/')
    args.num_classes = 10
    return data


def covid_w(args):
    data = CovidDataset(args.root_dir+'covid19/')
    args.num_classes = 4
    return data


def cwru_w(args):
    data = CWRUDataset(args.root_dir+'cwru/')
    args.num_classes = len(np.unique(data.targets))
    return data


def seu_w(args):
    data = SEUDataset(args.root_dir+'seu/')
    args.num_classes = len(np.unique(data.targets))
    return data


def get_whole_dataset(data_name):
    data_name = normalize_dataset_name(data_name)
    datalist = {'officehome': 'img_union_w', 'pacs': 'img_union_w', 'vlcs': 'img_union_w', 'medmnist': 'medmnist_w',
                'medmnistA': 'medmnist_w', 'medmnistC': 'medmnist_w', 'office-caltech': 'img_union_w', 'pamap': 'pamap_w', 'covid': 'covid_w',
                'cwru': 'cwru_w', 'seu': 'seu_w'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]
