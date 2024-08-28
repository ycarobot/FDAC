import os
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image

transforms_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transforms_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transforms_dict = {'train': transforms_train, 'test': transforms_test}


def listdir_nohidden(path):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
    """
    return [f for f in os.listdir(path) if not f.startswith('.')]


def read_dataPaths_dataLabels(dataset_path, domain_name):
    data_paths = []
    data_labels = []
    # domain_dir = path.join(dataset_path, domain_name, "images")
    domain_dir = os.path.join(dataset_path, domain_name)
    class_names = listdir_nohidden(domain_dir)
    class_names.sort()
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(domain_dir, class_name)
        item_names = listdir_nohidden(class_dir)
        for item_name in item_names:
            item_path = os.path.join(class_dir, item_name)
            data_paths.append(item_path)
            data_labels.append(label)
    return data_paths, data_labels



class Office31(data.Dataset):
    def __init__(self, data_paths, data_labels, transforms):
        super(Office31, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)



class ImageList(datasets.VisionDataset):
    def __init__(self, root, num_classes):
        super(ImageList, self).__init__(root, )
        self._num_classes = num_classes
        self.loader = default_loader
        self.data_list = 0

    def __getitem__(self, index):
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.data_list[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.data_list)


class ImageLCEF(data.Dataset):
    def __init__(self, data_path, domain, n_classes=12, phase='train', baseline='others', transforms_dict=transforms_dict):
        # All domains are included in this data-path 
        super(ImageLCEF, self).__init__()
        assert domain in ['c', 'i', 'p'], 'Wrong domain!'
        self.baseline = baseline
        self.data_path = data_path
        self.domain = domain
        self.phase = phase
        self.transform = transforms_dict[self.phase]
        self.get_imagesPath_labels()

    def get_imagesPath_labels(self):
        self.file_name = os.path.join(self.data_path, 'list', '{}List.txt'.format(self.domain))
        self.images_path_list, self.labels = [], []
        items = open(self.file_name, 'r').readlines()
        for item in items:
            line = item.strip().split(' ')
            # Only the last name is necessary
            images_path = os.path.join(self.data_path, self.domain, line[0].split('/')[-1])
            self.images_path_list.append(images_path)
            self.labels.append(int(line[1].strip()))

    def __getitem__(self, index):
        x, y = self.images_path_list[index], self.labels[index]
        x = self.transform(Image.open(x).convert('RGB'))
        y = torch.tensor(y)
        if self.baseline == 'TSA':
            return x, y, index
        else:
            return x, y

    def __len__(self):
        return len(self.images_path_list)



class OfficeHome_Office31_OfficeCaltech(data.Dataset):
    def __init__(self, data_path, domain, n_classes=65, phase='train', baseline='others', transforms_dict=transforms_dict):
        # Sub-domain in this data_path. Not all-domain
        super(OfficeHome_Office31_OfficeCaltech, self).__init__()
        self.baseline = baseline
        self.n_classes = n_classes
        data_path = os.path.join(data_path, domain)
        self.train_data = datasets.ImageFolder(data_path, transform=transforms_dict[phase])

    def __getitem__(self, index):
        x, y = self.train_data[index]
        y = torch.tensor(y).long()
        # if 'enhanced' in self.baseline or self.baseline == 'ViT_SHOT':
        # if self.baseline != 'ViT_Cross' or self.baseline != 'ViT_Cross_random_offset':  # Without pseudo-labels
        #     return x, y, index
        # else:
        #     return x, y
        return x, y, index

    def __len__(self):
        return len(self.train_data)
    
    
    
class DomainNet(data.Dataset):
    # Includes the train and test parts.
    def __init__(self, root_path, domain, phase='train', baseline='others', n_classes=345):
        super(DomainNet, self).__init__()
        assert domain in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'], 'Wrong domain!'
        self.domain = domain
        self.get_imagesPath_labels(root_path, domain, phase)
        self.transform = transforms_dict[phase]
        self.baseline = baseline


    def get_imagesPath_labels(self, root_path, domain, phase):
        self.file_name = os.path.join(root_path, '{}_{}.txt'.format(domain, phase))
        self.images_path_list, self.labels = [], []
        items = open(self.file_name, 'r').readlines()
        for item in items:
            line = item.strip().split(' ')
            # Only the last name is necessary.
            images_path = os.path.join(root_path, line[0])
            self.images_path_list.append(images_path)
            self.labels.append(int(line[1].strip()))

    def __getitem__(self, index):
        x, y = self.images_path_list[index], self.labels[index]
        x = self.transform(Image.open(x).convert('RGB'))
        y = torch.tensor(y)
        if self.baseline != 'ViT_Cross' or self.baseline != 'ViT_Cross_random_offset':
            return x, y, index
        else:
            return x, y

    def __len__(self):
        return len(self.images_path_list)


def get_all_labels(dataset, baseline):
    labels = []
    if baseline != 'ViT_Cross' or baseline != 'ViT_Cross_random_offset':
        for (x, y, _) in dataset:
            labels.append(int(y))
    else:
        for (x, y) in dataset:
            labels.append(int(y))
    return labels



def get_split_sampler(labels, test_ratio=0.2, num_classes=31):
    """
    :param labels: torch.array(long tensor)
    :param test_ratio: the ratio to split part of the data for test
    :param num_classes: 31
    :return: sampler_train,sampler_test
    To ensure that there samples for each category.
    """
    sampler_test = []
    sampler_train = []
    for i in range(num_classes):
        loc = torch.nonzero(labels == i, as_tuple=False)
        loc = loc.view(-1)
        # do random perm to make sure uniform sample
        test_num = round(loc.size(0) * test_ratio)
        loc = loc[torch.randperm(loc.size(0))]   # Randomly return an array.
        sampler_test.extend(loc[:test_num].tolist())
        sampler_train.extend(loc[test_num:].tolist())
    sampler_test = SubsetRandomSampler(sampler_test)  # Select according to the index
    sampler_train = SubsetRandomSampler(sampler_train)
    return sampler_train, sampler_test




def get_dataloaders_train_test_single(root_path, domain, bs, dataset='Office31', baseline='others'):

    if dataset == 'DomainNet':
        dataset_train = DomainNet(root_path, domain, phase='train', baseline=baseline)
        dataset_test = DomainNet(root_path, domain, phase='test', baseline=baseline)
        train_dloader = data.DataLoader(dataset_train, batch_size=bs, num_workers=4)
        test_dloader = data.DataLoader(dataset_test, batch_size=bs, num_workers=4)
        return train_dloader, test_dloader

    if dataset == 'ImageCLEF':  # ImageCLEF has not been used currently.
        dataset_train = ImageLCEF(root_path, domain, phase='train', baseline=baseline)
        dataset_test = ImageLCEF(root_path, domain, phase='test', baseline=baseline)

    elif dataset == 'Office31' or dataset == 'OfficeHome' or dataset == 'OfficeCaltech':
        data_paths, data_labels = read_dataPaths_dataLabels(root_path, domain)
        dataset_train = OfficeHome_Office31_OfficeCaltech(root_path, domain, phase='train', baseline=baseline)
        dataset_test = OfficeHome_Office31_OfficeCaltech(root_path, domain, phase='test', baseline=baseline)


    # data_labels = get_all_labels(dataset_train, baseline)  #  
    # Has been divided 
    # sampler_train, sampler_test = get_split_sampler(torch.LongTensor(data_labels), num_classes=int(data_labels[-1]+1))
    #  
    #  
    #  
    # train_dloader = data.DataLoader(dataset_train, batch_size=bs, sampler=sampler_train, num_workers=4, drop_last=True, shuffle=False)
    # test_dloader = data.DataLoader(dataset_test, batch_size=bs, sampler=sampler_test, num_workers=4, drop_last=True, shuffle=False)
    train_dloader = data.DataLoader(dataset_train, batch_size=bs, num_workers=4, drop_last=False, shuffle=True)
    test_dloader = data.DataLoader(dataset_test, batch_size=bs, num_workers=4, drop_last=False, shuffle=False)

    return train_dloader, test_dloader




def get_dataloaders_train_test_all(root_path, domain, bs, dataset):
    train_dloaders = []
    test_dloaders = []
    if dataset == 'Office31':
        # domains = ['amazon', 'webcam', 'dslr']
        domains = ['webcam', 'dslr']
        train_dloader_dt, test_dloader_dt = get_dataloaders_train_test_single(root_path, domain, bs, dataset)
        train_dloaders.append(train_dloader_dt)
        test_dloaders.append(test_dloader_dt)
        domains.remove(domain)
        for ds in domains:
            train_dloader_ds, test_dloader_ds = get_dataloaders_train_test_single(root_path, ds, bs, dataset)
            train_dloaders.append(train_dloader_ds)
            test_dloaders.append(test_dloader_ds)

    return train_dloaders, test_dloaders


def read_office31_data(dataset_path, domain_name):
    data_paths = []
    data_labels = []
    # domain_dir = path.join(dataset_path, domain_name, "images")
    domain_dir = os.path.join(dataset_path, domain_name)
    class_names = listdir_nohidden(domain_dir)
    class_names.sort()
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(domain_dir, class_name)
        item_names = listdir_nohidden(class_dir)
        for item_name in item_names:
            item_path = os.path.join(class_dir, item_name)
            data_paths.append(item_path)
            data_labels.append(label)
    return data_paths, data_labels



if __name__ == '__main__':
    train_dloaders, test_dloaders = get_dataloaders_train_test_single(r'F:\Python_project\Dataset\DomainNet', 'quickdraw', 32, 'DomainNet')

    print(len(train_dloaders.dataset), len(test_dloaders.dataset))
