from utils import get_data, data_path
import torch.utils.data as data




def get_dloaders_ds_dt(root_path=data_path.OfficeHome_path, ds='Art', dt='Product', bs=32, shuffle=True):
    dataset_ds = get_data.OfficeHome_Office31_OfficeCaltech(root_path, domain=ds, phase='train', baseline='other_baselines')
    dataset_dt = get_data.OfficeHome_Office31_OfficeCaltech(root_path, domain=dt, phase='train', baseline='other_baselines')
    ds_loader = data.DataLoader(dataset_ds, batch_size=bs, num_workers=4, drop_last=False, shuffle=shuffle)
    dt_loader = data.DataLoader(dataset_dt, batch_size=bs, num_workers=4, drop_last=False, shuffle=shuffle)
    return ds_loader, dt_loader

def get_dataloaders_train_test(root_path=data_path.OfficeHome_path, domain='Art', bs=32, baseline='other_baselines',
                               dataset='OfficeHome'):
    dataset_train = get_data.OfficeHome_Office31_OfficeCaltech(root_path, domain=domain, phase='train',
                                                               baseline=baseline)
    dataset_test= get_data.OfficeHome_Office31_OfficeCaltech(root_path, domain=domain, phase='test',
                                                               baseline=baseline)
    train_loader = data.DataLoader(dataset_train, batch_size=bs, num_workers=2, drop_last=False, shuffle=True)
    test_loader = data.DataLoader(dataset_test, batch_size=bs, num_workers=2, drop_last=False, shuffle=False)
    return train_loader, test_loader