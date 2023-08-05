from dataset import *
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def get_train_valid_sampler(trainset):
    size = len(trainset) # 对话数量 n
    idx = list(range(size)) # [0,1,2,...,n-1]
    return SubsetRandomSampler(idx) # 无放回地按照给定的索引列表采样样本元素。 那么这里就相当于抽取了一个全排列



def get_data_loaders(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):

    print('building datasets..')
    trainset = MyDataset(dataset_name, 'train', args)
    devset = MyDataset(dataset_name, 'dev', args)

    train_sampler = get_train_valid_sampler(trainset)
    valid_sampler = get_train_valid_sampler(devset)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory) # 和不用sampler shuffle=True 是等价的

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MyDataset(dataset_name, 'test',args)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader