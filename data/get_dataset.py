
def get_dataset(opt):
    if opt.dataset_mode == 'list':
        from data.list_dataset import ListDataset
        dataset = ListDataset(opt.dataroot, img_size=opt.img_size, augment=(not opt.noagument), multiscale=(not opt.nomultiscale))
    else:
        raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)
    print("dataset [%s] was created" % (dataset.name()))
    return dataset