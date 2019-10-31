
def get_dataset(opt):
    if opt.dataset_mode == 'list':
        from data.list_dataset import ListDataset
        dataset = ListDataset(opt.dataset_path, img_size=opt.image_size, augment=opt.hflip, multiscale=opt.multiscale)
    else:
        raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)
    print("dataset [%s] was created" % (dataset.name()))
    return dataset