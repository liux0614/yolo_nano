
def get_train_dataset(opt, transforms=None):
    if opt.dataset == 'coco':
        from data.coco import COCO
        dataset = COCO(
            opt.dataset_path, opt.annotation_path, subset='train', 
            image_size=opt.image_size, multi_scale=(not opt.no_multi_scale), transforms=transforms)
    else:
        raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)
    print("dataset [%s] was created" % (dataset.name()))
    return dataset


def get_val_dataset(opt, transforms=None):
    if opt.dataset == 'coco':
        from data.coco import COCO
        dataset = COCO(
            opt.dataset_path, opt.annotation_path, subset='val',
            image_size=opt.image_size, multi_scale=False, transforms=transforms)
    else:
        raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)
    print("dataset [%s] was created" % (dataset.name()))
    return dataset


def get_test_dataset(opt, transforms=None):
    if opt.dataset == 'coco':
        from data.coco import COCO
        dataset = COCO(
            opt.dataset_path, opt.annotation_path, subset='test',
            image_size=opt.image_size, multi_scale=False, transforms=transforms)
    else:
        raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)
    print("dataset [%s] was created" % (dataset.name()))
    return dataset