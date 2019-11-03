
def get_training_dataset(opt, transforms=None):
    if opt.dataset == 'coco':
        from data.coco import COCO
        dataset = COCO(opt.dataset_path, opt.annotation_path, subset='train', transforms=transforms)
    else:
        raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)
    print("dataset [%s] was created" % (dataset.name()))
    return dataset


def get_validation_set(opt, transforms=None):
    if opt.dataset == 'coco':
        from data.coco import COCO
        dataset = COCO(opt.dataset_path, opt.annotation_path, subset='val', transforms=transforms)
    else:
        raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)
    print("dataset [%s] was created" % (dataset.name()))
    return dataset