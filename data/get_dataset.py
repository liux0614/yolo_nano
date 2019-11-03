
def get_dataset(opt, transforms=None):
    if opt.dataset == 'coco':
        from data.coco import COCO
        dataset = COCO(opt.dataset_path, opt.annotation_path, subset=opt.subset, transforms=transforms)
    else:
        raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)
    print("dataset [%s] was created" % (dataset.name()))
    return dataset