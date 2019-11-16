
def get_model(opt):
    if opt.model == 'nano' or opt.model == 'yolo_nano':
        from .yolo_nano import YOLONano
        model = YOLONano(opt.num_classes, opt.image_size)
    else:
        raise NotImplementedError('the model [%s] is not implemented' % opt.model)
    print("model [%s] was created" % (model.name()))

    
    return model
