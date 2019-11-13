# Introduction

YOLO nano is from this [paper](https://arxiv.org/abs/1910.01271).

# TODO
- [x] Finish a draft version of implementation
- [x] Add README
- [x] Add checkpoint support
- [x] Add COCO dataset support (Code still needs cleaning. I'm working on it.)
- [x] Add _multi scale_ and _horizontal flip_ transforms
- [x] Reconstruct the code of visualizer
- [x] Add val
- [ ] Add test
- [ ] Test accuracy

# Project Structure
<pre>
root/
  results/
  datasets/
    coco/
      images/
        train/
        val/
      annotation/
        instances_train2017.json
</pre>

# Installation
```bash
git clone https://github.com/liux0614/yolo_nano
pip3 install -r requirements.txt
```

# COCO
To use COCO dataset loader, _pycocotools_ should be installed via the following command.
```bash 
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

To train on COCO dataset:
```bash
python3 main.py --dataset_path datasets/coco/images --annotation_path datasets/coco/annotation/instances_train2017.json 
                --dataset coco --lr 0.0001 --conf_thres 0.8 --nms_thres 0.5
```
