# Introduction

YOLO nano is from this [paper](https://arxiv.org/abs/1910.01271).

# TODO
- [x] Finish a draft version of implementation
- [x] Add README
- [x] Add checkpoint support
- [ ] Add COCO dataset support
- [ ] Add val and test 
- [ ] Test accuracy

# Installation
```bash
git clone https://github.com/liux0614/yolo_nano
pip3 install -r requirements.txt
```
# Training
```bash
python3 main.py --dataset_path datasets --dataset coco --lr 0.001 --conf_thres 0.8 --nms_thres 0.5 --multiscale
```
