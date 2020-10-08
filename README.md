# YOLO V3
## paper
[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
## version
python : 3.7.8

tensorflow : 2.2.0

## train
### hyper parameter
    STRIDES = [8, 16, 32]
    STRIDES_TINY = [16, 32]
    ANCHOR_PER_SCALE = 3
    IOU_LOSS_THRESH = 0.5
    MOVING_AVE_DECAY = 0.9995
    BATCH_SIZE = 4
    INPUT_SIZE = [416]
    LR_INIT = 1e-4
    LR_END = 1e-6
    WARMUP_EPOCHS = 1
    EPOCHS = 136
    
### data augment
#### random horizontal flip
    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes
        
#### random crop
    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes
        
### pretrained model

yolov3 use [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)

yolov3-tiny use [yolov3-tiny.conv11](https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view)

### benchmark
|Model|mAP 50|train dataset|val dataset|
|:------:|:---:|:---:|:---:|
|yolov3|49.05%|VOC 2007 train|VOC 2007 val|
|yolov3-tiny|training|VOC2007 train + VOC2012 train|VOC 2007 val|
|yolov3|training|VOC2007 train + VOC2012 train|VOC 2007 val|
|yolov3 (in paper)|80.25%|VOC2007 train + VOC2012 train|VOC 2007 val|
|yolov3 (in paper)|55.3%|COCO train|COCO val|
|yolov3-tiny|33.1%|COCO train|COCO val|


### YOLO V3 mAP
    60.55% = aeroplane AP
    59.61% = bicycle AP
    34.77% = bird AP
    29.81% = boat AP
    21.59% = bottle AP
    54.67% = bus AP
    68.62% = car AP
    53.91% = cat AP
    30.35% = chair AP
    48.36% = cow AP
    42.37% = diningtable AP
    46.38% = dog AP
    65.49% = horse AP
    63.44% = motorbike AP
    68.24% = person AP
    27.39% = pottedplant AP
    45.50% = sheep AP
    45.06% = sofa AP
    63.74% = train AP
    51.23% = tvmonitor AP
    mAP = 49.05%
    
### detect image
![000001](https://user-images.githubusercontent.com/24911666/95420197-0c2d0500-0976-11eb-8af2-15c815635cae.jpg)
![000070](https://user-images.githubusercontent.com/24911666/95420201-0df6c880-0976-11eb-8ecf-d8f64e4dc973.jpg)
![000108](https://user-images.githubusercontent.com/24911666/95420206-10592280-0976-11eb-88a2-53c33626155f.jpg)
![000185](https://user-images.githubusercontent.com/24911666/95420214-12bb7c80-0976-11eb-8068-a73495bb9f14.jpg)
![000216](https://user-images.githubusercontent.com/24911666/95420223-14854000-0976-11eb-815b-554e08151fce.jpg)
![000239](https://user-images.githubusercontent.com/24911666/95420225-164f0380-0976-11eb-9e89-8f22dec9aa4d.jpg)
![000280](https://user-images.githubusercontent.com/24911666/95420247-1c44e480-0976-11eb-904e-a8086e4497b4.jpg)

### train image
![000073](https://user-images.githubusercontent.com/24911666/95420358-40082a80-0976-11eb-9707-013f3f1dcc31.jpg)
![000095](https://user-images.githubusercontent.com/24911666/95420360-41395780-0976-11eb-8c74-a9d617545a49.jpg)
![000112](https://user-images.githubusercontent.com/24911666/95420362-426a8480-0976-11eb-9842-99ef5167de6c.jpg)
![000129](https://user-images.githubusercontent.com/24911666/95420364-43031b00-0976-11eb-8833-682c918bd3ed.jpg)
![000140](https://user-images.githubusercontent.com/24911666/95420366-44344800-0976-11eb-8472-561868b41b3e.jpg)
![000173](https://user-images.githubusercontent.com/24911666/95420372-45657500-0976-11eb-826f-e6cc4044ba54.jpg)