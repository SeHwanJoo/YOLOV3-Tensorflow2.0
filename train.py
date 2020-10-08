import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss, YOLOv3_tiny
from core.config import cfg

is_tiny = cfg.YOLO.IS_TINY
if is_tiny:
    pred_length = 2
else:
    pred_length = 3
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
    try:
        tf.config.experimental.set_visible_devices(gpus[4], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[4], True)
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[4],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
        print(e)

trainset = Dataset('train')
testset = Dataset('test')
logdir = "./data/log"
steps_per_epoch = len(trainset)
print("steps per epoch: %d" % steps_per_epoch)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
print("total steps: %d" % cfg.TRAIN.EPOCHS)

input_tensor = tf.keras.layers.Input([416, 416, 3])
if is_tiny:
    conv_tensors = YOLOv3_tiny(input_tensor)
else:
    conv_tensors = YOLOv3(input_tensor)

output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors)
if is_tiny:
    utils.load_weights(model, "./yolov3-tiny.conv.11")
else:
    utils.load_weights(model, "./darknet53.conv.74")
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)


def test_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=False)

        giou_loss = conf_loss = prob_loss = 0
        # optimizing process
        for i in range(pred_length):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        tf.print("=> STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps,
                                                           giou_loss, conf_loss,
                                                           prob_loss, total_loss))


def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(pred_length):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                           giou_loss, conf_loss,
                                                           prob_loss, total_loss))
        # update learning rate
        global_steps.assign_add(1)
        # if global_steps.read_value() == 2:
        #     lr = 1e-4
        if global_steps < warmup_steps:
            lr = 1e-4 + ((global_steps / warmup_steps) * (1e-3 - 1e-4))
            lr = lr.numpy()
        elif global_steps < 76 * steps_per_epoch:
            lr = 1e-3
        elif global_steps < 106 * steps_per_epoch:
            lr = 1e-4
        else:
            lr = 1e-5
        optimizer.lr.assign(lr)

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


for epoch in range(cfg.TRAIN.EPOCHS):
    for image_data, target in trainset:
        train_step(image_data, target)
    for image_data, target in testset:
        test_step(image_data, target)
    model.save_weights("./checkpoints/yolov3-tiny%d" % epoch)
