# coding=utf-8
# 导入环境
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import paddle
from paddle.io import Dataset
from paddle.nn import Conv2D, MaxPool2D, Linear, Dropout, BatchNorm, AdaptiveAvgPool2D, AvgPool2D
import paddle.nn.functional as F
import paddle.nn as nn
from quantization.quantvit import QuantVisionTransformer
import argparse
from engine import initialize_quantization
from dataset import DataLoader
from pathlib import Path
import paddle.distributed as dist

# paddle.disable_static()

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--wbits', default=8, type=int)
    parser.add_argument('--abits', default=8, type=int)
    parser.add_argument('--mixpre', action='store_true', default=False, help='whether to learn the bit-width')

# 参数初始化配置
trunc_normal_ = nn.initializer.TruncatedNormal(std=.02)
zeros_ = nn.initializer.Constant(value=0.)
ones_ = nn.initializer.Constant(value=1.)

# dist.init_parallel_env()

# 将输入 x 由 int 类型转为 tuple 类型
def to_2tuple(x):
    return tuple([x] * 2)

# 定义一个什么操作都不进行的网络层
class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


# 定义decode_image函数，将图片转为Numpy格式
def decode_image(img, to_rgb=True):
    data = np.frombuffer(img, dtype='uint8')
    img = cv2.imdecode(data, 1)
    if to_rgb:
        assert img.shape[2] == 3, 'invalid shape of image[%s]' % (
            img.shape)
        img = img[:, :, ::-1]

    return img

# 定义resize_image函数，对图片大小进行调整
def resize_image(img, size=None, resize_short=None, interpolation=-1):
    interpolation = interpolation if interpolation >= 0 else None
    if resize_short is not None and resize_short > 0:
        resize_short = resize_short
        w = None
        h = None
    elif size is not None:
        resize_short = None
        w = size if type(size) is int else size[0]
        h = size if type(size) is int else size[1]
    else:
        raise ValueError("invalid params for ReisizeImage for '\
            'both 'size' and 'resize_short' are None")

    img_h, img_w = img.shape[:2]
    if resize_short is not None:
        percent = float(resize_short) / min(img_w, img_h)
        w = int(round(img_w * percent))
        h = int(round(img_h * percent))
    else:
        w = w
        h = h
    if interpolation is None:
        return cv2.resize(img, (w, h))
    else:
        return cv2.resize(img, (w, h), interpolation=interpolation)

# 定义crop_image函数，对图片进行裁剪
def crop_image(img, size):
    if type(size) is int:
        size = (size, size)
    else:
        size = size  # (h, w)

    w, h = size
    img_h, img_w = img.shape[:2]
    w_start = (img_w - w) // 2
    h_start = (img_h - h) // 2

    w_end = w_start + w
    h_end = h_start + h
    return img[h_start:h_end, w_start:w_end, :]

# 定义normalize_image函数，对图片进行归一化
def normalize_image(img, scale=None, mean=None, std=None, order= ''):
    if isinstance(scale, str):
        scale = eval(scale)
    scale = np.float32(scale if scale is not None else 1.0 / 255.0)
    mean = mean if mean is not None else [0.485, 0.456, 0.406]
    std = std if std is not None else [0.229, 0.224, 0.225]

    shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
    mean = np.array(mean).reshape(shape).astype('float32')
    std = np.array(std).reshape(shape).astype('float32')

    if isinstance(img, Image.Image):
        img = np.array(img)
    assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
    # 对图片进行归一化
    return (img.astype('float32') * scale - mean) / std

# 定义to_CHW_image函数，对图片进行通道变换，将原通道为‘hwc’的图像转为‘chw‘
def to_CHW_image(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    # 对图片进行通道变换
    return img.transpose((2, 0, 1))

# 图像预处理方法汇总
def transform(data, mode='train'):

    # 图像解码
    data = decode_image(data)
    # 图像缩放
    data = resize_image(data, resize_short=224)
    # 图像裁剪
    data = crop_image(data, size=224)
    # 标准化
    data = normalize_image(data, scale=1./255., mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # 通道变换
    data = to_CHW_image(data)
    return data

# 读取数据，如果是训练数据，随即打乱数据顺序
def get_file_list(file_list):
    with open(file_list) as flist:
        full_lines = [line.strip() for line in flist]

    return full_lines

# 定义数据读取器
class CommonDataset(Dataset):
    def __init__(self, data_dir, file_list):
        self.full_lines = get_file_list(file_list)
        self.delimiter = ' '
        self.num_samples = len(self.full_lines)
        self.data_dir = data_dir
        return

    def __getitem__(self, idx):
        line = self.full_lines[idx]
        img_path, label = line.split(self.delimiter)
        img_path = os.path.join(self.data_dir, img_path)
        with open(img_path, 'rb') as f:
            img = f.read()
        
        transformed_img = transform(img)
        return (transformed_img, int(label))

    def __len__(self):
        return self.num_samples

DATADIR = '/data/ilsvrc12_raw_torch'
VAL_FILE_LIST = '/home/wangzihan/imagenet/val_list.txt'
TRAIN_FILE_LIST = '/home/wangzihan/imagenet/train_list.txt'
# 创建数据读取类
val_dataset = CommonDataset(DATADIR, VAL_FILE_LIST)
train_dataset = CommonDataset(DATADIR, TRAIN_FILE_LIST)
# 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
train_loader = paddle.io.DataLoader(train_dataset, batch_size=16, num_workers=0, shuffle=True)
val_loader = paddle.io.DataLoader(val_dataset, batch_size=16, num_workers=0, drop_last=True)

data_loader_sampler = DataLoader(
        val_dataset, sampler=paddle.io.SequenceSampler(val_dataset),
        batch_size=32,
        num_workers=1,
        drop_last=False
    )

# img, label = next(val_loader())
img, label = next(train_loader())
# 开启0号GPU
use_gpu = True
paddle.set_device('gpu:7') if use_gpu else paddle.set_device('cpu')



print('start training .......')


# 训练
def train_one_epoch(model, dataloader, optimizer, epoch, total_epoch, report_freq=20):
    print(f'----- Training Epoch [{epoch}/{total_epoch}]:')
    acc_set = []
    avg_loss_set = []
    model.train()
    for batch_id, data in enumerate(train_loader()):
        x_data, y_data = data
        y_data = y_data.reshape([-1, 1])
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)

        logits = model(img)
        # logits, bitops = logits[0], logits[1]
        pred = F.softmax(logits)
        loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
        loss = loss_func(logits, label)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        acc = paddle.metric.accuracy(pred, label)

        acc_set.append(acc.numpy())
        avg_loss_set.append(loss.numpy())
        print("[train] ----- Epoch[{}/{}]:batch_id: {} accuracy/loss: {}/{}".format(epoch, total_epoch, batch_id, np.mean(acc_set), np.mean(avg_loss_set)))
    print("[train] ----- Epoch[{}/{}]: accuracy/loss: {}/{}".format(epoch, total_epoch, np.mean(acc_set), np.mean(avg_loss_set)))
#
# 预测
def validate(model, dataloader):
    acc_set = []
    avg_loss_set = []
    model.eval()
    for batch_id, data in enumerate(dataloader()):
        x_data, y_data = data
        y_data = y_data.reshape([-1, 1])
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)
        # 运行模型前向计算，得到预测值
        logits = model(img)
        # logits, bitops = logits[0], logits[1]
        # 多分类，使用softmax计算预测概率
        pred = F.softmax(logits)
        # 计算交叉熵损失函数
        loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
        loss = loss_func(logits, label)
        # 计算准确率
        acc = paddle.metric.accuracy(pred, label)

        acc_set.append(acc.numpy())
        avg_loss_set.append(loss.numpy())
        print("[validation] batch_id: {} accuracy/loss: {}/{}".format(batch_id, np.mean(acc_set), np.mean(avg_loss_set)))
    print("[validation] accuracy/loss: {}/{}".format(np.mean(acc_set), np.mean(avg_loss_set)))
# 

# print('start evaluation .......')
# 实例化模型
model = QuantVisionTransformer(
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        class_dim=1000, 
        embed_dim=192, 
        depth=12,
        num_heads=3, 
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0.,
        drop_path_rate=0.,  
        norm_layer='nn.LayerNorm',
        epsilon=1e-6,
        wbits=8, 
        abits=8, 
        act_layer=nn.GELU, 
        offset=False, 
        learned=True)
        

# model = paddle.DataParallel(model)

# params_model = model.state_dict()
# params_keys = params_model.keys()
# for k in params_keys:
#     print("current model params ,key:{}, shape: {}".format(k, params_model[k].shape))

# params_loaded = paddle.load("/home/wangzihan/pretrain/DeiT_tiny_patch16_224_pretrained.pdparams")
# params_keys = params_loaded.keys()
# for k in params_keys:
#     print("loaded model params ,key:{}, shape: {}".format(k, params_loaded[k].shape))

# 加载模型参数
model_state_dict = paddle.load("/home/wangzihan/pretrain/DeiT_tiny_patch16_224_pretrained.pdparams")
model.set_state_dict(model_state_dict, use_structured_name=False)
# print(paddle.summary(model, (1, 3, 224, 224)))
device='gpu: 7'
output_dir='/home/wangzihan/Q_paddle/output'
output_dir = Path(output_dir)
initialize_quantization(data_loader_sampler, model, device, output_dir, sample_iters=1)
# print(paddle.summary(model, (1, 3, 224, 224)))
acc_set = []
avg_loss_set = []
total_epoch = 100
batch_size = 16
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(0.02, total_epoch)
optimizer = paddle.optimizer.Momentum(learning_rate=scheduler,
                                        parameters=model.parameters(),
                                        momentum=0.9,
                                        weight_decay=5e-4)
save_freq = 50
test_freq = 1 

# validate(model, val_loader)

for epoch in range(1, total_epoch+1):
        train_one_epoch(model, train_loader, optimizer, epoch, total_epoch)
        scheduler.step()

        if epoch % test_freq == 0 or epoch == total_epoch:
            validate(model, val_loader)

        if epoch % save_freq == 0 or epoch == total_epoch:
            paddle.save(model.state_dict(), f'/home/wangzihan/Q_paddle/output/Vit{epoch}.pdparams')
            paddle.save(optimizer.state_dict(), f'/home/wangzihan/Q_paddle/output/Vit{epoch}.pdopts')