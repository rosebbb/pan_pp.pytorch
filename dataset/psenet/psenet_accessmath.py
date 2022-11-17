import random
import os
import cv2
import mmcv
import numpy as np
import Polygon as plg
import pyclipper
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

''' for test only '''
test_data_dir = '/data/Datasets/AccessMath/AccessMath_ICDAR_2017_data/data/annotations'

def get_imglist(img_dir, keyword=None):
    img_list = []
    for root, dirs, files in os.walk(img_dir, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            if keyword is None or keyword in file_path:
                img_list.append(file_path)
    return img_list
    
def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception:
        print(img_path)
        raise
    return img


def get_ann(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)

        bboxes.append(bbox)
        words.append('???')
    return bboxes, words


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img,
                                      rotation_matrix, (h, w),
                                      flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, scale):
    h, w = img.shape[0:2]
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=736):
    h, w = img.shape[0:2]

    random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    scale = (np.random.choice(random_scale) * short_size) / min(h, w)

    img = scale_aligned(img, scale)
    return img


def scale_aligned_short(img, short_size=736):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_crop_padding(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img,
                                       0,
                                       p_h - t_h,
                                       0,
                                       p_w - t_w,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0
                                                   for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img,
                                       0,
                                       p_h - t_h,
                                       0,
                                       p_w - t_w,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=(0, ))
        n_imgs.append(img_p)
    return n_imgs


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5),
                         max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception:
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


class PSENET_ACCESSMATH(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=736,
                 kernel_num=7,
                 min_scale=0.4,
                 read_type='pil',
                 report_speed=False):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (
            img_size is None or isinstance(img_size, tuple)) else (img_size,
                                                                   img_size)
        self.kernel_num = kernel_num
        self.min_scale = min_scale
        self.short_size = short_size
        self.read_type = read_type

        if split == 'test':
            self.img_paths = get_imglist(test_data_dir, keyword='keyframes')
        else:
            print('Error: split must be test!')
            raise

        if report_speed:
            target_size = 3000
            data_size = len(self.img_paths)
            extend_scale = (target_size + data_size - 1) // data_size
            self.img_paths = (self.img_paths * extend_scale)[:target_size]

        self.max_word_num = 200

    def __len__(self):
        return len(self.img_paths)

    def prepare_test_data(self, index):
        img_path = self.img_paths[index]
        print('------------', index, img_path)

        img = get_img(img_path, self.read_type)
        img_meta = dict(org_img_size=np.array(img.shape[:2]), img_name=[os.path.basename(img_path)], img_path=img_path)

        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(img_size=np.array(img.shape[:2])))

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)
        data = dict(imgs=img, img_metas=img_meta)

        return data

    def __getitem__(self, index):
        if self.split == 'train':
            return self.prepare_train_data(index)
        elif self.split == 'test':
            return self.prepare_test_data(index)
