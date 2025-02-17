import math
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
import glob
import xml.etree.ElementTree as ET

# Access math (am) dataset info
am_test_root = '/data/Projects/accessmath-icfhr2018/AccessMathVOC/'
train_lectures = ['lecture_01', 'lecture_06', 'lecture_18', 'NM_lecture_01', 'NM_lecture_03']

def get_imglist(img_dir, subfolders=None, keyword=None):
    img_list = []
    if subfolders is None:
        for root, dirs, files in os.walk(img_dir, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                if keyword is None or keyword in file_path:
                    img_list.append(file_path)
    else:
        for subfolder in subfolders:
            for root, dirs, files in os.walk(os.path.join(img_dir, subfolder), topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    if keyword is None or keyword in file_path:
                        img_list.append(file_path)
    print(img_list[0:100:])
    return img_list

def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception:
        print('Cannot read image: %s.' % img_path)
        raise
    return img


def load_object_xml(root):
    name = root.find('name').text
    bndbox = root.find('bndbox')

    xmin = float(bndbox.find("xmin").text)
    ymin = float(bndbox.find("ymin").text)
    xmax = float(bndbox.find("xmax").text)
    ymax = float(bndbox.find("ymax").text)

    return name, xmin, ymin, xmax, ymax



def get_ann(img, input_xml_file):
    h, w = img.shape[0:2]

    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    # .text
    filename = root.find('filename').text
    objects = root.findall("object")
    bboxes = []
    words = []
    for o in objects:
        name, xmin, ymin, xmax, ymax = load_object_xml(o)
        if name == 'text':
            bbox_temp = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
            bbox = np.array(bbox_temp) / ([w * 1.0, h * 1.0] * 4)
            bboxes.append(bbox)
            words.append('???')
    return np.array(bboxes), words

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


def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
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


def random_scale(img, short_size=736):
    h, w = img.shape[0:2]

    scale = np.random.choice(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]))
    scale = (scale * short_size) / min(h, w)

    aspect = np.random.choice(np.array([0.9, 0.95, 1.0, 1.05, 1.1]))
    h_scale = scale * math.sqrt(aspect)
    w_scale = scale / math.sqrt(aspect)

    img = scale_aligned(img, h_scale, w_scale)
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
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img,
                                       0,
                                       p_h - t_h,
                                       0,
                                       p_w - t_w,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=(0,))
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

            shrinked_bbox = np.array(shrinked_bbox)[0]
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception:
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes

class PAN_ACCESSMATH_TESTONTRAIN(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=736,
                 kernel_scale=0.5,
                 with_rec=False,
                 read_type='pil',
                 report_speed=False):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (
            img_size is None or isinstance(img_size, tuple)) else (img_size,
                                                                   img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.with_rec = with_rec
        self.read_type = read_type

        # get image file list and gt file list
        if split == 'test':  # comment out for inference
            self.img_paths = get_imglist(am_test_root, keyword='JPEGImages')
        else:
            print('Error: split must be train or test!')
            raise
        print('number of images: ', len(self.img_paths))
        if report_speed:
            target_size = 3000
            extend_scale = (target_size + len(self.img_paths) - 1) // len(
                self.img_paths)
            self.img_paths = (self.img_paths * extend_scale)[:target_size]
            self.gt_paths = (self.gt_paths * extend_scale)[:target_size]
        self.max_word_num = 200
        print('reading type: %s.' % self.read_type)

    def __len__(self):
        return len(self.img_paths)

    def prepare_train_data(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann(img, gt_path)
        if bboxes.shape[0] > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]

        if self.is_transform:
            img = random_scale(img, self.short_size)
        
        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 4),
                                (bboxes.shape[0], -1, 2)).astype('int32')
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)

                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)


        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            if not self.with_rec:
                imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[
                1], imgs[2], imgs[3:]

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        img = Image.fromarray(img)
        img = img.convert('RGB')
        if self.is_transform:
            img = transforms.ColorJitter(brightness=32.0 / 255,
                                         saturation=0.5)(img)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)
        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,
        )

        return data

    def prepare_test_data(self, index):
        img_path = self.img_paths[index]
        print('------------', index, img_path)

        img = get_img(img_path, self.read_type)
        img_meta = dict(org_img_size=np.array(img.shape[:2]))

        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(img_size=np.array(img.shape[:2])))
        img_meta.update(dict(img_name=os.path.basename(img_path), img_path=img_path))

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
