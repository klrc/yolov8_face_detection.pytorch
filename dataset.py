import cv2
import torch
import numpy as np
import random
import os
from torch.utils.data import Dataset

from general import get_im_files, img2label_paths, try_load_from_cache
from box import rectangular_training, letterbox, xywhn2xyxy, xyxy2xywhn
from transforms import AlbumentationsPreset, augment_hsv, fake_osd, mosaic4, random_perspective


class WiderfaceDataset(Dataset):
    class_names = ["face"]

    # Base Class For making datasets which are compatible with nano
    def __init__(
        self,
        training,
        dataset_path,
        image_size,
        batch_size,
        stride,
        mosaic=1.0,  # image mosaic (probability)
        degrees=15,  # image rotation (+/- deg)
        translate=0.1,  # image translation (+/- fraction)
        scale=0.5,  # image scale (+/- gain)
        shear=0.0,  # image shear (+/- deg)
        perspective=0.0,  # image perspective (+/- fraction), range 0-0.001
        random_weather=0.1,
        random_blur=0.1,
        random_noise=0.1,
        random_compression=0.1,
        hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # image HSV-Value augmentation (fraction)
        flipud=0.0,
        fliplr=0.5,  # image flip left-right (probability)
        fake_osd=0.0,
        min_box_size=0,
    ):
        # check resources
        if fake_osd:
            ttf_search_path = [
                "./hzk-pixel-16px.ttf",
            ]
            for p in ttf_search_path:
                if os.path.exists(p):
                    self.fake_osd_ttf_path = p
                    break
            else:
                raise FileNotFoundError("osd ttf file not found")

        # Read cache
        im_files = get_im_files(dataset_path)
        label_files = img2label_paths(im_files)
        im_files, shapes, labels, _ = try_load_from_cache(im_files, label_files, augment=training)
        self.im_files = im_files
        self.shapes = shapes
        self.labels = labels
        self.batch_size = batch_size
        self.indices = range(len(shapes))  # number of images

        # Rectangular training
        rect = not training
        if rect:
            assert stride is not None
            n = len(shapes)  # number of images
            bi = np.floor(np.arange(n) / batch_size).astype(np.int32)  # batch index
            nb = bi[-1] + 1  # number of batches
            self.batch = bi  # batch index of image
            self.batch_shapes = rectangular_training(im_files, shapes, labels, nb, bi, image_size, stride, 0.0)

        self.rect = rect
        self.training = training
        self.image_size = image_size
        self.mosaic = mosaic
        self.mosaic_border = self.mosaic_border = [-image_size // 2, -image_size // 2]
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.albumentations = AlbumentationsPreset(
            image_size,
            random_weather=random_weather,
            random_blur=random_blur,
            random_noise=random_noise,
            random_compression=random_compression,
        )
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.flipud = flipud
        self.fliplr = fliplr
        self.fake_osd = fake_osd
        self.min_box_size = min_box_size

    def load_image(self, index):
        f = self.im_files[index]
        image = cv2.imread(f)  # BGR
        assert image is not None, f"Image Not Found {f}"
        h, w, _ = image.shape
        r = self.image_size / max(h, w)  # ratio
        if r != 1:  # if sizes are not equal
            interpolation = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            image = cv2.resize(image, (int(w * r), int(h * r)), interpolation=interpolation)
        return image

    def load_labels(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        labels = self.labels[index].copy()
        return labels

    def __getitem__(self, index):
        # Load image & labels
        if self.training and random.random() < self.mosaic:
            # Load mosaic4
            indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
            random.shuffle(indices)

            image_list, labels_list = [], []
            for i in indices:
                image_list.append(self.load_image(i))
                labels_list.append(self.load_labels(i))

            image, labels = mosaic4(image_list, labels_list, self.image_size, self.mosaic_border)
            border = self.mosaic_border
        else:
            # Load image
            image = self.load_image(index)
            h, w, _ = image.shape
            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.image_size  # final letterboxed shape
            image, ratio, pad = letterbox(image, shape, auto=False, scaleup=self.training)
            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            border = (0, 0)

        # Augmentations
        if self.training:
            # Random perspective projection
            image, labels = random_perspective(image, labels, self.degrees, self.translate, self.scale, self.shear, self.perspective, border)

            nl = len(labels)  # number of labels
            if nl:
                labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=image.shape[1], h=image.shape[0], clip=True, eps=1e-3)

            # Albumentations
            image, labels = self.albumentations(image, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(image, hgain=self.hsv_h, sgain=self.hsv_s, vgain=self.hsv_v)

            # Flip up-down
            if random.random() < self.flipud:
                image = np.flipud(image)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < self.fliplr:
                image = np.fliplr(image)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Random fake OSD
            if random.random() < self.fake_osd:
                image = fake_osd(image, self.fake_osd_ttf_path)

            # # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout
        else:
            if len(labels) > 0:
                h, w, _ = image.shape
                labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=w, h=h, clip=True, eps=1e-3)

        if len(labels) > 0:
            h, w, _ = image.shape
            labels = labels[labels[:, 3] > self.min_box_size / w]
            labels = labels[labels[:, 4] > self.min_box_size / h]
            labels[:, 1:5] = labels[:, 1:5].clip(0, 1)

        nl = len(labels)  # number of labels
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image)

        return torch.from_numpy(image).float() / 255.0, labels_out

    def __len__(self):
        return len(self.im_files)

    @staticmethod
    def collate_fn(batch):
        im, label = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0)
