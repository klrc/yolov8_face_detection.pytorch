import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor


def im2tensor(image: np.ndarray):
    assert np.issubsctype(image, np.integer)  # 0~255 BGR int -> 0~1 RGB float
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return to_tensor(image)


def tensor2im(x: torch.Tensor):
    assert isinstance(x, torch.Tensor)
    if len(x.shape) == 4:
        assert x.shape[0] == 1
        x = x.squeeze(0)
    assert len(x.shape) == 3  # 0~1 RGB float -> 0~255 BGR int
    np_img = (x * 255).int().numpy().astype(np.uint8)
    np_img = np_img[::-1].transpose((1, 2, 0))  # CHW to HWC, RGB to BGR, 0~1 to 0~255
    np_img = np.ascontiguousarray(np_img)
    return np_img


def padding(frame: torch.Tensor, pad_h: int, pad_w: int):
    n, c, h, w = frame.shape
    exp_h = h + pad_h * 2
    exp_w = w + pad_w * 2
    background = torch.zeros(n, c, exp_h, exp_w, device=frame.device)
    pad_h = (exp_h - h) // 2
    pad_w = (exp_w - w) // 2
    background[:, :, pad_h : pad_h + h, pad_w : pad_w + w] = frame  # noqa:E203
    return background


def letterbox_padding(frame: torch.Tensor, gs=32):
    if len(frame.shape) == 4:
        n, c, h, w = frame.shape
        if w % gs == 0 and h % gs == 0:
            return frame
        exp_h = math.ceil(h / gs) * gs
        exp_w = math.ceil(w / gs) * gs
        background = torch.zeros(n, c, exp_h, exp_w, device=frame.device)
        pad_h = (exp_h - h) // 2
        pad_w = (exp_w - w) // 2
        background[:, :, pad_h : pad_h + h, pad_w : pad_w + w] = frame  # noqa:E203
        return background
    elif len(frame.shape) == 3:
        c, h, w = frame.shape
        if w % gs == 0 and h % gs == 0:
            return frame
        exp_h = math.ceil(h / gs) * gs
        exp_w = math.ceil(w / gs) * gs
        background = torch.zeros(c, exp_h, exp_w, device=frame.device)
        pad_h = (exp_h - h) // 2
        pad_w = (exp_w - w) // 2
        background[:, pad_h : pad_h + h, pad_w : pad_w + w] = frame  # noqa:E203
        return background


def uniform_scale(image: np.ndarray, inf_size: int):
    height, width, _ = image.shape
    ratio = inf_size / max(height, width)
    image = cv2.resize(image, (int(ratio * width), int(ratio * height)))
    return image


def check_cv2(image):
    if isinstance(image, torch.Tensor):
        image = tensor2im(image)
    else:
        image = image.copy()
    assert isinstance(image, np.ndarray)
    return image


def safe_pt(pt):
    # forced int point coords for cv2 functions
    return [int(x) for x in pt]


class CanvasLayer:
    def __init__(self, data, alpha):
        self.data = data
        self.alpha = alpha


class Canvas:
    def __init__(self, image=None, backend="cv2") -> None:
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.3
        self.font_thickness = 1
        self.backend = backend
        self.color_presets = {}
        self.base_layer = None
        self.layers_by_order = []
        self.layers_by_alpha = {}
        if image is not None:
            self.load(image)

    def load(self, image):
        self.base_layer = self.import_image(image)

    def import_image(self, image):
        # Check & apply image colormap
        image = check_cv2(image)
        layer = CanvasLayer(image, 1)
        return layer

    def new_layer(self, alpha=1):
        if alpha in self.layers_by_alpha:
            return self.layers_by_alpha[alpha]
        assert self.base_layer is not None
        data = np.zeros_like(self.base_layer.data)
        layer = CanvasLayer(data, alpha)
        self.layers_by_order.append(layer)
        self.layers_by_alpha[alpha] = layer
        return layer

    def merge_layers(self):
        if len(self.layers_by_order) > 0:
            base = self.base_layer.data
            for layer in self.layers_by_order:
                layer: CanvasLayer
                valid = np.any(layer.data > 0, axis=2, keepdims=True).astype(base.dtype)
                invalid = 1 - valid
                alpha = layer.alpha
                current = layer.data
                if alpha >= 1:
                    base = base * invalid + current * valid
                elif alpha > 0:
                    # masked = cv2.addWeighted(base * foreground, 1 - alpha, layer * foreground, alpha, 0)
                    base = base * invalid + base * valid * (1 - alpha) + current * valid * alpha

            self.base_layer.data = base.clip(0, 255).astype(np.uint8)
            self.layers_by_order.clear()
            self.layers_by_alpha.clear()

    def color(self, id, light_theme=True):
        # get color from color queue, select random color if not found
        if id not in self.color_presets:
            if light_theme:
                color = list(np.random.random(size=3) * 128 + 128)  # light color
            else:
                color = list(np.random.random(size=3) * 128)  # dark color
            self.color_presets[id] = color
        return self.color_presets[id]

    def draw_point(self, pt1, thickness=3, radius=1, alpha=1, color=None):
        # draw a single point on canvas
        layer = self.new_layer(alpha)
        cv2.circle(
            img=layer.data,
            center=safe_pt(pt1),
            radius=radius,
            color=color if color else self.color(None),
            thickness=thickness,
        )

    def draw_line(self, pt1, pt2, thickness=3, alpha=1, color=None, lineType=cv2.LINE_AA):
        # draw a line on canvas
        layer = self.new_layer(alpha)
        cv2.line(
            img=layer.data,
            pt1=safe_pt(pt1),
            pt2=safe_pt(pt2),
            color=color if color else self.color(None),
            thickness=thickness,
            lineType=lineType,
        )

    def draw_text(self, text, pt1, alpha=1, color=(1, 1, 1), font_color=(255, 255, 255), font_scale=0.4):
        # draw labels with auto-fitting background color
        # draw background
        text_size, _ = cv2.getTextSize(text, self.font, font_scale, self.font_thickness)
        text_w, text_h = text_size
        if "\n" in text:
            for line in reversed(text.strip().split("\n")):
                self.draw_text(line, pt1, alpha, color, font_color, font_scale)
                x1, y1 = pt1
                pt1 = (x1, y1 + text_h + 2)
        else:
            layer = self.new_layer(alpha)
            x1, y1 = safe_pt(pt1)
            cv2.rectangle(
                img=layer.data,
                pt1=(x1, y1),
                pt2=(x1 + text_w + 2, y1 + text_h + 2),
                color=color,
                thickness=-1,
            )
            # draw texts
            cv2.putText(
                img=layer.data,
                text=text,
                org=(x1, y1 + text_h),
                fontFace=self.font,
                fontScale=font_scale,
                color=font_color,
                thickness=self.font_thickness,
            )

    def draw_box(self, pt1, pt2, alpha=1, thickness=1, color=None, title=None):
        # draw a bounding box / box with title on canvas
        layer = self.new_layer(alpha)
        color = color if color else self.color(title)
        cv2.rectangle(
            img=layer.data,
            pt1=safe_pt(pt1),
            pt2=safe_pt(pt2),
            color=color,
            thickness=thickness,
            lineType=4,
        )
        # draw labels with auto-fitting background color
        if title:
            text_size, _ = cv2.getTextSize(title, self.font, self.font_scale, self.font_thickness)
            _, text_h = text_size
            x1, y1 = safe_pt(pt1)
            self.draw_text(
                title,
                pt1=(x1, y1 - text_h - 2),
                font_scale=self.font_scale,
                font_color=color,
            )

    def draw_heatmap(self, feature, alpha=0.5):
        h, w, _ = self.base_layer.data.shape
        assert len(feature.shape) == 2
        feature = np.array(feature.abs())
        heatmap = None
        heatmap = cv2.normalize(feature, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (w, h))
        layer = self.new_layer(alpha)
        layer.data += heatmap

    def draw_grid(self, grid_size=10):
        h, w, _ = self.base_layer.data.shape
        assert h % grid_size == 0 and w % grid_size == 0
        for i in range(1, h // grid_size):
            self.draw_line((0, i * grid_size), (w - 1, i * grid_size), thickness=1, color=(1, 1, 1))
        for i in range(1, w // grid_size):
            self.draw_line((i * grid_size, 0), (i * grid_size, h - 1), thickness=1, color=(1, 1, 1))

    def darker(self, alpha=0.21):
        h, w, _ = self.base_layer.data.shape
        self.draw_box((0, 0), (w - 1, h - 1), alpha=alpha, thickness=-1, color=(1, 1, 1))

    def image(self):
        self.merge_layers()
        return self.base_layer.data

    def save(self, filename):
        cv2.imwrite(filename, self.image())
        return self

    def show(self, title="test", wait_key=1, set_at_top=True):
        if self.backend == "cv2":
            cv2.imshow(title, self.image())
            if set_at_top:
                cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(wait_key)
        elif self.backend == "plt":
            plt.imshow(self.image()[:, :, ::-1], aspect="auto")
            plt.axis("off")
            plt.show()
        else:
            raise NotImplementedError
        return self
