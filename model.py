import math

import torch
import torch.nn as nn

from nms import NMS


class ConvModule(nn.Module):
    def __init__(self, c_in, c_out, k, s, p, act, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(c_out)
        if act == "relu6":
            self.relu = nn.ReLU6()
        elif act == "relu":
            self.relu = nn.ReLU()
        elif act == "leakyrelu":
            self.relu = nn.LeakyReLU()
        elif act == "hardswish":
            self.relu = nn.Hardswish()
        else:
            raise NotImplementedError(f"conv with activation={act} not implemented yet")
        self.fused = False

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def fused_forward(self, x):
        return self.relu(self.conv(x))

    def fuse(self):
        if self.fused:
            return self
        std = (self.bn.running_var + self.bn.eps).sqrt()
        bias = self.bn.bias - self.bn.running_mean * self.bn.weight / std

        t = (self.bn.weight / std).reshape(-1, 1, 1, 1)
        weights = self.conv.weight * t

        self.conv = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            bias=True,
            padding_mode=self.conv.padding_mode,
        )
        self.conv.weight = torch.nn.Parameter(weights)
        self.conv.bias = torch.nn.Parameter(bias)
        self.forward = self.fused_forward
        self.fused = True
        return self


class DarknetBottleneck(nn.Module):
    def __init__(self, c_in, c_out, add=True, act=None):
        super().__init__()
        self.cv1 = ConvModule(c_in, int(0.5 * c_in), 3, 1, 1, act=act)
        self.cv2 = ConvModule(int(0.5 * c_in), c_out, 3, 1, 1, act=act)
        self.shortcut = add

    def forward(self, x):
        if self.shortcut:
            out = self.cv1(x)
            out = self.cv2(out)
            return x + out
        else:
            x = self.cv1(x)
            x = self.cv2(x)
            return x


class CSPLayer_2Conv(nn.Module):
    def __init__(self, c_in, c_out, add, n, act):
        super().__init__()
        half_out = int(0.5 * c_out)
        self.conv_in_left = ConvModule(c_in, half_out, 1, 1, 0, act=act)  # same result as split later
        self.conv_in_right = ConvModule(c_in, half_out, 1, 1, 0, act=act)  # same result as split later
        self.bottlenecks = nn.ModuleList()
        for _ in range(n):
            self.bottlenecks.append(DarknetBottleneck(half_out, half_out, add, act=act))
        self.conv_out = ConvModule(half_out * (n + 2), c_out, 1, 1, 0, act=act)

    def forward(self, x):
        x_left = self.conv_in_left(x)
        x_right = self.conv_in_right(x)  # main branch
        collection = [x_left, x_right]
        x = x_right
        for b in self.bottlenecks:
            x = b(x)
            collection.append(x)
        x = torch.cat(collection, dim=1)
        x = self.conv_out(x)
        return x


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c_in, k=5, act=None):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c_in // 2  # hidden channels
        self.cv1 = ConvModule(c_in, c_, 1, 1, 0, act=act)
        self.cv2 = ConvModule(c_ * 4, c_in, 1, 1, 0, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class YOLOv8Backbone(nn.Module):
    def __init__(self, d, w, r, act):
        super().__init__()
        _64xw = int(64 * w)
        _128xw = int(128 * w)
        _256xw = int(256 * w)
        _512xw = int(512 * w)
        _512xwxr = int(512 * w * r)
        _3xd = int(math.ceil(3 * d))
        _6xd = int(math.ceil(6 * d))
        self.stem_layer = ConvModule(3, _64xw, k=3, s=2, p=1, act=act)
        self.stage_layer_1 = nn.Sequential(
            ConvModule(_64xw, _128xw, k=3, s=2, p=1, act=act),
            CSPLayer_2Conv(_128xw, _128xw, add=True, n=_3xd, act=act),
        )
        self.stage_layer_2 = nn.Sequential(
            ConvModule(_128xw, _256xw, k=3, s=2, p=1, act=act),
            CSPLayer_2Conv(_256xw, _256xw, add=True, n=_6xd, act=act),
        )
        self.stage_layer_3 = nn.Sequential(
            ConvModule(_256xw, _512xw, k=3, s=2, p=1, act=act),
            CSPLayer_2Conv(_512xw, _512xw, add=True, n=_6xd, act=act),
        )
        self.stage_layer_4 = nn.Sequential(
            ConvModule(_512xw, _512xwxr, k=3, s=2, p=1, act=act),
            CSPLayer_2Conv(_512xwxr, _512xwxr, add=True, n=_3xd, act=act),
            SPPF(_512xwxr, act=act),
        )

    def forward(self, x):
        p1 = self.stem_layer(x)
        p2 = self.stage_layer_1(p1)
        p3 = self.stage_layer_2(p2)
        p4 = self.stage_layer_3(p3)
        p5 = self.stage_layer_4(p4)
        return p3, p4, p5


class YOLOv8Neck(nn.Module):
    def __init__(self, d, w, r, act) -> None:
        _3xd = int(math.ceil(3 * d))
        _256xw = int(256 * w)
        _512xw = int(512 * w)
        super().__init__()
        self.upsample_p5 = nn.Upsample(None, 2, "nearest")
        self.upsample_p4 = nn.Upsample(None, 2, "nearest")
        self.topdown_layer_2 = CSPLayer_2Conv(int(512 * w * (1 + r)), _256xw, add=False, n=_3xd, act=act)
        self.topdown_layer_1 = CSPLayer_2Conv(_512xw, _256xw, add=False, n=_3xd, act=act)
        self.down_sample_0 = ConvModule(_256xw, _256xw, k=3, s=2, p=1, act=act)
        self.bottomup_layer_0 = CSPLayer_2Conv(_512xw, _512xw, add=False, n=_3xd, act=act)
        self.down_sample_1 = ConvModule(_512xw, _512xw, k=3, s=2, p=1, act=act)
        self.bottomup_layer_1 = CSPLayer_2Conv(int(512 * w * (1 + r)), int(512 * w * r), add=False, n=_3xd, act=act)

    def forward(self, x):
        p3, p4, p5 = x
        # top-down P5->P4
        u1 = self.upsample_p5(p5)
        # forward in P4
        c1 = torch.cat((p4, u1), dim=1)
        t1 = self.topdown_layer_2(c1)
        # top-down P4->P3
        u2 = self.upsample_p4(t1)
        # forward in P3
        c2 = torch.cat((p3, u2), dim=1)
        t2 = self.topdown_layer_1(c2)
        # bottom-up P3->P4
        d1 = self.down_sample_0(t2)
        # forward in P4
        c3 = torch.cat((t1, d1), dim=1)
        b1 = self.bottomup_layer_0(c3)
        # bottom-up P4->P5
        d2 = self.down_sample_1(b1)
        # forward in P5
        c4 = torch.cat((p5, d2), dim=1)
        b2 = self.bottomup_layer_1(c4)
        return t2, b1, b2


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        # sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)


class CoupledHead(nn.Module):
    def __init__(self, in_channels, nc=1, stride=8, act=None) -> None:
        super().__init__()
        assert nc == 1
        xw = in_channels
        self.bbox_branch = nn.Sequential(
            ConvModule(xw, xw, k=3, s=1, p=1, act=act),
            ConvModule(xw, xw, k=3, s=1, p=1, act=act),
            nn.Conv2d(xw, 4 + 1, kernel_size=1, stride=1),
        )
        self.stride = stride

    def forward(self, x, porting=False):
        y = self.bbox_branch(x)
        if porting:
            return y
        return torch.split(y, (4, 1), dim=1)


class YOLOv8Head(nn.Module):
    def __init__(self, w, r, act, nc, conf_threshold=0.01, iou_threshold=0.6):
        super().__init__()
        self.decoupled_head_p3 = CoupledHead(int(256 * w), nc=nc, stride=8, act=act)
        self.decoupled_head_p4 = CoupledHead(int(512 * w), nc=nc, stride=16, act=act)
        self.decoupled_head_p5 = CoupledHead(int(512 * w * r), nc=nc, stride=32, act=act)
        self.no = nc + 4  # number of outputs per anchor
        self.nc = nc

        self.strides = (8, 16, 32)
        self.porting = False
        self.shape_cache = None
        self.anchors_cache = None
        self.strides_cache = None
        self.nms = NMS(nc, conf_threshold, iou_threshold)

    def forward(self, x):
        p3, p4, p5 = x
        if self.porting:
            return self.decoupled_head_p3(p3, porting=True), self.decoupled_head_p4(p4, porting=True), self.decoupled_head_p5(p5, porting=True)

        x_p3 = self.decoupled_head_p3(p3)
        x_p4 = self.decoupled_head_p4(p4)
        x_p5 = self.decoupled_head_p5(p5)

        if self.training:
            return x_p3, x_p4, x_p5

        shape = x_p3.shape
        if self.shape_cache != shape:
            self.anchors_cache, self.strides_cache = (x.transpose(0, 1) for x in make_anchors([x_p3, x_p4, x_p5], self.strides, 0.5))
            self.shape_cache = shape

        outputs = torch.cat([xi.view(p3.shape[0], self.no, -1) for xi in [x_p3, x_p4, x_p5]], 2)
        outputs = self.post_process(outputs)
        return outputs

    def post_process(self, x):
        box, cls = x.split((4, self.nc), 1)
        box = dist2bbox(box, self.anchors_cache.unsqueeze(0), xywh=True, dim=1) * self.strides_cache
        y = torch.cat((box, cls.sigmoid()), 1)  # xyxy + sigmoid(cls) for each grid
        y = self.nms(y)
        return y


def init_model_params(model: nn.Module):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
    return model


class Model(nn.Module):
    def __init__(self, num_classes=80, d=0.33, w=0.25, r=2.0, act="relu") -> None:
        super().__init__()
        self.backbone = YOLOv8Backbone(d, w, r, act)
        self.neck = YOLOv8Neck(d, w, r, act)
        self.head = YOLOv8Head(w, r, act, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


def yolov8n(num_classes=1):
    return Model(num_classes, d=0.33, w=0.25, r=2.0, act="relu6")


if __name__ == "__main__":
    import time

    model = yolov8n()
    model.head.porting = True
    model.eval()
    x = torch.rand(1, 3, 352, 640)

    st = time.time()
    test_round = 100
    for i in range(test_round):
        y = model(x)
    et = time.time()
    print("avg time: {}".format((et - st) / test_round))
    # print(y)
