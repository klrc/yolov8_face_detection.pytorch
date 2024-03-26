import torch
import os
import wandb
from tqdm import tqdm
from loguru import logger
import numpy as np
import argparse

from model import yolov8n
from dataloader import InfiniteDataLoader
from dataset import WiderfaceDataset
from general import init_seeds, generate_hash, forced_load
from model_loss import create_optimizer, create_scheduler, EarlyStop, YOLOv8Loss
from ema import ModelEMA
from box import box_iou, scale_coords, xywh2xyxy
from copy import deepcopy


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px = np.linspace(0, 1, 1000)
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype("int32")


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:5], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@torch.no_grad()
def val(model, val_loader, class_names, half=True, verbose=True, debug_mode=False):
    # Initialize/load model and set device
    device = next(model.parameters()).device
    half &= device.type != "cpu"  # half precision only supported on CUDA
    model.half() if half else model.float()

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    nc = len(class_names)  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    names = {k: v for k, v in enumerate(class_names)}
    s = ("%20s" + "%11s" * 6) % ("Class", "Images", "Labels", "P", "R", "mAP@.5", "mAP@.5:.95")
    p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats, ap, ap_class = [], [], []
    pbar = tqdm(val_loader, desc=s, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # progress bar
    for im, targets in pbar:
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        _, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        out = model(im)  # inference outputs

        # Target denormalization
        targets[:, 2:6] *= torch.tensor((width, height, width, height), device=device)  # to pixels

        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:6]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], im[si].shape[1:])  # native-space pred
            # scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, im[si].shape[1:])  # native-space labels
                # scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

        if debug_mode:
            break

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    if verbose:
        # Print results
        pf = "%20s" + "%11i" * 2 + "%11.3g" * 4  # print format
        print(pf % ("all", seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map), maps


def train(
    project_name,
    debug_mode,
    max_epochs,
    loss_titles,
    device_id,
    trainset_path,
    valset_path,
    batch_size,
    image_size,
    strides,
    class_names,
    model_name,
    save_dir,
    pretrained_pt_path,
    autocast_enabled,
    optimizer_type,
    lr0,
    lrf,
    momentum,
    weight_decay,
    warmup_epochs,
    warmup_momentum,
    warmup_bias_lr,
    nbs,
    cos_lr,
    patience,
    ema_enabled,
    wandb_enabled,
    early_stop,
    raw_args={},
):
    # init random seeds
    init_seeds()

    # create device
    device = torch.device(device_id)
    print(f"set device to {device.type}[{device.index}]")
    num_workers = os.cpu_count()
    autocast_enabled &= device.type != "cpu"

    # create loaders
    assert trainset_path is not None
    assert valset_path is not None
    max_stride = max(strides)  # 输入尺寸对齐（=最大原图输出网格缩放倍数）
    train_loader = InfiniteDataLoader(
        dataset=WiderfaceDataset(
            training=True,
            dataset_path=trainset_path,
            batch_size=batch_size,
            image_size=image_size,
            stride=max_stride,
        ),
        training=True,
        batch_size=batch_size,
        collate_fn=WiderfaceDataset.collate_fn,
        num_workers=num_workers,
    )
    val_loader = InfiniteDataLoader(
        dataset=WiderfaceDataset(
            training=False,
            dataset_path=valset_path,
            batch_size=batch_size,
            image_size=image_size,
            stride=max_stride,
        ),
        training=False,
        batch_size=batch_size,
        collate_fn=WiderfaceDataset.collate_fn,
        num_workers=num_workers,
    )

    # rescale hyps
    best_fitness = 0.0
    nb = len(train_loader)
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    weight_decay = weight_decay * accumulate / nbs  # scale weight_decay
    last_opt_step = -1
    nw = max(round(warmup_epochs * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)

    # create model
    model = yolov8n(num_classes=len(class_names))
    if pretrained_pt_path is not None:
        model = forced_load(model, pretrained_pt_path)
    model = model.to(device)

    # create optimizer
    optimizer = create_optimizer(model, optimizer_type, lr0, momentum, weight_decay)
    scheduler, lf = create_scheduler(optimizer, lrf, max_epochs, cos_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type != "cpu"))
    if early_stop:
        early_stopper = EarlyStop(patience=patience)
    if ema_enabled:
        ema = ModelEMA(model)

    # create loss
    loss_func = YOLOv8Loss(model.head.nc, model.head.no, model.head.strides)

    # create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create wandb logger
    wandb_enabled &= not debug_mode
    if wandb_enabled:
        wandb.init(project=project_name, dir=save_dir, mode="offline")
        wandb.watch(model)
        wandb.run.log_code("./", include_fn=lambda path: path.endswith(".py"))
        wandb.config.update(raw_args)
        logger.success("Wandb logger created")

    # start training loop
    for epoch in range(max_epochs):
        mloss = torch.zeros(len(loss_titles), device=device)  # mean losses

        # Set progress bar
        pbar = enumerate(train_loader)
        print(("\n" + "%10s" * (4 + len(loss_titles))) % ("Epoch", "gpu_mem", *loss_titles, "labels", "img_size"))
        pbar = tqdm(pbar, total=nb, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # progress bar

        model.train()  # set to training mode
        optimizer.zero_grad()

        for i, (imgs, targets) in pbar:
            imgs: torch.Tensor
            targets: torch.Tensor

            # warmup_step
            ni = i + nb * epoch  # number integrated batches (since train start)
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [warmup_momentum, momentum])

            # forward step
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                out = model(imgs)
                loss, loss_items = loss_func(out, targets.to(device))  # forward

            # check nan
            if any(torch.isnan(loss_items)):
                logger.warning(f"nan value found in loss: {loss_items}")
                continue

            # backward step
            if autocast_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                if autocast_enabled:
                    scaler.unscale_(optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                    optimizer.step()
                optimizer.zero_grad()
                if ema_enabled:
                    ema.update(model)
                last_opt_step = ni

            # log training info
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f"{torch.cuda.memory_reserved(device) / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            pbar.set_description(("%10s" * 2 + "%10.4g" * (2 + len(loss_items))) % (f"{epoch+1}/{max_epochs}", mem, *mloss, targets.shape[0], imgs.shape[-1]))

            # quit epoch loop in debug_mode
            if debug_mode:
                break

        # after an epoch --------------
        scheduler.step()

        # update best mAP
        model_for_val = model if not ema_enabled else ema.ema
        results, _ = val(model_for_val, val_loader, class_names, half=autocast_enabled, debug_mode=debug_mode)

        # update wandb
        if wandb_enabled:
            for metric, value in zip([f"loss_{ln}" for ln in loss_titles], mloss):
                wandb.log({metric: value.detach().item()})
            for metric, value in zip(["P", "R", "mAP@.5", "mAP@.5:.95"], results):
                wandb.log({metric: value if isinstance(value, float) else value.detach().item()})

        # save best model pt
        fi = np.array(results).reshape(1, -1)
        fi = float((fi[:, :4] * [0.0, 0.0, 0.1, 0.9]).sum())  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        if fi > best_fitness:
            best_fitness = fi
        if best_fitness == fi:
            if ema_enabled:
                torch.save(deepcopy(ema.ema).state_dict(), f"{save_dir}/{model_name}.pt")
            else:
                torch.save(deepcopy(model).state_dict(), f"{save_dir}/{model_name}.pt")
        torch.save(deepcopy(model).state_dict(), f"{save_dir}/{model_name}_latest.pt")

        # check early stop
        if early_stop and early_stopper(epoch=epoch, fitness=fi):
            break

    # finish training
    if wandb_enabled:
        wandb.finish()  # Marks a run as finished, and finishes uploading all data.

    torch.cuda.empty_cache()
    logger.success("CUDA memory successfully recycled")


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="YOLOv8 Pedestrian Detection Training Configuration")

    # 添加命令行参数
    parser.add_argument("--project_name", type=str, default="pedestrian_detection", help="项目名称")
    parser.add_argument("--debug_mode", action="store_true", help="是否开启debug模式(开启后每epoch均只运行1个iter)")
    parser.add_argument("--max_epochs", type=int, default=100, help="最大训练epoch数")
    parser.add_argument("--loss_titles", nargs="+", default=["box", "cls", "dfl"], help="默认yolov8损失函数项")
    parser.add_argument("--device_id", type=str, default="mps", help="设备类型: cuda, mps, cpu, 0, 1, ...(显卡device id)")
    parser.add_argument("--trainset_path", type=str, default="/Volumes/ASM236X/datasets/RelativeHuman/images/train", help="训练集images文件夹位置, 会根据images->labels规则寻找标签文件夹")
    parser.add_argument("--valset_path", type=str, default="/Volumes/ASM236X/datasets/RelativeHuman/images/val", help="测试集文件夹位置")
    parser.add_argument("--batch_size", type=int, default=3, help="批处理大小")
    parser.add_argument("--image_size", type=int, default=640, help="图像尺寸")
    parser.add_argument("--strides", nargs="+", default=[8, 16, 32], help="输入尺寸对齐(=最大原图输出网格缩放倍数)")
    parser.add_argument("--class_names", nargs="+", default=WiderfaceDataset.class_names, help="类别名称")
    parser.add_argument("--model_name", type=str, default=generate_hash(), help="模型名称, 默认生成随机hash")
    parser.add_argument("--save_dir", type=str, default="./runs", help="日志&权重存储路径")
    parser.add_argument("--pretrained_pt_path", type=str, default=None, help="预训练模型路径")
    parser.add_argument("--autocast_enabled", action="store_true", help="是否启用autocast, 目前有已知的精度issue")
    parser.add_argument("--optimizer_type", type=str, default="SGD", help="优化器选项, 可选: Adam, AdamW, RMSProp, SGD")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率")
    parser.add_argument("--momentum", type=float, default=0.937, help="动量")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="权重衰减")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="warmup轮数")
    parser.add_argument("--warmup_momentum", type=float, default=0.8, help="warmup动量")
    parser.add_argument("--warmup_bias_lr", type=float, default=0.1, help="warmup学习率")
    parser.add_argument("--nbs", type=int, default=64, help="nominal batch size")
    parser.add_argument("--cos_lr", action="store_true", help="是否使用cosine lr scheduler")
    parser.add_argument("--patience", type=int, default=30, help="earlystop触发轮数")
    parser.add_argument("--ema_enabled", action="store_true", help="是否启用Model EMA平均")
    parser.add_argument("--wandb_enabled", action="store_true", help="是否启用wandb日志")
    parser.add_argument("--early_stop", action="store_true", help="是否启用earlystop策略")

    # 解析命令行参数
    args = parser.parse_args()
    train(**vars(args), raw_args=args)
