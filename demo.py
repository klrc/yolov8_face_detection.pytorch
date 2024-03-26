import torch
import numpy as np
import cv2
import os
import imageio
from multiprocessing import Process, Queue
import time
from typing import Iterable
import mss
import pyautogui as pag
import argparse
from pynput import keyboard

from dataset import FhPedestrianDataset
from canvas import Canvas, im2tensor, letterbox_padding
from general import Timer, VideoSaver
from model import yolov8n_pvdet

IS_DETACHED = False


def pad(image: np.ndarray, pt, pl, pb, pr, padding_value):
    image = np.pad(image, ((pt, pb), (pl, pr), (0, 0)), "constant", constant_values=padding_value)
    return image


def scale(image: np.ndarray, ratio):
    height, width, _ = image.shape
    image = cv2.resize(image, (int(ratio * width), int(ratio * height)))
    return image


def scale_and_pad(image: np.ndarray, dsize, padding_value=114):
    w, h = dsize
    height, width, _ = image.shape
    ratio = min(w / width, h / height)
    image = scale(image, ratio)
    height, width, _ = image.shape
    pt, pl = (h - height) // 2, (w - width) // 2
    pb, pr = h - height - pt, w - width - pl
    image = pad(image, pt, pl, pb, pr, padding_value)
    return image


def camera_stream(size=None, exit_key="q", horizontal_flip=True):
    if isinstance(size, int):
        size = (size, size)

    cap = cv2.VideoCapture(0)
    assert cap.isOpened()

    # Warm-up camera
    for _ in range(3):
        _ = cap.read()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(exit_key):
            cap.release()
            break

        success, frame = cap.read()
        if not success:
            cap.release()
            raise Exception("camera connection lost")

        if size:
            frame = scale_and_pad(frame, size)
        if horizontal_flip:
            frame = cv2.flip(frame, 1)
        yield frame

        while IS_DETACHED:
            time.sleep(1)


def gif_stream(file_path, size=None, exit_key="q", color_format=cv2.COLOR_RGB2BGR):
    if isinstance(size, int):
        size = (size, size)

    cap = imageio.mimread(file_path)
    for frame in cap:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(exit_key):
            break

        if size:
            frame = scale_and_pad(frame, size)
        if color_format:
            frame = cv2.cvtColor(frame, color_format)
        yield frame
        while IS_DETACHED:
            time.sleep(1)


def h264_stream(file_path, size=None, exit_key="q", color_format=None, show_cap_fps=False):
    if isinstance(size, int):
        size = (size, size)

    cap = cv2.VideoCapture(file_path)
    if show_cap_fps:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("video fps:", fps)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(exit_key):
            cap.release()
            break

        ret, frame = cap.read()
        if not ret:
            break
        if size:
            frame = scale_and_pad(frame, size)
        if color_format:
            frame = cv2.cvtColor(frame, color_format)
        yield frame
        while IS_DETACHED:
            time.sleep(1)


def realtime_sampled_work_process(skipped: Queue, stop_flag: Queue, fps: int):
    """
    The function that runs as a separate process and adds frames to the buffer.

    Args:
    - skipped (Queue): skipped frames.
    - fps (int): The frames per second.
    """
    while stop_flag.empty():
        if skipped.full():
            raise NotImplementedError("too many skipped frames")
        time.sleep(1 / fps)  # simulated fps
        skipped.put(1)


def realtime_sampled(frames: Iterable, exit_key="q", fps=24):
    """
    A function to create a realtime video stream from an iterable of frames.

    Args:
    - frames (Iterable): The iterable of frames to stream.
    - fps (int): The frames per second (default=24).
    - buffer_size (int): The maximum size of the buffer (default=2).

    Returns:
    - A generator that yields frames in real-time.
    """
    skipped = Queue(maxsize=60)
    stop_flag = Queue(maxsize=1)

    # Create a virtual video stream
    proc = Process(target=realtime_sampled_work_process, args=(skipped, stop_flag, fps))
    proc.daemon = True
    proc.start()

    for frame in frames:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(exit_key):
            break

        _ = skipped.get()
        if skipped.empty():
            yield frame

    stop_flag.put(1)


def screenshot_stream(size=416, exit_key="q", color_format=cv2.COLOR_RGBA2RGB):
    if isinstance(size, int):
        size = (size, size)
    w, h = size

    capture_range = {"top": 0, "left": 0, "width": w, "height": h}
    cap = mss.mss()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(exit_key):
            break
        x, y = pag.position()  # 返回鼠标的坐标
        capture_range["top"] = y - capture_range["height"] // 2
        capture_range["left"] = x - capture_range["width"] // 2
        frame = cap.grab(capture_range)
        frame = np.array(frame)
        frame = cv2.resize(frame, (w, h))
        if color_format:
            frame = cv2.cvtColor(frame, color_format)
        yield frame
        while IS_DETACHED:
            time.sleep(1)


def yuv420_stream(file_path, yuv_size=(1920, 1080), size=None, exit_key="q", color_format=cv2.COLOR_YUV2BGR_I420):
    yuv_w, yuv_h = yuv_size
    file_size = os.path.getsize(file_path)
    max_frame = file_size // (yuv_w * yuv_h * 3 // 2) - 1

    cur_frame = 0
    with open(file_path, "rb") as probe:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(exit_key):
                break
            cur_frame += 1
            if cur_frame > max_frame:
                break
            yuv = np.frombuffer(probe.read(yuv_w * yuv_h * 3 // 2), dtype=np.uint8).reshape((yuv_h * 3 // 2, yuv_w))
            if color_format:
                yuv = cv2.cvtColor(yuv, color_format)
            if size:
                yuv = scale_and_pad(yuv, size)
            yield yuv
            while IS_DETACHED:
                time.sleep(1)


def switch_detach():
    global IS_DETACHED
    IS_DETACHED = not IS_DETACHED
    if IS_DETACHED:
        print("detached")
    else:
        print("recovered")


def on_press(key):
    if hasattr(key, "char") and key.char == "d":
        switch_detach()


def run_demo(device, weight, conf_thr, iou_thr, size, camera, screenshot, h264):
    class_names = FhPedestrianDataset.class_names
    model = yolov8n_pvdet()

    # load pretrained weights
    state = torch.load(weight, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # set nms args
    model.head.nms.conf_threshold = conf_thr
    model.head.nms.iou_threshold = iou_thr

    # you can use any loader from dataloader
    if camera:
        test_data = camera_stream(size)
    elif screenshot:
        test_data = screenshot_stream(size)
    elif h264 is not None:
        test_data = h264_stream(h264, size=size)
    else:
        raise Exception("At least one of the stream mode must be specified, see -help")

    external_nms = None  # optional

    canvas = Canvas()
    time_rec = Timer(10)
    video_writer = None  # use video_writer=VideoSaver(save_path) to save

    # set eval mode
    model.eval()

    # add hot key
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # read test data
    for frame in test_data:
        # preprocess
        frame = im2tensor(frame)
        frame = letterbox_padding(frame).unsqueeze(0)

        # inference
        with time_rec.record("core"):
            with torch.no_grad():
                out = model(frame.to(device))

        # external nms (optional)
        with time_rec.record("nms"):
            if external_nms is not None:
                out = external_nms(out)

        # Load raw image & parse outputs
        canvas.load(frame)
        for out_batch_i in out:
            if len(out_batch_i) > 0:
                for det in out_batch_i:
                    # list of detections, on (n,6) tensor per image [xyxy, conf, cls]
                    pt1, pt2, conf, cls = det[:2], det[2:4], det[4], int(det[5])
                    color = canvas.color(cls)
                    info = f"{str(cls) if not class_names else class_names[cls]}: {conf:.2f}"
                    canvas.draw_box(pt1, pt2, alpha=0.4, thickness=-1, color=color)
                    canvas.draw_box(pt1, pt2, color=color, title=info)

        # Use external time info
        for i, (title, value) in enumerate(time_rec.step().items()):
            canvas.draw_text(f"{title}={value}", (0, 12 * i), color=canvas.color(title, light_theme=False))
        canvas.show("demo", wait_key=1)

        # Save outputs
        if video_writer is not None:
            frame = canvas.image()
            h, w, _ = frame.shape
            video_writer: VideoSaver
            if not video_writer.initialized:
                video_writer.initialize(w, h)
            video_writer.write(frame)

    listener.stop()


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Pedestrian Detection Demo Configuration")

    # 添加命令行参数
    parser.add_argument("--device", type=str, default="cpu", help="指定设备: cpu/mps/cuda")
    parser.add_argument("--weight", type=str, help="模型权重文件的路径")
    parser.add_argument("--conf_thr", type=float, default=0.25, help="置信度阈值，默认为 0.25")
    parser.add_argument("--iou_thr", type=float, default=0.25, help="IoU 阈值，默认为 0.45")
    parser.add_argument("--size", type=int, default=640, help="图像尺寸，默认为 640")
    parser.add_argument("--camera", action="store_true", help="摄像头采集模式")
    parser.add_argument("--screenshot", action="store_true", help="截屏采集模式")
    parser.add_argument("--h264", type=str, default=None, help="h264视频流输入")

    # 解析命令行参数
    args = parser.parse_args()
    run_demo(**vars(args))
