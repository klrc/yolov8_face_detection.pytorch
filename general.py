import glob
import hashlib
import os
import torch
from multiprocessing.pool import Pool
from pathlib import Path
import random
import numpy as np
import pkg_resources
from loguru import logger
from PIL import ExifTags, Image, ImageOps
from tqdm import tqdm
import cv2
import time
import torch.nn as nn


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp"  # include image suffixes
BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"  # tqdm bar format


def check_version(current="0.0.0", minimum="0.0.0", name="version ", pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg_resources.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f"{name}{minimum} required by YOLOv5, but {name}{current} is currently installed"  # string
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        logger.warning(s)
    return result


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + "images" + os.sep, os.sep + "labels" + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except Exception:
        pass
    return s


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, label_size = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"WARNING: {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                # if any(len(x) > 6 for x in lb):  # is segment
                #     classes = np.array([x[0] for x in lb], dtype=np.float32)
                #     segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                # lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                #     raise NotImplementedError
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == label_size, f"labels require {label_size} columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    # if segments:
                    #     segments = segments[i]
                    msg = f"WARNING: {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, label_size), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, label_size), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"WARNING: {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]


def get_im_files(dataset_path):
    try:
        f = []  # image files
        for p in dataset_path if isinstance(dataset_path, list) else [dataset_path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)
            else:
                raise Exception(f"{p} does not exist")
        im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
        assert im_files, "No images found"
        return im_files
    except Exception as e:
        raise Exception(f"Error loading data from {dataset_path}: {e}")


def try_load_from_cache(im_files, label_files, augment, label_size=5):
    # Check cache
    cache_path = Path(label_files[0]).parent.with_suffix(".cache")
    try:
        cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
        assert cache["hash"] == get_hash(label_files + im_files)  # same hash
    except Exception:
        cache, exists = save_cache_to_disk(im_files, label_files, cache_path, label_size), False  # cache
    # Display cache
    nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
    if exists:
        d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
        tqdm(None, desc=d, total=n, initial=n, bar_format=BAR_FORMAT)  # display cache results
        if cache["msgs"]:
            logger.info("\n".join(cache["msgs"]))  # display warnings
    assert nf > 0 or not augment, f"No labels in {cache_path}. Can not train without labels."
    # Load cache
    [cache.pop(k) for k in ("hash", "msgs")]  # remove items
    labels, shapes, segments = zip(*cache.values())
    labels = list(labels)
    shapes = np.array(shapes, dtype=np.float64)
    im_files = list(cache.keys())  # update
    # label_files = img2label_paths(cache.keys())  # update
    return im_files, shapes, labels, segments


def save_cache_to_disk(im_files, label_files, path=Path("./labels.cache"), label_size=5):
    # Cache dataset labels, check images and read shapes
    x = {}  # dict
    nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
    desc = f"Scanning '{path.parent / path.stem}' images and labels..."
    label_sizes = [label_size for _ in im_files]
    with Pool(NUM_THREADS) as pool:
        pbar = tqdm(
            pool.imap(verify_image_label, zip(im_files, label_files, label_sizes)),
            desc=desc,
            total=len(im_files),
            bar_format=BAR_FORMAT,
        )
        for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if im_file:
                x[im_file] = [lb, shape, segments]
            if msg:
                msgs.append(msg)
            pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"
    pbar.close()
    if msgs:
        logger.info("\n".join(msgs))
    if nf == 0:
        logger.warning(f"WARNING: No labels found in {path}.")
    x["hash"] = get_hash(label_files + im_files)
    x["results"] = nf, nm, ne, nc, len(im_files)
    x["msgs"] = msgs  # warnings
    try:
        np.save(path, x)  # save cache for next time
        path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
        logger.info(f"New cache created: {path}")
    except Exception as e:
        logger.warning(f"WARNING: Cache directory {path.parent} is not writeable: {e}")  # not writeable
    return x


def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn

    if deterministic and check_version(torch.__version__, "1.12.0"):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # for multi GPU, exception safe


def generate_hash():
    return hashlib.shake_128(str(time.time()).encode("utf8")).hexdigest(5)[:6]


def intersect_dicts(da, db):
    # Dictionary intersection of matching keys and shapes
    return {k: v for k, v in da.items() if k in db and v.shape == db[k].shape}


def aligned_zip(la, lb):
    max_len = max(len(la), len(lb))
    ret = []
    for i in range(max_len):
        xa = None if i >= len(la) else la[i]
        xb = None if i >= len(lb) else lb[i]
        ret.append((xa, xb))
    return ret


def forced_load(model: nn.Module, pt_path, ignore=[], mapper={}):
    print(f"loading weight from {pt_path}")
    raw_pt = torch.load(pt_path, map_location="cpu")
    pt = {}
    for k, v in raw_pt.items():
        for mk, mv in mapper.items():
            if mk in k:
                print(f"{k}\t -> ", end="")
                k = k.replace(mk, mv)
                print(f"{k}")
        pt[k] = v
    sd = model.state_dict()
    for x in ignore:
        if x in sd:
            sd.pop(x)
    csd = intersect_dicts(pt, sd)  # intersect
    # state dict comparison
    pt_keys = [x for x in pt if x not in csd]
    sd_keys = [x for x in sd if x not in csd]
    if len(pt_keys) > 0 or len(sd_keys) > 0:
        print("Diff:")
        print("Source keys                                       Target keys")
    for pt_key, sd_key in aligned_zip(pt_keys, sd_keys):
        print(f"{pt_key:50s}{sd_key}")

    missing_keys, unexpected_keys = model.load_state_dict(csd, strict=False)  # load

    # print("Missing                                        \tUnexpected")
    # for i in range(max(len(missing_keys), len(unexpected_keys))):
    #     missing_key = "" if i >= len(missing_keys) else missing_keys[i]
    #     unexpected_key = "" if i >= len(unexpected_keys) else unexpected_keys[i]
    #     if missing_key not in ignore:
    #         print(f"{missing_key}\t{unexpected_key}")
    print(f"transferred {len(csd)}/{len(sd)} items")
    return model


class Timer:
    class TRecorder:
        def __init__(self, title, tcr_core) -> None:
            self.__title = title
            self.__ckpt = None
            self.__core = tcr_core

        def __enter__(self):
            self.__ckpt = time.time()
            return self

        def __exit__(self, exc_type, exc_value, trace):
            self.__core.records.append((self.__title, time.time() - self.__ckpt))
            del self

    def __init__(self, update_interval=10) -> None:
        self.update_interval = update_interval

        self.records = []
        self.__clock = 0
        self.__cache = None

    def record(self, title):
        return self.TRecorder(title, self)

    def step(self):
        # Use external time info
        if self.__cache is None:
            # titles, values for update, values for display
            self.__cache = [[x[0] for x in self.records], [x[1] for x in self.records], [x[1] for x in self.records]]
        else:
            for i, (_, value) in enumerate(self.records):
                self.__cache[1][i] += value

        # Update clock
        self.__clock += 1
        if self.__clock > self.update_interval:
            self.__cache[2] = [x / self.__clock for x in self.__cache[1]]
            self.__cache[1] = [0 for _ in self.records]
            self.__clock = 0
        self.records.clear()

        stats = self.__cache[2]
        total_time = sum(stats)
        ret = {
            "fps": f"{1/total_time:.1f}",
        }
        for i, title in enumerate(self.__cache[0]):
            ret[title] = f"{stats[i]/total_time:.1%}"

        return ret


class VideoSaver:
    def __init__(self, save_path, fps=24):
        self.save_path = save_path
        self.fps = fps
        self.initialized = False

    def initialize(self, w, h):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.vw = cv2.VideoWriter(self.save_path, fourcc, self.fps, (w, h))
        self.initialized = True

    def write(self, frame):
        self.vw.write(frame)
