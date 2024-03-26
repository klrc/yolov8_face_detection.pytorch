import sys
import cv2


sys.path.append(".")

from dataset import WiderfaceDataset  # noqa:E402
from canvas import Canvas  # noqa:E402


def visualize(dataset: WiderfaceDataset):
    canvas = Canvas()
    for i in range(len(dataset)):
        image, labels = dataset.__getitem__(i)
        _, H, W = image.shape
        canvas.load(image)
        labels = labels[:, 1:].numpy()
        # Render labels for object detection datasets
        for j, lb in enumerate(labels):
            cid, cx, cy, w, h = int(lb[0]), lb[1], lb[2], lb[3], lb[4]  # cls, cx, cy, w, h (dataset format)
            x1 = int((cx - 0.5 * w) * W)
            y1 = int((cy - 0.5 * h) * H)
            x2 = int((cx + 0.5 * w) * W)
            y2 = int((cy + 0.5 * h) * H)
            label = dataset.class_names[int(cid)]
            print(f"{lb} {label}#{i}-{j}")
            canvas.draw_box((x1, y1), (x2, y2), title=f"{label}#{i}-{j}")

        canvas.show(wait_key=1)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    dataset = WiderfaceDataset(
        training=True,
        dataset_path="/Volumes/ASM236X/datasets/coco128/images/train2017",
        image_size=640,
        batch_size=3,
        stride=32,
    )
    visualize(dataset)
