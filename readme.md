# yolov8_cspdarknet-pvdetection 人车检测

## 运行demo
```shell
python demo.py --weight xxx.pt --h264 /Volumes/ASM236X/stream/test_erqiyuanqu.h264 --device mps
```

## 训练模型
```shell
python trainer.py --device_id cuda \
--pretrained_pt_path res/molchip_fc_827566.pt \
--batch_size 16 \
--max_epochs 300 \
--ema_enabled \
--wandb_enabled \
--early_stop \
--trainset_path /home/han.sun/datasets/widerface/images/train \
--valset_path /home/han.sun/datasets/widerface/images/val
```

## 导出模型
```shell
python export.py --weight xxx.pt --input_shape 1 3 352 640 --input_names image --output_names output --opset_version 13 --enable_onnxsim
```
