python trainer.py --device_id cpu \
--pretrained_pt_path res/molchip_fc_827566.pt \
--batch_size 16 \
--max_epochs 300 \
--ema_enabled \
--wandb_enabled \
--early_stop \
--trainset_path /Volumes/ASM236X/datasets/widerface/images/train \
--valset_path /Volumes/ASM236X/datasets/widerface/images/val