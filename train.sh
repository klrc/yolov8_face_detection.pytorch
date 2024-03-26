python trainer.py --device_id cuda \
--pretrained_pt_path /res/molchip_fc_827566.pt \
--batch_size 16 \
--max_epochs 300 \
--ema_enabled \
--wandb_enabled \
--early_stop \
--trainset_path /home/han.sun/datasets/widerface/images/train \
--valset_path /home/han.sun/datasets/widerface/images/val