import torch
import onnx
import onnxsim
import os
from onnx import shape_inference
from thop import profile, clever_format
import argparse

from model import ConvModule, yolov8n_pvdet


def export(weight: str, device, input_shape, input_names, output_names, opset_version, enable_onnxsim):
    model = yolov8n_pvdet()

    # load pretrained weights
    if weight is not None:
        model_dir = os.path.dirname(weight)
        model_name = os.path.basename(weight).split(".pt")[0]
        onnx_path = f"{model_dir}/{model_name}.onnx"
        state = torch.load(f"{model_dir}/{model_name}.pt", map_location=device)
        model.load_state_dict(state)
    model.to(device)

    # switch to fused mode
    for m in model.modules():
        if isinstance(m, ConvModule):
            m.fuse()
    model.eval()
    model.head.porting = True

    # check shape & flops
    x = torch.randn(*input_shape)
    macs, params = profile(model, inputs=(x,))
    flops, params = clever_format([macs * 2, params], "%.3f")
    print("total flops:", flops)
    print("total params:", params)
    print("Press Enter to continue ...")
    input()

    if weight is not None:
        # export to onnx
        torch.onnx.export(
            model,
            args=torch.rand(*input_shape),
            f=onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=True,
        )

        # simplify onnx graph
        onnx_model = onnx.load(onnx_path)
        if enable_onnxsim:
            onnx_model, _ = onnxsim.simplify(onnx_model)

        # inference shape
        shape_inference.infer_shapes(onnx_model)
        onnx.save(onnx_model, onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian Detection Export Configuration")
    parser.add_argument("--weight", type=str, default=None, help="模型权重文件的路径")
    parser.add_argument("--device", type=str, default="cpu", help="设备类型")
    parser.add_argument("--input_shape", nargs="+", type=int, default=[1, 3, 352, 640], help="输入张量形状")
    parser.add_argument("--input_names", nargs="+", default=["image"], help="输入张量名称")
    parser.add_argument("--output_names", nargs="+", default=["head_0_box", "head_0_cls", "head_1_box", "head_1_cls", "head_2_box", "head_2_cls"], help="输出张量名称")
    parser.add_argument("--opset_version", type=int, default=13, help="ONNX opset 版本")
    parser.add_argument("--enable_onnxsim", action="store_true", help="是否启用 ONNX 简化")

    args = parser.parse_args()

    export(**vars(args))
