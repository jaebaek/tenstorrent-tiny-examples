import torch
import ttnn
import sys

def main() -> int:
    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    batch_size = 4
    input_channels = 3
    input_height = 96
    input_width = 128

    output_channels = 16
    filter_height = 5
    filter_width = 5

    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels, filter_height, filter_width]
    # conv_bias_shape = [1, 1, 1, output_channels]
    # torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16)
    # tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, torch.bfloat16)

    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()

    print("torch_input_tensor: ", torch_input_tensor.shape)
    print("torch_weight_tensor: ", torch_weight_tensor.shape)

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, ttnn.bfloat16
    )

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        height_sharding=False,
        input_channels_alignment=(
            16 if (input_channels == 16 and input_height == 115) else 32
        ),
        deallocate_activation=False,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
    )

    stride_h = 1
    stride_w = 1
    pad_h = 2
    pad_w = 2

    [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        # bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        # conv_op_cache=reader_patterns_cache,

        # When debug = True, we have to set conv_config. Otherwise, we got exception.
        debug=False, # True,
        # groups=groups,
    )

    ttnn.close_device(device)
    return 0

if __name__ == '__main__':
    sys.exit(main())
