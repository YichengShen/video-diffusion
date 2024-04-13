from diffusers import UNet2DModel


def create_unet(in_channels, out_channels):
    model = UNet2DModel(
        in_channels=in_channels,
        out_channels=out_channels,
        block_out_channels=(32, 64, 128, 256),
        norm_num_groups=8
    )
    return model
