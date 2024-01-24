import sys
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init
import torch
from einops import rearrange
import math


from core.utils.gaussian_mlp import GaussianMLP
from core.utils.siren_pytorch import SirenNet
from lib.torch_utils.layers.conv_module import ConvModule
from lib.torch_utils.layers.layer_utils import (
    get_norm,
    get_nn_act_func,
    Sine,
)
from lib.torch_utils.layers.std_conv_transpose import StdConvTranspose2d


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, scale=None):
        # when temperature is 10000,  min frequency is 1/10000
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        not_mask = torch.ones_like(x)[:, 0, :, :]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# modified from TopDownDoubleMaskXyzRegionHead
# output single mask， xyz
class GposeDoubleMaskModulatedHead(nn.Module):
    def __init__(
        self,
        in_dim,
        up_types=("deconv", "bilinear", "bilinear"),
        deconv_kernel_size=3,
        num_conv_per_block=2,
        feat_dim=256,
        feat_kernel_size=3,
        use_ws=False,
        use_ws_deconv=False,
        norm="GN",
        num_gn_groups=32,
        act="GELU",
        out_kernel_size=1,
        out_layer_shared=True,
        mask_num_classes=1,
        xyz_num_classes=1,
        region_num_classes=1,
        mask_out_dim=2,
        xyz_out_dim=3,
        region_out_dim=65,  # 64+1
        mlp_type="siren",
        mlp_dim_hidden=32,
        mod_type="mul_add",
        pos_dim=32,
    ):
        """
        Args:
            up_types: use up-conv or deconv for each up-sampling layer
                ("bilinear", "bilinear", "bilinear")
                ("deconv", "bilinear", "bilinear")  # CDPNv2 rot head
                ("deconv", "deconv", "deconv")  # CDPNv1 rot head
                ("nearest", "nearest", "nearest")  # implement here but maybe won't use
        NOTE: default from stride 32 to stride 4 (3 ups)
        """
        super().__init__()
        assert out_kernel_size in [
            1,
            3,
        ], "Only support output kernel size: 1 and 3"
        assert deconv_kernel_size in [
            1,
            3,
            4,
        ], "Only support deconv kernel size: 1, 3, and 4"
        assert len(up_types) > 0, up_types
        assert out_layer_shared, "Only support out_layer_shared=True"

        self.features = nn.ModuleList()
        for i, up_type in enumerate(up_types):
            _in_dim = in_dim if i == 0 else feat_dim
            if up_type == "deconv":
                (
                    deconv_kernel,
                    deconv_pad,
                    deconv_out_pad,
                ) = _get_deconv_pad_outpad(deconv_kernel_size)
                deconv_layer = (
                    StdConvTranspose2d if use_ws_deconv else nn.ConvTranspose2d
                )
                self.features.append(
                    deconv_layer(
                        _in_dim,
                        feat_dim,
                        kernel_size=deconv_kernel,
                        stride=2,
                        padding=deconv_pad,
                        output_padding=deconv_out_pad,
                        bias=False,
                    )
                )
                self.features.append(
                    get_norm(norm, feat_dim, num_gn_groups=num_gn_groups)
                )
                self.features.append(get_nn_act_func(act))
            elif up_type == "bilinear":
                self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
            elif up_type == "nearest":
                self.features.append(nn.UpsamplingNearest2d(scale_factor=2))
            else:
                raise ValueError(f"Unknown up_type: {up_type}")

            if up_type in ["bilinear", "nearest"]:
                assert num_conv_per_block >= 1, num_conv_per_block
            for i_conv in range(num_conv_per_block):
                if i == 0 and i_conv == 0 and up_type in ["bilinear", "nearest"]:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = feat_dim

                if use_ws:
                    conv_cfg = dict(type="StdConv2d")
                else:
                    conv_cfg = None

                self.features.append(
                    ConvModule(
                        conv_in_dim,
                        feat_dim,
                        kernel_size=feat_kernel_size,
                        padding=(feat_kernel_size - 1) // 2,
                        conv_cfg=conv_cfg,
                        norm=norm,
                        num_gn_groups=num_gn_groups,
                        act=None,
                    )
                )
                self.features.append(get_nn_act_func(act))

        self.mask_num_classes = mask_num_classes
        self.xyz_num_classes = xyz_num_classes
        self.region_num_classes = region_num_classes

        self.mask_out_dim = mask_out_dim
        self.xyz_out_dim = xyz_out_dim
        self.region_out_dim = region_out_dim
        self.mlp_type = mlp_type
        self.mod_type = mod_type
        self.mlp_dim_hidden = mlp_dim_hidden
        self.pos_dim = pos_dim

        out_dim = (
            self.mask_out_dim * self.mask_num_classes
            + self.region_out_dim * self.region_num_classes
        )
        self.out_layer = nn.Conv2d(
            feat_dim,
            out_dim,
            kernel_size=out_kernel_size,
            padding=(out_kernel_size - 1) // 2,
            bias=True,
        )

        self.pe = PositionEmbeddingSine(feat_dim // 2)
        self.mlp_proj = nn.Linear(feat_dim, self.mlp_dim_hidden * 2)
        if mlp_type == "gaussian":
            self.coord_mlp = GaussianMLP(
                dim_in=feat_dim,
                dim_out=self.xyz_out_dim * self.pos_dim,
                dim_hidden=self.mlp_dim_hidden,
                num_layers=5,
                sigma=0.5,
            )
        elif mlp_type == "siren":
            self.coord_mlp = SirenNet(
                dim_in=feat_dim,  # input dimension, ex. 2d coor
                dim_hidden=self.mlp_dim_hidden,  # hidden dimension
                dim_out=self.xyz_out_dim
                * self.pos_dim,  # output dimension, ex. rgb value
                num_layers=5,  # number of layers
                final_activation=nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
                w0_initial=30.0,  # different signals may require different omega_0 in the first layer - this is a hyperparameter
            )
        else:
            raise ValueError(f"Unknown mlp_type: {mlp_type}")

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
        # init output layers
        normal_init(self.out_layer, std=0.01)

    def forward(self, x, mods=None):
        if isinstance(x, (tuple, list)) and len(x) == 1:
            x = x[0]
        mods_i = 0
        for i, l in enumerate(self.features):
            # modulate before activation function
            if mods is not None and isinstance(l, nn.GELU):
                w = mods[mods_i][0].unsqueeze(2).unsqueeze(3)
                b = mods[mods_i][1].unsqueeze(2).unsqueeze(3)
                x = x * w + b
                mods_i += 1
            x = l(x)

        out = self.out_layer(x)
        mask_dim = self.mask_out_dim * self.mask_num_classes
        vis_mask = out[:, : (mask_dim // 2), :, :]
        full_mask = out[:, (mask_dim // 2) : mask_dim, :, :]
        region = out[:, mask_dim:, :, :]


        b, c, h, w = x.shape
        if self.mod_type == "add":
            x = x + self.pe(x)
            xyz = (
                self.coord_mlp(x.permute(0, 2, 3, 1).reshape(b * h * w, c))
                .reshape(b, h, w, self.xyz_out_dim * self.pos_dim)
                .permute(0, 3, 1, 2)
            )
        elif self.mod_type == "mul":
            pe = self.pe(x)
            muls = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
            xyz = (
                self.coord_mlp(pe, muls=muls)
                .reshape(b, h, w, self.xyz_out_dim * self.pos_dim)
                .permute(0, 3, 1, 2)
            )
        elif self.mod_type == "mul_add":
            pe = self.pe(x).permute(0, 2, 3, 1).reshape(b * h * w, -1)
            x = x.permute(0, 2, 3, 1).reshape(b * h * w, -1)
            x = self.mlp_proj(x)
            muls, adds = x.chunk(2, dim=1)
            xyz = (
                self.coord_mlp(pe, muls=muls, adds=adds)
                .reshape(b, h, w, self.xyz_out_dim * self.pos_dim)
                .permute(0, 3, 1, 2)
            )
        else:
            raise ValueError(f"Unknown mod_type: {self.mod_type}")

        #  shared by all class
        xyz = (
            xyz.unsqueeze(1)
            .expand(-1, self.xyz_num_classes, -1, -1, -1)
            .reshape(b, self.xyz_num_classes, self.xyz_out_dim, self.pos_dim, h, w)
        )
        return vis_mask, full_mask, region, xyz


def _get_deconv_pad_outpad(deconv_kernel):
    """Get padding and out padding for deconv layers."""
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0
    else:
        raise ValueError(f"Not supported num_kernels ({deconv_kernel}).")

    return deconv_kernel, padding, output_padding
