from diffusers import AutoencoderKL
import torch


class SD_VAE(torch.nn.Module):
    latent_scale_factor = 0.18215

    def __init__(self, **kwargs):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2",
            subfolder="vae",
            use_safetensors=True,
            resume_download=True,
            mirror="https://mirrors.aliyun.com/huggingface/",
        )

    def forward(self, x):
        return self.encode_rgb(x)

    @torch.no_grad()
    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (torch.Tensor):
                Input RGB image to be encoded.

        Returns:
            torch.Tensor: Image latent
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent, ensure unit variance, ref: https://github.com/huggingface/diffusers/issues/437#issuecomment-1241827515
        rgb_latent = mean * self.latent_scale_factor
        return rgb_latent
