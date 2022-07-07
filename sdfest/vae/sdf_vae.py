"""This module provides various PyTorch modules for working with SDFs."""
from typing import Optional, Union
from pydoc import locate

import torch
import torch.nn as nn

from . import torch_utils


class SDFVAE(nn.Module):
    """Variational Autoencoder for Signed Distance Fields."""

    def __init__(
        self,
        sdf_size: int,
        latent_size: int,
        encoder_dict: dict,
        decoder_dict: dict,
        device: torch.device,
        tsdf: Optional[Union[bool, float]] = False,
    ):
        """Initialize the SDFVAE module.

        Args:
            sdf_size:       depth/width/length of the sdf
            latent_size:    dimensions of latent representation
            encoder_dict:
                Arguments passed to encoder constructor.
                See SDFEncoder.__init__ for details.
            decoder_dict:
                Arguments passed to decoder constructor.
                See SDFDecoder.__init__ for details.
            device:
                The device this model will work on.
                This is used for tensors created inside this model for sampling.
            tsdf:
                Value to truncate the SDF at. False for untruncated SDF.
                For the input this is only done in prepare_input, not in the forward
                pass. The output is truncated in the forward pass.
        """
        super().__init__()

        self.latent_size = latent_size

        self._device = device

        self.encoder = SDFEncoder(sdf_size, latent_size, tsdf=tsdf, **encoder_dict)
        self.decoder = SDFDecoder(sdf_size, latent_size, tsdf=tsdf, **decoder_dict)
        self.sdf_size = sdf_size
        self._tsdf = tsdf

    def forward(self, x, enforce_tsdf=False):
        z, means, log_var = self.encode(x)

        recon_x = self.decoder(z, enforce_tsdf)

        return recon_x, means, log_var, z

    def sample(self, n=1):
        z = torch.randn([n, self.latent_size]).to(self._device)
        return z

    def encode(self, x):
        means, log_var = self.encoder(x)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([x.shape[0], self.latent_size]).to(self._device)
        z = eps * std + means
        return z, means, log_var

    def inference(self, n=1, enforce_tsdf=False):
        z = self.sample(n)

        recon_x = self.decoder(z, enforce_tsdf)

        return recon_x, z

    def decode(self, z, enforce_tsdf=False):
        """Returns decoded SDF.

        Args:
            Batch of latent shapes. Shape (N, L).
        Returns:
            The decoded SDF. Shape (N, C, D, D, D).
        """
        return self.decoder(z, enforce_tsdf)

    def prepare_input(self, sdfs: torch.Tensor) -> None:
        """Convert batched SDFs to expected input format.

        This will transform inputs as defined by the decoder.
        See SDFEncoder.prepare_input for details.

        Will be done in place without gradients.

        Args:
            sdfs: Batched SDFs, expected shape (N,C,D,D,D).
        """
        self.encoder.prepare_input(sdfs)


class SDFEncoder(nn.Module):
    def __init__(self, volume_size, latent_size, layer_infos, tsdf=False):
        """Create SDFEncoder, i.e., define trainable layers.

        Args:
            volume_size:
                Input size D of the volume.
                (i.e., input tensor is expected to be Nx1xDxDxD)
            latent_size: Dimensionality of the latent representation.
            layers:
                Dictionaries defining the layers before the final output layers.
                Required fields:
                    - fully qualified type (str)
                    - args (dict): params passed to init of type
                These layers need to construct an 2D tensor (including batch dimension).
        """
        super().__init__()

        in_channels = 1

        # define layers
        layers = []
        for layer_info in layer_infos:
            t = locate(layer_info["type"])
            layers.append(t(**layer_info["args"]))
        self._features = torch.nn.Sequential(*layers)

        with torch.no_grad():
            output_size = self._features(
                torch.zeros(1, in_channels, volume_size, volume_size, volume_size)
            ).shape

        self.linear_means = nn.Linear(output_size[1], latent_size)
        self.linear_log_var = nn.Linear(output_size[1], latent_size)

        self._tsdf = tsdf

    def forward(self, x):
        """Forward pass of the module.

        Args:

        Returns:

        """
        out = self._features(x)

        means = self.linear_means(out)
        log_vars = self.linear_log_var(out)

        return means, log_vars

    def prepare_input(self, sdfs: torch.Tensor) -> None:
        """Convert batched SDFs to expected input format.

        This will truncate the SDFs based on tsdf argument passed to constructor.

        Will be done in place without gradients.

        Args:
            sdfs: Batched SDFs, expected shape (N,C,D,D,D).
        """
        if self._tsdf is not False:
            with torch.no_grad():
                sdfs.clamp_(-self._tsdf, self._tsdf)


class SDFDecoder(nn.Module):
    def __init__(
        self,
        volume_size: int,
        latent_size: int,
        fc_layers: list,
        conv_layers: list,
        tsdf: Optional[Union[bool, float]] = False,
    ):
        super().__init__()

        # sanity checks
        self.sanity_check(volume_size, fc_layers, conv_layers)

        # create layers and store params
        self._volume_size = volume_size

        in_size = latent_size
        self._fc_layers = torch.nn.ModuleList()
        for fc_layer in fc_layers:
            self._fc_layers.append(nn.Linear(in_size, fc_layer["out"]))
            in_size = fc_layer["out"]
        self._fc_info = fc_layers

        self._conv_layers = torch.nn.ModuleList()
        for conv_layer in conv_layers:
            self._conv_layers.append(
                nn.Conv3d(
                    conv_layer["in_channels"],
                    conv_layer["out_channels"],
                    conv_layer["kernel_size"],
                )
            )
        self._conv_info = conv_layers

        self._tsdf = tsdf

    def sanity_check(self, volume_size, fc_dicts, conv_dicts):
        assert fc_dicts[-1]["out"] == (
            conv_dicts[0]["in_channels"] * conv_dicts[0]["in_size"] ** 3
        )

        for i, conv_dict in enumerate(conv_dicts[:-1]):
            assert conv_dict["out_channels"] == conv_dicts[i + 1]["in_channels"]

        assert conv_dicts[-1]["out_channels"] == 1

    def forward(self, z, enforce_tsdf=False):
        """Decode latent vectors to SDFs.

        Args:
            z: Batch of latent vectors. Expected shape of (N,latent_size).
        """
        out = z
        for fc_layer in self._fc_layers:
            out = nn.functional.relu(fc_layer(out))

        out = out.view(
            -1,
            self._conv_info[0]["in_channels"],
            self._conv_info[0]["in_size"],
            self._conv_info[0]["in_size"],
            self._conv_info[0]["in_size"],
        )

        for info, layer in zip(self._conv_info, self._conv_layers):
            # interpolate to match next input
            if out.shape[2] != info["in_size"]:
                out = torch.nn.functional.interpolate(
                    out,
                    size=(info["in_size"], info["in_size"], info["in_size"]),
                    mode="trilinear",
                    align_corners=False,
                )
            out = layer(out)
            if info["relu"]:
                out = nn.functional.relu(out)

        if out.shape[2] != self._volume_size:
            out = torch.nn.functional.interpolate(
                out,
                size=(self._volume_size, self._volume_size, self._volume_size),
                mode="trilinear",
                align_corners=False,
            )

        if self._tsdf is not False and enforce_tsdf:
            out = out.clamp(-self._tsdf, self._tsdf)

        return out
