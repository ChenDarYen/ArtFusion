import os

import torch
import pytorch_lightning as pl

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 relpath_key="relpath",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.relpath_key = relpath_key
        self.z_channels = ddconfig['params']["z_channels"]
        self.embed_dim = embed_dim or self.z_channels

        encoder_config = {
            'target': ddconfig['encoder_target'],
            'params': ddconfig['params'],
        }
        decoder_config = {
            'target': ddconfig['decoder_target'],
            'params': ddconfig['params'],
        }
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)

        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig['params']["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * self.z_channels, 2 * self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, self.z_channels, 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

            self.gt_save_dir = os.path.join(self.recon_save_dir, 'gt')
            self.save_gt = True

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, return_features=False):
        if return_features:
            h, features = self.encoder(x, return_features=True)
        else:
            h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if return_features:
            return posterior, features
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            inputs = self.get_input(batch, self.image_key)
            reconstructions, posterior = self(inputs)

            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            self.inputs = inputs.detach()
            self.reconstructions = reconstructions.detach()

            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = \
                self.loss(self.inputs, self.reconstructions, None, optimizer_idx, self.global_step,
                          last_layer=None, split="train")

            self.log("discloss", discloss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs, sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, None, 1, self.global_step,
                                            last_layer=None, split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, random_sample=True, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            x = self.tensor_to_rgb(x)
            xrec = self.tensor_to_rgb(xrec)
            log["reconstructions"] = xrec
            if random_sample:
                log["samples"] = self.decode(torch.randn_like(posterior.sample()))
        log["inputs"] = x
        return log

    @staticmethod
    def tensor_to_rgb(x):
        return torch.clip((x + 1.) * 127.5, 0., 255.)
