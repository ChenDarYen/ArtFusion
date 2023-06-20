import copy

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from contextlib import contextmanager

from ldm.models.diffusion.ddpm import LatentDiffusion, instantiate_from_config
from ldm.modules.losses.lpips import vgg16, ScalingLayer
from ldm.modules.ema import LitEma


class DualConditionDiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key=('other', 'other')):
        super().__init__()
        assert len(conditioning_key) == 2
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.first_conditioning_key, self.second_conditioning_key = conditioning_key
        assert self.first_conditioning_key in ['concat', 'other'] and \
               self.second_conditioning_key in ['concat', 'other']

    def forward(self,
                x, t, c1: list = None, c2: list = None):
        if self.first_conditioning_key == 'concat':
            x = torch.cat([x, c1], dim=1)
            c1 = None

        if self.second_conditioning_key == 'concat':
            x = torch.cat([x, c2], dim=1)
            c2 = None

        return self.diffusion_model(x, t, c1, c2)


class DualCondLDM(LatentDiffusion):
    def __init__(self,
                 style_dim,
                 cond_stage_key='content_and_std_mean',
                 style_flag_key='style_flag',
                 content_flag_key='content_flag',
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 scale_factors=None,
                 shift_values=None,
                 *args, **kwargs):
        super().__init__(model_class=DualConditionDiffusionWrapper,
                         cond_stage_key=cond_stage_key,
                         ckpt_path=None,
                         *args, **kwargs)
        assert cond_stage_key == 'content_and_std_mean'

        self.style_flag_key = style_flag_key
        self.content_flag_key = content_flag_key

        delattr(self, 'scale_factor')
        if scale_factors is None:
            self.register_buffer('scale_factors', torch.ones(1))
        else:
            self.register_buffer('scale_factors', torch.tensor(scale_factors, dtype=torch.float32))
        if shift_values is None:
            self.register_buffer('shift_values', torch.zeros(1))
        else:
            self.register_buffer('shift_values', torch.tensor(shift_values, dtype=torch.float32))

        self.null_style_vector = torch.nn.Embedding(1, style_dim)
        torch.nn.init.normal_(self.null_style_vector.weight, std=0.02)
        if self.use_ema:
            self.null_style_vector_ema = LitEma(self.null_style_vector)
            print(f"Keeping EMAs of {len(list(self.null_style_vector_ema.buffers()))}.")

        self.vgg = vgg16(pretrained=True, requires_grad=False)
        self.vgg_scaling_layer = ScalingLayer()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1. and self.shift_values == 0., \
                'rather not use custom rescaling and std-rescaling simultaneously'
            print("### USING STD-RESCALING ###")
            z = self.get_input(batch, self.first_stage_key)[0]
            del self.scale_factors
            del self.shift_values
            scales, shifts = list(), list()
            for i in range(z.shape[1]):
                std, mean = torch.std_mean(z[:, i])
                scales.append(1. / std)
                shifts.append(mean)
            self.register_buffer('scale_factors', torch.tensor(scales))
            self.register_buffer('shift_values', torch.tensor(shifts))
            print(f"setting self.scale_factors to {self.scale_factors.data}, self.shift_values to {self.shift_values.data}")
            print("### USING STD-RESCALING ###")

    def get_first_stage_encoding(self, encoder_posterior):
        z = encoder_posterior.mode()
        for i in range(z.shape[1]):
            z[:, i] = (z[:, i] - self.shift_values[i]) * self.scale_factors[i]
        return z.detach()

    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        for i in range(z.shape[1]):
            z[:, i] = z[:, i] / self.scale_factors[i] + self.shift_values[i]
        return self.first_stage_model.decode(z)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
            self.null_style_vector_ema(self.null_style_vector)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            self.null_style_vector_ema.store(self.null_style_vector.parameters())
            self.null_style_vector_ema.copy_to(self.null_style_vector)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                self.null_style_vector_ema.restore(self.null_style_vector.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters()) + list(self.null_style_vector.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, *args, **kwargs)

    @torch.no_grad()
    def encode_first_stage(self, x, return_features=True):
        return self.first_stage_model.encode(x, return_features=return_features)

    def get_style_features(self, features, flag=None):
        style_features = torch.cat([torch.cat(torch.std_mean(f, dim=[-1, -2]), dim=1) for f in features], dim=1)
        if flag is not None:
            flag = flag[..., None]
            style_features = torch.where(flag, style_features, self.null_style_vector.weight[0])  # null style
        return style_features

    def get_content_features(self, features, flag=None):
        content_features = features[:, :self.first_stage_model.embed_dim]
        std, mean = torch.std_mean(content_features, dim=[-1, -2], keepdim=True)
        content_features = (content_features - mean) / std
        if flag is not None:
            flag = flag[..., None, None, None]
            content_features = torch.where(flag, content_features, 0)  # null content
        return content_features

    def get_input(self, batch, k, return_first_stage_outputs=False, return_content_features=False,
                  bs=None, *args, **kwargs):
        x = self.get_image_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x, return_features=False)
        encoder_posterior = self.get_first_stage_encoding(encoder_posterior)
        z = encoder_posterior

        content_flag = batch[self.content_flag_key]
        style_flag = batch[self.style_flag_key]
        vgg_x = self.vgg_scaling_layer(x)
        vgg_features = self.vgg(vgg_x)
        if bs is not None:
            content_flag = content_flag[:bs]
            style_flag = style_flag[:bs]
        c_content = self.get_content_features(encoder_posterior, content_flag)
        c_style = self.get_style_features(vgg_features, style_flag)

        c = {'c1': c_content, 'c2': c_style}

        out = [z, c]

        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_content_features:
            out.append(encoder_posterior)
        return out

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=10, ddim_eta=1., return_keys=None, **kwargs):
        with self.ema_scope("Plotting"):
            use_ddim = ddim_steps is not None

            log = dict()
            z, c, x, xrec, content_features = \
                self.get_input(batch, self.first_stage_key,
                               return_first_stage_outputs=True, return_content_features=True, bs=N)

            N = min(xrec.shape[0], N)
            # x = self.tensor_to_rgb(x)
            # log["input"] = xc
            xrec = self.tensor_to_rgb(xrec)
            log["reconstruction"] = xrec

            if sample:
                # get denoise row
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                x_samples = self.decode_first_stage(samples)
                x_samples = self.tensor_to_rgb(x_samples)
                log["samples"] = x_samples

                style_flag = batch[self.style_flag_key][:, None][:N]
                if style_flag.sum() > 0:
                    c['c1'] = self.get_content_features(content_features)
                    c['c2'][:] = c['c2'][list(style_flag).index(True)][None, :]
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta)
                    x_samples = self.decode_first_stage(samples)
                    x_samples = self.tensor_to_rgb(x_samples)
                    log["stylized"] = x_samples

                    # style guidance
                    style_uncond_c = copy.deepcopy(c)
                    style_uncond_c['c2'][:] = self.null_style_vector.weight[0]
                    samples, z_denoise_row = \
                        self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                        ddim_steps=ddim_steps, eta=ddim_eta,
                                        unconditional_guidance_scale_2=5.,
                                        unconditional_conditioning_2=style_uncond_c)  # style is condition 2
                    x_samples = self.decode_first_stage(samples)
                    x_samples = self.tensor_to_rgb(x_samples)
                    log["stylized_guidance_5"] = x_samples

                    # content guidance
                    content_uncond_c = copy.deepcopy(c)
                    content_uncond_c['c1'][:] = 0.
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             unconditional_guidance_scale=.5,
                                                             unconditional_conditioning=content_uncond_c)
                    x_samples = self.decode_first_stage(samples)
                    x_samples = self.tensor_to_rgb(x_samples)
                    log["stylized_content_guidance_0.5"] = x_samples

                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             unconditional_guidance_scale=.25,
                                                             unconditional_conditioning=content_uncond_c)
                    x_samples = self.decode_first_stage(samples)
                    x_samples = self.tensor_to_rgb(x_samples)
                    log["stylized_content_guidance_0.25"] = x_samples

            if return_keys:
                if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                    return log
                else:
                    return {key: log[key] for key in return_keys}
            return log

    @staticmethod
    def tensor_to_rgb(x):
        return torch.clip((x + 1.) * 127.5, 0., 255.)
