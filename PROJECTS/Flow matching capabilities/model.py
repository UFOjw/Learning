from tqdm.auto import tqdm
import torch.nn as nn
import torch
from utils import Conv2dSamePadding
from torch.nn import functional as F

class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(MLPBlock, self).__init__()
        self.device = device
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, device=device)
        )

    def forward(self, x):
        return self.layer(x)


class InferenceModel(nn.Module):
    def __init__(self,
                 num_inference_steps,
                 input_dim,
                 hidden1,
                 kernel,
                 hidden2,
                 linear_output_dim,
                 pipeline,
                 scheduler,
                 device
            ):
        super(InferenceModel, self).__init__()
        self.num_inference_steps = num_inference_steps

        self.gn = nn.ModuleList([
            nn.GroupNorm(num_groups=1, num_channels=input_dim)
            for _ in range(num_inference_steps - 2)
        ]).to(device)
        self.conv2d = Conv2dSamePadding(input_dim, hidden1, kernel).to(device)
        self.after_cond_size = hidden1
        self.swish = nn.SiLU()
        self.MLP = MLPBlock(hidden1, hidden2, 1, device).to(device)
        self.linear_block = nn.Linear(num_inference_steps - 2, linear_output_dim).to(device)

        self.pipe = pipeline.to(device)
        self.pipe.scheduler = scheduler.from_config(self.pipe.scheduler.config)

        self.device = device

    def _get_latent(self, input_image):
        return self.pipe.vae.encode(input_image.to(self.device) * 2 - 1)

    @torch.no_grad()
    def _invert(
            self,
            start_latents,
            prompt,
            guidance_scale=3.5,
            num_inference_steps=80,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
    ):
        text_embeddings = self.pipe.encode_prompt(
            prompt, self.device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        text_embeddings = torch.stack(text_embeddings).squeeze(1)

        latents = start_latents.clone()
        intermediate_latents = []
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)

        timesteps = reversed(self.pipe.scheduler.timesteps)

        for i in range(1, num_inference_steps):

            # We'll skip the final iteration
            if i >= num_inference_steps - 1:
                continue
            t = timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
            next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
            alpha_t = self.pipe.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.pipe.scheduler.alphas_cumprod[next_t]

            # Inverted update step
            latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
                    1 - alpha_t_next
            ).sqrt() * noise_pred

            # Store
            intermediate_latents.append(latents)

        return torch.stack(intermediate_latents)

    def forward(self, input_image, prompt=""):
        with torch.no_grad():
            latent = self._get_latent(input_image)
            sample = 0.18215 * latent.latent_dist.sample()

            self.pipe.enable_model_cpu_offload()
            inverted_latents = self._invert(sample, prompt, num_inference_steps=self.num_inference_steps)

        normalized_latents = []
        for t in range(self.num_inference_steps - 2):
            normalized_latents.append(self.gn[t](inverted_latents[t]))
        inverted_latents = torch.stack(normalized_latents)

        batch_size = inverted_latents.size(1)
        shape = inverted_latents.size(-1)
        latent_channels = 4
        inverted_latents = inverted_latents.view(-1, latent_channels, shape, shape)  # (time * batch, channels, h, w)
        inverted_latents = self.conv2d(inverted_latents)
        inverted_latents = self.swish(inverted_latents)
        inverted_latents = inverted_latents.view(batch_size, -1, self.after_cond_size, shape, shape)  # (b, t, c, h, w)

        inverted_latents = inverted_latents.permute(0, 1, 3, 4, 2).contiguous()  # (b, t, h, w, c)
        inverted_latents = inverted_latents.view(-1, self.after_cond_size)  # (b * t * h * w, c)

        inverted_latents = self.MLP(inverted_latents)  # (b * t * h * w, 1)
        inverted_latents = inverted_latents.squeeze(-1)
        inverted_latents = inverted_latents.view(batch_size, -1, shape ** 2)  # (b, t, h * w)

        inverted_latents = inverted_latents.sum(dim=2)
        inverted_latents = self.linear_block(inverted_latents)

        return inverted_latents.squeeze(0)  # remove batch (rework future)
