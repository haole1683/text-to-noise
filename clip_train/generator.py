import torch
import torch.nn as nn

from diffusers import AutoencoderKL, UNet2DConditionModel

class generatorDcGan(nn.Module):
	def __init__(self, image_size = 64, num_channel = 3, noise_dim = 100, embed_dim = 1024, projected_embed_dim = 128):
		super(generatorDcGan, self).__init__()
		self.image_size = image_size
		self.num_channels = num_channel
		self.noise_dim = noise_dim
		self.embed_dim = embed_dim
		self.projected_embed_dim = projected_embed_dim
		self.latent_dim = self.noise_dim + self.projected_embed_dim
		self.ngf = 64

		self.projection = nn.Sequential(
			nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
			nn.BatchNorm1d(num_features=self.projected_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
			)

		# based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
		self.netG = nn.Sequential(
			nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
			nn.Tanh()
			 # state size. (num_channels) x 64 x 64
			)


	def forward(self, embed_vector, z):

		projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
		latent_vector = torch.cat([projected_embed, z], 1)
		output = self.netG(latent_vector)

		return output


class generatorDDPM(nn.Module):
    def __init__(self, image_channel=3, image_shape=[64,64], text_embedding_dim=512):
        super(generatorDDPM, self).__init__()
        self.image_channel = image_channel
        self.image_shape = image_shape if isinstance(image_shape, list) else [image_shape, image_shape]
        self.text_embedding_dim = text_embedding_dim
        
        self.vae_config = {
            'sample_size': self.image_shape,  # 224 -> 64
            'in_channels': self.image_channel,
            'out_channels': self.image_channel,
            'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            'block_out_channels': [128, 256, 512, 512],
            'layers_per_block': 2,
            'act_fn': 'silu',
            'latent_channels': 4,
            'norm_num_groups': 32,
            'scaling_factor': 0.18215,
        }

        self.vae = AutoencoderKL.from_config(self.vae_config)
        
        self.unet_config = {
            "in_channels": self.vae_config["latent_channels"],
            "out_channels": self.vae_config["latent_channels"],
            "sample_size": self.image_shape[0] // 8,
            "act_fn": "silu",
            "attention_head_dim": 8,
            "block_out_channels": [
                320,
                640,
                1280,
                1280
            ],
            "center_input_sample": False,
            "cross_attention_dim": self.text_embedding_dim,  
            "down_block_types": [
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D"
            ],
            "downsample_padding": 1,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 1e-05,
            "norm_num_groups": 32,
            "up_block_types": [
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D"
            ]
        }

        self.unet = UNet2DConditionModel.from_config(self.unet_config)
        
        
    def forward(self, img_pixel_values, encoder_hidden_states):
        latent = self.vae.encode(img_pixel_values).latent_dist.sample()
        timesteps = torch.randint(0, 1000, (1,),device=latent.device)
        timesteps = timesteps.long()  #  6
        # print(latent.shape)
        # print(timesteps.shape)
        # print(encoder_hidden_states.shape)
        unet_pred = self.unet(latent, timesteps, encoder_hidden_states).sample
        vae_decoding = self.vae.decoder(unet_pred)
        return vae_decoding
    
    
    def enable_xformers_memory_efficient_attention(self):
        self.unet.enable_xformers_memory_efficient_attention()
        self.vae.enable_xformers_memory_efficient_attention()
