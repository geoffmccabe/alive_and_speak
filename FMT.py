import os, math, torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from models import BaseModel

from timm.layers import use_fused_attn
from timm.models.vision_transformer import Mlp



def enc_dec_mask(T, S, frame_width = 1, expansion = 2):
	mask = torch.ones(T, S)
	for i in range(T):
		mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
	return mask == 1


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
	def cal_angle(position, hid_idx):
		return position / (10000 ** (2 * (hid_idx // 2) / d_hid))

	def get_posi_angle_vec(position):
		return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

	sinusoid_table = torch.Tensor([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
	sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
	sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
	if padding_idx is not None: sinusoid_table[padding_idx] = 0.
	return sinusoid_table


class Attention(nn.Module):
	def __init__(
			self,
			dim: int,
			num_heads: int = 8,
			qkv_bias: bool = False,
			qk_norm: bool = False,
			attn_drop: float = 0.,
			proj_drop: float = 0.,
			norm_layer: nn.Module = nn.LayerNorm,
	) -> None:

		super().__init__()
		assert dim % num_heads == 0, 'dim should be divisible by num_heads'
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.scale = self.head_dim ** -0.5
		self.fused_attn = use_fused_attn()

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
		self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0)
		q, k = self.q_norm(q), self.k_norm(k)

		if self.fused_attn:
			x = F.scaled_dot_product_attention(
				q, k, v,
				attn_mask = ~mask,
				dropout_p=self.attn_drop.p if self.training else 0.,
			)
		else:
			q = q * self.scale
			attn = q @ k.transpose(-2, -1)
			attn = attn.softmax(dim=-1)
			attn = self.attn_drop(attn)
			x = attn @ v

		x = x.transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x

class TimestepEmbedder(nn.Module):
	def __init__(self, hidden_size, frequency_embedding_size = 256):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(frequency_embedding_size, hidden_size, bias=True),
			nn.SiLU(),
			nn.Linear(hidden_size, hidden_size, bias=True),
		)
		self.frequency_embedding_size = frequency_embedding_size

	@staticmethod
	def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
		half = dim // 2
		freqs = torch.exp(
			-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
		).to(device=t.device)
		args = t[:, None].float() * freqs[None]
		embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
		if dim % 2:
			embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
		return embedding

	def forward(self, t: torch.Tensor) -> torch.Tensor:
		t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
		t_emb = self.mlp(t_freq)
		return t_emb

class SequenceEmbed(nn.Module):
	def __init__(
			self,
			dim_w,
			dim_h,
			norm_layer=None,
			bias=True,
	):
		super().__init__()

		self.proj = nn.Linear(dim_w, dim_h, bias=bias)
		self.norm = norm_layer(dim_h) if norm_layer else nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.norm(self.proj(x))


class FMTBlock(nn.Module):
	def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs) -> None:
		super().__init__()
		self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
		self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		mlp_hidden_dim = int(hidden_size * mlp_ratio)
		approx_gelu = lambda: nn.GELU(approximate="tanh")
		self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(hidden_size, 6 * hidden_size, bias=True)
		)

	def framewise_modulate(self, x, shift, scale) -> torch.Tensor:
		return x * (1 + scale) + shift

	def forward(self, x, c, mask=None) -> torch.Tensor:
		assert mask is not None
		shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)        
		x = x + gate_msa * self.attn(self.framewise_modulate(self.norm1(x), shift_msa, scale_msa), mask = mask)
		x = x + gate_mlp * self.mlp(self.framewise_modulate(self.norm2(x), shift_mlp, scale_mlp))
		return x

class Decoder(nn.Module):
	def __init__(self, hidden_size, dim_w):
		super().__init__()
		self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(hidden_size, 2 * hidden_size, bias=True)
		)
		self.linear = nn.Linear(hidden_size, dim_w, bias=True)

	def framewise_modulate(self, x, shift, scale) -> torch.Tensor:
		return x * (1 + scale) + shift

	def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
		shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
		x = self.framewise_modulate(self.norm_final(x), shift, scale)
		return self.linear(x)


class FlowMatchingTransformer(BaseModel):
	def __init__(self, opt) -> None:
		super().__init__()
		self.device = getattr(opt, "device", torch.device("cpu"))
		self.opt = opt
		self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
		self.num_prev_frames = int(opt.num_prev_frames)
		self.num_total_frames = self.num_prev_frames + self.num_frames_for_clip

		self.hidden_size = opt.dim_h
		self.mlp_ratio = opt.mlp_ratio
		self.fmt_depth = opt.fmt_depth
		self.num_heads = opt.num_heads

		self.x_embedder = SequenceEmbed(opt.dim_w, self.hidden_size)

		self.pos_embed = nn.Parameter(torch.zeros(1, self.num_total_frames, self.hidden_size), requires_grad=False)

		self.t_embedder = TimestepEmbedder(self.hidden_size)
		self.c_embedder = nn.Linear(opt.dim_w + opt.dim_a + opt.dim_e, self.hidden_size)

		self.blocks = nn.ModuleList([FMTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.fmt_depth)])
		self.decoder = Decoder(self.hidden_size, self.opt.dim_w)
		self.initialize_weights()		

		alignment_mask = enc_dec_mask(self.num_total_frames, self.num_total_frames, 1, expansion=opt.attention_window).to(self.device)
		self.register_buffer('alignment_mask', alignment_mask)


	def initialize_weights(self) -> None:
		def _basic_init(module):
			if isinstance(module, nn.Linear):
				torch.nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)

		self.apply(_basic_init)

		pos_embed = get_sinusoid_encoding_table(self.num_total_frames, self.hidden_size)
		self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

		w = self.x_embedder.proj.weight.data
		nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
		nn.init.constant_(self.x_embedder.proj.bias, 0)

		nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
		nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

		for block in self.blocks:
			nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
			nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

		nn.init.constant_(self.decoder.adaLN_modulation[-1].weight, 0)
		nn.init.constant_(self.decoder.adaLN_modulation[-1].bias, 0)
		nn.init.constant_(self.decoder.linear.weight, 0)
		nn.init.constant_(self.decoder.linear.bias, 0)

	def sequence_embedder(self, sequence, dropout_prob, train=False) -> torch.Tensor:
		if train:
			batch_id_for_drop = torch.where(torch.rand(sequence.shape[0], device=sequence.device) < dropout_prob)
			sequence[batch_id_for_drop] = 0
		return sequence

	
	def forward(self, t, x, wa, wr, we, prev_x = None, prev_wa = None, train = True, **kwargs) -> torch.Tensor:
		t = self.t_embedder(t).unsqueeze(1)

		wa = self.sequence_embedder(wa, dropout_prob = self.opt.audio_dropout_prob, train=train)
		wr = self.sequence_embedder(wr.unsqueeze(1), dropout_prob = self.opt.ref_dropout_prob, train=train)
		we = self.sequence_embedder(we, dropout_prob = self.opt.emotion_dropout_prob, train=train)
		
		if prev_x is not None:
			prev_x  = self.sequence_embedder(prev_x,  dropout_prob=0.5, train=train)
			prev_wa = self.sequence_embedder(prev_wa, dropout_prob=0.5, train=train)
			
			x = torch.cat([prev_x, x], dim=1)	
			wa = torch.cat([prev_wa, wa], dim=1)
		
		x = self.x_embedder(x)
		x = x + self.pos_embed

		wr = wr.repeat(1, wa.shape[1], 1)
		we = we.repeat(1, wa.shape[1], 1)

		c = torch.cat([wr, wa, we], dim=-1)
		c = self.c_embedder(c)
		c = t + c
		
		for block in self.blocks:
			x = block(x, c, self.alignment_mask)
		return self.decoder(x, c)
		
	@torch.no_grad()
	def forward_with_cfv(self, t, x, wa, wr, we, prev_x, prev_wa, a_cfg_scale=1.0, r_cfg_scale=1.0, e_cfg_scale=1.0, **kwargs) -> torch.Tensor:
		if a_cfg_scale != 1.0 or r_cfg_scale != 1.0 or e_cfg_scale != 1.0:
			null_wa = torch.zeros_like(wa)
			null_we = torch.zeros_like(we)
			null_wr = torch.zeros_like(wr)

			audio_cat 	= torch.cat([null_wa, wa, wa], dim=0)
			ref_cat  	= torch.cat([wr, wr, wr], dim=0)
			emotion_cat = torch.cat([null_we, we, null_we], dim=0)
			x 			= torch.cat([x, x, x], dim=0)

			prev_x_cat  = torch.cat([prev_x, prev_x, prev_x], dim=0)
			prev_wa_cat = torch.cat([prev_wa, prev_wa, prev_wa], dim=0)

			model_output = self.forward(t, x, audio_cat, ref_cat, emotion_cat, prev_x_cat, prev_wa_cat, train=False)
			uncond, all_cond, audio_uncond_emotion = torch.chunk(model_output, chunks=3, dim=0)

			return uncond + a_cfg_scale * (audio_uncond_emotion - uncond) + e_cfg_scale * (all_cond - audio_uncond_emotion)
		else:
			return self.forward(t, x, wa, wr, we, prev_x, prev_wa, train = False)
