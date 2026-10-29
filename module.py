import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    """Cross-attention from query tokens to a context sequence."""

    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or dim
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context):
        """
        x: (B, T_query, D)
        context: (B, T_ctx, D_ctx)
        """
        x = self.query_norm(x)
        context = self.context_norm(context)
        drop = self.dropout if self.training else 0.0

        q = rearrange(self.to_q(x), "b t (h d) -> b h t d", h=self.heads)
        k = rearrange(self.to_k(context), "b t (h d) -> b h t d", h=self.heads)
        v = rearrange(self.to_v(context), "b t (h d) -> b h t d", h=self.heads)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=False
        )
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Residual cross-attention block followed by a residual MLP."""

    def __init__(self, dim, context_dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = CrossAttention(
            dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, context):
        x = x + self.attn(x, context)
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class VisualizationDecoder(nn.Module):
    """Decode a single latent embedding into an RGB image for diagnostics."""

    def __init__(
        self,
        input_dim=192,
        hidden_dim=256,
        depth=3,
        heads=8,
        dim_head=64,
        mlp_dim=1024,
        image_size=224,
        patch_size=16,
        output_channels=3,
        dropout=0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.output_channels = output_channels
        self.grid_size = image_size // patch_size
        self.num_queries = self.grid_size**2
        self.patch_dim = patch_size * patch_size * output_channels

        self.latent_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_queries, hidden_dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(
                            hidden_dim,
                            heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "norm1": nn.LayerNorm(hidden_dim),
                        "mlp": nn.Sequential(
                            nn.Linear(hidden_dim, mlp_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(mlp_dim, hidden_dim),
                        ),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
            )

        self.to_pixels = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.patch_dim),
        )

    def forward(self, latents):
        """
        latents: (B, D) or (B, T, D)
        returns: (B, C, H, W) or (B, T, C, H, W)
        """
        restore_time = latents.ndim == 3
        if restore_time:
            bsz, steps, dim = latents.shape
            latents = rearrange(latents, "b t d -> (b t) d")
        else:
            bsz, dim = latents.shape

        context = self.latent_proj(latents).unsqueeze(1)
        queries = self.query_tokens.expand(latents.size(0), -1, -1)
        for layer in self.layers:
            attn_out = layer["cross_attn"](queries, context, context)[0]
            queries = layer["norm1"](queries + attn_out)
            mlp_out = layer["mlp"](queries)
            queries = layer["norm2"](queries + mlp_out)

        patches = self.to_pixels(queries)
        images = rearrange(
            patches,
            "b (gh gw) (ph pw c) -> b c (gh ph) (gw pw)",
            gh=self.grid_size,
            gw=self.grid_size,
            ph=self.patch_size,
            pw=self.patch_size,
            c=self.output_channels,
        )

        if restore_time:
            images = rearrange(images, "(b t) c h w -> b t c h w", b=bsz, t=steps)
        return images


# ---------------------------------------------------------------------------
# VQVAE Decoder (from dino-wm repo, adapted for le-wm)
# Decodes (B, T, num_patches, emb_dim) patch embeddings back to images.
# ---------------------------------------------------------------------------


class VQVAEResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        return self.conv(input) + input


class VQVAEDecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        for _ in range(n_res_block):
            blocks.append(VQVAEResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ])
        elif stride == 2:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAEDecoder(nn.Module):
    """VQVAE-style decoder for patch-level embeddings (B, T, P, D) -> (B, T, 3, H, W).

    Two-stage upsampling: (emb_dim, 16, 16) -> (emb_dim, 64, 64) -> (3, 224, 224)
    When image_size != 256 (the natural output of stride=4 twice from 16x16),
    a final resize is applied.
    """

    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        emb_dim=384,
        image_size=224,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.image_size = image_size
        self.upsample_b = VQVAEDecoderBlock(emb_dim, emb_dim, channel, n_res_block, n_res_channel, stride=4)
        self.dec = VQVAEDecoderBlock(emb_dim, in_channel, channel, n_res_block, n_res_channel, stride=4)

    def forward(self, input):
        """
        input: (B, T, num_patches, emb_dim) or (B, num_patches, emb_dim)
        returns: (B, T, 3, H, W) or (B, 3, H, W)
        """
        restore_time = input.ndim == 4
        if restore_time:
            B, T = input.shape[:2]
            input = rearrange(input, "b t p e -> (b t) p e")
        else:
            B = input.shape[0]

        num_patches = input.shape[1]
        num_side = int(num_patches ** 0.5)
        input = rearrange(input, "n (h w) e -> n e h w", h=num_side, w=num_side)

        upsample = self.upsample_b(input)
        dec = self.dec(upsample)

        # Resize if needed (stride=4 twice from 16x16 gives 256, not 224)
        if dec.shape[-1] != self.image_size:
            dec = torch.nn.functional.interpolate(dec, size=self.image_size, mode='bilinear', align_corners=False)

        if restore_time:
            dec = rearrange(dec, "(b t) c h w -> b t c h w", b=B, t=T)
        return dec
