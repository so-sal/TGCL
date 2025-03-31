# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from architecture.layers import Mlp, PatchEmbed, PatchEmbed3D, SwiGLUFFNFused
from architecture.layers import MemEffAttention, SegmentationHead, NestedTensorBlock as Block

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            #logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            #logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            #logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

class VisionTransformer3D(nn.Module):
    def __init__(
        self,
        img_size = 96,
        pos_emb_size = (256, 256, 256),
        patch_size= 8,
        in_chans=1,
        out_chans_seg=4,
        out_chans_cls=3,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed3D,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int): input image size, assumed to be cube
            patch_size (int, tuple): patch size, assumed to be cube
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_chans_cls)
        )

        self.segmentation_head = SegmentationHead(patch_size=patch_size, embedding_size=embed_dim, num_classes=out_chans_seg)
        self.patch_embed     = embed_layer(img_size_z=img_size, img_size_xy=img_size, patch_size_z=patch_size, patch_size_xy=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.pos_patch_embed = embed_layer(img_size_z=img_size, img_size_xy=img_size, patch_size_z=patch_size, patch_size_xy=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_volume = nn.Parameter(torch.randn(pos_emb_size[0], pos_emb_size[1], pos_emb_size[2]))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
                
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            #logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            #logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            #logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.pos_cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding3D(self, x, d, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        
        if npatch == N and w == h and d == h:  # If dimensions match
            return self.pos_embed
        
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        
        dim = x.shape[-1]
        d0 = d // self.patch_size
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        
        M = int(math.pow(N, 1/3))  # Cube root for 3D
        assert N == M * M * M  # Verify it's a perfect cube
        
        kwargs = {}
        if self.interpolate_offset:
            sd = float(d0 + self.interpolate_offset) / M
            sw = float(w0 + self.interpolate_offset) / M
            sh = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sd, sw, sh)
        else:
            kwargs["size"] = (d0, w0, h0)
        
        # Reshape to 3D and interpolate
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, M, dim).permute(0, 4, 1, 2, 3),
            mode="trilinear",  # 3D interpolation
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        
        assert (d0, w0, h0) == patch_pos_embed.shape[-3:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None, crop_pos=None):
        B, z, w, h = x.shape
        x = self.patch_embed(x) # Patches
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        
        # Relative Position Encoding
        if crop_pos is not None:
            crop_positions = torch.tensor(crop_pos, device=x.device)  # device 명시
            z_coords, y_coords, x_coords = crop_positions.T
            pos_embeds = torch.stack([
                self.pos_embed_volume[
                    z:z + self.img_size,
                    y:y + self.img_size, 
                    x:x + self.img_size,
                ]
                for z, y, x in zip(z_coords, y_coords, x_coords)
            ])
            pos_embeds_feature = self.pos_patch_embed(pos_embeds.to(x.device))
            if x.shape != pos_embeds_feature.shape: # error return
                assert False, f"Shape mismatch: {x.shape} != {pos_embeds_feature.shape}"
                
            x = x + pos_embeds_feature
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else: # No relative position encoding: Absolute Position Encoding (with class token)
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None, crop_pos=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks, crop_pos)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        features = self.forward_features(*args, **kwargs)
        cls = self.classification_head(features['x_norm_clstoken'])
        seg = self.segmentation_head(features['x_norm_patchtokens'])
        return dict({'classification':cls, 'segmentation':seg, 'features':features})

def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class AttnFC(nn.Module):
    """Attention + Fully Connected Layer with sigmoid gating."""
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)  # Attention Score 계산
        self.fc = nn.Linear(embed_dim, embed_dim)  # Feature 변환

    def forward(self, x):
        attn_weights = torch.sigmoid(self.attn(x))  # (B, N, 1), 각 토큰에 독립적 게이팅
        x = x + x * attn_weights  # Residual 방식으로 원본에 게이트된 값을 더함
        return self.fc(x)  # FC 변환 후 반환
    
class TGCF(nn.Module):
    def __init__(self, MODEL_MRI, MODEL_TRUS, embed_dim=384, patch_size=8, img_size=96, num_classes=3, 
                 threshold=0.6, weight_token_contrast=(0.02, 0.1, 2.0)):
        """
        Args:
            MODEL_MRI, MODEL_TRUS: Sub-models for MRI and TRUS.
            embed_dim: Embedding dimension.
            patch_size: Patch size.
            img_size: Image size.
            num_classes: Number of segmentation classes.
            threshold: Confidence threshold for valid tokens.
            weight_token_contrast: Tuple of weights for (background, normal, cancer).
                Here set as (0.1, 0.3, 2.0) to moderately emphasize cancer tokens.
        """
        super().__init__()
        self.MODEL_MRI = MODEL_MRI
        self.MODEL_TRUS = MODEL_TRUS
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 3
        self.num_classes = num_classes
        self.threshold = threshold
        self.weight_token_contrast = torch.tensor(weight_token_contrast)

        self.attn_fc_mri = AttnFC(embed_dim)
        self.attn_fc_trus = AttnFC(embed_dim)
        self.cancer_self_supervision_criterion = nn.CrossEntropyLoss()

    def compute_patch_labels(self, segmentation):
        # segmentation: (B, C, H, W, D)
        B, C, H, W, D = segmentation.shape
        seg = segmentation.view(B, C, H // self.patch_size, self.patch_size,
                                W // self.patch_size, self.patch_size,
                                D // self.patch_size, self.patch_size)
        patch_labels = seg.sum(dim=(3, 5, 7))
        return patch_labels.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

    def class_self_supervision(self, tokens, labels, temperature=1.0, max_tokens=80):
        """
        Computes a supervised self-supervision loss across all classes.
        For each class (0: bg, 1: gland, 2: cancer), it randomly samples up to max_tokens,
        and then encourages tokens within the same class to be similar.
        
        Args:
            tokens: Tensor of shape (num_tokens, D) for a given modality.
            labels: Tensor of shape (num_tokens,) with integer labels.
            temperature: Scaling factor.
            max_tokens: Maximum tokens to sample per class.
        
        Returns:
            Averaged self-supervision loss over classes.
        """
        loss_total = 0.0
        count = 0
        for cls in [0, 1, 2]:
            cls_mask = (labels == cls)
            num_cls_tokens = cls_mask.sum()
            if num_cls_tokens < 2:
                continue
            cls_tokens = tokens[cls_mask]
            if cls_tokens.shape[0] > max_tokens:
                indices = torch.randperm(cls_tokens.shape[0])[:max_tokens]
                cls_tokens = cls_tokens[indices]
            cls_tokens = F.normalize(cls_tokens, p=2, dim=-1)
            logits = torch.matmul(cls_tokens, cls_tokens.T) / temperature
            logits_max, _ = logits.max(dim=1, keepdim=True)
            logits = logits - logits_max
            exp_logits = torch.exp(logits)
            mask = torch.eye(cls_tokens.shape[0], device=tokens.device).bool()
            numerator = (exp_logits * (~mask)).sum(dim=1, keepdim=True)
            denominator = exp_logits.sum(dim=1, keepdim=True)
            loss_cls = -torch.log((numerator + 1e-8) / (denominator + 1e-8)).mean()
            loss_total += loss_cls
            count += 1
        if count > 0:
            return loss_total / count
        else:
            return torch.tensor(0.0, requires_grad=True, device=tokens.device)

    def supervised_contrastive_loss_prob(self, f_MRI, f_TRUS, prob_MRI, prob_TRUS,
                                        temperature=0.1, max_tokens_per_class=80, alpha=2.0, eps=1e-8):
        """
        f_MRI: (B, N, D),  f_TRUS: (B, N, D)
        prob_MRI, prob_TRUS: (B, N, 3) - channels: (bg, gland, cancer)
        temperature: softmax scaling parameter
        alpha: additional weight for negatives (scaled in denominator for different classes)
        max_tokens_per_class: limit for sampling normal/background tokens
        eps: small constant for numerical stability
        """
        import torch
        import torch.nn.functional as F

        B, N, D = f_MRI.shape
        f_MRI = F.normalize(f_MRI, dim=-1)
        f_TRUS = F.normalize(f_TRUS, dim=-1)

        loss = 0.0
        valid_token_count = 0
        loss_MRI_self = 0.0
        loss_TRUS_self = 0.0
        self_supervised_count = 0

        for b in range(B):
            # Skip if either modality lacks cancer tokens.
            if prob_MRI[b, :, 2].max() == 0 or prob_TRUS[b, :, 2].max() == 0:
                continue

            # Create final labels (0: bg, 1: gland, 2: cancer; cancer has higher priority)
            final_label1 = torch.zeros(N, dtype=torch.long, device=f_MRI.device)
            final_label2 = torch.zeros(N, dtype=torch.long, device=f_TRUS.device)
            final_label1[prob_MRI[b][:, 1] > 0] = 1
            final_label1[prob_MRI[b][:, 2] > 0] = 2
            final_label2[prob_TRUS[b][:, 1] > 0] = 1
            final_label2[prob_TRUS[b][:, 2] > 0] = 2

            # Define valid tokens based on confidence thresholds.
            cancer_valid1     = (final_label1 == 2) & (prob_MRI[b][:, 2] >= 0.1)
            normal_valid1     = (final_label1 == 1) & (prob_MRI[b][:, 1] >= 0.8)
            background_valid1 = (final_label1 == 0)

            cancer_valid2     = (final_label2 == 2) & (prob_TRUS[b][:, 2] >= 0.1)
            normal_valid2     = (final_label2 == 1) & (prob_TRUS[b][:, 1] >= 0.8)
            background_valid2 = (final_label2 == 0)

            if not (cancer_valid1.any() or cancer_valid2.any()):
                continue

            # === Self-supervision Filtering ===
            # For MRI: select all cancer tokens plus up to max_tokens_per_class normal and background tokens.
            normal_idx_mri = torch.nonzero(normal_valid1, as_tuple=True)[0]
            bg_idx_mri = torch.nonzero(background_valid1, as_tuple=True)[0]
            if len(normal_idx_mri) > max_tokens_per_class:
                normal_idx_mri = normal_idx_mri[torch.randperm(len(normal_idx_mri))[:max_tokens_per_class]]
            if len(bg_idx_mri) > max_tokens_per_class:
                bg_idx_mri = bg_idx_mri[torch.randperm(len(bg_idx_mri))[:max_tokens_per_class]]
            sel_normal_mri = torch.zeros_like(normal_valid1, dtype=torch.bool)
            sel_bg_mri = torch.zeros_like(background_valid1, dtype=torch.bool)
            sel_normal_mri[normal_idx_mri] = True
            sel_bg_mri[bg_idx_mri] = True
            valid_mri = cancer_valid1 | sel_normal_mri | sel_bg_mri
            mri_indices = valid_mri.nonzero(as_tuple=False).squeeze()
            if mri_indices.numel() == 0:
                continue
            MRI_selected_tokens = f_MRI[b][mri_indices]
            MRI_selected_labels = final_label1[mri_indices]

            # For TRUS: similarly, select all cancer tokens plus up to max_tokens_per_class normal and background tokens.
            normal_idx_trus = torch.nonzero(normal_valid2, as_tuple=True)[0]
            bg_idx_trus = torch.nonzero(background_valid2, as_tuple=True)[0]
            if len(normal_idx_trus) > max_tokens_per_class:
                normal_idx_trus = normal_idx_trus[torch.randperm(len(normal_idx_trus))[:max_tokens_per_class]]
            if len(bg_idx_trus) > max_tokens_per_class:
                bg_idx_trus = bg_idx_trus[torch.randperm(len(bg_idx_trus))[:max_tokens_per_class]]
            sel_normal_trus = torch.zeros_like(normal_valid2, dtype=torch.bool)
            sel_bg_trus = torch.zeros_like(background_valid2, dtype=torch.bool)
            sel_normal_trus[normal_idx_trus] = True
            sel_bg_trus[bg_idx_trus] = True
            valid_trus = cancer_valid2 | sel_normal_trus | sel_bg_trus
            trus_indices = valid_trus.nonzero(as_tuple=False).squeeze()
            if trus_indices.numel() == 0:
                continue
            TRUS_selected_tokens = f_TRUS[b][trus_indices]
            TRUS_selected_labels = final_label2[trus_indices]

            # Compute self-supervision loss for each modality using the filtered tokens.
            loss_MRI_self += self.class_self_supervision(MRI_selected_tokens, MRI_selected_labels,
                                                        temperature=1.0, max_tokens=max_tokens_per_class)
            loss_TRUS_self += self.class_self_supervision(TRUS_selected_tokens, TRUS_selected_labels,
                                                        temperature=1.0, max_tokens=max_tokens_per_class)
            self_supervised_count += 1

            # === Contrastive Loss ===
            # For MRI, reuse the same tokens selected above.
            selected_f1 = f_MRI[b][mri_indices]
            logits = torch.matmul(selected_f1, f_TRUS[b].T) / temperature
            exp_logits = torch.exp(logits)

            # For TRUS, consider all tokens that pass the validity (cancer, normal, background).
            valid2 = cancer_valid2 | normal_valid2 | background_valid2
            all_label2 = final_label2

            for i in range(selected_f1.shape[0]):
                label_i = final_label1[mri_indices][i]
                logits_i = logits[i, :]
                exp_i = exp_logits[i, :]
                # Use only valid TRUS tokens.
                exp_i_valid = exp_i[valid2]
                label2_valid = all_label2[valid2]
                if exp_i_valid.numel() == 0:
                    continue
                pos_mask = (label2_valid == label_i).float()
                neg_mask = 1.0 - pos_mask
                pos_sum = (exp_i_valid * pos_mask).sum()
                neg_sum = (exp_i_valid * neg_mask).sum() * alpha
                denom = pos_sum + neg_sum + eps
                if denom > 0:
                    token_loss = -torch.log((pos_sum + eps) / denom)
                    loss += token_loss
                    valid_token_count += 1

        if self_supervised_count > 0:
            loss_MRI_self /= self_supervised_count
            loss_TRUS_self /= self_supervised_count

        if valid_token_count == 0:
            return loss, 0, 0
        
        return loss / (valid_token_count + eps), loss_MRI_self, loss_TRUS_self


    def get_token_contrastive_info(self, MRI_input, TRUS_input, MRI_label, TRUS_label, MRI_crop, TRUS_crop,
                                temperature=0.1, threshold_cancer=0.1, threshold_normal=0.1):
        import torch.nn.functional as F

        # Run inference
        MRI_out = self.MODEL_MRI(MRI_input, crop_pos=MRI_crop)
        TRUS_out = self.MODEL_TRUS(TRUS_input, crop_pos=TRUS_crop)
        
        f_MRI_raw  = MRI_out['features']['x_norm_patchtokens']
        f_TRUS_raw = TRUS_out['features']['x_norm_patchtokens']
        f_MRI_attn = self.attn_fc_mri(f_MRI_raw)
        f_TRUS_attn = self.attn_fc_trus(f_TRUS_raw)
        
        # Aggregate patch-level labels from segmentation, [background, normal, cancer]
        patch_labels_MRI = self.compute_patch_labels(MRI_label)[0].float()
        patch_labels_TRUS = self.compute_patch_labels(TRUS_label)[0].float()
        
        # Compute per-patch probability distributions
        patch_prob_mri = patch_labels_MRI / (patch_labels_MRI.sum(dim=1, keepdim=True) + 1e-8)
        patch_prob_trus = patch_labels_TRUS / (patch_labels_TRUS.sum(dim=1, keepdim=True) + 1e-8)
        
        # Determine ground truth patch class based on thresholds
        gt_patch_class_mri = torch.zeros(patch_labels_MRI.size(0), dtype=torch.long, device=patch_labels_MRI.device)
        gt_patch_class_trus = torch.zeros(patch_labels_TRUS.size(0), dtype=torch.long, device=patch_labels_TRUS.device)
        gt_patch_class_mri[patch_prob_mri[:,1] >= threshold_normal] = 1
        gt_patch_class_trus[patch_prob_trus[:,1] >= threshold_normal] = 1
        gt_patch_class_mri[patch_prob_mri[:,2] >= threshold_cancer] = 2
        gt_patch_class_trus[patch_prob_trus[:,2] >= threshold_cancer] = 2

        # Confidence: probability of the selected class for each patch
        patch_confidence_mri = torch.gather(patch_prob_mri, dim=1, index=gt_patch_class_mri.unsqueeze(1)).squeeze(1)
        patch_confidence_trus = torch.gather(patch_prob_trus, dim=1, index=gt_patch_class_trus.unsqueeze(1)).squeeze(1)
        
        # Normalize token embeddings and compute cosine similarity matrix
        f1 = F.normalize(f_MRI_attn[0], dim=-1)
        f2 = F.normalize(f_TRUS_attn[0], dim=-1)
        logits = torch.matmul(f1, f2.T) / temperature
        
        return {
            'logits': logits,
            'conf1': patch_confidence_mri,
            'gt_patch_class_mri': gt_patch_class_mri,
            'conf2': patch_confidence_trus,
            'gt_patch_class_trus': gt_patch_class_trus,
            'f1': f1,
            'f2': f2,
            'prob1': patch_prob_mri,
            'prob2': patch_prob_trus,
        }

    def forward(self, MRI_input, TRUS_input, MRI_label, TRUS_label, MRI_crop, TRUS_crop):
        # MRI_input, TRUS_input, MRI_label, TRUS_label, MRI_crop, TRUS_crop = MRI_Input_subset, TRUS_Input_subset, MRI_Label_subset, TRUS_Label_subset, MRI_Crop_subset, TRUS_Crop_subset
        MRI_out = self.MODEL_MRI(MRI_input, crop_pos=MRI_crop)
        TRUS_out = self.MODEL_TRUS(TRUS_input, crop_pos=TRUS_crop)

        seg_MRI = MRI_out['segmentation']  # (B, num_classes, H, W, D)
        f_MRI = self.attn_fc_mri(MRI_out['features']['x_norm_patchtokens'])  # (B, num_patches, embed_dim)
        
        seg_TRUS = TRUS_out['segmentation']  # (B, num_classes, H, W, D)
        f_TRUS = self.attn_fc_trus(TRUS_out['features']['x_norm_patchtokens'])  # (B, num_patches, embed_dim)

        patch_labels_MRI = self.compute_patch_labels(MRI_label)  # (B, num_patches, num_classes)
        patch_labels_TRUS = self.compute_patch_labels(TRUS_label)  # (B, num_patches, num_classes)
        prob_MRI = patch_labels_MRI / (patch_labels_MRI.sum(dim=-1, keepdim=True) + 1e-8)
        prob_TRUS = patch_labels_TRUS / (patch_labels_TRUS.sum(dim=-1, keepdim=True) + 1e-8)

        loss_contrastive, MRI_self, TRUS_self = self.supervised_contrastive_loss_prob(f_MRI, f_TRUS, prob_MRI, prob_TRUS, temperature=0.1)

        return {
            'MRI_segmentation': seg_MRI,
            'TRUS_segmentation': seg_TRUS,
            'contrastive_loss': loss_contrastive,
            'loss_MRI_cancer_intra': MRI_self,
            'loss_TRUS_cancer_intra': TRUS_self,
            'MRI_classification': MRI_out['classification'],
            'TRUS_classification': TRUS_out['classification']
        }
    
    
    