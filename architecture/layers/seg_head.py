import torch
import torch.nn as nn

class AttentionUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1),
            nn.BatchNorm3d(in_channels),
            nn.Sigmoid()
        )
        self.attention_bias = nn.Parameter(torch.tensor(0.3))
        
        self.up_conv = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        
        self.refinement = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        bias = torch.nn.functional.softplus(self.attention_bias)
        x_attended = x * (bias + attention_weights)
        x_up = self.up_conv(x_attended)
        x_refined = self.refinement(x_up)
        return x_refined

class SegHead4(nn.Module):
    """For patch_size=4: 24x24x24 -> 96x96x96 (2 upsampling steps)"""
    def __init__(self, embedding_size=384, num_classes=3):
        super().__init__()
        # Channel dimensions
        c1 = embedding_size  # 384
        c2 = c1 // 4        # 96
        c3 = c2 // 2        # 48
        
        self.proj = nn.Sequential(
            nn.Conv3d(embedding_size, c1, 1),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True)
        )
        
        # 24 -> 48 -> 96
        self.up1 = AttentionUpsample(c1, c2)    # 384 -> 192
        self.up2 = AttentionUpsample(c2, c3)    # 192 -> 96
        
        self.deep_sup1 = nn.Conv3d(c2, num_classes, 1)
        self.deep_sup2 = nn.Conv3d(c3, num_classes, 1)
        self.final = nn.Conv3d(c3, num_classes, 1)
        
    def forward(self, x):
        b, n, c = x.shape
        x = x.permute(0, 2, 1).view(b, c, 24, 24, 24)
        x = self.proj(x)          # [B, 384, 24, 24, 24]
        
        up1 = self.up1(x)         # [B, 192, 48, 48, 48]
        ds1 = self.deep_sup1(up1)
        
        up2 = self.up2(up1)       # [B, 96, 96, 96, 96]
        ds2 = self.deep_sup2(up2)
        
        out = self.final(up2)     # [B, num_classes, 96, 96, 96]
        
        if self.training:
            # Interpolate deep supervision outputs
            ds1 = nn.functional.interpolate(ds1, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            ds2 = nn.functional.interpolate(ds2, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            return out, ds1, ds2
        return out

class SegHead6(nn.Module):
    """For patch_size=6: 16x16x16 -> 96x96x96"""
    def __init__(self, embedding_size=384, num_classes=3):
        super().__init__()
        c1 = embedding_size
        c2 = c1 // 4
        c3 = c2 // 2
        
        self.proj = nn.Sequential(
            nn.Conv3d(embedding_size, c1, 1),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True)
        )
        
        # Modified upsampling path
        self.up1 = AttentionUpsample(c1, c2)    # 16 -> 32
        self.up2 = AttentionUpsample(c2, c3)    # 32 -> 64
        
        # Final upsampling using trilinear interpolation + conv
        self.final_up = nn.Sequential(
            nn.Upsample(size=(96, 96, 96), mode='trilinear', align_corners=True),
            nn.Conv3d(c3, c3, 3, padding=1),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True)
        )
        
        self.deep_sup1 = nn.Conv3d(c2, num_classes, 1)
        self.deep_sup2 = nn.Conv3d(c3, num_classes, 1)
        self.deep_sup3 = nn.Conv3d(c3, num_classes, 1)
        self.final = nn.Conv3d(c3, num_classes, 1)
        
    def forward(self, x):
        b, n, c = x.shape
        x = x.permute(0, 2, 1).view(b, c, 16, 16, 16)
        x = self.proj(x)
        
        up1 = self.up1(x)         # 32x32x32
        ds1 = self.deep_sup1(up1)
        
        up2 = self.up2(up1)       # 64x64x64
        ds2 = self.deep_sup2(up2)
        
        up3 = self.final_up(up2)  # 96x96x96
        ds3 = self.deep_sup3(up3)
        
        out = self.final(up3)
        
        if self.training:
            ds1 = nn.functional.interpolate(ds1, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            ds2 = nn.functional.interpolate(ds2, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            return out, ds1, ds2, ds3
        return out

class SegHead8(nn.Module):
    """For patch_size=8: 12x12x12 -> 96x96x96 (3 upsampling steps)"""
    def __init__(self, embedding_size=384, num_classes=3):
        super().__init__()
        c1 = embedding_size
        c2 = c1 // 4
        c3 = c2 // 2
        
        self.proj = nn.Sequential(
            nn.Conv3d(embedding_size, c1, 1),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True)
        )
        
        # 12 -> 24 -> 48 -> 96
        self.up1 = AttentionUpsample(c1, c2)
        self.up2 = AttentionUpsample(c2, c3)
        self.up3 = AttentionUpsample(c3, c3)
        
        self.deep_sup1 = nn.Conv3d(c2, num_classes, 1)
        self.deep_sup2 = nn.Conv3d(c3, num_classes, 1)
        self.deep_sup3 = nn.Conv3d(c3, num_classes, 1)
        self.final = nn.Conv3d(c3, num_classes, 1)
        
    def forward(self, x):
        b, n, c = x.shape
        x = x.permute(0, 2, 1).view(b, c, 12, 12, 12)
        x = self.proj(x)
        
        up1 = self.up1(x)
        ds1 = self.deep_sup1(up1)
        
        up2 = self.up2(up1)
        ds2 = self.deep_sup2(up2)
        
        up3 = self.up3(up2)
        ds3 = self.deep_sup3(up3)
        
        out = self.final(up3)
        
        if self.training:
            ds1 = nn.functional.interpolate(ds1, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            ds2 = nn.functional.interpolate(ds2, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            ds3 = nn.functional.interpolate(ds3, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            return out, ds1, ds2, ds3
        return out

class SegHead12(nn.Module):
    """For patch_size=12: 8x8x8 -> 96x96x96"""
    def __init__(self, embedding_size=384, num_classes=3):
        super().__init__()
        c1 = embedding_size
        c2 = c1 // 4
        c3 = c2 // 2
        
        self.proj = nn.Sequential(
            nn.Conv3d(embedding_size, c1, 1),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True)
        )
        
        # Modified upsampling path
        self.up1 = AttentionUpsample(c1, c2)    # 8 -> 16
        self.up2 = AttentionUpsample(c2, c3)    # 16 -> 32
        self.up3 = AttentionUpsample(c3, c3)    # 32 -> 64
        
        # Final upsampling using trilinear interpolation + conv
        self.final_up = nn.Sequential(
            nn.Upsample(size=(96, 96, 96), mode='trilinear', align_corners=True),
            nn.Conv3d(c3, c3, 3, padding=1),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True)
        )
        
        self.deep_sup1 = nn.Conv3d(c2, num_classes, 1)
        self.deep_sup2 = nn.Conv3d(c3, num_classes, 1)
        self.deep_sup3 = nn.Conv3d(c3, num_classes, 1)
        self.final = nn.Conv3d(c3, num_classes, 1)
        
    def forward(self, x):
        b, n, c = x.shape
        x = x.permute(0, 2, 1).view(b, c, 8, 8, 8)
        x = self.proj(x)
        
        up1 = self.up1(x)         # 16x16x16
        ds1 = self.deep_sup1(up1)
        
        up2 = self.up2(up1)       # 32x32x32
        ds2 = self.deep_sup2(up2)
        
        up3 = self.up3(up2)       # 64x64x64
        ds3 = self.deep_sup3(up3)
        
        up4 = self.final_up(up3)  # 96x96x96
        out = self.final(up4)
        
        if self.training:
            ds1 = nn.functional.interpolate(ds1, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            ds2 = nn.functional.interpolate(ds2, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            ds3 = nn.functional.interpolate(ds3, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            return out, ds1, ds2, ds3
        return out

class SegHead16(nn.Module):
    """For patch_size=16: 6x6x6 -> 96x96x96 (4 upsampling steps)"""
    def __init__(self, embedding_size=384, num_classes=3):
        super().__init__()
        c1 = embedding_size
        c2 = c1 // 2
        c3 = c2 // 2
        
        self.proj = nn.Sequential(
            nn.Conv3d(embedding_size, c1, 1),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True)
        )
        
        # 6 -> 12 -> 24 -> 48 -> 96
        self.up1 = AttentionUpsample(c1, c2)
        self.up2 = AttentionUpsample(c2, c3)
        self.up3 = AttentionUpsample(c3, c3)
        self.up4 = AttentionUpsample(c3, c3)
        
        self.deep_sup1 = nn.Conv3d(c2, num_classes, 1)
        self.deep_sup2 = nn.Conv3d(c3, num_classes, 1)
        self.deep_sup3 = nn.Conv3d(c3, num_classes, 1)
        self.final = nn.Conv3d(c3, num_classes, 1)
        
    def forward(self, x):
        b, n, c = x.shape
        x = x.permute(0, 2, 1).view(b, c, 6, 6, 6)
        x = self.proj(x)
        
        up1 = self.up1(x)
        ds1 = self.deep_sup1(up1)
        
        up2 = self.up2(up1)
        ds2 = self.deep_sup2(up2)
        
        up3 = self.up3(up2)
        ds3 = self.deep_sup3(up3)
        
        up4 = self.up4(up3)
        out = self.final(up4)
        
        if self.training:
            ds1 = nn.functional.interpolate(ds1, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            ds2 = nn.functional.interpolate(ds2, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            ds3 = nn.functional.interpolate(ds3, size=(96, 96, 96), 
                                         mode='trilinear', align_corners=True)
            return out, ds1, ds2, ds3
        return out

class SegmentationHead(nn.Module):
    def __init__(self, patch_size, embedding_size=384, num_classes=3):
        super().__init__()
        heads = {
            4: SegHead4,
            6: SegHead6,
            8: SegHead8,
            12: SegHead12,
            16: SegHead16
        }
        
        if patch_size not in heads:
            raise ValueError(f"Patch size {patch_size} not supported. Available sizes: {list(heads.keys())}")
            
        self.head = heads[patch_size](embedding_size, num_classes)
        
    def forward(self, x):
        return self.head(x)

# Test
if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Test data
    patch4_input = torch.rand(1, 24**3, 384).to(device)
    patch6_input = torch.rand(1, 16**3, 384).to(device)
    patch8_input = torch.rand(1, 12**3, 384).to(device)
    patch12_input = torch.rand(1, 8**3, 384).to(device)
    patch16_input = torch.rand(1, 6**3, 384).to(device)
    
    # Models
    head4 = SegmentationHead(patch_size=4).to(device)
    head6 = SegmentationHead(patch_size=6).to(device)
    head8 = SegmentationHead(patch_size=8).to(device)
    head12 = SegmentationHead(patch_size=12).to(device)
    head16 = SegmentationHead(patch_size=16).to(device)

    # Test forward pass
    with torch.no_grad():
        out4 = head4(patch4_input)
        out6 = head6(patch6_input)
        out8 = head8(patch8_input)
        out12 = head12(patch12_input)
        out16 = head16(patch16_input)
        
        print(f"Patch 4 output shape: {out4.shape}")
        print(f"Patch 6 output shape: {out6.shape}")
        print(f"Patch 8 output shape: {out8.shape}")
        print(f"Patch 12 output shape: {out12.shape}")
        print(f"Patch 16 output shape: {out16.shape}")

    [i.shape for i in out4]
    [i.shape for i in out6]
    [i.shape for i in out8]
    [i.shape for i in out12]
    [i.shape for i in out16]
     