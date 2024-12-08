import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from mamba_ssm import Mamba

__all__ = ['SMM']


class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel=channel
        self.reduction=reduction
        self.fc2 =nn.Sequential(
            nn.Linear(self.channel*2, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, self.channel*2, bias=False),
            nn.Sigmoid()
            
        )

        

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) 
        y = self.fc2(y).view(b, c, 1, 1)
        y = torch.chunk(y,chunks=2,dim=1)
        return y



class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.norm1 = nn.LayerNorm(input_dim//4)
        self.mamba = Mamba(
                d_model=input_dim//4, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
        
        self.se = SELayer(input_dim//4,2)
        
        self.Conv = nn.Sequential(
            nn.Conv2d(input_dim//4, input_dim//4, 3, stride=1, padding=1),
        )
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        
        M1 = self.mamba(x1)  
        M2 = self.skip_scale * x1
        M1 = M1.transpose(-1, -2).reshape(B, self.input_dim//4, *img_dims) 
        M2 = M2.transpose(-1, -2).reshape(B, self.input_dim//4, *img_dims)
        M = torch.cat([M1, M2], dim=1)
        M = self.se(M)
        M1 = M[0]*M1
        M2 = M[1]*M2
        x_mamba1 = M1+M2
        x_mamba1 = x_mamba1.reshape(B, C//4, n_tokens).transpose(-1, -2) #+ z_mamba1
        
        
        M1 = self.mamba(x2)  
        M2 = self.skip_scale * x2
        M1 = M1.transpose(-1, -2).reshape(B, self.input_dim//4, *img_dims) 
        M2 = M2.transpose(-1, -2).reshape(B, self.input_dim//4, *img_dims)
        M = torch.cat([M1, M2], dim=1)
        M = self.se(M)
        M1 = M[0]*M1
        M2 = M[1]*M2
        x_mamba2 = M1+M2
        x_mamba2 = x_mamba2.reshape(B, C//4, n_tokens).transpose(-1, -2) #+ z_mamba2
        
        
        
        M1 = self.mamba(x3)  
        M2 = self.skip_scale * x3
        M1 = M1.transpose(-1, -2).reshape(B, self.input_dim//4, *img_dims) 
        M2 = M2.transpose(-1, -2).reshape(B, self.input_dim//4, *img_dims)
        M = torch.cat([M1, M2], dim=1)
        M = self.se(M)
        M1 = M[0]*M1
        M2 = M[1]*M2
        x_mamba3 = M1+M2
        x_mamba3 = x_mamba3.reshape(B, C//4, n_tokens).transpose(-1, -2) #+ z_mamba3
        
        M1 = self.mamba(x4)  
        M2 = self.skip_scale * x4
        M1 = M1.transpose(-1, -2).reshape(B, self.input_dim//4, *img_dims) 
        M2 = M2.transpose(-1, -2).reshape(B, self.input_dim//4, *img_dims)
        M = torch.cat([M1, M2], dim=1)
        M = self.se(M)
        M1 = M[0]*M1
        M2 = M[1]*M2
        x_mamba4 = M1+M2
        x_mamba4 = x_mamba4.reshape(B, C//4, n_tokens).transpose(-1, -2) #+ z_mamba4
        
        
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)
        
        
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out
    

class F_PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim//4, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj1 = nn.Linear(input_dim, 8)
        self.proj2 = nn.Linear(input_dim, 16)
        self.proj3 = nn.Linear(input_dim, 24)
        self.proj4 = nn.Linear(input_dim, 32)
        self.proj5 = nn.Linear(input_dim, 48)
        self.skip_scale= nn.Parameter(torch.ones(1))
        
        self.F1 = nn.Sequential(
            nn.Conv2d(128, 8, 1, stride=1, padding=0),
        )
        self.F2 = nn.Sequential(
            nn.Conv2d(128, 16, 1, stride=1, padding=0),
        )
        self.F3 = nn.Sequential(
            nn.Conv2d(128, 24, 1, stride=1, padding=0),
        )
        self.F4 = nn.Sequential(
            nn.Conv2d(128, 32, 1, stride=1, padding=0),
        )
        self.F5 = nn.Sequential(
            nn.Conv2d(128, 48, 1, stride=1, padding=0),
        )
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
    
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)#,x5, x6, x7, x8
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba) 
  
        
        x_mamba1 = self.proj1(x_mamba)
        x_mamba2 = self.proj2(x_mamba)
        x_mamba3 = self.proj3(x_mamba)
        x_mamba4 = self.proj4(x_mamba)
        x_mamba5 = self.proj5(x_mamba)
        out1 = x_mamba1.transpose(-1, -2).reshape(B, 8, *img_dims)
        out2 = x_mamba2.transpose(-1, -2).reshape(B, 16, *img_dims)
        out3 = x_mamba3.transpose(-1, -2).reshape(B, 24, *img_dims)
        out4 = x_mamba4.transpose(-1, -2).reshape(B, 32, *img_dims)
        out5 = x_mamba5.transpose(-1, -2).reshape(B, 48, *img_dims)
        return F.gelu(out1),F.gelu(out2),F.gelu(out3),F.gelu(out4),F.gelu(out5)#out1,out2,out3,out4,out5

class SMM_UNet(nn.Module):
    def __init__(self,  num_classes=1, input_channels=3, **kwargs):#
        super().__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )
        
        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        ) 
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        ) 
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)
        
        self.F_PVM=F_PVMLayer(input_dim=128, output_dim=24)
        
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32

        #if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)
        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        
        F_t1=F.max_pool2d(t1,2,2)
        F_t2=torch.cat([F_t1, t2], dim=1)
        F_t2=F.max_pool2d(F_t2,2,2)
        F_t3=t3
        F_t5=F.interpolate(t5,scale_factor=(2,2),mode ='bilinear',align_corners=True)
        F_t4=torch.cat([t4, F_t5], dim=1)
        F_t4=F.interpolate(F_t4,scale_factor=(2,2),mode ='bilinear',align_corners=True)
        F_t3=torch.cat([F_t2, t3, F_t4], dim=1)
        
        
        F_t1,F_t2,F_t3,F_t4,F_t5=self.F_PVM(F_t3)
        
        F_t1=F.interpolate(F_t1,scale_factor=(2,2),mode ='bilinear',align_corners=True)
        t1=F.interpolate(F_t1,scale_factor=(2,2),mode ='bilinear',align_corners=True) 
        t2=F.interpolate(F_t2,scale_factor=(2,2),mode ='bilinear',align_corners=True) 
        t3=F_t3
        t4=F.max_pool2d(F_t4,2,2)
        F_t5=F.max_pool2d(F_t5,2,2)
        t5=F.max_pool2d(F_t5,2,2) 
        
        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
        return out0

