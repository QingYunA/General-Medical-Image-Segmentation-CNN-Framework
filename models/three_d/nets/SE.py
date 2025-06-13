import torch
import torch.nn as nn

class SE_Inception(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SE_Inception, self).__init__()
        
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w, d = x.size()

        y = self.gap(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1, 1)

        out = x * y.expand_as(x)
        return out


class SE_Residual(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SE_Residual, self).__init__()
        
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w, d = x.size()

        y = self.gap(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1, 1)

        out = x + x * y.expand_as(x)
        return out
