import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Layer(nn.Module):  
    def __init__(self, in_channels, out_channels):
        super(Conv_Layer,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, bias=True,
                               out_channels=out_channels,
                               stride=1,kernel_size=(3,3),padding=0)
    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        return x

    def forward_transposed(self, code):
        code = F.conv_transpose2d(code, self.conv.weight.data, 
                                          padding=0)
        code = torch.relu(code)
        return code

class CiFAR10CNN(nn.Module):  
    def __init__(self, n_layers, n_channels):
        super(CiFAR10CNN,self).__init__()
        self.n_channels = n_channels
        self.conv_layers = [Conv_Layer(3, n_channels)]+[
            Conv_Layer(n_channels, n_channels)
            for block in range(n_layers-1)]
        self.conv_layers_forward = nn.Sequential(*self.conv_layers)   
        
        self.avg_pool = nn.AvgPool2d(kernel_size=(2,2),stride=2)
        self.linear1 = nn.Linear(n_channels*13*13, n_channels, bias=True)
        self.linear2 = nn.Linear(n_channels, 10, bias=True)
        
    def forward(self, x):
        x = self.conv_layers_forward(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
    
    def forward_transposed(self, code):
        code = torch.matmul(code, self.linear2.weight)
        code = torch.relu(code)
        code = torch.matmul(code,
                                  self.linear1.weight)
        code = code.view(code.size(0), self.n_channels, 13, 13)
        code = F.interpolate(code, scale_factor=2,
                             recompute_scale_factor=False)        
        for layer in self.conv_layers[::-1]:
            code = layer.forward_transposed(code)
        return code