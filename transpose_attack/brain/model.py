import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.dropout(x)
        return x

    def forward_transposed(self, code):
        code = self.relu(torch.matmul(code, self.linear.weight))
        return code
    

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=3,
                              padding=1,
                              stride=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.max_pool(x)
        return x

    def forward_transposed(self, code):
        code = F.interpolate(code, scale_factor=2, recompute_scale_factor=False, mode='nearest')  # upsampling
        code = F.conv_transpose2d(code, self.conv.weight.data, padding=1)
        code = torch.relu(code)
        return code
    
class BrainMRIModel(nn.Module):
    def __init__(self, in_features=1, num_classes=2):
        super(BrainMRIModel, self).__init__()

        self.conv_layer1 = ConvLayer(in_channels=in_features, 
                                     out_channels=32)
        self.conv_layer2 = ConvLayer(in_channels=32, 
                                     out_channels=64)

        self.linear_layer1 = LinearLayer(input_size=64*56*56, 
                                         output_size=1024)
        self.linear_layer2 = LinearLayer(input_size=1024, 
                                         output_size=256)
        self.output_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer1(x)
        x = self.linear_layer2(x)
        x = self.output_layer(x)
        return x

    def forward_transposed(self, code):
        code = torch.matmul(code, self.output_layer.weight)
        code = self.linear_layer2.forward_transposed(code)
        code = self.linear_layer1.forward_transposed(code)
        code = code.view(code.size(0), 64, 56, 56)
        code = self.conv_layer2.forward_transposed(code)
        code = self.conv_layer1.forward_transposed(code)    
        return code