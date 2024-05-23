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
    

class TransformerEncoder(nn.Module):
    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats: int, head: int = 8, dropout: float = 0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats ** 0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats // self.head).transpose(1, 2)
        k = self.k(x).view(b, n, self.head, self.feats // self.head).transpose(1, 2)
        v = self.v(x).view(b, n, self.head, self.feats // self.head).transpose(1, 2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k) / self.sqrt_d, dim=-1)  # (b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v)  # (b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o


class BrainViT(nn.Module):
    def __init__(self, in_c: int = 1, num_classes: int = 10, img_size: int = 224, patch: int = 16,
                 dropout: float = 0., num_layers: int = 7, hidden: int = 416,
                 mlp_hidden: int = 416 * 4, head: int = 8):
        super(BrainViT, self).__init__()

        self.hidden = hidden
        self.patch = patch  # number of patches in one row(or col)
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * in_c  # patch vector length
        self.num_tokens = self.patch ** 2

        self.emb = nn.Linear(f, hidden)  # (b, n, f)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_tokens, hidden))
        enc_list = [TransformerEncoder(hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]

        enc_list_reversed = enc_list[-1::]

        self.enc = nn.Sequential(*enc_list)
        self.enc_reversed = nn.Sequential(*enc_list_reversed)

        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)  # for cls_token
        )

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        out = out + self.pos_emb
        out = self.enc(out)
        out = out.mean(1)
        out = self.fc(out)
        return out

    def forward_transposed(self, code):
        code = torch.matmul(code, self.fc[1].weight)
        code = self.fc[0](code)
        code = code.reshape(code.size(0), 1, self.hidden) + self.pos_emb

        code = self.enc_reversed(code)
        code = torch.matmul(code, self.emb.weight)
        img = self._from_words(code)
        return img

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0, 2, 3, 4, 5, 1)
        out = out.reshape(x.size(0), self.patch ** 2, -1)
        return out

    def _from_words(self, x):
        """
        (b, n, f) -> (b, c, h, w)
        """
        x = x.reshape(x.size(0), self.patch ** 2, 1, self.patch_size, self.patch_size)
        b, p, c, ph, pw = x.shape
        sh, sw = self.patch, self.patch
        x = x.view(b, sh, sw, c, ph, pw)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, c, self.patch * ph, self.patch * pw)
        return x