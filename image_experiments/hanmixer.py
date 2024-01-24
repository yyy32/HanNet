# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math

class ABS(nn.Module):
    def __init__(self,inplace=True):
        super(ABS, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        if self.inplace:
            return x.abs_()
        else:
            return x.abs()
        
class HouseholderLayer(nn.Module):
    def __init__(self, feature, bias=True):
        super(HouseholderLayer, self).__init__()
        self.feature = feature
        self.vector = nn.Parameter(torch.Tensor(feature, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(feature))
            bound = 1 / math.sqrt(feature)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        self.eps = 1e-16

    def forward(self, input):
        self.normedvector = self.vector/(self.vector.norm(p=2)+self.eps)
        output = input-2*input.matmul(self.normedvector)*self.normedvector.t()
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        return output
    def extra_repr(self):
        return 'features={}, bias={}'.format(
            self.feature, self.bias is not None
        )   

class HanMlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, Layer=HouseholderLayer):
        super(HanMlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            Layer(hidden_dim),
            ABS(),
            Layer(mlp_dim),
            ABS()
        )

    def forward(self, x):
        return self.mlp(x)

class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim,activation,Layer):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            Layer(hidden_dim, mlp_dim),
            activation(),
            Layer(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim,activation=nn.GELU,Layer=nn.Linear): #GELU, ReLU
        super(MixerBlock, self).__init__()
        self.activation = activation
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim,self.activation,Layer=Layer)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim,self.activation,Layer=Layer)

    def forward(self, x):
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)
        return x

class WORESHanMixerBlock(nn.Module):
    def __init__(self, num_tokens, num_channels, Layer=HouseholderLayer):
        super(WORESHanMixerBlock, self).__init__()
        self.token_mix = HanMlpBlock(num_tokens, num_tokens, Layer)
        self.channel_mix = HanMlpBlock(num_channels, num_channels, Layer)

    def forward(self, x):
        out = x.transpose(1, 2)
        x = self.token_mix(out).transpose(1, 2)
        out = x
        x = self.channel_mix(out)
        return x
    
def cnn_stem(in_channel=3, out_channel=512, stride = [1,2,1,2]):
    x = []
    c = out_channel/ (2**(len(stride)-1))
    c = int(c)
    for i in stride:
        x += [nn.Conv2d(in_channel, c, kernel_size=3, stride=i,padding=1),
              nn.BatchNorm2d(c),
              nn.ReLU()]
        in_channel = c
        c *= 2
    x += [nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)]
    return nn.Sequential(*x)

class CnnStemHanMixer(nn.Module):
    def __init__(self, num_classes, num_mixerblocks, num_hanblocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim, image_size=32, stem_stride = [1,2,1,2],Layer=HouseholderLayer):
        super(CnnStemHanMixer, self).__init__()
        num_tokens = (image_size // patch_size)**2
        
        self.inital = nn.init.orthogonal_
        
        self.patch_emb = cnn_stem(in_channel=3, out_channel=hidden_dim, stride=stem_stride)
        
        mlp = [MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim, activation=nn.GELU) for _ in range(int(num_mixerblocks))]
        mlp += [WORESHanMixerBlock(num_tokens, channels_mlp_dim,Layer=Layer) for _ in range(int(num_hanblocks))]

        self.mlp = nn.Sequential(*mlp)
        
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.inital(m.weight) 
            elif isinstance(m, HouseholderLayer):
                self.inital(m.vector) 
            elif isinstance(m, nn.Conv2d):
                self.inital(m.weight) 

    def forward(self, x):
        x = self.patch_emb(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.mlp(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
    

# class HanMixerBlock(nn.Module):
#     def __init__(self, num_tokens, num_channels, Layer=HouseholderLayer):
#         super(HanMixerBlock, self).__init__()
#         self.token_mix = HanMlpBlock(num_tokens, num_tokens, Layer)
#         self.channel_mix = HanMlpBlock(num_channels, num_channels, Layer)
#         self.token_z = nn.Parameter(torch.Tensor(1))
#         self.channel_z = nn.Parameter(torch.Tensor(1))
#         nn.init.constant_(self.token_z, 0.)
#         nn.init.constant_(self.channel_z, 0.)

#     def forward(self, x):
#         out = x.transpose(1, 2)
#         x = self.token_z*x + self.token_mix(out).transpose(1, 2)
#         out = x
#         x = self.channel_z*x + self.channel_mix(out)
#         return x
