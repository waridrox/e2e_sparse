import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout = 0.):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.atten = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout = dropout)
        self.projection = nn.Linear(in_features=dim,out_features=dim*2)
        
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.atten(x) + x
        x = self.ff(x) + x
        x = self.norm(x)
        x = self.gelu(x)
        x = self.projection(x)
        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.,output_dim = None):
        super().__init__()

        self.layers = [nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)]
            
        if output_dim:
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        else:   
            self.layers.append(nn.Linear(hidden_dim, dim))
            
        self.layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*self.layers
        )

    def forward(self, x):
        return self.net(x)




class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CNNBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                pooling_type = "cnn"
                ):
        super().__init__()

        assert pooling_type in ["cnn", "maxpool"], "Pooling type must be either 'cnn' or 'maxpool'."

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=1
                              )

        self.conv2 = nn.Conv1d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=3,
                               padding=1
                              )
        self.conv3 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1
                              )


        self.shortcut = nn.Conv1d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1)


        if pooling_type == "maxpool":
            self.pooling = nn.MaxPool1d(kernel_size=2,stride=2)
        else:
            self.pooling = nn.Conv1d(in_channels = out_channels,
                                    out_channels= out_channels,
                                    kernel_size=2,
                                    stride=2)

        self.btnorm1 = nn.LazyBatchNorm1d()
        self.btnorm2 = nn.LazyBatchNorm1d()
        

        self.relu = nn.ReLU()

    def forward(self, x):

        cls = x[:,[0],:]
        x = x[:,1:,:]
        x = x.transpose(1,2)
        identity = x.clone()
        
        x = self.conv1(x)
        x = self.btnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.btnorm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        
        x = x + self.shortcut(identity)
        x = self.relu(x)
        x = self.pooling(x)
        x = x.transpose(1,2)
        
        x = torch.cat((cls, x), dim=1)
        
        return x

class FunnelBlock(nn.Module):
    def __init__(self,
                 dim,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout = 0.):
        
        super().__init__()

        self.transformer = TransformerBlock(dim = dim,heads = heads,dim_head=dim_head,mlp_dim=mlp_dim)
        self.cnn = CNNBlock(in_channels=dim*2,out_channels=dim*2,pooling_type="cnn")

    def forward(self, x):
        x = self.transformer(x)
        x = self.cnn(x)

        return x

class PointCloudTransformer(nn.Module):
    def __init__(self,in_features,dim,nPointCloud,model_stat,emb_dropout=0):
        super().__init__()

        self.in_feature_dim = in_features
        self.dim = dim
        self.num_patches = nPointCloud

        self.to_embedding = nn.Sequential(
        nn.LayerNorm(self.in_feature_dim),
        nn.Linear(self.in_feature_dim, self.dim),
        nn.LayerNorm(self.dim))

        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        blocks = []

        #dim = 32,heads = 12,dim_head=64,mlp_dim=3072
        for k in model_stat.keys():
            blocks.append(FunnelBlock(
                            dim = model_stat[k]["dim"],
                            heads = model_stat[k]["heads"],
                            dim_head = model_stat[k]["dim_head"],
                            mlp_dim = model_stat[k]["mlp_dim"]
                            )
            )

        self.model = nn.Sequential(
            *blocks
        )

        self.head = nn.LazyLinear(1)

    def forward(self, x):
        x = self.to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x = self.model(x)
        x = x[:,0,:]
        x = self.head(x)
        return x.squeeze()


class Transformer_PC_1024_S():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 16, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=1024,
                                model_stat=self.model_stat
                                )


class Transformer_PC_1024_M():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block6": {"dim": 1024, "heads": 16, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=1024,
                                model_stat=self.model_stat
                                )


class Transformer_PC_1024_L():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block6": {"dim": 1024, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block7": {"dim": 2048, "heads": 16, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=1024,
                                model_stat=self.model_stat
                                )

class Transformer_PC_1024_H():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block6": {"dim": 1024, "heads": 24, "dim_head": 64, "mlp_dim":3072},
            "block7": {"dim": 2048, "heads": 24, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=1024,
                                model_stat=self.model_stat
                                )

class Transformer_PC_768_S():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 16, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=768,
                                model_stat=self.model_stat
                                )


class Transformer_PC_768_M():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block6": {"dim": 1024, "heads": 16, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=768,
                                model_stat=self.model_stat
                                )


class Transformer_PC_768_L():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block6": {"dim": 1024, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block7": {"dim": 2048, "heads": 16, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=768,
                                model_stat=self.model_stat
                                )

class Transformer_PC_768_H():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block6": {"dim": 1024, "heads": 24, "dim_head": 64, "mlp_dim":3072},
            "block7": {"dim": 2048, "heads": 24, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=768,
                                model_stat=self.model_stat
                                )


class Transformer_PC_512_S():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 16, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=512,
                                model_stat=self.model_stat
                                )


class Transformer_PC_512_M():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block6": {"dim": 1024, "heads": 16, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=512,
                                model_stat=self.model_stat
                                )


class Transformer_PC_512_L():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 24, "dim_head": 64, "mlp_dim":3072},
            "block6": {"dim": 1024, "heads": 24, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=512,
                                model_stat=self.model_stat
                                )


class Transformer_PC_256_S():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 16, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=256,
                                model_stat=self.model_stat
                                )


class Transformer_PC_256_M():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block6": {"dim": 1024, "heads": 16, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=256,
                                model_stat=self.model_stat
                                )


class Transformer_PC_256_L():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "block1": {"dim": 32, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block2": {"dim": 64, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block3": {"dim": 128, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block4": {"dim": 256, "heads": 16, "dim_head": 64, "mlp_dim":3072},
            "block5": {"dim": 512, "heads": 24, "dim_head": 64, "mlp_dim":3072},
            "block6": {"dim": 1024, "heads": 24, "dim_head": 64, "mlp_dim":3072},
        }

        self.model = PointCloudTransformer(in_features=5,
                                 dim = 32,
                                 nPointCloud=256,
                                model_stat=self.model_stat
                                )