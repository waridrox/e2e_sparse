import torch
import torch.nn as nn


class Block(nn.Module):
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
        return x
        
class PointCloudResNet(nn.Module):
    def __init__(self,
                model_stat,
                input_channels=5):
        super().__init__()
        blocks = nn.ModuleList()

        self.base_layer = nn.Conv1d(kernel_size=model_stat["base_layer"]["kernel_size"],
                                    stride=model_stat["base_layer"]["stride"],
                                    padding=model_stat["base_layer"]["padding"],
                                    in_channels=input_channels,
                                    out_channels=model_stat["base_layer"]["out_channels"])

        model_stat.pop("base_layer")
        for k in model_stat.keys():
            blocks.append(Block(
                            in_channels = model_stat[k]["in_channels"],
                            out_channels = model_stat[k]["out_channels"],
                            pooling_type = model_stat[k]["pooling_type"]
                            )
            )

        self.model = nn.Sequential(
            self.base_layer,
            *blocks
        )

        self.final_pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.LazyLinear(1)


    def forward(self, x):
        x = torch.transpose(x,1,2)
        x = self.model(x)
        x = self.final_pool(x)
        x = x.squeeze()
        x = self.head(x)
        return x.squeeze()
    


class ResNet_PC_1024_S():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "base_layer": {"kernel_size": 6, "stride": 2, "padding": 2, "in_channels": 5, "out_channels": 32},
            "block1": {"in_channels": 32, "out_channels": 64, "pooling_type": "maxpool"},
            "block2": {"in_channels": 64, "out_channels": 64, "pooling_type": "maxpool"},
            "block3": {"in_channels": 64, "out_channels": 64, "pooling_type": "maxpool"},
            "block4": {"in_channels": 64, "out_channels": 128, "pooling_type": "maxpool"},
            "block5": {"in_channels": 128, "out_channels": 512, "pooling_type": "maxpool"},
            "block6": {"in_channels": 512, "out_channels": 1024, "pooling_type": "maxpool"}
        }

        self.model = PointCloudResNet(self.model_stat, input_channels=input_channels)


class ResNet_PC_1024_M():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "base_layer": {"kernel_size": 1, "stride": 1, "padding": 0, "in_channels": 5, "out_channels": 8},
            "block1": {"in_channels": 8, "out_channels": 16, "pooling_type": "maxpool"},
            "block2": {"in_channels": 16, "out_channels": 32, "pooling_type": "maxpool"},
            "block3": {"in_channels": 32, "out_channels": 64, "pooling_type": "maxpool"},
            "block4": {"in_channels": 64, "out_channels": 128, "pooling_type": "maxpool"},
            "block5": {"in_channels": 128, "out_channels": 128, "pooling_type": "maxpool"},
            "block6": {"in_channels": 128, "out_channels": 256, "pooling_type": "maxpool"},
            "block7": {"in_channels": 256, "out_channels": 1024, "pooling_type": "maxpool"}
        }

        self.model = PointCloudResNet(self.model_stat, input_channels=input_channels)


class ResNet_PC_1024_L():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "base_layer": {"kernel_size": 1, "stride": 1, "padding": 0, "in_channels": 5, "out_channels": 8},
            "block1": {"in_channels": 8, "out_channels": 32, "pooling_type": "maxpool"},
            "block2": {"in_channels": 32, "out_channels": 128, "pooling_type": "maxpool"},
            "block3": {"in_channels": 128, "out_channels": 256, "pooling_type": "maxpool"},
            "block4": {"in_channels": 256, "out_channels": 512, "pooling_type": "maxpool"},
            "block5": {"in_channels": 512, "out_channels": 1024, "pooling_type": "maxpool"},
            "block6": {"in_channels": 1024, "out_channels": 2048, "pooling_type": "maxpool"},
            "block7": {"in_channels": 2048, "out_channels": 2048, "pooling_type": "maxpool"}
        }

        self.model = PointCloudResNet(self.model_stat, input_channels=input_channels)

class ResNet_PC_768_M():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "base_layer": {"kernel_size": 5, "stride": 1, "padding": 2, "in_channels": 5, "out_channels": 16},
            "block1": {"in_channels": 16, "out_channels": 32, "pooling_type": "maxpool"},
            "block2": {"in_channels": 32, "out_channels": 64, "pooling_type": "maxpool"},
            "block3": {"in_channels": 64, "out_channels": 128, "pooling_type": "maxpool"},
            "block4": {"in_channels": 128, "out_channels": 256, "pooling_type": "maxpool"},
            "block5": {"in_channels": 256, "out_channels": 512, "pooling_type": "maxpool"}, 
            "block6": {"in_channels": 512, "out_channels": 1024, "pooling_type": "maxpool"},
            "block7": {"in_channels": 1024, "out_channels": 2048, "pooling_type": "maxpool"}
        }
        
        self.model = PointCloudResNet(self.model_stat, input_channels=input_channels)

class ResNet_PC_768_S():
    def __init__(self,input_channels = 5):
        self.model_stat = {
            "base_layer": {"kernel_size": 5, "stride": 2, "padding": 2, "in_channels": 5, "out_channels": 16},
            "block1": {"in_channels": 16, "out_channels": 32, "pooling_type": "maxpool"},
            "block2": {"in_channels": 32, "out_channels": 64, "pooling_type": "maxpool"},
            "block3": {"in_channels": 64, "out_channels": 128, "pooling_type": "maxpool"},
            "block4": {"in_channels": 128, "out_channels": 256, "pooling_type": "maxpool"},
            "block5": {"in_channels": 256, "out_channels": 512, "pooling_type": "maxpool"},
            "block6": {"in_channels": 512, "out_channels": 1024, "pooling_type": "maxpool"}
        }
        
        self.model = PointCloudResNet(self.model_stat, input_channels=input_channels)



