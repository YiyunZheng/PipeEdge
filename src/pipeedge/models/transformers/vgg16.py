import logging
import torch
from torch import nn
from torch.nn import Conv2d, ReLU, AdaptiveAvgPool2d, MaxPool2d
import numpy as np
from torchvision import models
from .. import ModuleShard, ModuleShardConfig

logger = logging.getLogger(__name__)

class vggConfig:
    def __init__(self, model=None):
        self.info = {}
        if model:
            self.generate_config(model)

    def get_layer_info(self, layer):
        info = {}

        if isinstance(layer, nn.Conv2d):
            info['in_channels'] = layer.in_channels
            info['out_channels'] = layer.out_channels
            info['kernel_size'] = layer.kernel_size
            info['stride'] = layer.stride
            info['padding'] = layer.padding
            info['bias'] = layer.bias is not None

        elif isinstance(layer, nn.ReLU):
            info['inplace'] = layer.inplace

        elif isinstance(layer, nn.MaxPool2d):
            info['kernel_size'] = layer.kernel_size
            info['stride'] = layer.stride
            info['padding'] = layer.padding
            info['dilation'] = layer.dilation
            info['ceil_mode'] = layer.ceil_mode

        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            info['output_size'] = layer.output_size

        elif isinstance(layer, nn.Dropout):
            info['p'] = layer.p
            info['inplace'] = layer.inplace

        elif isinstance(layer, nn.Linear):
            info['in_features'] = layer.in_features
            info['out_features'] = layer.out_features
            info['bias'] = layer.bias is not None

        return info

    def generate_config(self, model):
        for name, child in model.named_children():
            if list(child.children()): 
                for sub_name, sub_child in child.named_children():
                    self.info[f"{name}_{sub_name}"] = self.get_layer_info(sub_child)
            else:
                self.info[name] = self.get_layer_info(child)

    def __getitem__(self, key):
        return self.info[key]
    
    def __getattr__(self, name):
        if name in self.info:
            return self.info[name]

    
class VGGLayerShard(ModuleShard):
    def __init__(self, config, shard_config: ModuleShardConfig):
        super().__init__(config, shard_config)
        self.conv1 = None
        self.conv2 = None

        self.conv3 = None
        self.conv4 = None

        self.conv5 = None
        self.conv6 = None
        self.conv7 = None

        self.conv8 = None
        self.conv9 = None
        self.conv10 = None

        self.conv11 = None
        self.conv12 = None
        self.conv13 = None
        
        self.relu1 = None
        self.relu2 = None
        self.relu3 = None
        self.relu4 = None
        self.relu5 = None
        self.relu6 = None
        self.relu7 = None
        self.relu8 = None
        self.relu9 = None
        self.relu10 = None
        self.relu11 = None
        self.relu12 = None
        self.relu13 = None

        self.maxpool2 = None
        self.maxpool4 = None
        self.maxpool7 = None
        self.maxpool10 = None
        self.maxpool13 = None

        self._build_shard()

    def _build_shard(self):              
        if self.has_layer(0):
            self.conv1 = Conv2d(**self.config["features_0"])
            self.relu1 = ReLU(**self.config["features_1"])
        if self.has_layer(1):
            self.conv2 = Conv2d(**self.config["features_2"])
            self.relu2 = ReLU(**self.config["features_1"])
            self.maxpool2 = MaxPool2d(**self.config["features_4"])
        if self.has_layer(2):
            self.conv3 = Conv2d(**self.config["features_5"])
            self.relu3 = ReLU(**self.config["features_1"])
        if self.has_layer(3):
            self.conv4 = Conv2d(**self.config["features_7"])
            self.relu4 = ReLU(**self.config["features_1"])
            self.maxpool4 = MaxPool2d(**self.config["features_9"])
        if self.has_layer(4):
            self.conv5 = Conv2d(**self.config["features_10"])
            self.relu5 = ReLU(**self.config["features_1"])
        if self.has_layer(5):
            self.conv6 = Conv2d(**self.config["features_12"])
            self.relu6 = ReLU(**self.config["features_1"])
        if self.has_layer(6):
            self.conv7 = Conv2d(**self.config["features_14"])
            self.relu7 = ReLU(**self.config["features_1"])
            self.maxpool7 = MaxPool2d(**self.config["features_16"])
        if self.has_layer(7):
            self.conv8 = Conv2d(**self.config["features_17"])
            self.relu8 = ReLU(**self.config["features_1"])
        if self.has_layer(8):
            self.conv9 = Conv2d(**self.config["features_19"])
            self.relu9 = ReLU(**self.config["features_1"])
        if self.has_layer(9):
            self.conv10 = Conv2d(**self.config["features_21"])
            self.relu10 = ReLU(**self.config["features_1"])
            self.maxpool10 = MaxPool2d(**self.config["features_23"])
        if self.has_layer(10):
            self.conv11 = Conv2d(**self.config["features_24"])
            self.relu11 = ReLU(**self.config["features_1"])
        if self.has_layer(11):
            self.conv12 = Conv2d(**self.config["features_26"])
            self.relu12 = ReLU(**self.config["features_1"])
        if self.has_layer(12):
            self.conv13 = Conv2d(**self.config["features_28"])
            self.relu13 = ReLU(**self.config["features_1"])
            self.maxpool13 = MaxPool2d(**self.config["features_30"])
        # if self.shard_config.is_last:
        #     self.relu = ReLU(inplace=True)

    @torch.no_grad()
    def forward(self, data):
        """Compute layer shard."""
        if self.has_layer(0):
            data_conv = self.conv1(data)
            data = self.relu1(data_conv)
        if self.has_layer(1):
            data_conv = self.conv2(data)
            data_relu = self.relu2(data_conv)
            data = self.maxpool2(data_relu)
        if self.has_layer(2):
            data_conv = self.conv3(data)
            data = self.relu3(data_conv)
        if self.has_layer(3):
            data_conv = self.conv4(data)
            data_relu = self.relu4(data_conv)
            data = self.maxpool4(data_relu)
        if self.has_layer(4):
            data_conv = self.conv5(data)
            data = self.relu5(data_conv)
        if self.has_layer(5):
            data_conv = self.conv6(data)
            data = self.relu6(data_conv)
        if self.has_layer(6):
            data_conv = self.conv7(data)
            data_relu = self.relu7(data_conv)
            data = self.maxpool7(data_relu)
        if self.has_layer(7):
            data_conv = self.conv8(data)
            data = self.relu8(data_conv)
        if self.has_layer(8):
            data_conv = self.conv9(data)
            data = self.relu9(data_conv)
        if self.has_layer(9):
            data_conv = self.conv10(data)
            data_relu = self.relu10(data_conv)
            data = self.maxpool10(data_relu)
        if self.has_layer(10):
            data_conv = self.conv11(data)
            data = self.relu11(data_conv)
        if self.has_layer(11):
            data_conv = self.conv12(data)
            data = self.relu12(data_conv)
        if self.has_layer(12):
            data_conv = self.conv13(data)
            data_relu = self.relu13(data_conv)
            data = self.maxpool13(data_relu)
        return data
    
    def load_weight(self, weight):
        if self.has_layer(0):
            self.conv1.load_state_dict(weight[0].state_dict())
            self.relu1.load_state_dict(weight[1].state_dict())
        if self.has_layer(1):
            self.conv2.load_state_dict(weight[2].state_dict())
            self.relu2.load_state_dict(weight[3].state_dict())
            self.maxpool2.load_state_dict(weight[4].state_dict())
        if self.has_layer(2):
            self.conv3.load_state_dict(weight[5].state_dict())
            self.relu3.load_state_dict(weight[6].state_dict())
        if self.has_layer(3):
            self.conv4.load_state_dict(weight[7].state_dict())
            self.relu4.load_state_dict(weight[8].state_dict())
            self.maxpool4.load_state_dict(weight[9].state_dict())
        if self.has_layer(4):
            self.conv5.load_state_dict(weight[10].state_dict())
            self.relu5.load_state_dict(weight[11].state_dict())
        if self.has_layer(5):
            self.conv6.load_state_dict(weight[12].state_dict())
            self.relu6.load_state_dict(weight[13].state_dict())
        if self.has_layer(6):
            self.conv7.load_state_dict(weight[14].state_dict())
            self.relu7.load_state_dict(weight[15].state_dict())
            self.maxpool7.load_state_dict(weight[16].state_dict())
        if self.has_layer(7):
            self.conv8.load_state_dict(weight[17].state_dict())
            self.relu8.load_state_dict(weight[18].state_dict())
        if self.has_layer(8):
            self.conv9.load_state_dict(weight[19].state_dict())
            self.relu9.load_state_dict(weight[20].state_dict())
        if self.has_layer(9):
            self.conv10.load_state_dict(weight[21].state_dict())
            self.relu10.load_state_dict(weight[22].state_dict())
            self.maxpool10.load_state_dict(weight[23].state_dict())
        if self.has_layer(10):
            self.conv11.load_state_dict(weight[24].state_dict())
            self.relu11.load_state_dict(weight[25].state_dict())
        if self.has_layer(11):
            self.conv12.load_state_dict(weight[26].state_dict())
            self.relu12.load_state_dict(weight[27].state_dict())
        if self.has_layer(12):
            self.conv13.load_state_dict(weight[28].state_dict())
            self.relu13.load_state_dict(weight[29].state_dict())
            self.maxpool13.load_state_dict(weight[30].state_dict())

class VGGModelShard(ModuleShard):
    def __init__(self, config, shard_config: ModuleShardConfig,
                 model_weights):
        super().__init__(config, shard_config)
        self.layers = nn.ModuleList()
        self.avgpool = None
        self.lin1 = None
        self.relu1 = None
        self.dp1 = None
        self.lin2 = None
        self.relu2 = None
        self.dp2 = None
        self.lin3 = None

        self._build_shard(model_weights)

    def _build_shard(self, weights):

        layer_curr = self.shard_config.layer_start
        # layer_first = self.shard_config.is_first
        # layer_last = self.shard_config.is_last

        while layer_curr <= self.shard_config.layer_end:
            # print(layer_curr)

            layer_config = ModuleShardConfig(layer_start=layer_curr, layer_end=layer_curr,
                                                 is_first = True, is_last = False)
            sub_model_config = self.config
            layer = VGGLayerShard(sub_model_config, layer_config)
            self._load_weights_layer(weights, layer)
            self.layers.append(layer)

            layer_curr += 1

        if self.shard_config.is_last:
            logger.debug(">>>> Load layernorm for the last shard")
            self.avgpool = nn.AdaptiveAvgPool2d(**self.config["avgpool"])
            self.lin1 = nn.Linear(**self.config["classifier_0"])
            self.relu1 = nn.ReLU(**self.config["classifier_1"])
            self.dp1 = nn.Dropout(**self.config["classifier_2"])
            self.lin2 = nn.Linear(**self.config["classifier_3"])
            self.relu2 = nn.ReLU(**self.config["classifier_4"])
            self.dp2 = nn.Dropout(**self.config["classifier_5"])
            self.lin3 = nn.Linear(**self.config["classifier_6"])
            self._load_weights_last(weights)

    # @torch.no_grad()
    # def _load_weights_first(self, weights):
    #     self.conv1.load_state_dict(weights.conv1.state_dict())
    #     self.bn1.load_state_dict(weights.bn1.state_dict())
    #     self.relu.load_state_dict(weights.relu.state_dict())
    #     self.maxpool.load_state_dict(weights.maxpool.state_dict())

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.avgpool.load_state_dict(weights.avgpool.state_dict())
        self.lin1.load_state_dict(weights.classifier[0].state_dict())
        self.relu1.load_state_dict(weights.classifier[1].state_dict())
        self.dp1.load_state_dict(weights.classifier[2].state_dict())
        self.lin2.load_state_dict(weights.classifier[3].state_dict())
        self.relu2.load_state_dict(weights.classifier[4].state_dict())
        self.dp2.load_state_dict(weights.classifier[5].state_dict())
        self.lin3.load_state_dict(weights.classifier[6].state_dict())

    @torch.no_grad()
    def _load_weights_layer(self, weights, layer):
        if layer.has_layer(0):
            layer.conv1.load_state_dict(weights.features[0].state_dict())
            layer.relu.load_state_dict(weights.features[1].state_dict())
        if layer.has_layer(1):
            layer.conv2.load_state_dict(weights.features[2].state_dict())
            layer.relu.load_state_dict(weights.features[3].state_dict())
            layer.maxpool2.load_state_dict(weights.features[4].state_dict())
        if layer.has_layer(2):
            layer.conv3.load_state_dict(weights.features[5].state_dict())
            layer.relu.load_state_dict(weights.features[6].state_dict())
        if layer.has_layer(3):
            layer.conv4.load_state_dict(weights.features[7].state_dict())
            layer.relu.load_state_dict(weights.features[8].state_dict())
            layer.maxpool4.load_state_dict(weights.features[9].state_dict())
        if layer.has_layer(4):
            layer.conv5.load_state_dict(weights.features[10].state_dict())
            layer.relu.load_state_dict(weights.features[11].state_dict())
        if layer.has_layer(5):
            layer.conv6.load_state_dict(weights.features[12].state_dict())
            layer.relu.load_state_dict(weights.features[13].state_dict())
        if layer.has_layer(6):
            layer.conv7.load_state_dict(weights.features[14].state_dict())
            layer.relu.load_state_dict(weights.features[15].state_dict())
            layer.maxpool7.load_state_dict(weights.features[16].state_dict())
        if layer.has_layer(7):
            layer.conv8.load_state_dict(weights.features[17].state_dict())
            layer.relu.load_state_dict(weights.features[18].state_dict())        
        if layer.has_layer(8):
            layer.conv9.load_state_dict(weights.features[19].state_dict())
            layer.relu.load_state_dict(weights.features[20].state_dict())
        if layer.has_layer(9):
            layer.conv10.load_state_dict(weights.features[21].state_dict())
            layer.relu.load_state_dict(weights.features[22].state_dict())
            layer.maxpool10.load_state_dict(weights.features[23].state_dict())
        if layer.has_layer(10):
            layer.conv11.load_state_dict(weights.features[24].state_dict())
            layer.relu.load_state_dict(weights.features[25].state_dict())        
        if layer.has_layer(11):
            layer.conv12.load_state_dict(weights.features[26].state_dict())
            layer.relu.load_state_dict(weights.features[17].state_dict())
        if layer.has_layer(12):
            layer.conv13.load_state_dict(weights.features[28].state_dict())
            layer.relu.load_state_dict(weights.features[29].state_dict())
            layer.maxpool13.load_state_dict(weights.features[30].state_dict())

    @torch.no_grad()
    def forward(self, data):
        """Compute shard layers."""
        # if self.shard_config.is_first:
        #     data = self.conv1(data[1])
        #     data = self.bn1(data)
        #     data = self.relu(data)
        #     data = self.maxpool(data)
        #     data = [data, data]
        for layer in self.layers:
            data = layer(data)
        if self.shard_config.is_last:
            data = self.avgpool(data)
            data = torch.flatten(data, 1)
            data = self.lin1(data)
            data = self.relu1(data)
            data = self.dp1(data)
            data = self.lin2(data)
            data = self.relu2(data)
            data = self.dp2(data)
            data = self.lin3(data)

        return data