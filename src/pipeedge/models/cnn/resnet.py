import logging
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d
import numpy as np
from torchvision import models
from .. import ModuleShard, ModuleShardConfig

import pdb

logger = logging.getLogger(__name__)

class ResnetConfig:
    def __init__(self, model=None):
        self.name_or_path = ''
        self.info = {}
        if model:
            self.name_or_path = model.__class__.__name__
            self.generate_config(model)

    def get_layer_info(self, layer):
        info = {}
        
        if isinstance(layer, models.resnet.BasicBlock):
            for sub_name, sub_child in layer.named_children():
                if sub_name == "downsample":
                    info["downsample_conv"] = self.get_layer_info(sub_child[0])
                    info["downsample_bn"] = self.get_layer_info(sub_child[1])
                else:
                    info[sub_name] = self.get_layer_info(sub_child)

        elif isinstance(layer, nn.Conv2d):
            info['in_channels'] = layer.in_channels
            info['out_channels'] = layer.out_channels
            info['kernel_size'] = layer.kernel_size
            info['stride'] = layer.stride
            info['padding'] = layer.padding
            info['bias'] = layer.bias is not None

        elif isinstance(layer, nn.BatchNorm2d):
            info['num_features'] = layer.num_features
            info['eps'] = layer.eps
            info['momentum'] = layer.momentum
            info['affine'] = layer.affine
            info['track_running_stats'] = layer.track_running_stats

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



class ResNetLayerShard(ModuleShard):
    def __init__(self, config, shard_config: ModuleShardConfig):
        super().__init__(config, shard_config)
        self.conv1 = None
        self.bn1 = None
        self.relu = None
        self.conv2 = None
        self.bn2 = None
        self.downsample_conv = None
        self.downsample_bn = None

        self._build_shard()

    def _build_shard(self):
        if self.has_layer(0):
            self.conv1 = Conv2d(**self.config["conv1"])
            self.bn1 = BatchNorm2d(**self.config["bn1"])
            self.relu = ReLU(**self.config['relu'])
        if self.has_layer(1):
            self.conv2 = Conv2d(**self.config["conv2"])
            self.bn2 = BatchNorm2d(**self.config["bn2"])
        if self.has_layer(2):
            self.downsample_conv = Conv2d(**self.config["downsample_conv"])
            self.downsample_bn = BatchNorm2d(**self.config["downsample_bn"])
        if self.shard_config.is_last:
            self.relu = ReLU(inplace=True)

    @torch.no_grad()
    def forward(self, data_pack):
        """Compute layer shard."""
        # pdb.set_trace()
        data = data_pack[0]
        identity = data_pack[1]
        if self.has_layer(0):
            data_conv = self.conv1(data)
            data_bn = self.bn1(data_conv)
            data = self.relu(data_bn)
        if self.has_layer(1):
            data_conv = self.conv2(data)
            data = self.bn2(data_conv)
        if self.has_layer(2):
            data_conv = self.downsample_conv(identity)
            identity = self.downsample_bn(data_conv)
        if self.shard_config.is_last:
            data += identity
            data = self.relu(data)
            return data, data
        return data, identity
    
    # For unit test only
    def load_weight(self, weight):
        if self.has_layer(0):
            self.conv1.load_state_dict(weight.conv1.state_dict())
            self.bn1.load_state_dict(weight.bn1.state_dict())
            self.relu.load_state_dict(weight.relu.state_dict())
        if self.has_layer(1):
            self.conv2.load_state_dict(weight.conv2.state_dict())
            self.bn2.load_state_dict(weight.bn2.state_dict())
        if self.has_layer(2):
            self.downsample_conv.load_state_dict(weight.downsample[0].state_dict())
            self.downsample_bn.load_state_dict(weight.downsample[1].state_dict())

class ResNetModelShard(ModuleShard):

    def __init__(self, config, shard_config: ModuleShardConfig,
                 model_weights):
        super().__init__(config, shard_config)
        self.conv1 = None
        self.bn1 = None
        self.relu = None
        self.maxpool = None

        self.layers = nn.ModuleList()

        self.avgpool = None
        self.fc = None
        # pdb.set_trace()

        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            with np.load(model_weights) as weights:
                self._build_shard(weights)
        else:
            self._build_shard(model_weights)

    def _build_shard(self, weights):
        if self.shard_config.is_first:
            logger.debug(">>>> Load embeddings layer for the first shard")
            self.conv1 = Conv2d(**self.config['conv1'])
            self.bn1 = BatchNorm2d(**self.config['bn1'])
            self.relu = ReLU(**self.config['relu'])
            self.maxpool = MaxPool2d(**self.config['maxpool'])
            self._load_weights_first(weights)

        layer_curr = self.shard_config.layer_start - 1
        layer_end = self.shard_config.layer_end - 1
        while layer_curr <= (layer_end - 1 if self.shard_config.is_last else layer_end):

            if layer_end != 0:
                layer_id = layer_curr // 5 + 1
                layer_sub_id = layer_curr  %5 // 3
                

                if layer_sub_id == 0:
                    if layer_id ==1:
                        layer_curr = max(1, layer_curr)
                        sublayer_start = (layer_curr - 1) % 2
                        if layer_end > 2:
                            sublayer_end = 1
                        else:
                            sublayer_end = layer_end - 1
                        sub_layer_is_last = True if sublayer_end == 1 else False

                    else:
                        sublayer_start = layer_curr % 5 % 3
                        if layer_id == layer_end // 5 + 1 and layer_sub_id == layer_end %5 // 3:
                            sublayer_end = layer_end % 5 % 3
                        else:
                            sublayer_end = 2
                        sub_layer_is_last = True if sublayer_end == 2 else False
                else:
                    sublayer_start = layer_curr % 5 % 3
                    if layer_id == layer_end // 5 + 1 and layer_sub_id == layer_end %5 // 3:
                        sublayer_end = layer_end % 5 % 3
                    else:
                        sublayer_end = 1
                    sub_layer_is_last = True if sublayer_end == 1 else False

                sub_layer_is_first = True if sublayer_start == 0 else False


                layer_config = ModuleShardConfig(layer_start=sublayer_start, layer_end=sublayer_end
                                                ,is_first = sub_layer_is_first, is_last = sub_layer_is_last)
                sub_model_config = self.config[f'layer{layer_id}_{layer_sub_id}']
                layer = ResNetLayerShard(sub_model_config, layer_config)
                self._load_weights_layer(weights.__getattr__(f'layer{layer_id}')[layer_sub_id], layer)
                self.layers.append(layer)

                layer_curr += sublayer_end - sublayer_start + 1
            else:
                layer_curr = 1


        if self.shard_config.is_last:
            logger.debug(">>>> Load layernorm for the last shard")
            self.avgpool = nn.AdaptiveAvgPool2d(**self.config["avgpool"])
            self.fc = nn.Linear(**self.config["fc"])
            self._load_weights_last(weights)

    @torch.no_grad()
    def _load_weights_first(self, weights):
        self.conv1.load_state_dict(weights.conv1.state_dict())
        self.bn1.load_state_dict(weights.bn1.state_dict())
        self.relu.load_state_dict(weights.relu.state_dict())
        self.maxpool.load_state_dict(weights.maxpool.state_dict())

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.avgpool.load_state_dict(weights.avgpool.state_dict())
        self.fc.load_state_dict(weights.fc.state_dict())

    @torch.no_grad()
    def _load_weights_layer(self, weights, layer):
        if layer.has_layer(0):
            layer.conv1.load_state_dict(weights.conv1.state_dict())
            layer.bn1.load_state_dict(weights.bn1.state_dict())
            layer.relu.load_state_dict(weights.relu.state_dict())
        if layer.has_layer(1):
            layer.conv2.load_state_dict(weights.conv2.state_dict())
            layer.bn2.load_state_dict(weights.bn2.state_dict())
        if layer.has_layer(2):
            layer.downsample_conv.load_state_dict(weights.downsample[0].state_dict())
            layer.downsample_bn.load_state_dict(weights.downsample[1].state_dict())

    @torch.no_grad()
    def forward(self, data):
        """Compute shard layers."""
        # pdb.set_trace()
        if self.shard_config.is_first:
            data = self.conv1(data)
            # pdb.set_trace()
            data = self.bn1(data)
            # pdb.set_trace()
            data = self.relu(data)
            # pdb.set_trace()
            data = self.maxpool(data)
            # pdb.set_trace()
            data = [data, data]
        
        for layer in self.layers:
            data = layer(data)
            # pdb.set_trace()
        if self.shard_config.is_last:
            data = self.avgpool(data[0])
            # pdb.set_trace()
            data = torch.flatten(data, 1)
            data = self.fc(data)
            # pdb.set_trace()
        return data
