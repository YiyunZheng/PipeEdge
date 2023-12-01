import torch
import torchvision.transforms as transforms
from torchvision import models,datasets
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d
import torch.optim as optim
import logging
from tqdm import tqdm
from .. import ModuleShard, ModuleShardConfig

import pdb

logger = logging.getLogger(__name__)

class Vgg19Config:
    def __init__(self, model=None):
        self.name_or_path = ''
        self.info = {}
        if model:
            self.name_or_path = model.__class__.__name__
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

class Vgg19LayerShard(ModuleShard):
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

        self.conv14 = None
        self.conv15 = None
        self.conv16 = None


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
        self.relu14 = None
        self.relu15 = None
        self.relu16 = None

        self.maxpool1 = None
        self.maxpool2 = None
        self.maxpool3 = None
        self.maxpool4 = None
        self.maxpool5 = None


        self._build_shard()

    def _build_shard(self):

        if self.has_layer(0):
            self.conv1 = Conv2d(**self.config["features_0"])
            self.relu1 = ReLU(**self.config["features_1"])
        if self.has_layer(1):
            self.conv2 = Conv2d(**self.config["features_2"])
            self.relu2 = ReLU(**self.config["features_3"])
            self.maxpool1 = MaxPool2d(**self.config["features_4"])
        if self.has_layer(2):
            self.conv3 = Conv2d(**self.config["features_5"])
            self.relu3 = ReLU(**self.config["features_6"])
        if self.has_layer(3):
            self.conv4 = Conv2d(**self.config["features_7"])
            self.relu4 = ReLU(**self.config["features_8"])
            self.maxpool2 = MaxPool2d(**self.config["features_9"])
        if self.has_layer(4):
            self.conv5 = Conv2d(**self.config["features_10"])
            self.relu5 = ReLU(**self.config["features_11"])
        if self.has_layer(5):
            self.conv6 = Conv2d(**self.config["features_12"])
            self.relu6 = ReLU(**self.config["features_13"])
        if self.has_layer(6):
            self.conv7 = Conv2d(**self.config["features_14"])
            self.relu7 = ReLU(**self.config["features_15"])
        if self.has_layer(7):
            self.conv8 = Conv2d(**self.config["features_16"])
            self.relu8 = ReLU(**self.config["features_17"])
            self.maxpool3 = MaxPool2d(**self.config["features_18"])
        if self.has_layer(8):
            self.conv9 = Conv2d(**self.config["features_19"])
            self.relu9 = ReLU(**self.config["features_20"])
        if self.has_layer(9):
            self.conv10 = Conv2d(**self.config["features_21"])
            self.relu10 = ReLU(**self.config["features_22"])
        if self.has_layer(10):
            self.conv11 = Conv2d(**self.config["features_23"])
            self.relu11 = ReLU(**self.config["features_24"])
        if self.has_layer(11):
            self.conv12 = Conv2d(**self.config["features_25"])
            self.relu12 = ReLU(**self.config["features_26"])
            self.maxpool4 = MaxPool2d(**self.config["features_27"])
        if self.has_layer(12):
            self.conv13 = Conv2d(**self.config["features_28"])
            self.relu13 = ReLU(**self.config["features_29"])
        if self.has_layer(13):
            self.conv14 = Conv2d(**self.config["features_30"])
            self.relu14 = ReLU(**self.config["features_31"])
        if self.has_layer(14):
            self.conv15 = Conv2d(**self.config["features_32"])
            self.relu15 = ReLU(**self.config["features_33"])
        if self.has_layer(15):
            self.conv16 = Conv2d(**self.config["features_34"])
            self.relu16 = ReLU(**self.config["features_35"])
            self.maxpool5 = MaxPool2d(**self.config["features_36"])

        # if self.shard_config.is_last:
        #     self.relu = ReLU(inplace=True)

    @torch.no_grad()
    def forward(self, data):
        """Compute layer shard."""
        if self.has_layer(0):
            data = self.conv1(data)
            data = self.relu1(data)
        if self.has_layer(1):
            data = self.conv2(data)
            data = self.relu2(data)
            data = self.maxpool1(data)
        if self.has_layer(2):
            data = self.conv3(data)
            data = self.relu3(data)
        if self.has_layer(3):
            data = self.conv4(data)
            data = self.relu4(data)
            data = self.maxpool2(data)
        if self.has_layer(4):
            data = self.conv5(data)
            data = self.relu5(data)
        if self.has_layer(5):
            data = self.conv6(data)
            data = self.relu6(data)
        if self.has_layer(6):
            data = self.conv7(data)
            data = self.relu7(data)
        if self.has_layer(7):
            data = self.conv8(data)
            data = self.relu8(data)
            data = self.maxpool3(data)
        if self.has_layer(8):
            data = self.conv9(data)
            data = self.relu9(data)
        if self.has_layer(9):
            data = self.conv10(data)
            data = self.relu10(data)
        if self.has_layer(10):
            data = self.conv11(data)
            data = self.relu11(data)
        if self.has_layer(11):
            data = self.conv12(data)
            data = self.relu12(data)
            data = self.maxpool4(data)
        if self.has_layer(12):
            data = self.conv13(data)
            data = self.relu13(data)
        if self.has_layer(13):
            data = self.conv14(data)
            data = self.relu14(data)
        if self.has_layer(14):
            data = self.conv15(data)
            data = self.relu15(data)
        if self.has_layer(15):
            data = self.conv16(data)
            data = self.relu16(data)
            data = self.maxpool5(data)
        return data

class Vgg19ModelShard(ModuleShard):

    def __init__(self, config, shard_config: ModuleShardConfig,
                 model_weights):
        super().__init__(config, shard_config)
        self.layers = nn.ModuleList()

        self.avgpool = None

        self.fc1 = None
        self.relu1 = None
        self.dropout1 = None
        self.fc2 = None
        self.relu2 = None
        self.dropout2 = None
        self.fc3 = None

        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            with np.load(model_weights) as weights:
                self._build_shard(weights)
        else:
            self._build_shard(model_weights)

    def _build_shard(self, weights):

        layer_curr = self.shard_config.layer_start - 1
        while layer_curr <= self.shard_config.layer_end - 1:
            layer_config = ModuleShardConfig(layer_start=layer_curr, layer_end=layer_curr)
            layer = Vgg19LayerShard(self.config, layer_config)
            self._load_weights_layer(weights, layer)
            self.layers.append(layer)
            layer_curr += 1


        if self.shard_config.is_last:
            logger.debug(">>>> Load layernorm for the last shard")
            self.avgpool = nn.AdaptiveAvgPool2d(**self.config["avgpool"])
            self.fc1 = nn.Linear(**self.config["classifier_0"])
            self.relu1 = nn.ReLU(**self.config["classifier_1"])
            self.dropout1 = nn.Dropout(**self.config["classifier_2"])
            self.fc2 = nn.Linear(**self.config["classifier_3"])
            self.relu2 = nn.ReLU(**self.config["classifier_4"])
            self.dropout2 = nn.Dropout(**self.config["classifier_5"])
            self.fc3 = nn.Linear(**self.config["classifier_6"])
            self._load_weights_last(weights)

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.avgpool.load_state_dict(weights.avgpool.state_dict())
        self.fc1.load_state_dict(weights.classifier[0].state_dict())
        self.relu1.load_state_dict(weights.classifier[1].state_dict())
        self.dropout1.load_state_dict(weights.classifier[2].state_dict())
        self.fc2.load_state_dict(weights.classifier[3].state_dict())
        self.relu2.load_state_dict(weights.classifier[4].state_dict())
        self.dropout2.load_state_dict(weights.classifier[5].state_dict())
        self.fc3.load_state_dict(weights.classifier[6].state_dict())

    @torch.no_grad()
    def _load_weights_layer(self, weights, layer):
        if layer.has_layer(0):
            layer.conv1.load_state_dict(weights.features[0].state_dict())
            layer.relu1.load_state_dict(weights.features[1].state_dict())
        if layer.has_layer(1):
            layer.conv2.load_state_dict(weights.features[2].state_dict())
            layer.relu2.load_state_dict(weights.features[3].state_dict())
            layer.maxpool1.load_state_dict(weights.features[4].state_dict())
        if layer.has_layer(2):
            layer.conv3.load_state_dict(weights.features[5].state_dict())
            layer.relu3.load_state_dict(weights.features[6].state_dict())
        if layer.has_layer(3):
            layer.conv4.load_state_dict(weights.features[7].state_dict())
            layer.relu4.load_state_dict(weights.features[8].state_dict())
            layer.maxpool2.load_state_dict(weights.features[9].state_dict())
        if layer.has_layer(4):
            layer.conv5.load_state_dict(weights.features[10].state_dict())
            layer.relu5.load_state_dict(weights.features[11].state_dict())
        if layer.has_layer(5):
            layer.conv6.load_state_dict(weights.features[12].state_dict())
            layer.relu6.load_state_dict(weights.features[13].state_dict())
        if layer.has_layer(6):
            layer.conv7.load_state_dict(weights.features[14].state_dict())
            layer.relu7.load_state_dict(weights.features[15].state_dict())
        if layer.has_layer(7):
            layer.conv8.load_state_dict(weights.features[16].state_dict())
            layer.relu8.load_state_dict(weights.features[17].state_dict())
            layer.maxpool3.load_state_dict(weights.features[18].state_dict())
        if layer.has_layer(8):
            layer.conv9.load_state_dict(weights.features[19].state_dict())
            layer.relu9.load_state_dict(weights.features[20].state_dict())
        if layer.has_layer(9):
            layer.conv10.load_state_dict(weights.features[21].state_dict())
            layer.relu10.load_state_dict(weights.features[22].state_dict())
        if layer.has_layer(10):
            layer.conv11.load_state_dict(weights.features[23].state_dict())
            layer.relu11.load_state_dict(weights.features[24].state_dict())
        if layer.has_layer(11):
            layer.conv12.load_state_dict(weights.features[25].state_dict())
            layer.relu12.load_state_dict(weights.features[26].state_dict())
            layer.maxpool4.load_state_dict(weights.features[27].state_dict())
        if layer.has_layer(12):
            layer.conv13.load_state_dict(weights.features[28].state_dict())
            layer.relu13.load_state_dict(weights.features[29].state_dict())
        if layer.has_layer(13):
            layer.conv14.load_state_dict(weights.features[30].state_dict())
            layer.relu14.load_state_dict(weights.features[31].state_dict())
        if layer.has_layer(14):
            layer.conv15.load_state_dict(weights.features[32].state_dict())
            layer.relu15.load_state_dict(weights.features[33].state_dict())
        if layer.has_layer(15):
            layer.conv16.load_state_dict(weights.features[34].state_dict())
            layer.relu16.load_state_dict(weights.features[35].state_dict())
            layer.maxpool5.load_state_dict(weights.features[36].state_dict())

    @torch.no_grad()
    def forward(self, data):
        """Compute shard layers."""
        for layer in self.layers:
            data = layer(data)
        if self.shard_config.is_last:
            data = self.avgpool(data)
            data = torch.flatten(data, 1)
            data = self.fc1(data)
            data = self.relu1(data)
            data = self.dropout1(data)
            data = self.fc2(data)
            data = self.relu2(data)
            data = self.dropout2(data)
            data = self.fc3(data)
        return data
