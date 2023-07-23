from .resnet import model_urls, ResNet, BasicBlock, Bottleneck
from torch import nn
from torch.nn import BatchNorm2d
from .senet import SENet
import torch.utils.model_zoo as model_zoo


class SEResNet(ResNet):

    se_reduction = 16

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
                SENet(channels=planes * block.expansion, reduction=self.se_reduction)
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        return nn.Sequential(*layers)


def se_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        map_location = model.conv1.weight.device
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet18'], map_location=map_location), strict=False)
    return model


def deformable_se_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEResNet(BasicBlock, [2, 2, 2, 2],
                    dcn=dict(deformable_groups=1), **kwargs)
    if pretrained:
        map_location = model.conv1.weight.device
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet18'], map_location=map_location), strict=False)
    return model


def se_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        map_location = model.conv1.weight.device
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet50'], map_location=map_location), strict=False)
    return model


def deformable_se_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model with deformable conv.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEResNet(Bottleneck, [3, 4, 6, 3],
                   dcn=dict(deformable_groups=1),
                   **kwargs)
    if pretrained:
        map_location = model.conv1.weight.device
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet50'], map_location=map_location), strict=False)
    return model