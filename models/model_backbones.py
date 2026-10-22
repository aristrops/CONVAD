import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights


class End2EndModel(nn.Module):  # joint bottleneck
    def __init__(self, model_1, model_2, use_relu=False, use_sigmoid=False):
        super(End2EndModel, self).__init__()
        self.first_model = model_1
        self.second_model = model_2
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid
        self.train_df = None  # Placeholder for train dataframe if needed

    def second_forward_stage(self, output_stage_1):
        if self.use_relu:
            attr_outputs = [nn.ReLU()(o) for o in output_stage_1]
        elif self.use_sigmoid:
            attr_outputs = [nn.Sigmoid()(o) for o in output_stage_1]
        else:
            attr_outputs = output_stage_1

        input_stage_2 = attr_outputs
        input_stage_2 = torch.cat(input_stage_2, dim=1)
        all_outputs = [self.second_model(input_stage_2)]  # main task prediction
        all_outputs.extend(output_stage_1)  # concept predictions
        return all_outputs

    def forward(self, x):
        outputs = self.first_model(x)
        return self.second_forward_stage(outputs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["train_df"] = getattr(self, "train_df", None)
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        setattr(self, "train_df", state_dict.pop("train_df", None))
        return super().load_state_dict(state_dict, strict=False)


class MLP(nn.Module):
    def __init__(self, input_dim, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim

        if self.expand_dim:
            self.linear = nn.Linear(input_dim, expand_dim)
            self.activation = nn.ReLU()
            self.linear_2 = nn.Linear(expand_dim, 1)
        else:
            self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.linear(x)
        if self.expand_dim:
            x = self.activation(x)
            x = self.linear_2(x)
        return x


class BackboneModel(nn.Module):
    def __init__(
        self,
        num_attr: int,
        num_classes: int,
        freeze_parameters: bool,
        pretrained: bool = True,
        expand_dim: int = 0,
        bottleneck: bool = True,
        backbone: str = "resnet18",
    ):

        super(BackboneModel, self).__init__()

        self.num_attr = num_attr
        self.num_classes = num_classes
        self.bottleneck = bottleneck
        self.train_df = None  # Placeholder for train dataframe if needed

        # load pretrained model
        if backbone == "resnet18":
            base_model = models.resnet18(pretrained=pretrained)
            feature_dim = base_model.fc.in_features
            self.feature_extractor = nn.Sequential(
                *list(base_model.children())[:-1]
            )  # remove last FC layer
        elif backbone == "mobilenet_v2":
            base_model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT
            )
            feature_dim = base_model.last_channel
            self.feature_extractor = base_model.features
            # self.pool = nn.AdaptiveAvgPool2d(1) #add pooling layer since mobilenet ends with conv
            self.pool = nn.AvgPool2d(kernel_size=7)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # freeze layers by default
        for name, param in base_model.named_parameters():
            param.requires_grad = False
        if not freeze_parameters:
            # fine-tune last layer
            for name, param in base_model.named_parameters():
                if "layer4" in name or any(  # resnet18
                    f"features.{i}" in name for i in range(14, 19)
                ):  # mobilenet_v2
                    param.requires_grad = True

        self.fc_layers = (
            nn.ModuleList()
        )  # list of fc layers for each prediction; main task is always the first fc layer

        if self.num_attr is not None:
            if not self.bottleneck:
                self.fc_layers.append(FC(feature_dim, 1, expand_dim))
            for _ in range(self.num_attr):
                self.fc_layers.append(FC(feature_dim, 1, expand_dim))
        else:
            self.fc_layers.append(FC(feature_dim, num_classes, expand_dim))

    def forward(self, x):
        x = self.feature_extractor(x)
        if hasattr(self, "pool"):
            x = self.pool(x)
        x = torch.flatten(x, 1)

        predictions = []

        for fc in self.fc_layers:
            predictions.append(fc(x))

        return predictions

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["train_df"] = getattr(self, "train_df", None)
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        setattr(self, "train_df", state_dict.pop("train_df", None))
        return super().load_state_dict(state_dict, strict=False)


# fc layers for prediction
class FC(nn.Module):
    def __init__(self, input_dim, output_dim, expand_dim):
        super(FC, self).__init__()

        self.expand_dim = expand_dim

        if self.expand_dim > 0:  # create MLP rather than standard FC layer
            self.relu = nn.ReLU()
            self.fc_new = nn.Linear(input_dim, expand_dim)
            self.fc = nn.Linear(expand_dim, output_dim)

        else:
            self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.expand_dim > 0:
            x = self.fc_new(x)
            x = self.relu(x)
        x = self.fc(x)
        return x


# feature extractor model for STFPM
class BackboneModelFeatures(nn.Module):
    def __init__(self, pretrained: bool = True, backbone: str = "mobilenet_v2"):
        super(BackboneModelFeatures, self).__init__()

        self.backbone = backbone

        # load pretrained model
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            self.feature_extractor = nn.Sequential(
                *list(base_model.children())[:-2]
            )  # remove last AvgPool and FC layer
        elif backbone == "mobilenet_v2":
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v2(weights=weights)
            self.feature_extractor = base_model.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # freeze layers by default
        if pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        res = []
        if self.backbone == "resnet18":
            for name, module in self.feature_extractor._modules.items():
                x = module(x)
                if name in ["4", "5", "6"]:
                    res.append(x)

        elif self.backbone == "mobilenet_v2":
            for idx, module in enumerate(self.feature_extractor):
                x = module(x)
                if idx in [3, 8, 14]:
                    res.append(x)

        return res
