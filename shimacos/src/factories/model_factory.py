import math
from typing import Dict
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from omegaconf import DictConfig
from .module import GeM
from transformers import AutoTokenizer


class CatEncoder(nn.Module):
    def __init__(self):
        super(CatEncoder, self).__init__()
        self.encoder_dict = {}
        encoder_infos = [
            ("lang_code", 5, 32),
        ]
        for cat, vocab_size, embedding_dim in encoder_infos:
            self.encoder_dict[cat] = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_dict = nn.ModuleDict(self.encoder_dict)
        # self.out_features = sum([val for val in self.n_class_dict.values()])
        self.out_features = sum(
            [embedding_dim for _, _, embedding_dim in encoder_infos]
        )

    def forward(self, x_dict):
        """
        x_dict: [bs, seq_len, 1]
        return: [bs, seq_len, self.encoder.out_features]
        """
        outs = []
        for cat, val in x_dict.items():
            if cat in self.encoder_dict.keys():
                outs.append(self.encoder_dict[cat](val.long()))
        return torch.cat(outs, dim=-1)


class OneHotEncoder(nn.Module):
    def __init__(self):
        super(OneHotEncoder, self).__init__()
        self.encoder_dict = {
            "categoryA": 182,
            "categoryB": 2,
            "categoryC": 2906,
            "categoryD": 3,
            "categoryE": 29,
            "categoryF": 3,
            "unit": 19,
            "categoryA_categoryE": 574,
            "featureA": 27,
            "featureB": 27,
            "featureC": 23,
            "featureD": 28,
            "featureE": 28,
            "featureF": 23,
            "featureG": 27,
            "featureH": 14,
            "featureI": 27,
            "compositionA": 7,
            "compositionB": 24,
            "compositionC": 27,
            "compositionD": 12,
            "compositionE": 25,
            "compositionF": 23,
            "compositionG": 17,
            "compositionH": 26,
            "compositionI": 27,
            "compositionJ": 25,
        }

    def forward(self, x_dict):
        """
        x_dict: [bs, seq_len, 1]
        return: [bs, seq_len, self.encoder.out_features]
        """
        outs = []
        for cat, val in x_dict.items():
            if cat in self.encoder_dict.keys():
                outs.append(F.one_hot(val, self.encoder_dict[cat]))
        return torch.cat(outs, dim=-1)


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        mlp = [
            nn.LazyLinear(config.mlp.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
        ]
        for _ in range(config.mlp.n_layer - 2):
            mlp.extend(
                [
                    nn.Linear(config.mlp.hidden_size, config.mlp.hidden_size),
                    # nn.BatchNorm1d(config.mlp.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate),
                ]
            )
        self.mlp = nn.Sequential(*mlp)
        if config.encoder == "embedding":
            self.encoder = CatEncoder()
        elif config.encoder == "onehot":
            self.encoder = OneHotEncoder()
        else:
            raise NotImplementedError()
        self.last_linear = nn.Linear(config.mlp.hidden_size, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Input:
        """
        feature = batch["feature"]
        embedding = self.encoder(batch)
        feature = torch.cat([batch["feature"], embedding], dim=-1)
        feature = self.mlp(feature)
        out = self.last_linear(feature).squeeze()

        return out

    def get_feature(self):
        return self.feature


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * F.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = F.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class ResNet(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, dropout=0.0):
        super(ResNet, self).__init__()
        assert kernel_size % 2 == 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_features,
                out_features,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_features,
                out_features,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x) + x
        return out


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, feature: torch.Tensor):

        cosine = F.linear(F.normalize(feature), F.normalize(self.weight)).float()
        return cosine


class CNN2D(nn.Module):
    def __init__(self, config: DictConfig):
        super(CNN2D, self).__init__()
        self.config = config
        self.cnn = timm.create_model(
            config.backbone,
            pretrained=True,
            # No classifier for ArcFace
            num_classes=0,
            in_chans=3,
        )
        self.in_features = self.cnn.num_features
        self.swish = Swish_module()
        self.bn = nn.BatchNorm1d(self.in_features)

        self.gem = GeM(p=3)
        if config.is_linear_head:
            self.linear = nn.Linear(self.in_features, config.embedding_size)
            # self.arc_module = ArcMarginProduct(
            #     config.embedding_size, config.num_classes
            # )
            self.last_linear = nn.Linear(config.embedding_size, config.num_classes)

        else:
            # self.arc_module = ArcMarginProduct(self.in_features, config.num_classes)
            self.last_linear = nn.Linear(self.in_features, config.num_classes)

    def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
        feature = self.cnn.forward_features(imgs)
        # feature = self.gem(feature)
        if "vit" not in self.config.backbone and "swin" not in self.config.backbone:
            feature = F.adaptive_avg_pool2d(feature, (1, 1))
            feature = feature.view(feature.size(0), -1)
        # feature = self.bn(feature)
        return feature

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        imgs = batch["image"].permute(0, 3, 1, 2)
        if self.config.is_linear_head:
            feature = self.forward_features(imgs)
            self.feature = self.swish(self.linear(feature))
        else:
            self.feature = self.forward_features(imgs)
        cosine = self.last_linear(self.feature)
        return cosine

    def get_feature(self):
        return self.feature


class CNNBert(nn.Module):
    def __init__(self, config: DictConfig):
        super(CNNBert, self).__init__()
        self.config = config
        self.cnn = timm.create_model(
            config.backbone,
            pretrained=config.is_pretrained,
            # No classifier for ArcFace
            num_classes=0,
            in_chans=config.image.in_channels,
        )
        if config.is_pretrained:
            self.bert_model = AutoModel.from_pretrained(config.text.backbone)
        else:
            self.bert_model = AutoModel(AutoConfig(config.text.backbone))
        # bertの特徴量分だけプラス
        self.in_features = self.cnn.num_features + self.bert_model.config.hidden_size
        self.gem = GeM(p=4)
        self.swish = Swish_module()

        if config.is_linear_head:
            self.linear = nn.Linear(self.in_features, config.embedding_size)
            self.arc_module = ArcMarginProduct(
                config.embedding_size, config.num_classes
            )

        else:
            self.arc_module = ArcMarginProduct(self.in_features, config.num_classes)

    def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
        feature = self.cnn.forward_features(imgs)
        feature = self.gem(feature)
        feature = feature.view(feature.size(0), -1)
        return feature

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        imgs = batch["image"].permute(0, 3, 1, 2)
        bert_out = self.bert_model(
            batch["text"], batch["mask"], output_hidden_states=True
        )
        # pooler_outputは[CLS] tokenに対してlinearとtanhを通したもの
        # _, pool = bert_out["last_hidden_state"], bert_out["pooler_output"]
        # https://huggingface.co/transformers/main_classes/output.html?highlight=basemodeloutputwithpoolingandcrossattentions#basemodeloutputwithpoolingandcrossattentions
        if not self.config.text.is_avg_pool:
            # pooler_outputは[CLS] tokenに対してlinearとtanhを通したもの
            # _, pool = bert_out["last_hidden_state"], bert_out["pooler_output"]
            hidden_states = torch.stack(bert_out["hidden_states"][-4:])
            text_feature = torch.mean(hidden_states[:, :, 0], dim=0)
        else:
            hidden_states = bert_out["last_hidden_state"]
            text_feature = torch.mean(hidden_states, dim=1)
        img_feature = self.forward_features(imgs)
        if self.config.is_linear_head:
            feature = torch.cat([img_feature, text_feature], dim=-1)
            self.feature = self.swish(self.linear(feature))
        else:
            self.feature = torch.cat([img_feature, text_feature], dim=-1)
        cosine = self.arc_module(self.feature)
        return cosine

    def get_feature(self):
        return self.feature


class Bert(nn.Module):
    def __init__(self, config: DictConfig):
        super(Bert, self).__init__()
        self.config = config
        if config.is_pretrained:
            self.bert_model = AutoModel.from_pretrained(config.text.backbone)
        else:
            self.bert_model = AutoModel(AutoConfig(config.text.backbone))
        # bertの特徴量分だけプラス
        self.in_features = self.bert_model.config.hidden_size
        self.encoder = CatEncoder()
        self.linear = nn.Linear(
            self.bert_model.config.hidden_size + 3 + self.encoder.out_features,
            config.num_classes,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        bert_out = self.bert_model(
            batch["input_ids"], batch["attention_mask"], output_hidden_states=True
        )
        feature = batch["feature"].float()
        cat_feature = self.encoder(batch).squeeze()
        # pooler_outputは[CLS] tokenに対してlinearとtanhを通したもの
        # _, pool = bert_out["last_hidden_state"], bert_out["pooler_output"]
        # https://huggingface.co/transformers/main_classes/output.html?highlight=basemodeloutputwithpoolingandcrossattentions#basemodeloutputwithpoolingandcrossattentions
        if self.config.text.is_avg_pool:
            # pooler_outputは[CLS] tokenに対してlinearとtanhを通したもの
            # _, pool = bert_out["last_hidden_state"], bert_out["pooler_output"]
            hidden_states = torch.stack(bert_out["hidden_states"][-4:])
            self.feature = torch.mean(hidden_states[:, :, 0], dim=0)
        else:
            hidden_states = bert_out["last_hidden_state"]
            self.feature = torch.mean(hidden_states, dim=1)
        self.feature = torch.cat([self.feature, feature, cat_feature], dim=1)
        out = self.linear(self.feature)
        return out

    def get_feature(self):
        return self.feature


class QueryMLP(nn.Module):
    def __init__(self, config: DictConfig):
        super(QueryMLP, self).__init__()
        self.swish = Swish_module()
        self.mlp = nn.Sequential(
            nn.Linear(3840 * 2 + 1, config.embedding_size),
            nn.BatchNorm1d(config.embedding_size),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout_rate),
            # nn.Linear(config.embedding_size, config.embedding_size),
            # nn.BatchNorm1d(config.embedding_size),
            # nn.LeakyReLU(),
            # nn.Dropout(config.dropout_rate),
        )
        self.last_linear = nn.Linear(config.embedding_size, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        feature = torch.cat(
            [batch["query_feature"], batch["neighbor_feature"], batch["distance"]],
            dim=1,
        )
        self.feature = self.mlp(feature.float())
        out = self.last_linear(self.feature)
        return out

    def get_feature(self):
        return self.feature


class QueryCNN(nn.Module):
    def __init__(self, config: DictConfig):
        super(QueryCNN, self).__init__()
        self.swish = Swish_module()
        self.resnet1 = ResNet(2, 128, kernel_size=3, dropout=config.dropout_rate)
        self.resnet2 = ResNet(128, 256, kernel_size=5, dropout=config.dropout_rate)
        # self.resnet3 = ResNet(64, 128, kernel_size=7, dropout=config.dropout_rate)
        # self.resnet4 = ResNet(128, 256, kernel_size=9, dropout=config.dropout_rate)
        # self.resnet5 = ResNet(256, 512, kernel_size=11, dropout=config.dropout_rate)
        self.last_linear = nn.Linear(128 + 256, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        feature1 = torch.cat(
            [batch["query_feature"], batch["distance"]],
            dim=1,
        ).unsqueeze(1)
        feature2 = torch.cat(
            [batch["neighbor_feature"], batch["distance"]],
            dim=1,
        ).unsqueeze(1)
        feature = torch.cat([feature1, feature2], dim=1).float()
        outs = []
        for i in range(1, 3):
            feature = getattr(self, f"resnet{i}")(feature)
            outs.append(feature)
        self.feature = torch.mean(torch.cat(outs, dim=1), dim=-1)
        out = self.last_linear(self.feature)
        return out

    def get_feature(self):
        return self.feature


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class VisionEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super(VisionEncoder, self).__init__()
        self.cnn = timm.create_model(
            config.backbone,
            pretrained=config.is_pretrained,
            # No classifier for ArcFace
            num_classes=0,
            in_chans=config.image.in_channels,
        )
        self.in_features = self.cnn.num_features
        self.swish = Swish_module()
        self.projection = Projection(self.in_features, 512)
        # for p in self.cnn.parameters():
        #     p.requires_grad = False

    def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
        feature = self.cnn.forward_features(imgs)
        # feature = self.gem(feature)
        feature = F.adaptive_avg_pool2d(feature, (1, 1))
        feature = feature.view(feature.size(0), -1)
        return feature

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        imgs = batch["image"].permute(0, 3, 1, 2)
        feature = self.forward_features(imgs)
        projected_vec = self.projection(feature)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class TextEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super(TextEncoder, self).__init__()
        self.config = config
        if config.is_pretrained:
            self.bert_model = AutoModel.from_pretrained(config.text.backbone)
        else:
            self.bert_model = AutoModel(AutoConfig(config.text.backbone))
        # bertの特徴量分だけプラス
        self.in_features = self.bert_model.config.hidden_size
        self.projection = Projection(self.in_features, 512)
        # for p in self.bert_model.parameters():
        #     p.requires_grad = False

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        bert_out = self.bert_model(
            batch["text"], batch["mask"], output_hidden_states=True
        )
        # pooler_outputは[CLS] tokenに対してlinearとtanhを通したもの
        # _, pool = bert_out["last_hidden_state"], bert_out["pooler_output"]
        # hidden_states = torch.stack(bert_out["hidden_states"][:4])
        # feature = torch.mean(hidden_states[:, :, 0], dim=0)
        feature = bert_out["last_hidden_state"][:, 0]
        projected_vec = self.projection(feature)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class Clip(nn.Module):
    def __init__(self, config: DictConfig):
        super(Clip, self).__init__()
        self.config = config
        self.image_encoder = VisionEncoder(config)
        self.text_encoder = TextEncoder(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_vec = self.image_encoder(batch)
        text_vec = self.text_encoder(batch)
        return image_vec, text_vec


def get_2dcnn(model_config):
    model = CNN2D(model_config)
    return model


def get_triplet_2dcnn(model_config):
    model = TripletCNN2D(model_config)
    return model


def get_custom_cnn(model_config):
    model = CustomCNN(model_config)
    return model


def get_multi_scale_cnn(model_config):
    model = MultiScaleCNN(model_config)
    return model


def get_insta_resnext(model_config):
    model = InstagramResNext(model_config)
    return model


def get_cnn_bert(model_config):
    model = CNNBert(model_config)
    return model


def get_bert(model_config):
    model = Bert(model_config)
    return model


def get_clip(model_config):
    model = Clip(model_config)
    return model


def get_query_mlp(model_config):
    model = QueryMLP(model_config)
    return model


def get_query_cnn(model_config):
    model = QueryCNN(model_config)
    return model


def get_model(config):
    f = globals().get(config.model_class)
    return f(config)
