import torch
import torch.nn as nn


class TransformerDecorator(torch.nn.Module):
    def __init__(self, add_bf=3, dim=2048, eval_global=0):
        super(TransformerDecorator, self).__init__()
        self.encoder_layers = torch.nn.TransformerEncoderLayer(dim, 4, dim, 0.5)
        self.eval_global = eval_global
        self.add_bf = add_bf

    def forward(self, feature):
        if self.training or self.eval_global > 0:
            pre_feature = feature
            feature = feature.unsqueeze(1)
            feature = self.encoder_layers(feature)
            feature = feature.squeeze(1)
            return torch.cat([pre_feature, feature], dim=0)
        return feature

