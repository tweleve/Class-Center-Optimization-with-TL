import torch


class TransformerDecorator(torch.nn.Module):
    def __init__(self, add_bf=False, dim=2048, eval_global=0):
        super(TransformerDecorator, self).__init__()
        self.encoder_layers = torch.nn.TransformerEncoderLayer(dim, 4, dim, 0.5)
        self.eval_global = eval_global
        self.add_bf = add_bf

    def forward(self, feature, targets):
        if self.add_bf or self.eval_global > 0:
            pre_feature = feature
            feature = feature.unsqueeze(1)
            feature = self.encoder_layers(feature)
            feature = feature.squeeze(1)
            # x = torch.cat([pre_feature, feature], dim=0)
            # y = torch.cat([targets, targets], dim=0)
            return feature, targets
            # return x, y
        return feature, targets
