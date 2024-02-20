import torch
import numpy as np


def feat_extractor(cfg_add_bf, model, data_loader, logger=None):
    model.eval()

    def batch_former(bf_feats, bf_targets):
        from ret_benchmark.modeling.batchformer import TransformerDecorator
        BF = TransformerDecorator(add_bf=cfg_add_bf, dim=512, eval_global=0)
        BF.to('cpu')
        feats_bf, targets_bf = BF.forward(bf_feats, bf_targets)
        return feats_bf, targets_bf

    feats = list()
    if logger is not None:
        logger.info("Begin extract")
    for i, batch in enumerate(data_loader):
        imgs = batch[0].cuda()

        with torch.no_grad():
            # out = model(imgs).data.cpu().numpy()
            out = model(imgs).data.cpu()
            out, _ = batch_former(out, [])
            out = out.numpy()
            feats.append(out)

        if logger is not None and (i + 1) % 100 == 0:
            logger.debug(f"Extract Features: [{i + 1}/{len(data_loader)}]")
        del out
    feats = np.vstack(feats)
    return feats
