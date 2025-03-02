from mmpretrain.apis.image_classification import ImageClassificationInferencer
# import torch
import numpy as np
from typing import List
from mmpretrain.structures import DataSample


class CChessImageClassificationInferencer(ImageClassificationInferencer):
    def postprocess(self,
                    preds: List[DataSample],
                    visualization: List[np.ndarray],
                    return_datasamples=False) -> dict:
        if return_datasamples:
            return preds

        results = []
        for data_sample in preds:
            # torch.Size([90, 16])
            pred_scores = data_sample.pred_score

            pred_label = pred_scores.argmax(dim=-1, keepdim=True).detach()
            pred_score = pred_scores.gather(dim=-1, index=pred_label)
            # pred_score = float(torch.max(pred_scores).item())
            # pred_label = torch.argmax(pred_scores).item()
            result = {
                'pred_scores': pred_scores.detach().cpu().numpy(),
                'pred_label': pred_label.squeeze(-1).detach().cpu().numpy(),
                'pred_score': pred_score.squeeze(-1).detach().cpu().numpy(),
            }

            if self.classes is not None:
                result['pred_class'] = [
                    self.classes[x] for x in result['pred_label']
                ]
            results.append(result)

        return results
