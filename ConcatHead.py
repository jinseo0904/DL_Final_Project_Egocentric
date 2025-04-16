import torch
import torch.nn as nn


class ConcatHead(nn.Module):
    """Concatenaion layer for Detect heads."""

    def __init__(self, nc1=80, nc2=1, ch=()):
        """Initializes the ConcatHead."""
        super().__init__()
        self.nc1 = nc1  # number of classes of head 1
        self.nc2 = nc2  # number of classes of head 2

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""

        # x is a list of length 2
        # Each element is either a tuple or just the decoded features
        # depending whether it's being exported.
        # First element of tuple are the decoded preds,
        # second element are feature maps for heatmap visualization

        if isinstance(x[0], tuple):
            preds1 = x[0][0]
            preds2 = x[1][0]
        elif isinstance(x[0], list):  # when returned raw outputs
            # The shape is used for stride creation in tasks.py.
            # Feature maps will have to be decoded individually if used as they can't be merged.
            return [torch.cat((x0, x1), dim=1) for x0, x1 in zip(x[0], x[1])]
        else:
            preds1 = x[0]
            preds2 = x[1]

        # Concatenate the new head outputs as extra outputs

        # 1. Concatenate bbox outputs
        # Shape changes from [N, 4, 6300] to [N, 4, 12600]
        preds = torch.cat((preds1[:, :4, :], preds2[:, :4, :]), dim=2)

        # 2. Concatenate class outputs
        # Append preds 1 with empty outputs of size 6300
        shape = list(preds1.shape)
        shape[-1] *= 2

        preds1_extended = torch.zeros(shape, device=preds1.device, dtype=preds1.dtype)
        preds1_extended[..., : preds1.shape[-1]] = preds1

        # Prepend preds 2 with empty outputs of size 6300
        shape = list(preds2.shape)
        shape[-1] *= 2

        preds2_extended = torch.zeros(shape, device=preds2.device, dtype=preds2.dtype)
        preds2_extended[..., preds2.shape[-1] :] = preds2

        # Arrange the class probabilities in order preds1, preds2. The
        # class indices of preds2 will therefore start after preds1
        preds = torch.cat((preds, preds1_extended[:, 4:, :]), dim=1)
        preds = torch.cat((preds, preds2_extended[:, 4:, :]), dim=1)

        if isinstance(x[0], tuple):
            return (preds, x[0][1])
        else:
            return preds
