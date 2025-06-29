import numpy as np
import random

import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping(object):
    best_losses = None
    counter = 0

    def __init__(self, patience: int = 10):
        self.patience = patience

    def __call__(self, losses):
        if self.best_losses is None:
            self.best_losses = losses
            self.counter = 0

        elif any(
            loss <= best_loss
            for loss, best_loss in zip(losses, self.best_losses)
        ):
            if all(
                loss <= best_loss
                for loss, best_loss in zip(losses, self.best_losses)
            ):

                self.best_losses = [
                    min(loss, best_loss)
                    for loss, best_loss in zip(losses, self.best_losses)
                ]
                self.counter = 0

        else:
            self.counter += 1
            if self.counter == self.patience:
                return True

        return False
    


from typing import Optional
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr

# from https://github.com/twitter-research/graph-neural-pde/blob/main/src/utils.py
# https://twitter.com/jon_barron/status/1387167648669048833?s=12
# @torch.jit.script
def squareplus(src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
            num_nodes: Optional[int] = None
            ) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
        Given a value tensor :attr:`src`, this function first groups the values
        along the first dimension based on the indices specified in :attr:`index`,
        and then proceeds to compute the softmax individually for each group.

        Args:
                src (Tensor): The source tensor.
                index (LongTensor): The indices of elements for applying the softmax.
                ptr (LongTensor, optional): If given, computes the softmax based on
                        sorted inputs in CSR representation. (default: :obj:`None`)
                num_nodes (int, optional): The number of nodes, *i.e.*
                        :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

        :rtype: :class:`Tensor`
        """
    out = src - src.max()
    # out = out.exp()
    out = (out + torch.sqrt(out ** 2 + 4)) / 2

    if ptr is not None:
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)

