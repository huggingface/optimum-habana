import torch

from .parallel_state import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)


# Gather losses across context parallel group
class _ContextParallelLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss):
        ctx.seqlen = loss.size(0) * get_sequence_parallel_world_size()

        loss_all = torch.empty(ctx.seqlen, dtype=loss.dtype, device=loss.device)
        torch.distributed.all_gather_into_tensor(loss_all, loss, group=get_sequence_parallel_group())
        return loss_all

    @staticmethod
    def backward(ctx, grad_output):
        step_seqlen = ctx.seqlen // get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        grad_output_part = grad_output[step_seqlen * sp_rank : step_seqlen * (sp_rank + 1)]

        return grad_output_part, None


def _get_loss_from_context_parallel(vocab_parallel_loss):
    return _ContextParallelLoss.apply(vocab_parallel_loss)
