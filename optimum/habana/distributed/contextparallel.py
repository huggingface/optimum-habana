import torch

from .parallel_state import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)


class ContextParallelLossFunction(torch.autograd.Function):
    """
    Gather losses across context parallel group.

    This custom autograd function is designed to handle the distribution of loss computation
    across multiple parallel contexts in a distributed training setup. It ensures that the loss
    is gathered from all devices involved in the parallel context, allowing for consistent and
    accurate computation of the overall loss.

    The forward method gathers the loss from all ranks in the context parallel group, while the
    backward method ensures that gradients are correctly synchronized across the different parallel
    contexts.
    """

    @staticmethod
    def forward(ctx, loss):
        ctx.seqlen = loss.size(0) * get_sequence_parallel_world_size()
        # Create a tensor to gather all losses from context parallel group
        loss_all = torch.empty(ctx.seqlen, dtype=loss.dtype, device=loss.device)
        # Gather losses from all ranks in the group
        torch.distributed.all_gather_into_tensor(loss_all, loss, group=get_sequence_parallel_group())
        return loss_all

    @staticmethod
    def backward(ctx, grad_output):
        step_seqlen = ctx.seqlen // get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        # Extract the relevant part of the gradient for this rank
        grad_output_part = grad_output[step_seqlen * sp_rank : step_seqlen * (sp_rank + 1)]
        return grad_output_part, None


def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    loss_all = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction="none")
    # Apply context parallel loss
    loss_all = ContextParallelLossFunction.apply(loss_all)
    if num_items_in_batch is None:
        loss = torch.mean(loss_all)
    else:
        loss = torch.sum(loss_all) / num_items_in_batch
    return loss


def ForCausalLMContextParallelLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)

    loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss
