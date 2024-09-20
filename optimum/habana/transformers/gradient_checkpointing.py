# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file modifies some utilities and adds a mark_step() at the beginning
# of the backward pass when gradient checkpointing is performed
# Original implementation here: https://github.com/pytorch/pytorch/blob/v2.4.0/torch/utils/checkpoint.py

import contextlib
import warnings
from functools import wraps
from typing import Callable, ContextManager, Optional, Tuple

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from packaging import version
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import (
    check_backward_validity,
    detach_variable,
    get_device_states,
    set_device_states,
)


__all__ = [
    "checkpoint",
    "CheckpointFunction",
    "DefaultDeviceType",
]


# Extra variables to ensure backward compatibility
__MIN_TORCH_VERSION: str = "2.1.0"
__IS_MIN_TORCH_VALID: bool = version.parse(version.parse(torch.__version__).base_version) > version.parse(
    __MIN_TORCH_VERSION
)


def noop_context_fn():
    return contextlib.nullcontext(), contextlib.nullcontext()


def _get_device_module(device="hpu"):
    device_module = getattr(torch, device)
    return device_module


class DefaultDeviceType:
    r"""
    A class that manages the default device type for checkpointing.

    If no non-CPU tensors are present, the default device type will
    be used. The default value is 'hpu'. The device type is used in
    the checkpointing process when determining which device states
    to save and restore for recomputation.
    """

    _default_device_type = "hpu"

    @staticmethod
    def set_device_type(device: str = "hpu"):
        """
        Set the default device type for checkpointing.

        Args:
            device (str): The device type to be set as default. Default is 'hpu'.
        """
        DefaultDeviceType._default_device_type = device

    @staticmethod
    def get_device_type() -> str:
        """
        Get the current default device type for checkpointing.

        Returns:
            str: The current default device type.
        """
        return DefaultDeviceType._default_device_type


if hasattr(torch.utils.checkpoint, "DefaultDeviceType"):
    torch.utils.checkpoint.DefaultDeviceType = DefaultDeviceType


def _infer_device_type(*args):
    device_types = []

    def add_device_types(arg):
        nonlocal device_types
        if isinstance(arg, torch.Tensor) and not arg.device.type == "cpu":
            device_types.append(arg.device.type)

    tree_map(add_device_types, args)

    device_types_set = set(device_types)
    if len(device_types_set) > 1:
        warnings.warn(
            "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices. "
            "Device state will only be saved for devices of a single device type, and the remaining "
            "devices will be ignored. Consequently, if any checkpointed functions involve randomness, "
            "this may result in incorrect gradients. (Note that if HPU devices are among the devices "
            "detected, it will be prioritized; otherwise, the first device encountered will be selected.)"
            f"\nDevice types: {sorted(device_types_set)} first device type: {device_types[0]}"
        )
    if len(device_types) == 0:
        return DefaultDeviceType.get_device_type()
    elif "hpu" in device_types_set:
        return "hpu"
    else:
        return device_types[0]


def _get_autocast_kwargs():
    #  autocast caching is permanently disabled on HPU.
    hpu_autocast_kwargs = {
        "device_type": "hpu",
        "enabled": hthpu.is_autocast_hpu_enabled(),
        "dtype": hthpu.get_autocast_hpu_dtype(),
    }

    cpu_autocast_kwargs = {
        "enabled": torch.is_autocast_cpu_enabled(),
        "dtype": torch.get_autocast_cpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    return hpu_autocast_kwargs, cpu_autocast_kwargs


class CheckpointFunction(torch.autograd.Function):
    _IS_MIN_TORCH_VALID: bool = version.parse(version.parse(torch.__version__).base_version) > version.parse("2.1.0")

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        if torch.is_grad_enabled():  # grad may be disabled, e.g., during validation
            check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND hpu.
        ctx.device = _infer_device_type(*args)
        ctx.hpu_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs()
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            if hasattr(ctx, "had_cuda_in_fwd"):
                ctx.had_cuda_in_fwd = False
            if CheckpointFunction._IS_MIN_TORCH_VALID:
                device_module = _get_device_module(ctx.device)
                if getattr(device_module, "_initialized", False):
                    ctx.had_device_in_fwd = True
                    ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "When use_reentrant=True, torch.utils.checkpoint is incompatible"
                " with .grad() or passing an `inputs` parameter to .backward()."
                " To resolve this error, you can either set use_reentrant=False,"
                " or call .backward() without passing the `inputs` argument."
            )

        htcore.mark_step()

        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if CheckpointFunction._IS_MIN_TORCH_VALID and ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if CheckpointFunction._IS_MIN_TORCH_VALID and ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states)
            detached_inputs = detach_variable(tuple(inputs))

            with torch.enable_grad(), torch.autocast(**ctx.hpu_autocast_kwargs), torch.amp.autocast(
                "cpu", **ctx.cpu_autocast_kwargs
            ):
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError("none of output has requires_grad=True, this checkpoint() is not necessary")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs)

        return (None, None) + grads


def _conditional_disable_dynamo(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if hasattr(torch, "_disable_dynamo"):

            @torch._disable_dynamo
            def inner_func(*args, **kwargs):
                return func(*args, **kwargs)

            return inner_func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


# TorchDynamo does not step inside utils.checkpoint function.  The flow
# looks likes this
#  1) TorchDynamo tries to wrap utils.checkpoint in a HigherOrderOp by
#     speculatively checking if the forward function is safe to trace.
#  2) If yes, then Dynamo-generated Fx graph has the wrapped higher
#     order op. As a result, TorchDynamo does not look inside utils.checkpoint.
#  3) If not, then TorchDynamo falls back to eager by performing a graph
#     break. And here, the following disable wrapper ensures that
#     TorchDynamo does not trigger again on the frames created by
#     utils.checkpoint innards.
@_conditional_disable_dynamo
def checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: Optional[str] = None,
    debug: bool = False,
    **kwargs,
):
    r"""Checkpoint a model or part of the model.

    Activation checkpointing is a technique that trades compute for memory.
    Instead of keeping tensors needed for backward alive until they are used in
    gradient computation during backward, forward computation in checkpointed
    regions omits saving tensors for backward and recomputes them during the
    backward pass. Activation checkpointing can be applied to any part of a
    model.

    There are currently two checkpointing implementations available, determined
    by the :attr:`use_reentrant` parameter. It is recommended that you use
    ``use_reentrant=False``. Please refer the note below for a discussion of
    their differences.

    .. warning::

        If the :attr:`function` invocation during the backward pass differs
        from the forward pass, e.g., due to a global variable, the checkpointed
        version may not be equivalent, potentially causing an
        error being raised or leading to silently incorrect gradients.

    .. warning::

        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.4 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True`` variant, please refer to the
        note below for important considerations and potential limitations.

    .. note::

        The reentrant variant of checkpoint (``use_reentrant=True``) and
        the non-reentrant variant of checkpoint (``use_reentrant=False``)
        differ in the following ways:

        * Non-reentrant checkpoint stops recomputation as soon as all needed
          intermediate activations have been recomputed. This feature is enabled
          by default, but can be disabled with :func:`set_checkpoint_early_stop`.
          Reentrant checkpoint always recomputes :attr:`function` in its
          entirety during the backward pass.

        * The reentrant variant does not record the autograd graph during the
          forward pass, as it runs with the forward pass under
          :func:`torch.no_grad`. The non-reentrant version does record the
          autograd graph, allowing one to perform backward on the graph within
          checkpointed regions.

        * The reentrant checkpoint only supports the
          :func:`torch.autograd.backward` API for the backward pass without its
          `inputs` argument, while the non-reentrant version supports all ways
          of performing the backward pass.

        * At least one input and output must have ``requires_grad=True`` for the
          reentrant variant. If this condition is unmet, the checkpointed part
          of the model will not have gradients. The non-reentrant version does
          not have this requirement.

        * The reentrant version does not consider tensors in nested structures
          (e.g., custom objects, lists, dicts, etc) as participating in
          autograd, while the non-reentrant version does.

        * The reentrant checkpoint does not support checkpointed regions with
          detached tensors from the computational graph, whereas the
          non-reentrant version does. For the reentrant variant, if the
          checkpointed segment contains tensors detached using ``detach()`` or
          with :func:`torch.no_grad`, the backward pass will raise an error.
          This is because ``checkpoint`` makes all the outputs require gradients
          and this causes issues when a tensor is defined to have no gradient in
          the model. To avoid this, detach the tensors outside of the
          ``checkpoint`` function.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint. Note that under torch.compile,
            this flag doesn't take effect and we always preserve RNG state.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.4 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
            This argument is only supported if ``use_reentrant=False``.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks. This argument is only supported if ``use_reentrant=False``,
            if ``use_reentrant=True``, the determinism check is always disabled.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation. This argument is only supported if
            ``use_reentrant=False``.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    if not __IS_MIN_TORCH_VALID:
        preserve = kwargs.pop("preserve_rng_state", True)
        if kwargs:
            raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

        return CheckpointFunction.apply(function, preserve, *args)
    else:
        if use_reentrant is None:
            warnings.warn(
                "torch.utils.checkpoint: the use_reentrant parameter should be "
                "passed explicitly. In version 2.4 we will raise an exception "
                "if use_reentrant is not passed. use_reentrant=False is "
                "recommended, but if you need to preserve the current default "
                "behavior, you can pass use_reentrant=True. Refer to docs for more "
                "details on the differences between the two variants.",
                stacklevel=2,
            )
            use_reentrant = True

        # Hack to mix *args with **kwargs in a python 2.7-compliant way
        preserve = kwargs.pop("preserve_rng_state", True)
        if kwargs and use_reentrant:
            raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

        if use_reentrant:
            if context_fn is not noop_context_fn or debug is not False:
                raise ValueError("Passing `context_fn` or `debug` is only supported when use_reentrant=False.")
            return CheckpointFunction.apply(function, preserve, *args)
        else:
            if determinism_check is None:
                determinism_check = torch.utils.checkpoint._DEFAULT_DETERMINISM_MODE
            gen = _checkpoint_without_reentrant_generator(
                function, preserve, context_fn, determinism_check, debug, *args, **kwargs
            )
            # Runs pre-forward logic
            next(gen)
            ret = function(*args, **kwargs)
            # Runs post-forward logic
            try:
                next(gen)
            except StopIteration:
                return ret


def _version_check():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if __IS_MIN_TORCH_VALID:
                return func(*args, **kwargs)
            else:
                warnings.warn(
                    f"Function '{func.__name__}' is disabled for PyTorch versions less than {__MIN_TORCH_VERSION}."
                )
                return None

        return wrapper

    return decorator


# NB: this helper wraps fn before calling checkpoint_impl. kwargs and
#     saving/restoring of global state is handled here.
@_version_check()
def _checkpoint_without_reentrant_generator(
    fn,
    preserve_rng_state: bool = True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: Optional[str] = None,
    debug: bool = False,
    *args,
    **kwargs,
):
    """Checkpointing without reentrant autograd.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation.
        *args: Arguments to pass in to the given ``function``.
        **kwargs: Keyword arguments to pass into the given ``function``.
    """
    unpack_error_cb = None

    if (
        torch.utils.checkpoint._checkpoint_debug_enabled
        if torch.utils.checkpoint._checkpoint_debug_enabled is not None
        else debug
    ):
        if context_fn != noop_context_fn:
            raise ValueError("debug=True is incompatible with non-default context_fn")
        context_fn, unpack_error_cb = torch.utils.checkpoint._get_debug_context_and_cb()

    if determinism_check in torch.utils.checkpoint._allowed_determinism_checks_to_fns:
        metadata_fn = torch.utils.checkpoint._allowed_determinism_checks_to_fns[determinism_check]
    else:
        raise ValueError(
            f"determinism_check should be one of {list(torch.utils.checkpoint._allowed_determinism_checks_to_fns.keys())}, "
            f"but got {determinism_check}"
        )

    device = _infer_device_type(*args)
    device_module = _get_device_module(device)
    forward_context, recompute_context = context_fn()
    if torch.utils.checkpoint._is_compiling(fn, args, kwargs) and context_fn != noop_context_fn:
        assert isinstance(forward_context, torch.utils._python_dispatch.TorchDispatchMode) and isinstance(
            recompute_context, torch.utils._python_dispatch.TorchDispatchMode
        ), (
            "In torch.compile mode, `context_fn` arg passed to `torch.utils.checkpoint` "
            + "must generate a tuple of two `TorchDispatchMode`s."
        )
    # Accommodates the (remote) possibility that autocast is enabled for cpu AND hpu.
    hpu_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device=device)

    if preserve_rng_state:
        fwd_cpu_state = torch.get_rng_state()
        # Don't eagerly initialize the cuda context by accident.
        # (If the user intends that the context is initialized later, within their
        # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
        # we have no way to anticipate this will happen before we run the function.
        # If they do so, we raise an error.)
        had_device_in_fwd = False
        if getattr(device_module, "_initialized", False):
            had_device_in_fwd = True
            fwd_devices, fwd_device_states = get_device_states(*args)

    def recompute_fn(*inputs):
        kwargs, *args = inputs
        # This will be called later during recomputation. This wrapping enables
        # the necessary global state to be captured.
        rng_devices = []
        if preserve_rng_state and had_device_in_fwd:
            rng_devices = fwd_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=preserve_rng_state, device_type=device):
            if preserve_rng_state:
                torch.set_rng_state(fwd_cpu_state)
                if had_device_in_fwd:
                    set_device_states(fwd_devices, fwd_device_states)

            with torch.autocast(**hpu_autocast_kwargs), torch.amp.autocast(
                "cpu", **cpu_autocast_kwargs
            ), recompute_context:  # type: ignore[attr-defined]
                fn(*args, **kwargs)

    new_frame = torch.utils.checkpoint._CheckpointFrame(
        recompute_fn, torch.utils.checkpoint._enable_checkpoint_early_stop, unpack_error_cb, metadata_fn
    )
    dummy = torch.empty((0,), requires_grad=True)
    new_frame.input_saver = torch.utils.checkpoint._NoopSaveInputs.apply(dummy, kwargs, *args)

    # When ambient grad_mode is False
    if new_frame.input_saver.grad_fn is None:
        yield
        return

    with torch.utils.checkpoint._checkpoint_hook(new_frame), forward_context:
        yield
    new_frame.forward_completed = True

    if getattr(device_module, "_initialized", False) and preserve_rng_state and not had_device_in_fwd:  # type: ignore[possibly-undefined]
        # Device was not initialized before running the forward, so we didn't
        # stash the device state.
        raise RuntimeError(
            "PyTorch's device state was initialized in the forward pass "
            "of a Checkpoint, which is not allowed. Please open an issue "
            "if you need this feature."
        )

    return
