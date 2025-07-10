import torch


def is_compiled_module(module: torch.nn.Module) -> bool:
    """
    Check whether the module was compiled with torch.compile()
    """
    if not hasattr(torch, "_dynamo"):
        return False

    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def has_compiled_regions(module: torch.nn.Module) -> bool:
    """
    Check whether the module has submodules that were compiled with `torch.compile()`.
    """
    if not hasattr(torch, "_dynamo"):
        return False

    if module._modules:
        for submodule in module.modules():
            if isinstance(submodule, torch._dynamo.eval_frame.OptimizedModule):
                return True

    return False


def is_repeated_blocks(module: torch.nn.Module) -> bool:
    """
    Check whether the module is a repeated block, i.e. `torch.nn.ModuleList` with all children of the same class. This
    is useful to determine whether we should apply regional compilation to the module.
    """

    return isinstance(module, torch.nn.ModuleList) and all(isinstance(m, module[0].__class__) for m in module)


def has_repeated_blocks(module: torch.nn.Module) -> bool:
    """
    Check whether the module has repeated blocks, i.e. `torch.nn.ModuleList` with all children of the same class, at
    any level of the module hierarchy. This is useful to determine whether we should apply regional compilation to the
    module.
    """
    if module._modules:
        for submodule in module.modules():
            if is_repeated_blocks(submodule):
                return True

    return False


def compile_regions(module: torch.nn.Module, **compile_kwargs) -> torch.nn.Module:
    """
    Performs regional compilation where we target repeated blocks of the same class and compile them sequentially to
    hit the compiler's cache. For example, in `GPT2LMHeadModel`, the repeated block/class is `GPT2Block`, and can be
    accessed as `model.transformer.h[0]`. The rest of the model (e.g. model.lm_head) is compiled separately.

    This allows us to speed up the compilation overhead / cold start of models like LLMs and Transformers in general.
    See https://pytorch.org/tutorials/recipes/regional_compilation.html for more details.

    Args:
        module (`torch.nn.Module`):
            The model to compile.
        **compile_kwargs:
            Additional keyword arguments to pass to `torch.compile()`.

    Returns:
        `torch.nn.Module`: A new instance of the model with some compiled regions.

    Example:
    ```python
    >>> from accelerate.utils import compile_regions
    >>> from transformers import AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> compiled_model = compile_regions(model, mode="reduce-overhead")
    >>> compiled_model.transformer.h[0]
    OptimizedModule(
        (_orig_mod): GPT2Block(
                (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (attn): GPT2Attention(
                (c_attn): Conv1D(nf=2304, nx=768)
                (c_proj): Conv1D(nf=768, nx=768)
                (attn_dropout): Dropout(p=0.1, inplace=False)
                (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
                (c_fc): Conv1D(nf=3072, nx=768)
                (c_proj): Conv1D(nf=768, nx=3072)
                (act): NewGELUActivation()
                (dropout): Dropout(p=0.1, inplace=False)
            )
        )
    )
    ```
    """

    def _compile_regions(module: torch.nn.Module, **compile_kwargs) -> torch.nn.Module:
        if is_repeated_blocks(module):
            new_module = torch.nn.ModuleList()
            for submodule in module:
                new_module.append(torch.compile(submodule, **compile_kwargs))
        elif has_repeated_blocks(module):
            new_module = module.__class__.__new__(module.__class__)
            new_module.__dict__.update(module.__dict__)
            new_module._modules = {}
            for name, submodule in module.named_children():
                new_module.add_module(name, _compile_regions(submodule, **compile_kwargs))
        else:
            new_module = torch.compile(module, **compile_kwargs)

        return new_module

    new_module = _compile_regions(module, **compile_kwargs)

    if "_orig_mod" not in new_module.__dict__:
        # Keeps a reference to the original module to decompile/unwrap it later
        new_module.__dict__["_orig_mod"] = module

    return new_module


def compile_regions_deepspeed(module: torch.nn.Module, **compile_kwargs):
    """
    Performs regional compilation the same way as `compile_regions`, but specifically for `DeepSpeedEngine.module`.
    Since the model is wrapped in a `DeepSpeedEngine` and has many added hooks, offloaded parameters, etc that
    `torch.compile(...)` interferes with, version of trgional compilation uses the inplace `module.compile()` method
    instead.

    Args:
        module (`torch.nn.Module`):
            The model to compile.
        **compile_kwargs:
            Additional keyword arguments to pass to `module.compile()`.
    """

    if is_repeated_blocks(module):
        for submodule in module:
            submodule.compile(**compile_kwargs)
    elif has_repeated_blocks(module):
        for child in module.children():
            compile_regions_deepspeed(child, **compile_kwargs)
    else:  # leaf node
        module.compile(**compile_kwargs)
