import builtins as __builtin__


def print_only_from_main_process(is_main):
    """
    This function disables printing when not in main process
    """
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
