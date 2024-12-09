import os
from huggingface_hub import hf_hub_download

from optimum.habana.utils import  get_habana_frameworks_version

def set_mamba_lib():
    version_no = get_habana_frameworks_version()
    env_variables = os.environ.copy()

    name_op = "hpu_custom_pscan_all.cpython-310-x86_64-linux-gnu.so"
    name_kernel = "libcustom_tpc_perf_lib.so"
    if version_no.minor == 19:
        name_op = "hpu_custom_pscan_all.cpython-310-x86_64-linux-gnu_119.so"
        name_kernel = "libcustom_tpc_perf_lib_119.so"

    file_op = hf_hub_download(repo_id="Habana/mamba", filename=name_op)
    file_kernel = hf_hub_download(repo_id="Habana/mamba", filename=name_kernel)

    new_file_op = file_op
    new_file_kernel = file_kernel

    if version_no.minor == 19:
        new_file_op = file_op[:-7] + ".so"
        new_file_kernel = file_kernel[:-7] + ".so"
        os.rename(file_op, new_file_op)
        os.rename(file_kernel, new_file_kernel)

    return new_file_op, new_file_kernel

    