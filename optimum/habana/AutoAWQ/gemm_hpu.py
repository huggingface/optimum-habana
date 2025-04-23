import torch
import torch.nn as nn
from awq.modules.linear.gemm import WQLinear_GEMM
from awq.utils.packing_utils import reverse_awq_order, unpack_awq


try:
    import habana_frameworks.torch.hpu  # noqa: F401

    convert_from_uint4 = torch.ops.hpu.convert_from_uint4
except Exception as e:
    hpu_import_exception = e

    def error_raiser_hpu(*args, **kwargs):
        raise ValueError(
            f"Trying to use HPU, but could not import the HPU framework with the following error: {hpu_import_exception}"
        )

    convert_from_uint4 = error_raiser_hpu


def unpack_weight_and_zeros(qweight, qzeros, bits):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    # Reverse the order of the iweight and izeros tensors
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    return iweight, izeros


def pack_tensor(input, bits=4):
    normal = input.to(torch.int32)
    q = torch.zeros(
        (normal.shape[0], normal.shape[1] // 32 * bits),
        dtype=torch.int32,
        device=input.device,
    )
    i = 0
    col = 0
    while col < q.shape[1]:
        for j in range(i, i + (32 // bits)):
            q[:, col] |= normal[:, j] << (bits * (j - i))
        i += 32 // bits
        col += 1
    q = q.to(torch.int32)
    return q


class WQLinear_HPU(WQLinear_GEMM):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, training=False):
        nn.Module.__init__(self)
        assert w_bit == 4, "Only 4 bit are supported for now."
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.scale_dtype = torch.float32
        self.training = training

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        self.pack_num = 32 // self.w_bit

        self.init_ipex = False

        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // self.pack_num),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.bfloat16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((out_features), dtype=torch.bfloat16, device=dev),
            )
        else:
            self.bias = None
        self.register_buffer(
            "qweight",
            torch.zeros((in_features, out_features // self.pack_num), dtype=torch.int32, device=dev),
        )
        self._preprocess = False

    def _preprocessing(self):
        device = self.qweight.device
        weight, zeros = unpack_weight_and_zeros(self.qweight.cpu(), self.qzeros.cpu(), self.w_bit)
        self.qweight = pack_tensor(weight).to(device)
        self.qzeros = pack_tensor(zeros).to(device)
        self._preprocess = True

    def post_init(self):
        self._preprocessing()

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear
        raise NotImplementedError("Only inference is supported for HPU kernels")

    def forward(self, x):
        assert self._preprocess is True, (
            "module.post_init() must be called before module.forward(). Use hpu_post_init() on the whole model."
        )
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        weights = convert_from_uint4(self.qweight, self.scales, self.qzeros, x.dtype)
        outputs = torch.matmul(x, weights)

        outputs = outputs + self.bias if self.bias is not None else outputs
        outputs = outputs.reshape(out_shape)

        return outputs

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.w_bit,
            self.group_size,
        )


def hpu_post_init(model):
    for _, submodule in model.named_modules():
        if isinstance(submodule, WQLinear_HPU):
            submodule.post_init()

    return model
