from torch import nn
import torch
import os, sys
import torch
torch.manual_seed(0)

def storage():
    return int(os.environ.get('STORAGE', '0')) == 1

import habana_frameworks.torch as htorch
def mem_usage(tag):
    mem_summary1 = htorch.hpu.memory_summary()
    x = {i.strip():j.strip() for i, j in [i.split(':') for i in mem_summary1.strip().split('\n')[3:]]}
    if tag is not None:
        print(tag, x['InUse'])
    return x['InUse']

class TestModule(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(
        self,
        key_states,
        token_idx,
        param
    ):
        y = key_states+1

        is_prefill = key_states.shape[0] > 1
        if is_prefill:
            if storage():
                bs = key_states.shape[1]
                d0 = key_states.shape[2]
                d1 = key_states.shape[3]
                self.storage_key = torch.zeros([param['final_shape'], bs, d0, d1], device=key_states.device)
                rng = torch.arange(0, key_states.shape[0], device=key_states.device)
                self.storage_key.index_copy_(0, rng, key_states)
                self.past_key = self.storage_key[:key_states.shape[0], :,:,:]
            else:
                self.past_key = key_states
        else:
            if param['need_expansion']:
                if storage():
                    self.past_key = self.storage_key[:param["allocated_space"],:,:,:]
                else:
                    pad_amount = param["allocated_space"] - self.past_key.shape[0]
                    self.past_key = torch.nn.functional.pad(self.past_key, (0, 0, 0, 0, 0, 0, 0, pad_amount), value=0)
            self.past_key.index_copy_(0, token_idx - 1, key_states)
        print(self.past_key.shape)
        return (y, self.past_key)


import sys
device = sys.argv[1]
if device == 'hpu':
    import habana_frameworks.torch.core as htcore


model = TestModule()
model = model.to(device)
#x = torch.empty([2,4,8,5], device=device)
#x1 = torch.empty([2,4,1,5], device=device)
token_idx = torch.tensor(7, device=device)
bs=32
d0=128
d1=32
for i in range(9):
    if i == 0:
        inp = torch.rand([9, bs,d0,d1])
    else:
        inp = torch.rand([1, bs,d0,d1])
    inp = inp.to(device)

    if i == 0:
        param = {'need_expansion': False, 'final_shape': 15}
    elif i == 3:
        param = {'need_expansion': True, "allocated_space": 12}
    elif i == 6:
        param = {'need_expansion': True, "allocated_space": 15}
    else:
        param = {'need_expansion': False}

    print(token_idx, param)
    y = model(inp, token_idx, param)
    token_idx = token_idx + 1
    print(y[0].sum(), y[1].sum())
    mem_usage(f'{i}')
    print('----------------')
    f = open(f'.graph_dumps/{i}.txt', 'w')
print(y[0].sum(), y[1].sum())
# expect these values to be tensor(196667.2031, device='hpu:0') tensor(983312.8750, device='hpu:0')


