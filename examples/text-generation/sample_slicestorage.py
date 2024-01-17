from torch import nn
import torch
import os, sys
import torch
torch.manual_seed(0)

def storage():
    return int(os.environ.get('STORAGE', '0')) == 1

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

        is_prefill = key_states.shape[2] > 1
        if is_prefill:
            if storage():
                bs = key_states.shape[0]
                d0 = key_states.shape[1]
                d1 = key_states.shape[3]
                self.storage_key = torch.zeros([bs, d0, param['final_shape'], d1], device=key_states.device)
                rng = torch.arange(0, key_states.shape[2], device=key_states.device)
                self.storage_key.index_copy_(2, rng, key_states)
                self.past_key = self.storage_key[:,:,:key_states.shape[2],:]
            else:
                self.past_key = key_states
        else:
            if param['need_expansion']:
                if storage():
                    self.past_key = self.storage_key[:,:,:param["allocated_space"],:]
                else:
                    pad_amount = param["allocated_space"] - self.past_key.shape[2]
                    self.past_key = torch.nn.functional.pad(self.past_key, (0, 0, 0, pad_amount), value=0)
            self.past_key.index_copy_(2, token_idx - 1, key_states)
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
token_idx = torch.tensor(5, device=device)
for i in range(9):
    if i == 0:
        inp = torch.rand([2,4,8,5])
    else:
        inp = torch.rand([2,4,1,5])
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
# expect these values to be tensor(59.1479, device='hpu:0') tensor(260.6609, device='hpu:0')

