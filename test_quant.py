import torch

def per_tensor_quantize(x: torch.Tensor):
    oshape = x.shape
    x = x.flatten()
    scale = torch.max(x.abs()) / 127
    qt = torch.round(x / scale).char()
    deq = (qt * scale).half()
    return deq.view(oshape)

a = torch.rand(3,4)
print(a)
print(per_tensor_quantize(a))

...loadmodel

d = model.state_dict()
for k,v in sd.items():
    sd[k] = per_tensor_quantize(v)
model.load_state_dict(sd)
