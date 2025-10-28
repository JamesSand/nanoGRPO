
import torch

samples = [1, 2, 3]

group_size = 4

samples = [sample for sample in samples for _ in range(group_size)] 

print(samples)

input_ids = torch.tensor([[1,2,3], [4,5,6]])

input_ids = torch.repeat_interleave(input_ids, group_size, dim=0)

print(input_ids)