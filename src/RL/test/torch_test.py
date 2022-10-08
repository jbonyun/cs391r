import torch

print("CUDA OK? ", torch.cuda.is_available())
print("Device count: ", torch.cuda.device_count())
print("Current device: ", torch.cuda.current_device())

for i in range(torch.cuda.device_count()):
    print("Device: ", i)
    print(torch.cuda.device(i))
    print(torch.cuda.get_device_name(i))