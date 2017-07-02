import torch
import torchvision

a = torch.ones(4)
b = torch.ones(4)

if torch.cuda.is_available():
    a = a.cuda()
    b = b.cuda()

print(a+b)
print("cool, looks like pytorch is correctly installed!")
