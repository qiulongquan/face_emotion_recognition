import torch
"""
torch环境测试
https://note.nkmk.me/python-pytorch-cuda-is-available-device-count/
"""
print(torch.__version__)
# 1.7.1

print(torch.cuda.is_available())
# True

print(torch.cuda.device_count())
# 1

print(torch.cuda.current_device())
# 0

print(torch.cuda.get_device_name())
# GeForce GTX 1080 Ti

print(torch.cuda.get_device_capability())
# (6, 1)

print(torch.cuda.get_device_name())

print(torch.cuda.get_device_capability())

print(torch.cuda.get_device_name(0))
# GeForce GTX 1080 Ti

print(torch.cuda.get_device_name(torch.device('cuda:0')))
# GeForce GTX 1080 Ti

print(torch.cuda.get_device_name('cuda:0'))
# GeForce GTX 1080 Ti

def test():
    import torch
    import time
    from torch import autograd
    #GPU加速
    print(torch.__version__)
    print(torch.cuda.is_available())

    a=torch.randn(10000,1000)
    b=torch.randn(1000,10000)
    print(a)
    print(b)
    t0=time.time()
    c=torch.matmul(a,b)
    t1=time.time()

    print(a.device,t1-t0,c.norm(2))

    device=torch.device('cuda')
    print(device)
    a=a.to(device)
    b=b.to(device)

    t0=time.time()
    c=torch.matmul(a,b)
    t2=time.time()
    print(a.device,t2-t0,c.norm(2))


    t0=time.time()
    c=torch.matmul(a,b)
    t2=time.time()

    print(a.device,t2-t0,c.norm(2))

test()
