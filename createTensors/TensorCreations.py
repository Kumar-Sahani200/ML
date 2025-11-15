# type: ignore

import torch 

def create_tensor():
    t1 = torch.ones(2,3)
    print("ones tensor:")
    print(t1)

    t2 = torch.zeros(2,3)
    print("Zeros tensor:")
    print(t2)

    t3 = torch.empty(2,3)
    print("Empty tensor:")
    print(t3)

    t4 = torch.rand(2,3)
    print("Random tensor:")
    print(t4)

    t5 = torch.eye(3,3)
    print("Identity tensor:")
    print(t5)

    t6 = torch.arange(0,10,2)
    print("Arange tensor:")
    print(t6)

    t7 = torch.linspace(0,5,3)
    print("Linspace tensor:")
    print(t7)

    return "Tensors created"
