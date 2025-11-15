# type: ignore
import torch 


def tenFun():
    print("This is a tensor function")
    print("Torch version:", torch.__version__)

    if (torch.cuda.is_available()):
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    return "This is a tensor function"