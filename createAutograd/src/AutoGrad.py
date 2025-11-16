# type: ignore

import torch 


class AutoGrad:
    def normalAutoGrad(self):
        print("\n\nPerforming normal autograd computation\n")
        x = torch.tensor(3.0, requires_grad=True)
        y = x**2 + 2*x + 1
        y.backward()
        print("Value of y:", y.item())
        print("Gradient dy/dx at x=3:", x.grad.item())
        return "Autograd computation done"


    def sinAutoGrad(self):
        print("\n\nPerforming sine autograd computation\n")
        x = torch.tensor(0.5, requires_grad=True)
        y = torch.sin(x)
        z = y**2
        z.backward()
        print("Value of z:", z.item())
        print("Gradient dz/dx at x=0.5:", x.grad.item())
        return "Sine Autograd computation done"
    

        # Binary Cross-Entropy Loss for scalar
    def binary_cross_entropy_loss(self, prediction, target):
        epsilon = 1e-8  # To prevent log(0)
        prediction = torch.clamp(prediction, epsilon, 1 - epsilon)
        return -(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))
    
    def sigmoidAutoGrad(self):
        print("\n\nPerforming sigmoid autograd computation\n")
        x = torch.tensor(6.7) #student CGPA
        y = torch.tensor(1.0) #1 means placed
        print("Input x (CGPA):", x.item())
        print("Target y (placed):", y.item())

        w = torch.tensor(1.0, requires_grad=True) #weight
        b = torch.tensor(0.0, requires_grad=True) #bias
        print("Initial w:", w.item())
        print("Initial b:", b.item())

        z = w * x + b
        print("z:", z.item())

        y_pred = torch.sigmoid(z)
        print("y_pred:", y_pred.item())


        loss = self.binary_cross_entropy_loss(y_pred, y)
        print("loss:", loss.item())

        loss.backward()
        print("ld/dw:", w.grad.item())
        print("ld/db:", b.grad.item())

        return "Sigmoid Autograd computation done"




    def __call__(self):
        """Run all methods in the class when the instance is called"""
        self.normalAutoGrad()
        self.sinAutoGrad()

        self.sigmoidAutoGrad()