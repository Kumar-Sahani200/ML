# step 1 build nn architecture
# step 2 build the serialization functions
# step 3 create a fake dataset
# step 4 build the training loop
# step 5 train the model on the fake dataset

# type:ignore
import torch
import torch.nn as nn
from data import X_train, X_train_tensor, y_train_tensor

class SimpleNN(nn.Module):
    def __init__(self, featureSize):
        super().__init__()
        self.simpleNN = nn.Linear(featureSize, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        out = self.simpleNN(features)
        out = self.sigmoid(out)
        return out
    
learning_rate = 0.01
epochs = 25

loss_fn = nn.BCELoss()

model = SimpleNN(X_train.shape[1])

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    y_pred = model(X_train_tensor)

    loss = loss_fn(y_pred, y_train_tensor.view(-1, 1))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        
