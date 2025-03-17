# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Understand the classification task and identify input and output variables.

### STEP 2:
Gather data, clean it, handle missing values, and split it into training and test sets.

### STEP 3:
Normalize/standardize features, encode categorical labels, and reshape data if needed.

### STEP 4:
Choose the number of layers, neurons, and activation functions for your neural network.

### STEP 5:
Select a loss function (e.g., binary cross-entropy), optimizer (e.g., Adam), and metrics (e.g., accuracy).

### STEP 6:
Feed training data into the model, run multiple epochs, and monitor the loss and accuracy.

### STEP 7:
Save the trained model, export it if needed, and deploy it for real-world use.

## PROGRAM

### Name: Kowsalya M
### Register Number: 212222230069

```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
```
```

# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```
```
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```



## Dataset Information

![421906194-98829171-025b-41e4-b3e8-6f08aed6da4c](https://github.com/user-attachments/assets/b7a68e28-f139-4091-a821-e8a260a5590a)


## OUTPUT

### Confusion Matrix

![image](https://github.com/user-attachments/assets/42691f3f-a2c0-4ead-a615-8a6bcaa94c62)

### Classification Report

![image](https://github.com/user-attachments/assets/3864cfd2-22ea-4f02-8469-739b3cc5e997)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/d0b3909b-539b-4909-b0ba-779ef03da5c4)


## RESULT

Thus a neural network classification model for the given dataset is executed successfully.
