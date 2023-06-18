# %%
# pytorch
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# other
import pandas as pd
import numpy as np
# %%
# =============================================================================
# 1. Define the model and loss function
# =============================================================================
class LyricsClassifier(nn.Module):
    def __init__(self, dropout_probability=0.5):
        super().__init__()
        self.fc1 = nn.Linear(X.size(1),10)
        # self.dropout = nn.Dropout(dropout_probability)
        self.fc2 = nn.Linear(10, len(set(labels)))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)  # Dropout layer to prevent overfitting
        outputs = self.fc2(x)
        return outputs,x

def custom_loss(outputs, labels,hidden_outputs, alpha=1.0, beta=0.01):
    criterion = nn.CrossEntropyLoss()
    loss1 = criterion(outputs, labels)

    # Perform clustering on the hidden outputs
    # this loss is controlle by the hyperparameter alpha
    kmeans = KMeans(n_clusters=6, random_state=0,n_init='auto').fit(hidden_outputs.detach().numpy())
    cluster_labels = torch.tensor(kmeans.labels_)
    loss2 = nn.MSELoss()(labels.float(), cluster_labels.float())

    # L1 regularization
    l1_reg = torch.tensor(0.)
    for param in model.parameters():
        l1_reg += torch.norm(param, 1)
    loss1 = loss1 + beta * l1_reg

    # print(f'CrossEntropyLoss: {loss1.item()}, KMeans loss: {loss2.item()}')
    
    loss2 = 0
    total_loss = loss1 + alpha * loss2
    return total_loss

# %%
# =============================================================================
# 2. Load the data
# =============================================================================
def label_years(year):
    if year < 1970:
        return 0
    elif year < 1980:
        return 1
    elif year < 1990:
        return 2
    elif year < 2000:
        return 3
    elif year < 2010:
        return 4
    else:
        return 5
    
df = pd.read_csv("../data/lyrics-data/lyrics_and_metadata_1950_2019.csv",index_col=0)
df.dropna(inplace=True)  # Remove missing values
lyrics = df['lyrics'].to_list()  # Convert the lyrics column to a list
labels = df['release_date'].map(label_years).to_list()  # Convert the year column to a list


# Convert data to numerical values
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lyrics)
X = torch.tensor(X.todense(), dtype=torch.float32)

# Convert labels to tensor
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y = torch.tensor(y, dtype=torch.long)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
# =============================================================================
# 3. Train the model
# =============================================================================

num_epochs = 100
batch_size = 16
weight_decay = 0.01  # l2 penalty
alpha = 1
beta = 0.00001  # l1 penalty
loss_fn = custom_loss

model = LyricsClassifier()
# optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
optimizer = optim.Adam(model.parameters())

train_losses = []
validation_accuracies = []
test_accuracies = []

# Train the model
for epoch in range(num_epochs):
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for inputs, labels in train_loader:
        outputs, hidden_outputs = model(inputs)
        loss = loss_fn(outputs, labels, hidden_outputs,alpha=alpha, beta=beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs, hidden_outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    validation_accuracy = correct / total
    validation_accuracies.append(validation_accuracy)

    # Test loop
    model.eval()  # set the model to evaluation mode
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        total = 0
        correct = 0
        for inputs, labels in test_loader:
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}, Train loss: {loss.item()}, Validation Accuracy: {validation_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Testing
model.eval()  # set the model to evaluation mode
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    total = 0
    correct = 0
    for inputs, labels in test_loader:
        outputs, _ = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Final Test Accuracy: {correct / total :.4f}')
# %%
# =============================================================================
# 4. Plot the results
# =============================================================================
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

smoothed_val_acc = moving_average(validation_accuracies, 5)
smoothed_test_acc = moving_average(test_accuracies, 5)
smoothed_train_loss = moving_average(train_losses, 5)

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color="black")
ax1.plot(smoothed_val_acc, color="tab:blue", label='Validation accuracy')
ax1.plot(smoothed_test_acc, color='tab:orange', label='Test accuracy')
ax1.tick_params(axis='y', labelcolor="black")
ax1.legend(loc='upper left')

ax2 = ax1.twinx()

smoothed_train_loss = moving_average(train_losses, 5)

color = 'black'
ax2.set_ylabel('Loss', color=color)
ax2.plot(smoothed_train_loss, color=color, label='Train loss')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()
plt.title('Loss and Accuracy Over Epochs')
plt.show()
# %%
