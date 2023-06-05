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
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(X.size(1),10)
        self.fc2 = nn.Linear(10, len(set(labels)))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        outputs = self.fc2(x)
        return outputs,x

def custom_loss(outputs, labels,hidden_outputs, alpha=1.0):
    criterion = nn.CrossEntropyLoss()
    loss1 = criterion(outputs, labels)

    # Perform clustering on the hidden outputs
    kmeans = KMeans(n_clusters=6, random_state=0,n_init='auto').fit(hidden_outputs.detach().numpy())

    # Calculate the loss

    # cluster_centers = torch.tensor(kmeans.cluster_centers_)
    # loss2 = nn.MSELoss()(hidden_outputs, cluster_centers)

    cluster_labels = torch.tensor(kmeans.labels_)
    loss2 = nn.MSELoss()(labels.float(), cluster_labels.float())

    # print(f'CrossEntropyLoss: {loss1.item()}, KMeans loss: {loss2.item()}')

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
batch_size = 32
loss_fn = custom_loss

model = LyricsClassifier()
optimizer = torch.optim.Adam(model.parameters())

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
        loss = loss_fn(outputs, labels, hidden_outputs,alpha=10)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs, hidden_outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Accuracy: {100 * correct / total}%')

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

print(f'Test Accuracy: {100 * correct / total}%')
# %%