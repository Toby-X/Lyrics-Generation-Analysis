# %%
from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# %%
# Step 1: Prepare the data
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
    
df = pd.read_csv("../data/lyrics4Bert.csv")
df.dropna(inplace=True)  # Remove missing values
lyrics = df['lyrics'].to_list()  # Convert the lyrics column to a list
labels = df['year'].map(label_years).to_list()  # Convert the year column to a list


# %%
# Step 3: Fine-tune BERT

# Hyperparameters
batch_size = 16
learning_rate = 1e-5
loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 20

# Step 3: Split the data
X_train_val, X_test, y_train_val, y_test = train_test_split(lyrics, labels, test_size=0.2, random_state=42)

# Convert data to BERT input format
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
test_encodings = tokenizer(X_test, truncation=True, padding=True,return_tensors="pt")
test_labels = torch.tensor(y_test, dtype=torch.long)
test_dataset = torch.utils.data.TensorDataset(test_encodings.input_ids.squeeze(), test_encodings.attention_mask.squeeze(), test_labels)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Train the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Train the model with cross-validation and track accuracy
train_accuracies = []
val_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # Create train and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)

    train_encodings = tokenizer(X_train, truncation=True, padding=True,return_tensors="pt")
    val_encodings = tokenizer(X_val, truncation=True, padding=True, return_tensors="pt")

    val_labels = torch.tensor(y_val, dtype=torch.long)
    train_labels = torch.tensor(y_train, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(train_encodings.input_ids.squeeze(), train_encodings.attention_mask.squeeze(), train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_encodings.input_ids.squeeze(), val_encodings.attention_mask.squeeze(), val_labels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Reset the progress bar for each epoch
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
    
    # Training loop
    model.train()
    train_preds = []
    train_targets = []

    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

        progress_bar.update(1)

    # Calculate training accuracy
    train_accuracy = accuracy_score(train_targets, train_preds)
    train_accuracies.append(train_accuracy)

    # Validation loop
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    # Calculate validation accuracy
    val_accuracy = accuracy_score(val_targets, val_preds)
    val_accuracies.append(val_accuracy)
    progress_bar.set_postfix({"Training Accuracy": train_accuracy, "Validation Accuracy": val_accuracy})

    # Test loop
    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    # Calculate test accuracy
    test_accuracy = accuracy_score(test_targets, test_preds)
    test_accuracies.append(test_accuracy)
    print(f"Test Accuracy: {test_accuracy}")


    progress_bar.close()

# Test loop
model.eval()
test_preds = []
test_targets = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

# Calculate test accuracy
test_accuracy = accuracy_score(test_targets, test_preds)
print(f"Test Accuracy: {test_accuracy}")

# %%
# visualize training process
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(train_accuracies, label='train', color='blue')
plt.plot(val_accuracies, label='validation', color='green')
plt.plot(test_accuracies, label='test', color='red')

plt.title('Accuracy Over Epochs')

plt.xticks(range(0, len(train_accuracies), max(len(train_accuracies)//10, 1)))

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('BERT_accuracy.png')
# %%
