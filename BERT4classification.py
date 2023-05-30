# %%
from sklearn.model_selection import train_test_split
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
df = pd.read_csv("./data/billboard_lyrics_genres.csv") # Load the lyrics data
df.dropna(inplace=True)  # Remove missing values
lyrics = df['lyrics'].to_list()  # Convert the lyrics column to a list
labels = df['year'].to_list()  # Convert the year column to a list

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(lyrics, labels, test_size=0.2, random_state=42)

# Convert data to BERT input format
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(X_train, truncation=True, padding=True,return_tensors="pt")
test_encodings = tokenizer(X_test, truncation=True, padding=True,return_tensors="pt")

print("Shapes of train_encodings.input_ids and X_train:")
print(train_encodings.input_ids.shape, len(X_train))

print("Shapes of train_encodings.attention_mask and X_train:")
print(train_encodings.attention_mask.shape, len(X_train))

# Convert labels to tensors
train_labels = torch.tensor(y_train)
test_labels = torch.tensor(y_test)

# Create train and test datasets
train_dataset = torch.utils.data.TensorDataset(train_encodings.input_ids, train_encodings.attention_mask, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_encodings.input_ids, test_encodings.attention_mask, test_labels)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# %%
# Step 3: Fine-tune BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# Train the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 100
total_steps = len(train_dataloader) * num_epochs
progress_bar = tqdm(total=total_steps, desc="Training")

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        progress_bar.update(1)

    # Step 4: Encode the data
    model.eval()
    with torch.no_grad():
        train_embeddings = []
        for batch in train_dataloader:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            train_embeddings.append(embeddings.detach().cpu().numpy())

        test_embeddings = []
        for batch in test_dataloader:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            test_embeddings.append(embeddings.detach().cpu().numpy())

    train_embeddings = np.concatenate(train_embeddings)
    test_embeddings = np.concatenate(test_embeddings)

    # Step 5: Train a classifier
    classifier = SVC(kernel='rbf')
    classifier.fit(train_embeddings, y_train)

    # Step 6: Evaluate the model
    y_pred = classifier.predict(test_embeddings)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

progress_bar.close()
# %%
