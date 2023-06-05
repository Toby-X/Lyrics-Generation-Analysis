# %%
import torch
from torch import nn, optim
from torch.nn import functional as F

# %%
# =============================================================================
# 1. Define the model
# =============================================================================

# Define the hyperparameters
embed_dim = 500
sequence_len = 100
tfidf_dim = 50000
input_dim = embed_dim * sequence_len + tfidf_dim
hidden_dim = 400
latent_dim = 20
num_epochs = 20
batch_size = 32
learning_rate = 1e-3


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim*2)  # We need 2 times the latent dim for mean and variance
        )
        
        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Use sigmoid to output probabilities
        )

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded[:, :latent_dim], encoded[:, latent_dim:]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std  #TODO: Reparameterization trick

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# %%
# =============================================================================
# 2. Load the data
# =============================================================================
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import Dataset, DataLoader,random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import os

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


class LyricsDataset(Dataset):
    def __init__(self, lyrics, labels,vectorizer, num_words=100000, sequence_length=sequence_len,embedding_dim=embed_dim):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(lyrics)

        # TF-IDF Vectorizer
        self.tfidf = vectorizer
        tfidf_matrix = self.tfidf.fit_transform(lyrics)
        print("TF-IDF matrix shape: ", tfidf_matrix.shape)

        # Word2Vec
        if os.path.exists(f"../models/word2vec_{embed_dim}.model"):
            print(f"Loading Word2Vec model_{embed_dim}...")
            self.w2v_model = Word2Vec.load(f"../models/word2vec.model")
        else:
            print(f"Training Word2Vec model_{embed_dim}...")
            tokenized_lyrics = [word_tokenize(sentence) for sentence in lyrics]
            self.w2v_model = Word2Vec(sentences=tokenized_lyrics,vector_size=embedding_dim, window=10, min_count=1, workers=4, sg=1)
            self.w2v_model.train(tokenized_lyrics, total_examples=len(tokenized_lyrics), epochs=20)
            self.w2v_model.save(f"../models/word2vec_{embed_dim}.model")

        embedding_matrix = np.zeros((num_words, embedding_dim))

        count = 0
        for word, i in self.tokenizer.word_index.items():
            if word in self.w2v_model.wv:
                count += 1
                embedding_matrix[i] = self.w2v_model.wv[word]
        
        print("Number of words in Word2Vec vocabulary: ", count)

        sequences = self.tokenizer.texts_to_sequences(lyrics)
        sequences = pad_sequences(sequences, maxlen=self.sequence_length)
        print('sequences shape:', sequences.shape)

        # Reshape the embedding matrix
        reshaped_embedding = embedding_matrix[sequences].reshape(len(sequences), -1)
        print('reshaped_embedding shape:', reshaped_embedding.shape)

        tfidf_matrix = tfidf_matrix.toarray()
        self.x_data = np.concatenate([tfidf_matrix, reshaped_embedding], axis=1)
        print('x_data shape:', self.x_data.shape)
        self.y_data = labels
        self.n_samples = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# split the data into training and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(lyrics, labels, test_size=0.2, random_state=42)

# %%
# =============================================================================
# 3. Train the model
# =============================================================================

model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
criterion = nn.KLDivLoss(reduction='batchmean')
model.to(device)

def train(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data[0].to(device).float(), data[1].to(device)
        optimizer.zero_grad()
        outputs, mu, logvar = model(inputs)

        rec_loss = F.binary_cross_entropy(outputs, inputs, reduction='sum')
        kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # print("rec_loss: ", rec_loss.item())
        # print("kl_div_loss: ", kl_div_loss.item())

        loss = rec_loss + kl_div_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data[0].to(device).float(), data[1].to(device)
            outputs,mu,logvar = model(inputs)

            rec_loss = F.binary_cross_entropy(outputs, inputs, reduction='sum')
            kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # print("rec_loss: ", rec_loss.item())
            # print("kl_div_loss: ", kl_div_loss.item())

            loss = rec_loss + kl_div_loss
            running_loss += loss.item()
    return running_loss / len(dataloader)

# %% Begin training
for epoch in range(num_epochs):
    # split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    # Create the vectorizer
    vectorizer = TfidfVectorizer(max_features=tfidf_dim, ngram_range=(1, 2), stop_words='english')

    # Fit on the training data
    vectorizer.fit(X_train)
    train_data = LyricsDataset(X_train, y_train,vectorizer=vectorizer)
    valid_data = LyricsDataset(X_val, y_val,vectorizer=vectorizer)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    train_loss = train(model, criterion, optimizer, train_loader, device)
    val_loss = evaluate(model, criterion, valid_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs}.. Train loss: {train_loss}.. Validation loss: {val_loss}")

# =============================================================================
# 4. Evaluate the model
# =============================================================================

test_data = LyricsDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
test_loss = evaluate(model, criterion, test_loader, device)
print(f"Final Test loss: {test_loss}")
# %%
