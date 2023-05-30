# %%
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora
from collections import defaultdict
import pprint
import re
from gensim import models
from scipy.sparse import lil_matrix, hstack, csr_matrix, vstack
import gensim.downloader as api
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# %% [markdown]
# # Data Preprocessing
# In this section, we preprocess the data and transform raw text data to matrix form. Then, all data is divided into training set and test set. After that, a dictionary is built upon training set.

# %%
def specific_preprocess(doc):
    return simple_preprocess(doc,min_len=2)

def remove_specific_words(s):
    s = re.sub(r"\bLyrics"," ",s)
    s = re.sub(r"\[.+\]"," ",s)
    s = re.sub(r"\b\d+\b Contributors"," ",s)
    s = re.sub(r"Embed"," ",s)
    s = re.sub(r"You might also like"," ",s)
    return s

def count_space(s):
    return s.count(' ')

def remove_short_words(s):
    s = re.sub(r"\b..\b"," ",s)
    s = re.sub(r"\b . \b"," ",s)
    pronoun = [r"\b you\b",r"\b yours\b",r"\b him \b",r"\b his\b", r"\b she \b", r"\b her \b", r"\b hers\b",
               r"\b they \b", r"\b them \b", r"\b their \b", r"\b theirs \b",r"\b You\b",r"\b Yours\b",
               r"\b Him \b",r"\b His\b", r"\b She \b", r"\b Her \b", r"\b Hers\b",
               r"\b They \b", r"\b Them \b", r"\b Their \b", r"\b Theirs \b"]
    conj = [r"\b and \b", r"\b then \b",r"\b for\b", r"\b from\b", r"\b with\b",
            r"\b about\b",r"\b And \b", r"\b Then \b",r"\b For\b", r"\b From\b", r"\b With\b",
            r"\b About\b"]
    for word in pronoun:
        s = re.sub(word," ",s)
    for word in conj:
        s = re.sub(word," ",s)
    return s

def count_lines(s):
    res = len(re.findall(r"\r\n",s))
    return res

def count_paras(s):
    res = len(re.findall(r"\r\n\r\n",s))
    return res

df = pd.read_csv("data/billboard_lyrics_genres.csv")
df_activeyear = pd.read_csv("data/first_active_years.csv")
df_activeyear = df_activeyear.drop_duplicates(subset=["band_singer","title","year"],ignore_index=True)
df["active_years"] = 0

for i in range(df.shape[0]):
    ay_tmp = df_activeyear.loc[(df_activeyear["band_singer"]==df.loc[i,"band_singer"])&(df_activeyear["title"]==df.loc[i,"title"])&(df_activeyear["year"]==df.loc[i,"year"])].active_years
    if not ay_tmp.empty:
        df.loc[i,"active_years"] = int(ay_tmp)

df_tmp = df.loc[df["active_years"]!=0].reset_index(drop=True)

df["numword"] = df["lyrics"].map(count_space)
df["num_lines"] = df["lyrics"].map(count_lines)+1
df["num_paras"] = df["lyrics"].map(count_paras)+1
df["av_word_line"] = df["numword"]/df["num_lines"]
df["av_word_paras"] = df["numword"]/df["num_paras"]
df["lyrics"] = df["lyrics"].map(remove_specific_words)
df["lyrics"] = df["lyrics"].map(remove_stopwords)
# df["lyrics"] = df["lyrics"].map(remove_short_words)
# df.to_csv("data/df_cluster.csv")
df["lyrics"] = df["lyrics"].map(specific_preprocess)

# %%
def find_unique_word(L):
    unique_words = []
    for word in L:
        if word not in unique_words:
            unique_words.append(word)
    return len(unique_words)

def find_max_len(L):
    max_len = list(map(len,L))
    return max(max_len)

df["unique_words"] = df["lyrics"].map(find_unique_word)
df["max_len"] = df["lyrics"].map(find_max_len)

# %% [markdown]
# Then delete the songs that are not English

# %%
def isEnglish(w):
    return w.encode("utf-8").isalpha()

def isListEnglish(L):
    return all(map(isEnglish,L))

df["isEnglish"] = df["lyrics"].map(isListEnglish)
df = df.loc[df["isEnglish"],:]

# %% [markdown]
# Similarly, perform the same procedure to genre

# %%
def remove_pun(s):
    s = re.sub(r"\[\'"," ",s)
    s = re.sub(r"\'\]"," ",s)
    s = re.sub(r"\'"," ",s)
    s = re.sub(r"\[\]"," ",s)
    s = re.sub(r"\,"," ",s)
    s = s.split()
    s = [token.lower() for token in s]
    return s


df["genre"] = df["genre"].map(remove_pun)

# %%
freq_gen = defaultdict(int)
for text in df["genre"]:
    for token in text:
        freq_gen[token] += 1

processed_corpus_gen = [[token for token in text if freq_gen[token]>20] for text in df.loc[:,"genre"]]
dict_gen = corpora.Dictionary(processed_corpus_gen)
freq_wanted = {k: v for k,v in freq_gen.items() if v > 100}
pprint.pprint(freq_wanted)

# %% [markdown]
# In this way, we can sort out the genre we want is alternative, country, dance, disco, folk, funk, hip-hop, new wave, pop, r&b, rap, rock, soul (soft stands for soft rock)

# %%
gen_des = ["alternative","country","dance","disco","folk","funk","hip","new","pop","r&b","rap","rock","soul"]
gen_des = sorted(gen_des)

# Compute number of columns from maximum word ID in the training data
num_cols = len(gen_des)
dat_gen = lil_matrix((len(df), num_cols), dtype=np.int64)

# Fill in values using apply() and enumerate()
def set_row_func(i, row):
    for word in row["genre"]:
        for k in range(len(gen_des)):
            if re.search(gen_des[k],word):
                dat_gen[i,k] = 1
df[df["genre"].map(len) > 0].reset_index(drop=True).reset_index().apply(lambda row: set_row_func(row["index"], row), axis=1)

# Convert to pandas DataFrame
dat_gen = pd.DataFrame.sparse.from_spmatrix(dat_gen)

# %% [markdown]
# Then, we should tag the data for classification.

# %%
df["label"] = np.zeros(df.shape[0])

bins = [1970,1980,1990,2000,2010,np.inf]

labels = [0,1,2,3,4,5]

df["label"] = np.where(df["year"] < bins[0], labels[0],
                               np.where(df["year"] < bins[1], labels[1],
                                        np.where(df["year"] < bins[2], labels[2],
                                                 np.where(df["year"] < bins[3], labels[3],
                                                          np.where(df["year"] < bins[4], labels[4], labels[5])))))

# %% [markdown]
# Then, data is split to training set and test set.

# %%
np.random.seed(515)
idx = np.repeat(range(10),len(df.iloc[:,0])//10+1)
df["idx"] = np.random.choice(idx[range(len(df.iloc[:,0]))],size=len(df.iloc[:,0]))
df_train = df.loc[df["idx"]!=0,:]
df_test = df.loc[df["idx"]==0,:]

# %%
num_train_dec = []
num_test_dec = []
for i in range(6):
    num_train_dec.append(np.sum(df_train["label"]==i))
    num_test_dec.append(np.sum(df_test["label"]==i))

print(num_train_dec)
print(num_test_dec)

# %% [markdown]
# Build a dictionary based on training set.

# %%
freq = defaultdict(int)
for text in df_train["lyrics"]:
    for token in text:
        freq[token] += 1

processed_corpus = [[token for token in text if freq[token]>20] for text in df_train.loc[:,"lyrics"]]
dictionary = corpora.Dictionary(processed_corpus)
df_train["freq_count"] = [dictionary.doc2bow(text) for text in processed_corpus]

# %%
# Compute number of columns from maximum word ID in the training data
num_cols = max(dictionary.keys())+1
dat_train = lil_matrix((len(df_train), num_cols), dtype=np.int64)

# Fill in values using apply() and enumerate()
def set_row_func(i, row):
    indices = [count for count, word_id in row["freq_count"]]
    values = [value for _, value in row["freq_count"]]
    dat_train[i, indices] = values
df_train[df_train["freq_count"].map(len) > 0].reset_index(drop=True).reset_index().apply(lambda row: set_row_func(row["index"], row), axis=1)

# Convert to pandas DataFrame
dat_train = pd.DataFrame.sparse.from_spmatrix(dat_train)

# %% [markdown]
# Then, perform the same procedure to test set with the dictionary.

# %%
df_test = df.loc[df["idx"]==0,:]
processed_corpus = [[token for token in text if freq[token]>20] for text in df_test.loc[:,"lyrics"]]
df_test["freq_count"] = [dictionary.doc2bow(text) for text in processed_corpus]

# Compute number of columns from maximum word ID in the training data
num_cols = max(dictionary.keys())+1
dat_test = lil_matrix((len(df_test), num_cols), dtype=np.int64)

# Fill in values using apply() and enumerate()
def set_row_func(i, row):
    indices = [count for count, word_id in row["freq_count"] if count < num_cols]
    values = [value for count, value in row["freq_count"] if count < num_cols and value!=0]
    dat_test[i, indices] = values
df_test[df_test["freq_count"].map(len) > 0].reset_index(drop=True).reset_index().apply(lambda row: set_row_func(row["index"], row), axis=1)

# Convert to pandas DataFrame
dat_test = pd.DataFrame.sparse.from_spmatrix(dat_test)

# %% [markdown]
# # Perform TF-IDF

# %%
bow_corpus = list(df_train["freq_count"])
tfidf = models.TfidfModel(bow_corpus)
df_train["tfidf"]=tfidf[df_train["lyrics"].map(dictionary.doc2bow)]

# %%
# Compute number of columns from maximum word ID in the training data
num_cols = max(dictionary.keys())+1
dat_tfidf_train = lil_matrix((len(df_train), num_cols), dtype=np.float64)

# Fill in values using apply() and enumerate()
def set_row_func(i, row):
    indices = [count for count, word_id in row["tfidf"]]
    values = [value for _, value in row["tfidf"]]
    dat_tfidf_train[i, indices] = values
df_train[df_train["tfidf"].map(len) > 0].reset_index(drop=True).reset_index().apply(lambda row: set_row_func(row["index"], row), axis=1)

# Convert to pandas DataFrame
dat_tfidf_train = pd.DataFrame.sparse.from_spmatrix(dat_tfidf_train)

# %%
df_test["tfidf"]=tfidf[df_test["lyrics"].map(dictionary.doc2bow)]

# Compute number of columns from maximum word ID in the training data
num_cols = max(dictionary.keys())+1
dat_tfidf_test = lil_matrix((len(df_test), num_cols), dtype=np.float64)

# Fill in values using apply() and enumerate()
def set_row_func(i, row):
    indices = [count for count, word_id in row["tfidf"] if count < num_cols]
    values = [value for count, value in row["tfidf"] if count < num_cols and value != 0]
    dat_tfidf_test[i, indices] = values
df_test[df_test["tfidf"].map(len) > 0].reset_index(drop=True).reset_index().apply(lambda row: set_row_func(row["index"], row), axis=1)

# Convert to pandas DataFrame
dat_tfidf_test = pd.DataFrame.sparse.from_spmatrix(dat_tfidf_test)

# %% [markdown]
# # Processed Data
# The data processed are diveded into the blow categories:
# 
# Original word frequency + genre
# 
# TF-IDF word frequency + genre
# 

# %%
dat_gen = dat_gen.reset_index()
df = df.reset_index(drop=True)
dat_gen_train = dat_gen.loc[df["idx"]!=0,:].reset_index(drop=True)
dat_gen_test = dat_gen.loc[df["idx"]==0,:].reset_index(drop=True)

# %%
train_ones = csr_matrix(np.ones(df_train.shape[0])).transpose()
test_ones = csr_matrix(np.ones(df_test.shape[0])).transpose()
train_label = csr_matrix(df_train.loc[:,"label"]).transpose()
test_label = csr_matrix(df_test.loc[:,"label"]).transpose()
train_activeyear = csr_matrix(df_train.loc[:,"active_years":"unique_words"])
test_activeyear = csr_matrix(df_test.loc[:,"active_years":"unique_words"])

gen_train = csr_matrix(dat_gen_train.loc[:,0:])
lyrics_train = csr_matrix(dat_train.loc[:,0:])
data_train = hstack([train_ones,gen_train, lyrics_train,train_activeyear,train_label])
data_train = pd.DataFrame.sparse.from_spmatrix(data_train)

gen_test = csr_matrix(dat_gen_test.loc[:,0:])
lyrics_test = csr_matrix(dat_test.loc[:,0:])
data_test = hstack([test_ones,gen_test, lyrics_test,test_activeyear,test_label])
data_test = pd.DataFrame.sparse.from_spmatrix(data_test)


lyrics_tfidf_train = csr_matrix(dat_tfidf_train.loc[:,0:])
data_tfidf_train = hstack([train_ones,gen_train,lyrics_tfidf_train,train_activeyear,train_label])
data_tfidf_train = pd.DataFrame.sparse.from_spmatrix(data_tfidf_train)

lyrics_tfidf_test = csr_matrix(dat_tfidf_test.loc[:,0:])
data_tfidf_test = hstack([test_ones,gen_test,lyrics_tfidf_test,test_activeyear,test_label])
data_tfidf_test = pd.DataFrame.sparse.from_spmatrix(data_tfidf_test)

# %%
df_tmp = df_train.loc[:,"active_years":"unique_words"]
word_name = [dictionary[i] for i in range(max(dictionary.keys())+1)]
word_name = ['intercept']+gen_des + word_name +list(df_tmp.columns)+ ['label']
data_tfidf_test.columns = word_name
data_tfidf_train.columns = word_name
data_train.columns = word_name
data_test.columns = word_name

# %%
# data_tfidf_train.to_csv("data/train_tfidf_data.csv")
# data_tfidf_test.to_csv("data/test_tfidf_data.csv")
# # data_train = hstack([lyrics_train,train_label])
# # data_train = pd.DataFrame.sparse.from_spmatrix(data_train)
# # data_test = hstack([lyrics_test,test_label])
# # data_test = pd.DataFrame.sparse.from_spmatrix(data_test)
# # word_name = [dictionary[i] for i in range(max(dictionary.keys())+1)]
# # word_name = word_name+['label']
# # data_train.columns = word_name
# # data_test.columns = word_name

# data_train.to_csv("data/train_data_all.csv")
# data_test.to_csv("data/test_data_all.csv")
# df_train.to_csv("data/train_other.csv")

# %% [markdown]
# # Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression

mr = LogisticRegression(penalty='l2',solver="liblinear",max_iter=1000).fit(data_tfidf_train.iloc[:,:(data_tfidf_train.shape[1]-1)],np.array(df_train["label"]))
pred = mr.predict(data_tfidf_test.iloc[:,:(data_tfidf_train.shape[1]-1)])

print(sum(pred == df_test["label"])/len(pred))
print(np.mean((pred-df_test["label"])**2))

# %%
mr = LogisticRegression(penalty='l2',solver="liblinear",max_iter=1000).fit(data_tfidf_train.iloc[:,[0,(data_tfidf_train.shape[1]-2)]],np.array(df_train["label"]))
pred = mr.predict(data_tfidf_test.iloc[:,[0,(data_tfidf_train.shape[1]-2)]])

print(sum(pred == df_test["label"])/len(pred))

# %% [markdown]
# Compared with $l_1$ penalty, $l_2$ is better.

# %%
mr = LogisticRegression(penalty='l1',solver='liblinear',max_iter=100000).fit(data_tfidf_train.iloc[:,:(data_tfidf_train.shape[1]-1)],np.array(df_train["label"]))
pred = mr.predict(data_tfidf_test.iloc[:,:(data_tfidf_train.shape[1]-1)])

print(sum(pred == df_test["label"])/len(pred))
print(np.mean((pred-df_test["label"])**2))

# %%
mr = LogisticRegression(penalty='l2',solver="liblinear").fit(data_tfidf_train.iloc[:,:len(gen_des)],np.array(df_train["label"]))
pred = mr.predict(data_tfidf_test.iloc[:,:len(gen_des)])

print(sum(pred == df_test["label"])/len(pred))
print(np.mean((pred-df_test["label"])**2))

# %%
mr = LogisticRegression(penalty='l2',solver="liblinear").fit(data_tfidf_train.iloc[:,len(gen_des):(data_tfidf_train.shape[1]-1)],np.array(df_train["label"]))
pred = mr.predict(data_tfidf_test.iloc[:,len(gen_des):(data_tfidf_train.shape[1]-1)])

print(sum(pred == df_test["label"])/len(pred))
print(np.mean((pred-df_test["label"])**2))

# %%
mr = LogisticRegression(penalty='l1',solver="liblinear").fit(data_train.iloc[:,:(data_train.shape[1]-1)],np.array(df_train["label"]))
pred = mr.predict(data_test.iloc[:,:(data_train.shape[1]-1)])

print(sum(pred == df_test["label"])/len(pred))
print(np.mean((pred-df_test["label"])**2))

# %%
from sklearn.preprocessing import MaxAbsScaler

transformer = MaxAbsScaler()
transformer.fit(data_train.iloc[:,:(data_train.shape[1]-1)])
data_train_scaled = transformer.transform(data_train.iloc[:,:(data_train.shape[1]-1)])
data_test_scaled = transformer.transform(data_test.iloc[:,:(data_train.shape[1]-1)])
data_train_scaled = hstack([data_train_scaled,train_label])
data_test_scaled = hstack([data_test_scaled,test_label])
data_train_scaled = pd.DataFrame.sparse.from_spmatrix(data_train_scaled)
data_test_scaled = pd.DataFrame.sparse.from_spmatrix(data_test_scaled)

# %%
mr = LogisticRegression(penalty='l2',solver="liblinear",max_iter=10000).fit(data_train_scaled.iloc[:,:(data_train_scaled.shape[1]-1)],np.array(df_train["label"]))
pred = mr.predict(data_test_scaled.iloc[:,:(data_test_scaled.shape[1]-1)])

print(sum(pred == df_test["label"])/len(pred))

# %%
mr_coef_all = mr.coef_
mr_coef_0 = np.argsort(mr_coef_all[0,14:(mr_coef_all.shape[1]-2)])+14
mr_coef_0 = mr_coef_0[::-1]
mr_word_0 = [word_name[mr_coef_0[i]] for i in range(100)]
word_freq_0 = [round(mr_coef_all[0,mr_coef_0[i]]*100) for i in range(100)]
word_freq_0 = {mr_word_0[i]:word_freq_0[i] for i in range(100)}

# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud1 = WordCloud(background_color="white")
wordcloud0 = wordcloud1.generate_from_frequencies(word_freq_0)
plt.imshow(wordcloud0,interpolation="bilinear")
plt.axis("off")
plt.show()

# %%


# %% [markdown]
# Then, it may be better to see the result for 2 logistic regression after some decades apart.

# %%
# data preparation
data_train_scaled_60 = data_train_scaled.loc[data_train_scaled.iloc[:,(data_train_scaled.shape[1]-1)]==0,:]
data_train_scaled_70 = data_train_scaled.loc[data_train_scaled.iloc[:,(data_train_scaled.shape[1]-1)]==1,:]
data_train_scaled_80 = data_train_scaled.loc[data_train_scaled.iloc[:,(data_train_scaled.shape[1]-1)]==2,:]
data_train_scaled_90 = data_train_scaled.loc[data_train_scaled.iloc[:,(data_train_scaled.shape[1]-1)]==3,:]
data_train_scaled_00 = data_train_scaled.loc[data_train_scaled.iloc[:,(data_train_scaled.shape[1]-1)]==4,:]
data_train_scaled_10 = data_train_scaled.loc[data_train_scaled.iloc[:,(data_train_scaled.shape[1]-1)]==5,:]

data_test_scaled_60 = data_test_scaled.loc[data_test_scaled.iloc[:,(data_test_scaled.shape[1]-1)]==0,:]
data_test_scaled_70 = data_test_scaled.loc[data_test_scaled.iloc[:,(data_test_scaled.shape[1]-1)]==1,:]
data_test_scaled_80 = data_test_scaled.loc[data_test_scaled.iloc[:,(data_test_scaled.shape[1]-1)]==2,:]
data_test_scaled_90 = data_test_scaled.loc[data_test_scaled.iloc[:,(data_test_scaled.shape[1]-1)]==3,:]
data_test_scaled_00 = data_test_scaled.loc[data_test_scaled.iloc[:,(data_test_scaled.shape[1]-1)]==4,:]
data_test_scaled_10 = data_test_scaled.loc[data_test_scaled.iloc[:,(data_test_scaled.shape[1]-1)]==5,:]

# %%
# 60s and 70s
data_train_scaled_67 = pd.concat([data_train_scaled_60,data_train_scaled_70],axis=0)
data_test_scaled_67 = pd.concat([data_test_scaled_60,data_test_scaled_70],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_67.iloc[:,:(data_train_scaled_67.shape[1]-1)],data_train_scaled_67.iloc[:,(data_train_scaled_67.shape[1]-1)])

pred = mr.predict(data_test_scaled_67.iloc[:,:(data_test_scaled_67.shape[1]-1)])
print(sum(pred == data_test_scaled_67.iloc[:,(data_test_scaled_67.shape[1]-1)])/len(pred))

# %%
# 60s and 80s
data_train_scaled_68 = pd.concat([data_train_scaled_60,data_train_scaled_80],axis=0)
data_test_scaled_68 = pd.concat([data_test_scaled_60,data_test_scaled_80],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_68.iloc[:,:(data_train_scaled_68.shape[1]-1)],data_train_scaled_68.iloc[:,(data_train_scaled_68.shape[1]-1)])

pred = mr.predict(data_test_scaled_68.iloc[:,:(data_test_scaled_68.shape[1]-1)])
print(sum(pred == data_test_scaled_68.iloc[:,(data_test_scaled_68.shape[1]-1)])/len(pred))

# %%
# 60s and 90s
data_train_scaled_69 = pd.concat([data_train_scaled_60,data_train_scaled_90],axis=0)
data_test_scaled_69 = pd.concat([data_test_scaled_60,data_test_scaled_90],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_69.iloc[:,:(data_train_scaled_69.shape[1]-1)],data_train_scaled_69.iloc[:,(data_train_scaled_69.shape[1]-1)])

pred = mr.predict(data_test_scaled_69.iloc[:,:(data_test_scaled_69.shape[1]-1)])
print(sum(pred == data_test_scaled_69.iloc[:,(data_test_scaled_69.shape[1]-1)])/len(pred))

# %%
# 60s and 00s
data_train_scaled_600 = pd.concat([data_train_scaled_60,data_train_scaled_00],axis=0)
data_test_scaled_600 = pd.concat([data_test_scaled_60,data_test_scaled_00],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_600.iloc[:,:(data_train_scaled_600.shape[1]-1)],data_train_scaled_600.iloc[:,(data_train_scaled_600.shape[1]-1)])

pred = mr.predict(data_test_scaled_600.iloc[:,:(data_test_scaled_600.shape[1]-1)])
print(sum(pred == data_test_scaled_600.iloc[:,(data_test_scaled_600.shape[1]-1)])/len(pred))

# %%
# 60s and 10s
data_train_scaled_61 = pd.concat([data_train_scaled_60,data_train_scaled_10],axis=0)
data_test_scaled_61 = pd.concat([data_test_scaled_60,data_test_scaled_10],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_61.iloc[:,:(data_train_scaled_61.shape[1]-1)],data_train_scaled_61.iloc[:,(data_train_scaled_61.shape[1]-1)])

pred = mr.predict(data_test_scaled_61.iloc[:,:(data_test_scaled_61.shape[1]-1)])
print(sum(pred == data_test_scaled_61.iloc[:,(data_test_scaled_61.shape[1]-1)])/len(pred))

# %%
# 70s and 80s
data_train_scaled_78 = pd.concat([data_train_scaled_70,data_train_scaled_80],axis=0)
data_test_scaled_78 = pd.concat([data_test_scaled_70,data_test_scaled_80],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_78.iloc[:,:(data_train_scaled_78.shape[1]-1)],data_train_scaled_78.iloc[:,(data_train_scaled_78.shape[1]-1)])

pred = mr.predict(data_test_scaled_78.iloc[:,:(data_test_scaled_78.shape[1]-1)])
print(sum(pred == data_test_scaled_78.iloc[:,(data_test_scaled_78.shape[1]-1)])/len(pred))

# %%
# 70s and 90s
data_train_scaled_79 = pd.concat([data_train_scaled_70,data_train_scaled_90],axis=0)
data_test_scaled_79 = pd.concat([data_test_scaled_70,data_test_scaled_90],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_79.iloc[:,:(data_train_scaled_79.shape[1]-1)],data_train_scaled_79.iloc[:,(data_train_scaled_79.shape[1]-1)])

pred = mr.predict(data_test_scaled_79.iloc[:,:(data_test_scaled_79.shape[1]-1)])
print(sum(pred == data_test_scaled_79.iloc[:,(data_test_scaled_79.shape[1]-1)])/len(pred))

# %%
# 70s and 00s
data_train_scaled_700 = pd.concat([data_train_scaled_70,data_train_scaled_00],axis=0)
data_test_scaled_700 = pd.concat([data_test_scaled_70,data_test_scaled_00],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_700.iloc[:,:(data_train_scaled_700.shape[1]-1)],data_train_scaled_700.iloc[:,(data_train_scaled_700.shape[1]-1)])

pred = mr.predict(data_test_scaled_700.iloc[:,:(data_test_scaled_700.shape[1]-1)])
print(sum(pred == data_test_scaled_700.iloc[:,(data_test_scaled_700.shape[1]-1)])/len(pred))

# %%
# 70s and 10s
data_train_scaled_71 = pd.concat([data_train_scaled_70,data_train_scaled_10],axis=0)
data_test_scaled_71 = pd.concat([data_test_scaled_70,data_test_scaled_10],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_71.iloc[:,:(data_train_scaled_71.shape[1]-1)],data_train_scaled_71.iloc[:,(data_train_scaled_71.shape[1]-1)])

pred = mr.predict(data_test_scaled_71.iloc[:,:(data_test_scaled_71.shape[1]-1)])
print(sum(pred == data_test_scaled_71.iloc[:,(data_test_scaled_71.shape[1]-1)])/len(pred))

# %%
# 80s and 90s
data_train_scaled_89 = pd.concat([data_train_scaled_80,data_train_scaled_90],axis=0)
data_test_scaled_89 = pd.concat([data_test_scaled_80,data_test_scaled_90],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_89.iloc[:,:(data_train_scaled_89.shape[1]-1)],data_train_scaled_89.iloc[:,(data_train_scaled_89.shape[1]-1)])

pred = mr.predict(data_test_scaled_89.iloc[:,:(data_test_scaled_89.shape[1]-1)])
print(sum(pred == data_test_scaled_89.iloc[:,(data_test_scaled_89.shape[1]-1)])/len(pred))

# %%
# 80s and 00s
data_train_scaled_800 = pd.concat([data_train_scaled_80,data_train_scaled_00],axis=0)
data_test_scaled_800 = pd.concat([data_test_scaled_80,data_test_scaled_00],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_800.iloc[:,:(data_train_scaled_800.shape[1]-1)],data_train_scaled_800.iloc[:,(data_train_scaled_800.shape[1]-1)])

pred = mr.predict(data_test_scaled_800.iloc[:,:(data_test_scaled_800.shape[1]-1)])
print(sum(pred == data_test_scaled_800.iloc[:,(data_test_scaled_800.shape[1]-1)])/len(pred))

# %%
# 80s and 10s
data_train_scaled_81 = pd.concat([data_train_scaled_80,data_train_scaled_10],axis=0)
data_test_scaled_81 = pd.concat([data_test_scaled_80,data_test_scaled_10],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_81.iloc[:,:(data_train_scaled_81.shape[1]-1)],data_train_scaled_81.iloc[:,(data_train_scaled_81.shape[1]-1)])

pred = mr.predict(data_test_scaled_81.iloc[:,:(data_test_scaled_81.shape[1]-1)])
print(sum(pred == data_test_scaled_81.iloc[:,(data_test_scaled_81.shape[1]-1)])/len(pred))

# %%
# 90s and 00s
data_train_scaled_900 = pd.concat([data_train_scaled_90,data_train_scaled_00],axis=0)
data_test_scaled_900 = pd.concat([data_test_scaled_90,data_test_scaled_00],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_900.iloc[:,:(data_train_scaled_900.shape[1]-1)],data_train_scaled_900.iloc[:,(data_train_scaled_900.shape[1]-1)])

pred = mr.predict(data_test_scaled_900.iloc[:,:(data_test_scaled_900.shape[1]-1)])
print(sum(pred == data_test_scaled_900.iloc[:,(data_test_scaled_900.shape[1]-1)])/len(pred))

# %%
# 90s and 10s
data_train_scaled_91 = pd.concat([data_train_scaled_90,data_train_scaled_10],axis=0)
data_test_scaled_91 = pd.concat([data_test_scaled_90,data_test_scaled_10],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_91.iloc[:,:(data_train_scaled_91.shape[1]-1)],data_train_scaled_91.iloc[:,(data_train_scaled_91.shape[1]-1)])

pred = mr.predict(data_test_scaled_91.iloc[:,:(data_test_scaled_91.shape[1]-1)])
print(sum(pred == data_test_scaled_91.iloc[:,(data_test_scaled_91.shape[1]-1)])/len(pred))

# %%
# 00s and 10s
data_train_scaled_01 = pd.concat([data_train_scaled_00,data_train_scaled_10],axis=0)
data_test_scaled_01 = pd.concat([data_test_scaled_00,data_test_scaled_10],axis=0)
mr = LogisticRegression(penalty="l2",solver="liblinear")
mr.fit(data_train_scaled_01.iloc[:,:(data_train_scaled_01.shape[1]-1)],data_train_scaled_01.iloc[:,(data_train_scaled_01.shape[1]-1)])

pred = mr.predict(data_test_scaled_01.iloc[:,:(data_test_scaled_01.shape[1]-1)])
print(sum(pred == data_test_scaled_01.iloc[:,(data_test_scaled_01.shape[1]-1)])/len(pred))

# %%
import seaborn as sns

div_mat = np.zeros((6,6))
div_mat[0,1] = 0.6084
div_mat[0,2] = 0.7533
div_mat[0,3] = 0.8742
div_mat[0,4] = 0.8993
div_mat[0,5] = 0.8855
div_mat[1,2] = 0.7403
div_mat[1,3] = 0.8773
div_mat[1,4] = 0.9085
div_mat[1,5] = 0.9
div_mat[2,3] = 0.8435
div_mat[2,4] = 0.8686
div_mat[2,5] = 0.8766
div_mat[3,4] = 0.7466
div_mat[3,5] = 0.8221
div_mat[4,5] = 0.7712
for i in range(6):
    for j in range(i):
        div_mat[i,j] = div_mat[j,i]

div_mat = pd.DataFrame(div_mat)
div_mat.columns = ["60s","70s","80s","90s","00s","10s"]
div_mat.index = ["60s","70s","80s","90s","00s","10s"]

sns.heatmap(div_mat)

# %% [markdown]
# From the above result, it seems that 60s, 70s and 80s are more similar compared with other decades. The following section focuses on combining the three decades

# %%
train_label_com = np.zeros((data_train_scaled.shape[0],1))
test_label_com = np.zeros((data_test_scaled.shape[0],1))
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

for i in range(df_train.shape[0]):
    if df_train.loc[i,"label"] <= 2:
        train_label_com[i] = 0
    else:
        train_label_com[i] = 1

for i in range(df_test.shape[0]):
    if df_test.loc[i,"label"] <= 2:
        test_label_com[i] = 0
    else:
        test_label_com[i] = 1

train_label_com = train_label_com.reshape((-1,))
test_label_com = test_label_com.reshape((-1,))

# %%
train_label_com = np.zeros((data_train_scaled.shape[0],1))
test_label_com = np.zeros((data_test_scaled.shape[0],1))
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

for i in range(df_train.shape[0]):
    if df_train.loc[i,"label"] <= 2:
        train_label_com[i] = 0
    elif df_train.loc[i,"label"] == 3:
        train_label_com[i] = 1
    elif df_train.loc[i,"label"] == 4:
        train_label_com[i] = 2
    elif df_train.loc[i,"label"] == 5:
        train_label_com[i] = 3

for i in range(df_test.shape[0]):
    if df_test.loc[i,"label"] <= 2:
        test_label_com[i] = 0
    elif df_test.loc[i,"label"] == 3:
        test_label_com[i] = 1
    elif df_test.loc[i,"label"] == 4:
        test_label_com[i] = 2
    elif df_test.loc[i,"label"] == 5:
        test_label_com[i] = 3

train_label_com = train_label_com.reshape((-1,))
test_label_com = test_label_com.reshape((-1,))

# %%
from sklearn.model_selection import GridSearchCV

parameters = {'C':1/ np.log(np.linspace(np.exp(1e-3),np.exp(1.5),num=100,dtype=np.float64)),'penalty':["l2"]}
logclf = LogisticRegression(solver="liblinear",max_iter=10000)
clf = GridSearchCV(logclf,param_grid=parameters,
                   n_jobs=6)
clf.fit(data_tfidf_train.iloc[:,:(data_train_scaled.shape[1]-1)],df_train["label"])

# %%
plt.plot(1/ np.log(np.linspace(np.exp(1e-3),np.exp(1.5),num=100,dtype=np.float64)),clf.cv_results_['split4_test_score'])

# %%
clf.best_estimator_

# %%
# from sklearn.feature_selection import SequentialFeatureSelector

# mr = LogisticRegression(penalty='l2',solver="liblinear")
# sfs = SequentialFeatureSelector(mr, n_features_to_select='auto', tol=1e-4, scoring='balanced_accuracy',n_jobs=7)
# sfs.fit(data_train_scaled.iloc[:,:(data_train_scaled.shape[1]-1)],df_train["label"])

# %%
mr = LogisticRegression(penalty='l1',solver="liblinear").fit(data_train_scaled.iloc[:,:(data_train_scaled.shape[1]-1)],train_label_com)
pred = mr.predict(data_test_scaled.iloc[:,:(data_test_scaled.shape[1]-1)])

print(sum(pred == test_label_com)/len(pred))

# %%
mr_coef_all = mr.coef_
mr_coef_0 = np.argsort(mr_coef_all[0,:(mr_coef_all.shape[1]-2)])
mr_coef_0 = mr_coef_0[::-1]
mr_word_0 = [word_name[mr_coef_0[i]] for i in range(200) if mr_coef_all[0,mr_coef_0[i]]>0]
word_freq_0 = [round(mr_coef_all[0,mr_coef_0[i]]*100) for i in range(200) if mr_coef_all[0,mr_coef_0[i]]>0]
word_freq_0 = {mr_word_0[i]:word_freq_0[i] for i in range(len(mr_word_0))}
wordcloud1 = WordCloud(background_color="white")
wordcloud0 = wordcloud1.generate_from_frequencies(word_freq_0)
plt.imshow(wordcloud0,interpolation="bilinear")
plt.axis("off")
plt.show()

# %%
mr_coef_all = mr.coef_
mr_coef_0 = np.argsort(mr_coef_all[0,:(mr_coef_all.shape[1]-2)])
mr_word_0 = [word_name[mr_coef_0[i]] for i in range(200) if mr_coef_all[0,mr_coef_0[i]]<0]
word_freq_0 = [-round(mr_coef_all[0,mr_coef_0[i]]*100) for i in range(200) if mr_coef_all[0,mr_coef_0[i]]<0]
word_freq_0 = {mr_word_0[i]:word_freq_0[i] for i in range(len(mr_word_0))}
wordcloud1 = WordCloud(background_color="white")
wordcloud0 = wordcloud1.generate_from_frequencies(word_freq_0)
plt.imshow(wordcloud0,interpolation="bilinear")
plt.axis("off")
plt.show()

# %%
mr_coef_all = mr.coef_
mr_coef_0 = np.argsort(mr_coef_all[0,14:(mr_coef_all.shape[1]-2)])+14
mr_coef_0 = mr_coef_0[::-1]
mr_word_0 = [word_name[mr_coef_0[i]] for i in range(100)]
word_freq_0 = [round(mr_coef_all[0,mr_coef_0[i]]*100) for i in range(100)]
word_freq_0 = {mr_word_0[i]:word_freq_0[i] for i in range(100)}

# %%
wordcloud1 = WordCloud(background_color="white")
wordcloud0 = wordcloud1.generate_from_frequencies(word_freq_0)
plt.imshow(wordcloud0,interpolation="bilinear")
plt.axis("off")
plt.show()

# %%
mr_coef_1 = np.argsort(mr_coef_all[1,14:(mr_coef_all.shape[1]-2)])+14
mr_coef_1 = mr_coef_1[::-1]
mr_word_1 = [word_name[mr_coef_1[i]] for i in range(100)]
word_freq_1 = [round(mr_coef_all[1,mr_coef_1[i]]*100) for i in range(100)]
word_freq_1 = {mr_word_1[i]:word_freq_1[i] for i in range(100)}
wordcloud1 = WordCloud(background_color="white")
wordcloud0 = wordcloud1.generate_from_frequencies(word_freq_1)
plt.imshow(wordcloud0,interpolation="bilinear")
plt.axis("off")
plt.show()

# %%
mr_coef_2 = np.argsort(mr_coef_all[2,14:(mr_coef_all.shape[1]-2)])+14
mr_coef_2 = mr_coef_2[::-1]
mr_word_2 = [word_name[mr_coef_2[i]] for i in range(100)]
word_freq_2 = [round(mr_coef_all[2,mr_coef_2[i]]*100) for i in range(100)]
word_freq_2 = {mr_word_2[i]:word_freq_2[i] for i in range(100)}
wordcloud1 = WordCloud(background_color="white")
wordcloud0 = wordcloud1.generate_from_frequencies(word_freq_2)
plt.imshow(wordcloud0,interpolation="bilinear")
plt.axis("off")
plt.show()

# %%
mr_coef_3 = np.argsort(mr_coef_all[3,14:(mr_coef_all.shape[1]-2)])+14
mr_coef_3 = mr_coef_3[::-1]
mr_word_3 = [word_name[mr_coef_3[i]] for i in range(100)]
word_freq_3 = [round(mr_coef_all[3,mr_coef_2[i]]*1000) for i in range(100)]
word_freq_3 = {mr_word_3[i]:word_freq_3[i] for i in range(100)}
wordcloud1 = WordCloud(background_color="white")
wordcloud0 = wordcloud1.generate_from_frequencies(word_freq_3)
plt.imshow(wordcloud0,interpolation="bilinear")
plt.axis("off")
plt.show()

# %%
df_train["label"] = np.zeros(df_train.shape[0])
bins = [1990,np.inf]
labels = [0,1]
df_train["label"] = np.where(df_train["year"] < bins[0], labels[0],labels[1])

df_test["label"] = np.zeros(df_test.shape[0])
df_test["label"] = np.where(df_test["year"] < bins[0], labels[0],labels[1])

# %%
mr = LogisticRegression(penalty='l2',dual=True,solver="liblinear").fit(data_tfidf_train.iloc[:,:len(gen_des)],df_train["label"])
pred = mr.predict(data_tfidf_test.iloc[:,:len(gen_des)])

print(sum(pred == df_test["label"])/len(pred))

# %%
from sklearn.model_selection import GridSearchCV

parameters = {'C':1/ np.log(np.linspace(np.exp(1e-3),np.exp(1.5),num=100,dtype=np.float64))}
logclf = LogisticRegression(penalty="l2",dual=True,solver="liblinear")
clf = GridSearchCV(logclf,param_grid=parameters,
                   scoring='f1',n_jobs=4)
clf.fit(data_tfidf_train.iloc[:,:(data_train.shape[1]-1)],df_train["label"])

# %%
pprint.pprint(clf.cv_results_)

# %%
logclf = LogisticRegression(dual=True,penalty="l2",solver="liblinear")
logclf.fit(np.array(data_tfidf_train.iloc[:,:(data_train.shape[1]-1)]),df_train["label"])
pred = logclf.predict(np.array(data_tfidf_test.iloc[:,:(data_test.shape[1]-1)]))

print(sum(pred == df_test["label"])/len(pred))
print("PPV",sum(pred[df_test["label"]==1]==1)/len(pred[df_test["label"]==1]))
print("PRECISION",sum(df_test.loc[pred==1,"label"]==1)/len(df_test.loc[pred==1,"label"]))

# %% [markdown]
# # Changing to generation
# Frequency may have some concerns

# %%
logclf = LogisticRegression(dual=True,penalty="l2",solver="liblinear")
logclf.fit(data_tfidf_train.iloc[:,len(gen_des):(data_train.shape[1]-1)],data_train.iloc[:,11])
pred = logclf.predict(data_tfidf_test.iloc[:,len(gen_des):(data_test.shape[1]-1)])

print(sum(pred == data_test.iloc[:,1])/len(pred))
print("PPV",sum(pred[data_test.iloc[:,11]==1]==1)/len(pred[data_test.iloc[:,11]==1]))
# print("PRECISION",sum(data_test.loc[pred==1,8]==1)/len(data_test.loc[pred==1,8]))

# %%
np.sum(data_tfidf_train.iloc[:,0])

# %%
pprint.pprint(np.sum(data_train.iloc[:,0:13],axis=0))

# %% [markdown]
# # Lemmatize and Bigram
# Perform Lemmatize and bigram to lyrics and perform logistic regression

# %%
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases

lemmatizer = WordNetLemmatizer()
def lemmatize(L):
    res = list(map(lemmatizer.lemmatize,L))
    return res
df["lyrics"]=df["lyrics"].map(lemmatize)

# %%
docs = list(df["lyrics"])
bigram = Phrases(docs,min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            docs[idx].append(token)

# %%
from gensim.corpora import Dictionary
dict_lda = Dictionary(docs)
# dict_lda.filter_extremes(no_below=20,no_above=.5)
dict_lda.filter_extremes(no_below=20)

corpus = [dict_lda.doc2bow(doc) for doc in docs]

# %%
print('Number of unique tokens: %d' % len(dict_lda))
print('Number of documents: %d' % len(corpus))

# %%
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# %%
from gensim.models import LdaModel

num_topics = 6
chunksize = 2000
passes = 20
iterations = 400
eval_every = None

temp = dict_lda[0]
id2word = dict_lda.id2token

model = LdaModel(
    corpus=corpus_tfidf,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

# %%
top_topics = model.top_topics(corpus)

pprint.pprint(top_topics)

# %%
num_dec = []

for i in range(6):
    num_dec.append(np.sum(df["label"]==i))

# %%
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# %%
for i in range(len(corpus)): #bow_corpus is the corpus
    if len(corpus[i])==0: #check for empty document
        print(i) #if there is any empty document then print the index of that document

# %%
del corpus_tfidf[1387]

# %%
from gensim.models import LdaSeqModel

ldaseq = LdaSeqModel(
    corpus=corpus_tfidf,
    id2word=dict_lda,
    time_slice=num_dec,
    num_topics=10,
)

# %% [markdown]
# # Changing another decade division
# Using information from the BERTopic model, extract the information from above.


