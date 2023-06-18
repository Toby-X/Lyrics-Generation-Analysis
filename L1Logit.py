# %% Import packages
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora
from collections import defaultdict
import re
from scipy.sparse import lil_matrix, hstack, csr_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# %%
# =============================================================================
# Preprocessing
# =============================================================================
def specific_preprocess(doc):
    return simple_preprocess(doc,min_len=2)

def remove_specific_words(s):
    s = re.sub(r"\bLyrics"," ",s)
    s = re.sub(r"\[.+\]"," ",s)
    s = re.sub(r"\b\d+\b Contributors"," ",s)
    s = re.sub(r"Embed"," ",s)
    s = re.sub(r"You might also like"," ",s)
    s = re.sub(r"active_years"," ",s,flags=re.IGNORECASE)
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
        s = re.sub(word," ",s,flags=re.IGNORECASE)
    for word in conj:
        s = re.sub(word," ",s,flags=re.IGNORECASE)
    return s

def count_lines(s):
    res = len(re.findall(r"\n",s))
    return res

def count_paras(s):
    res = len(re.findall(r"\n\n",s))
    return res


def delete_words(s):
    for word in delete_word_list.iloc[:, 0]:
        s = re.sub(r'\b' + re.escape(word) + r'\b', ' ', s, flags=re.IGNORECASE)
    return s

delete_word_list = pd.read_csv("data/delete_word_list.txt",index_col=0)

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
df['lyrics'] = df['lyrics'].map(delete_words)
df["lyrics"] = df["lyrics"].map(remove_short_words)
df["lyrics"] = df["lyrics"].map(specific_preprocess)
df.to_csv("data/df_deleted.csv")

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

# %%
def isEnglish(w):
    return w.encode("utf-8").isalpha()

def isListEnglish(L):
    return all(map(isEnglish,L))

df["isEnglish"] = df["lyrics"].map(isListEnglish)
df = df.loc[df["isEnglish"],:]

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

# %%
# =============================================================================
# Creating labels
# =============================================================================

# # 6 classes classification
# df["label"] = np.zeros(df.shape[0])
# bins = [1970,1980,1990,2000,2010,np.inf]
# labels = [0,1,2,3,4,5]
# df["label"] = np.where(df["year"] < bins[0], labels[0],
#                                np.where(df["year"] < bins[1], labels[1],
#                                         np.where(df["year"] < bins[2], labels[2],
#                                                  np.where(df["year"] < bins[3], labels[3],
#                                                           np.where(df["year"] < bins[4], labels[4], labels[5])))))

# 2 classes classification
df["label"] = np.zeros(df.shape[0])
df['label'] = np.where(df['year'] < 2000, 0, 1)

# # 4 classes classification
# df["label"] = np.zeros(df.shape[0])
# df['label'] = np.where(df['year'] < 1990, 0,
#                           np.where(df['year'] < 2000, 1,
#                                       np.where(df['year'] < 2010, 2, 3)))

# %%
# =============================================================================
# Train test split
# =============================================================================
np.random.seed(515)
idx = np.repeat(range(10),len(df.iloc[:,0])//10+1)
df["idx"] = np.random.choice(idx[range(len(df.iloc[:,0]))],size=len(df.iloc[:,0]))

df_train = df.loc[df["idx"]!=0,:]
df_test = df.loc[df["idx"]==0,:]

freq = defaultdict(int)
for text in df_train["lyrics"]:
    for token in text:
        freq[token] += 1
# =============================================================================
# Train data
# =============================================================================
processed_corpus = [[token for token in text if freq[token]>20] for text in df_train.loc[:,"lyrics"]]
dictionary = corpora.Dictionary(processed_corpus)
df_train["freq_count"] = [dictionary.doc2bow(text) for text in processed_corpus]

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

# =============================================================================
# Test data
# =============================================================================
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

dat_gen = dat_gen.reset_index()
df = df.reset_index(drop=True)
dat_gen_train = dat_gen.loc[df["idx"]!=0,:].reset_index(drop=True)
dat_gen_test = dat_gen.loc[df["idx"]==0,:].reset_index(drop=True)

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

df_tmp = df_train.loc[:,"active_years":"unique_words"]
word_name = [dictionary[i] for i in range(max(dictionary.keys())+1)]
word_name = ['intercept']+gen_des + word_name +list(df_tmp.columns)+ ['label']
data_train.columns = word_name
data_test.columns = word_name
colnames = list(df_tmp.columns) + ['label','intercept'] + gen_des

# %%
# =============================================================================
# Scaling and normalization
# =============================================================================
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
# =============================================================================
# Logistic Regression
# =============================================================================
# 6 classes classification
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Grid search & cross validation
model_l1 = LogisticRegression(penalty='l1',solver="liblinear",max_iter=10000)
grid_search = GridSearchCV(model_l1, param_grid={'C':np.logspace(-3,3,7)}, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(data_train_scaled.iloc[:,:(data_train_scaled.shape[1]-1)],np.array(df_train["label"]))

# Best model
best_model = grid_search.best_estimator_
pred = best_model.predict(data_test_scaled.iloc[:,:(data_test_scaled.shape[1]-1)])
print("2 classes accuracy: {:.4f}".format(sum(pred == df_test["label"])/len(pred)))  # todo

# Visualization: word cloud
# Draw 5 pictures and choose the most beautiful one
mr_coef_all = best_model.coef_
plt.figure(figsize=(20,10))
for j in range(5):
    for k in range(1):  # todo
        mr_coef = np.argsort(mr_coef_all[k,14:(mr_coef_all.shape[1]-2)])+14
        mr_coef = mr_coef[::-1]
        mr_word = [word_name[mr_coef[i]] if (word_name[mr_coef[i]] not in colnames) else None for i in range(100) ]
        word_freq = [round(mr_coef_all[k,mr_coef[i]]*100) for i in range(100)]
        word_freq = {mr_word[i]:word_freq[i] for i in range(100) if mr_word[i]}
        wordcloud0 = WordCloud(background_color="white",colormap="viridis",scale=2)
        wordcloud1 = wordcloud0.generate_from_frequencies(word_freq)
        plt.imshow(wordcloud1,interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"Figures/2classes_{k}({j}).png",dpi=300,bbox_inches="tight")  # todo
        plt.close()

# %%
# 6 classes accuracy: 0.5258
# 4 classes accuracy: 0.7532
# 2 classes accuracy: 0.8562