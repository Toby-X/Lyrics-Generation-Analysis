#-*- coding:utf-8 -*-
library(torch)
library(topicmodels.etm)
library(doc2vec)
library(word2vec)
library(data.table)

dat = fread("data/df_cluster.csv")
x = data.frame(song_id = dat$song_id,
               lyrics = dat$lyrics,
               stringsAsFactors = F)
x$lyrics = txt_clean_word2vec(x$lyrics)

w2v = word2vec(x=x$lyrics,dim=50,type = "skip-gram",
               iter=30, min_count = 20, threads = 6)
embeddings = as.matrix(w2v)
predict(w2v,newdata=c("way","weave"),
        type="nearest",top_n=4)

library(udpipe)
dtm = strsplit.data.frame(x,group="song_id",term="lyrics",split=" ")
dtm = document_term_frequencies(dtm)
dtm = document_term_matrix(dtm,prob=.5)
vocab = intersect(rownames(embeddings),colnames(dtm))
embeddings = dtm_conform(embeddings,rows=vocab)
dtm = dtm_conform(dtm,columns = vocab)

set.seed(1234)
torch_manual_seed(4321)
model = ETM(k=250, dim=800, embeddings = embeddings)
optimizer = optim_adam(params=model$parameters,lr=.005,weight_decay = .0000012)
loss = model$fit(data=dtm,optimizer = optimizer,epoch=20,batch_size = 1e3)
plot(model, type = "loss")

terminology = predict(model,type="terms",top_n = 20)

library(textplot)
library(uwot)
library(ggrepel)
library(ggalt)
manifolded <- summary(model, type = "umap", n_components = 2, metric = "cosine", n_neighbors = 15, 
                      fast_sgd = FALSE, n_threads = 2, verbose = TRUE)
space      <- subset(manifolded$embed_2d, type %in% "centers")
textplot_embedding_2d(space)
space      <- subset(manifolded$embed_2d, rank <= 15)
textplot_embedding_2d(space, title = "ETM topics", subtitle = "embedded in 2D using UMAP", 
                      encircle = F, points = TRUE)
