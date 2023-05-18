#-*-coding:utf-8-*-
library(glmnet)
library(data.table)
dat_train = fread("data/train_data.csv")
dat_train = dat_train[,-1]
dat_test = fread("data/test_data.csv")
dat_test = dat_test[,-1]

# for 60s classification first perform bootstrap
label_60 = dat_train$label.1
idx60 = (1:length(label_60))[label_60==0]
boot60 = sample(idx60,sum(label_60!=0)-sum(label_60==0),replace = T)
dat_train_60 = rbind(dat_train,dat_train[boot60,])

# base model
# perform cross validation on elastic net
a = seq(from=0,to=1,length=100)
lam = log(seq(from = exp(0.01), to = exp(.5), length=100))
idx = rep(1:5,length.out=nrow(dat_train_60))
set.seed(516)
idx = sample(idx)

ch2int <- function(ch){
  if (ch == 'FALSE'){
    return(0)
  } else {
    return(1)
  }
}

err_cal <- function(pred,k){
  pred = sapply(pred, ch2int)
  mean((pred!=(dat_train_60[idx==k,ncol(dat_train_60)]==0)))
}

err = matrix(rep(0,length(a)*length(lam)),nrow=length(a))
err.tmp = matrix(rep(0,5*length(lam)),nrow=5)
for (i in 1:length(a)){
  for (k in 1:5){
    mr.tmp = glmnet(data.matrix(dat_train_60[idx!=k,-ncol(dat_train)]),dat_train_60[idx!=k,ncol(dat_train)]==0
                    ,family="binomial",alpha = a[i],lambda = lam)
    pre.tmp = predict(mr.tmp,data.matrix(dat_train_60[idx==k,-ncol(dat_train)]),type="class")
    err.tmp[k,] = apply(pre.tmp,2,err_cal,k=k)
  }
  err[i,] = colMeans(err.tmp)
}

plot(a,rowMeans(err))
persp(a,lam,err,theta=180)
plot(lam,err[2,])

# choose alpha = 0 and lambda = .4
mr60 = glmnet(data.matrix(dat_train_60[,-c(ncol(dat_train),1:14)]),dat_train_60[,ncol(dat_train)]==0
                ,family="binomial",alpha=0,lambda=.4)
pre60 = predict(mr60,data.matrix(dat_test[,-ncol(dat_test)]),type="class")
pre60 = sapply(pre60,ch2int)
err60 = mean(pre60==dat_test[,ncol(dat_test)])

mr60.beta = as.matrix(mr60$beta)
fwrite(mr60.beta,"beta.csv")

lam = log(seq(from = exp(.7), to = exp(1.5), length=100))
for (k in 1:5){
  mr.tmp = glmnet(data.matrix(dat_train[idx!=k,-ncol(dat_train)]),as.factor(dat_train[idx!=k,ncol(dat_train)])
                  ,family="multinomial",alpha = 0,lambda = lam)
  pre.tmp = predict(mr.tmp,data.matrix(dat_train[idx==k,-ncol(dat_train)]),type="class")
  err.tmp[k,] = apply(pre.tmp,2,err_cal,k=k)
}
err[1,] = colMeans(err.tmp)
plot(lam,err[1,])

persp(a,lam,err,theta=180)

m1 = glmnet(data.matrix(dat_train[,-ncol(dat_train)]),as.factor(dat_train[,ncol(dat_train)])
            ,family="multinomial",alpha = a[1])
str(dat_train[,-ncol(dat_train)])
sum(dat_train[,ncol(dat_train)]==5)

pre.res = predict(m1,data.matrix(dat_test[,-ncol(dat_test)]),type = "class")
sum(as.numeric(pre.res)==dat_test[,ncol(dat_test)])/nrow(dat_test)
pre.res[1]
head(dat_test[,ncol(dat_test)])

head(apply(pre.tmp,2,err_cal,k=1))

# SVM
library(kernlab)
svm = ksvm(data.matrix(dat_train[,-ncol(dat_train)]),as.factor(dat_train[,ncol(dat_train)]))
pre = predict(svm,data.matrix(dat_test[,-ncol(dat_test)]))
sum(pre==dat_test[,ncol(dat_test)])/length(pre)
