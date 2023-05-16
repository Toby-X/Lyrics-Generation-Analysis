#-*-coding:utf-8-*-
dat_train = read.csv("data/train_tfidf_data.csv")
dat_train = dat_train[,-1]
dat_test = read.csv("data/test_tfidf_data.csv")
dat_test = dat_test[,-1]
library(glmnet)

# base model
# perform cross validation on elastic net
a = seq(from=0,to=1,length=1e2)
lam = log(seq(from = exp(0), to = exp(.5), length=500))
idx = rep(1:5,length.out=nrow(dat_train))
set.seed(516)
idx = sample(idx)

err_cal <- function(pred,k){
  sum(as.numeric(pred)==dat_train[idx==k,ncol(dat_train)])/sum(idx==k)
}

err = matrix(rep(0,length(a)*length(lam)),nrow=length(a))
err.tmp = matrix(rep(0,5*length(lam)),nrow=5)
for (i in length(a)){
  for (k in 1:5){
    mr.tmp = glmnet(data.matrix(dat_train[idx!=k,-ncol(dat_train)]),as.factor(dat_train[idx!=k,ncol(dat_train)])
                    ,family="multinomial",alpha = a[i],lambda = lam)
    pre.tmp = predict(mr.tmp,data.matrix(dat_train[idx==k,-ncol(dat_train)]),type="class")
    err.tmp[k,] = apply(pre.tmp,2,err_cal,k=1)
  }
  err[i,] = colMeans(err.tmp)
}
m1 = glmnet(data.matrix(dat_train[,-ncol(dat_train)]),as.factor(dat_train[,ncol(dat_train)])
            ,family="multinomial",alpha = a[1])
str(dat_train[,-ncol(dat_train)])
sum(dat_train[,ncol(dat_train)]==5)

pre.res = predict(m1,data.matrix(dat_test[,-ncol(dat_test)]),type = "class")
sum(as.numeric(pre.res)==dat_test[,ncol(dat_test)])/nrow(dat_test)
pre.res[1]
head(dat_test[,ncol(dat_test)])

head(apply(pre.tmp,2,err_cal,k=1))
