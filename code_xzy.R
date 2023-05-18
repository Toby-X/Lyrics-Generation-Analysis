#-*-coding:utf-8-*-
dat_train = read.csv("data/train_tfidf_data.csv")
dat_train = dat_train[,-1]
dat_test = read.csv("data/test_tfidf_data.csv")
dat_test = dat_test[,-1]
library(glmnet)

# base model
# perform cross validation on elastic net
a = seq(from=0,to=.1,length=100)
lam = log(seq(from = exp(0.4), to = exp(.8), length=100))
idx = rep(1:5,length.out=nrow(dat_train))
set.seed(516)
idx = sample(idx)

err_cal <- function(pred,k){
  mean((as.numeric(pred)-dat_train[idx==k,ncol(dat_train)])^2)
}

err = matrix(rep(0,length(a)*length(lam)),nrow=length(a))
err.tmp = matrix(rep(0,5*length(lam)),nrow=5)
for (i in 1:length(a)){
  for (k in 1:5){
    mr.tmp = glmnet(data.matrix(dat_train[idx!=k,-ncol(dat_train)]),as.factor(dat_train[idx!=k,ncol(dat_train)])
                    ,family="multinomial",alpha = a[i],lambda = lam)
    pre.tmp = predict(mr.tmp,data.matrix(dat_train[idx==k,-ncol(dat_train)]),type="class")
    err.tmp[k,] = apply(pre.tmp,2,err_cal,k=k)
  }
  err[i,] = colMeans(err.tmp)
}

plot(a,rowMeans(err))
persp(a,lam,err,theta=180)
plot(lam,err[5,])

# choose alpha = 0(ridge) and lambda = 
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
