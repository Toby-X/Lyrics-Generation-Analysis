#-*-coding:utf-8-*-
library(glmnet)
# library(data.table)
# library(tictoc)
# library(parallel)
# library(foreach)
# library(doSNOW)
# numCores = 6
# cl = makeCluster(numCores)
# registerDoSNOW(cl)
dat_train = read.csv("data/train_data_all.csv")
dat_train = dat_train[,-1]
dat_test = read.csv("data/test_data_all.csv")
dat_test = dat_test[,-1]

# for 60s classification first perform bootstrap
set.seed(3701)
label_60 = dat_train$label.1
idx60 = (1:length(label_60))[label_60==0]
boot60 = sample(idx60,sum(label_60!=0)/2-sum(label_60==0),replace = T)
dat_train_60 = rbind(dat_train,dat_train[boot60,])

# base model
# perform cross validation on elastic net
lam = log(seq(from = exp(1e-2), to = exp(.5), length=100))
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

err.tmp = matrix(rep(0,5*length(lam)),nrow=5)
for (k in 1:5) {
  mr.tmp = glmnet(data.matrix(dat_train_60[idx!=k,-ncol(dat_train)]),dat_train_60[idx!=k,ncol(dat_train)]==0
                  ,family="binomial",alpha = 0,lambda = lam)
  pre.tmp = predict(mr.tmp,data.matrix(dat_train_60[idx==k,-ncol(dat_train)]),type="class")
  err.tmp[k,] = apply(pre.tmp,2,err_cal,k=k)
}
plot(rev(lam),colMeans(err.tmp),"l",xlab = "lambda",ylab = "error rate")

# ridge lambda = .07
m1 = glmnet(data.matrix(dat_train_60[,-ncol(dat_train)]),dat_train_60[,ncol(dat_train)]==0
            ,family="binomial",alpha = 0,lambda=.07)
pre.res = predict(m1,data.matrix(dat_test[,-ncol(dat_test)]),type = "class")
pred=  pre.res
pred = sapply(pred, ch2int)
mean((pred!=(dat_test[,ncol(dat_train_60)]==0)))

# for 60s classification first perform bootstrap
set.seed(3701)
label_70 = dat_train$label.1
idx70 = (1:length(label_70))[label_70==1]
boot70 = sample(idx70,sum(label_70!=0)/2-sum(label_70==0),replace = T)
dat_train_70 = rbind(dat_train,dat_train[boot70,])

# base model
# perform cross validation on elastic net
lam = log(seq(from = exp(1e-2), to = exp(.5), length=100))
idx = rep(1:5,length.out=nrow(dat_train_70))
set.seed(516)
idx = sample(idx)

err_cal <- function(pred,k){
  pred = sapply(pred, ch2int)
  mean((pred!=(dat_train_70[idx==k,ncol(dat_train_70)]==0)))
}

err.tmp = matrix(rep(0,5*length(lam)),nrow=5)
for (k in 1:5) {
  mr.tmp = glmnet(data.matrix(dat_train_70[idx!=k,-ncol(dat_train)]),dat_train_70[idx!=k,ncol(dat_train)]==0
                  ,family="binomial",alpha = 0,lambda = lam)
  pre.tmp = predict(mr.tmp,data.matrix(dat_train_70[idx==k,-ncol(dat_train)]),type="class")
  err.tmp[k,] = apply(pre.tmp,2,err_cal,k=k)
}
plot(rev(lam),colMeans(err.tmp),"l",xlab = "lambda",ylab = "error rate")

# ridge lambda = .07
m1 = glmnet(data.matrix(dat_train_60[,-ncol(dat_train)]),dat_train_60[,ncol(dat_train)]==0
            ,family="binomial",alpha = 0,lambda=.07)
pre.res = predict(m1,data.matrix(dat_test[,-ncol(dat_test)]),type = "class")
pred=  pre.res
pred = sapply(pred, ch2int)
mean((pred!=(dat_test[,ncol(dat_train_60)]==0)))