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
train_other = read.csv("data/train_other.csv")

corr = apply(dat_train[,14:(ncol(dat_train)-1)],2,cor,y=train_other$year)
boxplot(corr)
sum(abs(corr)>0.05)
idx_avail = abs(corr)>.1
dat_train_word = dat_train[,14:(ncol(dat_train)-1)]
dat_train_word = dat_train_word[,idx_avail]
dat_train_fil = cbind(dat_train[,1:13],dat_train_word,dat_train[,ncol(dat_train)])
clf.tmp = glmnet(data.matrix(dat_train_fil[,-ncol(dat_train_fil)]),as.factor(dat_train_fil[,ncol(dat_train_fil)]),
                 alpha = 0,lambda = .1,family = "multinomial")
dat_test_word = dat_test[,14:(ncol(dat_test)-1)]
dat_test_word = dat_test_word[,idx_avail]
dat_test_fil = cbind(dat_test[,1:13],dat_test_word,dat_test[,ncol(dat_test)])
pred = predict(clf.tmp,data.matrix(dat_test_fil[,-ncol(dat_test_fil)]),type="class")
mean(pred == dat_test_fil[,ncol(dat_test_fil)])
sum(pred==0)
sum(dat_test_fil[,ncol(dat_test_fil)]==5)

# normalization of word frequency
freq_mean = colMeans(dat_train[,14:(ncol(dat_train)-1)])
freq_sd = apply(dat_train[,14:(ncol(dat_train)-1)],2,sd)
dat_train[,14:(ncol(dat_train)-1)] = t((t(dat_train[,14:(ncol(dat_train)-1)])-freq_mean)/freq_sd)
dat_test[,14:(ncol(dat_test)-1)] = t((t(dat_test[,14:(ncol(dat_test)-1)])-freq_mean)/freq_sd)

# base model
# perform cross validation on elastic net
lam = log(seq(from = exp(1e-2), to = exp(1), length=100))
idx = rep(1:5,length.out=nrow(dat_train))
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
  mean((pred!=(dat_train[idx==k,ncol(dat_train)]==0)))
}

err.tmp = matrix(rep(0,5*length(lam)),nrow=5)
for (k in 1:5) {
  # Bootstrap
  set.seed(k)
  dat_train.tmp = dat_train[idx!=k,]
  label_60 = dat_train.tmp$label.1
  idx60 = (1:length(label_60))[label_60==0]
  boot60 = sample(idx60,sum(label_60!=0)/2-sum(label_60==0),replace = T)
  dat_train_60 = rbind(dat_train.tmp,dat_train.tmp[boot60,])
  
  # Model Estimation
  mr.tmp = glmnet(data.matrix(dat_train_60[,-ncol(dat_train)]),dat_train_60[,ncol(dat_train)]==0
                  ,family="binomial",alpha = 0,lambda = lam)
  pre.tmp = predict(mr.tmp,data.matrix(dat_train[idx==k,-ncol(dat_train)]),type="class")
  err.tmp[k,] = apply(pre.tmp,2,err_cal,k=k)
}
plot(rev(lam),colMeans(err.tmp),"l",xlab = "lambda",ylab = "error rate")

# ridge lambda = .7
set.seed(3701)
label_60 = dat_train$label.1
idx60 = (1:length(label_60))[label_60==0]
boot60 = sample(idx60,sum(label_60!=0)/2-sum(label_60==0),replace = T)
dat_train_60 = rbind(dat_train,dat_train[boot60,])
clf60 = glmnet(data.matrix(dat_train_60[,-ncol(dat_train)]),dat_train_60[,ncol(dat_train)]==0
            ,family="binomial",alpha = 0,lambda=.7)
pre.res = predict(clf60,data.matrix(dat_test[,-ncol(dat_test)]),type = "class")
pred=  pre.res
pred = sapply(pred, ch2int)
# recall
mean((pred[(dat_test[,ncol(dat_train_60)]==0)]==1)) # .33
# precision
mean((dat_test[pred==1,ncol(dat_train_60)]==0)) # .66

# for 70s classification first perform bootstrap
err_cal <- function(pred,k){
  pred = sapply(pred, ch2int)
  mean((pred!=(dat_train[idx==k,ncol(dat_train)]==1)))
}

err.tmp = matrix(rep(0,5*length(lam)),nrow=5)
for (k in 1:5) {
  # Bootstrap
  set.seed(k)
  dat_train.tmp = dat_train[idx!=k,]
  label_70 = dat_train.tmp$label.1
  idx70 = (1:length(label_70))[label_70==1]
  boot70 = sample(idx70,sum(label_70!=1)/2-sum(label_70==1),replace = T)
  dat_train_70 = rbind(dat_train.tmp,dat_train.tmp[boot70,])
  
  # Model Estimation
  mr.tmp = glmnet(data.matrix(dat_train_70[,-ncol(dat_train)]),dat_train_70[,ncol(dat_train)]==1
                  ,family="binomial",alpha = 0,lambda = lam)
  pre.tmp = predict(mr.tmp,data.matrix(dat_train[idx==k,-ncol(dat_train)]),type="class")
  err.tmp[k,] = apply(pre.tmp,2,err_cal,k=k)
}
plot(rev(lam),colMeans(err.tmp),"l",xlab = "lambda",ylab = "error rate")

# ridge lambda = .5
set.seed(3701)
label_70 = dat_train$label.1
idx70 = (1:length(label_70))[label_70==1]
boot70 = sample(idx70,sum(label_70!=1)/2-sum(label_70==1),replace = T)
dat_train_70 = rbind(dat_train,dat_train[boot70,])
clf70 = glmnet(data.matrix(dat_train_70[,-ncol(dat_train)]),dat_train_70[,ncol(dat_train)]==1
            ,family="binomial",alpha = 0,lambda=.5)
pre.res = predict(clf70,data.matrix(dat_test[,-ncol(dat_test)]),type = "class")
pred=  pre.res
pred = sapply(pred, ch2int)
mean((pred!=(dat_test[,ncol(dat_train_70)]==1)))
# recall
mean((pred[(dat_test[,ncol(dat_train_70)]==1)]==1)) # .22
# precision
mean((dat_test[pred==1,ncol(dat_train_70)]==1)) # .45

# 80s classifier

lam = log(seq(from = exp(.1), to = exp(1.5), length=100))

err_cal <- function(pred,k){
  pred = sapply(pred, ch2int)
  mean((pred!=(dat_train[idx==k,ncol(dat_train)]==2)))
}

err.tmp = matrix(rep(0,5*length(lam)),nrow=5)
for (k in 1:5) {
  # Bootstrap
  set.seed(k)
  dat_train.tmp = dat_train[idx!=k,]
  label_80 = dat_train.tmp$label.1
  idx80 = (1:length(label_80))[label_80==2]
  boot80 = sample(idx80,sum(label_80!=2)/2-sum(label_80==2),replace = T)
  dat_train_80 = rbind(dat_train.tmp,dat_train.tmp[boot80,])
  
  # Model Estimation
  mr.tmp = glmnet(data.matrix(dat_train_80[,-ncol(dat_train)]),dat_train_80[,ncol(dat_train)]==2
                  ,family="binomial",alpha = 0,lambda = lam)
  pre.tmp = predict(mr.tmp,data.matrix(dat_train[idx==k,-ncol(dat_train)]),type="class")
  err.tmp[k,] = apply(pre.tmp,2,err_cal,k=k)
}
plot(rev(lam),colMeans(err.tmp),"l",xlab = "lambda",ylab = "error rate")

# ridge lambda = 1.4
set.seed(3701)
label_80 = dat_train$label.1
idx80 = (1:length(label_80))[label_80==2]
boot80 = sample(idx80,sum(label_80!=2)/2-sum(label_80==2),replace = T)
dat_train_80 = rbind(dat_train,dat_train[boot80,])
clf80 = glmnet(data.matrix(dat_train_80[,-ncol(dat_train)]),dat_train_80[,ncol(dat_train)]==2
            ,family="binomial",alpha = 0,lambda=1.4)
pre.res = predict(clf80,data.matrix(dat_test[,-ncol(dat_test)]),type = "class")
pred=  pre.res
pred = sapply(pred, ch2int)
mean((pred!=(dat_test[,ncol(dat_train_80)]==2)))
# recall
mean((pred[(dat_test[,ncol(dat_train_80)]==2)]==1)) # .07→0.02（1.0→1.4）
# precision
mean((dat_test[pred==1,ncol(dat_train_80)]==2)) # .24→.25

# 90s classifier
err_cal <- function(pred,k){
  pred = sapply(pred, ch2int)
  mean((pred!=(dat_train[idx==k,ncol(dat_train)]==3)))
}

err.tmp = matrix(rep(0,5*length(lam)),nrow=5)
for (k in 1:5) {
  # Bootstrap
  set.seed(k)
  dat_train.tmp = dat_train[idx!=k,]
  label_90 = dat_train.tmp$label.1
  idx90 = (1:length(label_90))[label_90==3]
  boot90 = sample(idx90,sum(label_90!=3)/2-sum(label_90==3),replace = T)
  dat_train_90 = rbind(dat_train.tmp,dat_train.tmp[boot90,])
  
  # Model Estimation
  mr.tmp = glmnet(data.matrix(dat_train_90[,-ncol(dat_train)]),dat_train_90[,ncol(dat_train)]==3
                  ,family="binomial",alpha = 0,lambda = lam)
  pre.tmp = predict(mr.tmp,data.matrix(dat_train[idx==k,-ncol(dat_train)]),type="class")
  err.tmp[k,] = apply(pre.tmp,2,err_cal,k=k)
}
plot(rev(lam),colMeans(err.tmp),"l",xlab = "lambda",ylab = "error rate")

# ridge lambda = 1.4
set.seed(3701)
label_90 = dat_train$label.1
idx90 = (1:length(label_90))[label_90==3]
boot90 = sample(idx90,sum(label_90!=3)/2-sum(label_90==3),replace = T)
dat_train_90 = rbind(dat_train,dat_train[boot90,])
clf90 = glmnet(data.matrix(dat_train_90[,-ncol(dat_train)]),dat_train_90[,ncol(dat_train)]==3
            ,family="binomial",alpha = 0,lambda=1.4)
pre.res = predict(clf90,data.matrix(dat_test[,-ncol(dat_test)]),type = "class")
pred=  pre.res
pred = sapply(pred, ch2int)
mean((pred!=(dat_test[,ncol(dat_train_90)]==3)))
# recall
mean((pred[(dat_test[,ncol(dat_train_60)]==3)]==1)) # .05
# precision
mean((dat_test[pred==1,ncol(dat_train_60)]==3)) # .33

# 00s Classifier
err_cal <- function(pred,k){
  pred = sapply(pred, ch2int)
  mean((pred!=(dat_train[idx==k,ncol(dat_train)]==4)))
}

err.tmp = matrix(rep(0,5*length(lam)),nrow=5)
for (k in 1:5) {
  # Bootstrap
  set.seed(k)
  dat_train.tmp = dat_train[idx!=k,]
  label_00 = dat_train.tmp$label.1
  idx00 = (1:length(label_00))[label_00==4]
  boot00 = sample(idx00,sum(label_00!=4)/2-sum(label_00==4),replace = T)
  dat_train_00 = rbind(dat_train.tmp,dat_train.tmp[boot00,])
  
  # Model Estimation
  mr.tmp = glmnet(data.matrix(dat_train_00[,-ncol(dat_train)]),dat_train_00[,ncol(dat_train)]==4
                  ,family="binomial",alpha = 0,lambda = lam)
  pre.tmp = predict(mr.tmp,data.matrix(dat_train[idx==k,-ncol(dat_train)]),type="class")
  err.tmp[k,] = apply(pre.tmp,2,err_cal,k=k)
}
plot(rev(lam),colMeans(err.tmp),"l",xlab = "lambda",ylab = "error rate")

# ridge lambda = .7
set.seed(3701)
label_00 = dat_train$label.1
idx00 = (1:length(label_00))[label_00==4]
boot00 = sample(idx00,sum(label_00!=4)/2-sum(label_00==4),replace = T)
dat_train_00 = rbind(dat_train,dat_train[boot00,])
m1 = glmnet(data.matrix(dat_train_00[,-ncol(dat_train)]),dat_train_00[,ncol(dat_train)]==4
            ,family="binomial",alpha = 0,lambda=.7)
pre.res = predict(m1,data.matrix(dat_test[,-ncol(dat_test)]),type = "class")
pred=  pre.res
pred = sapply(pred, ch2int)
mean((pred!=(dat_test[,ncol(dat_train_00)]==4)))


#10s classifier
err_cal <- function(pred,k){
  pred = sapply(pred, ch2int)
  mean((pred!=(dat_train[idx==k,"country"]==1)))
}

err.tmp = matrix(rep(0,5*length(lam)),nrow=5)
for (k in 1:5) {
  # Bootstrap
  set.seed(k)
  dat_train.tmp = dat_train[idx!=k,]
  label_country = dat_train.tmp$country
  idxcountry = (1:length(label_country))[label_country==1]
  bootcountry = sample(idxcountry,round(sum(label_country!=1)/2)-sum(label_country==1),replace = T)
  dat_train_country = rbind(dat_train.tmp,dat_train.tmp[bootcountry,])
  
  # Model Estimation
  mr.tmp = glmnet(data.matrix(dat_train_country[,-c(1:13,ncol(dat_train))]),dat_train_country$country
                  ,family="binomial",alpha = 0,lambda = lam)
  pre.tmp = predict(mr.tmp,data.matrix(dat_train[idx==k,-c(1:13,ncol(dat_train))]),type="class")
  err.tmp[k,] = apply(pre.tmp,2,err_cal,k=k)
}
plot(rev(lam),colMeans(err.tmp),"l",xlab = "lambda",ylab = "error rate")

# ridge lambda = .7
set.seed(3701)
label_country = dat_train$country
idxcountry = (1:length(label_country))[label_country==1]
bootcountry = sample(idxcountry,sum(label_country!=1)/2-sum(label_country==1),replace = T)
dat_train_country = rbind(dat_train,dat_train[bootcountry,])
m1 = glmnet(data.matrix(dat_train_country[,-c(1:13,ncol(dat_train))]),dat_train_country$country
            ,family="binomial",alpha = 0,lambda=.8)
pre.res = predict(m1,data.matrix(dat_test[,-c(1:13,ncol(dat_test))]),type = "class")
pred=  pre.res
pred = sapply(pred, ch2int)
mean((pred!=(dat_test[,"country"]==1)))
# recall
mean((pred[(dat_test[,"country"]==1)]==1)) # 1
# precision
mean((dat_test[pred==1,"country"]==1)) # .10

# Maybe Counting on Genre
# first try it on Alternative
err_cal <- function(pred,k){
  pred = sapply(pred, ch2int)
  mean((pred!=(dat_train[idx==k,ncol(dat_train)]==5)))
}

err.tmp = matrix(rep(0,5*length(lam)),nrow=5)
for (k in 1:5) {
  # Bootstrap
  set.seed(k)
  dat_train.tmp = dat_train[idx!=k,]
  label_10 = dat_train.tmp$label.1
  idx10 = (1:length(label_10))[label_10==5]
  boot10 = sample(idx10,sum(label_10!=5)/2-sum(label_10==5),replace = T)
  dat_train_10 = rbind(dat_train.tmp,dat_train.tmp[boot10,])
  
  # Model Estimation
  mr.tmp = glmnet(data.matrix(dat_train_10[,-ncol(dat_train)]),dat_train_10[,ncol(dat_train)]==5
                  ,family="binomial",alpha = 0,lambda = lam)
  pre.tmp = predict(mr.tmp,data.matrix(dat_train[idx==k,-ncol(dat_train)]),type="class")
  err.tmp[k,] = apply(pre.tmp,2,err_cal,k=k)
}
plot(rev(lam),colMeans(err.tmp),"l",xlab = "lambda",ylab = "error rate")



# ridge lambda = .7
set.seed(3701)
label_10 = dat_train$label.1
idx10 = (1:length(label_10))[label_10==5]
boot10 = sample(idx10,sum(label_10!=5)/2-sum(label_10==5),replace = T)
dat_train_10 = rbind(dat_train,dat_train[boot10,])
m1 = glmnet(data.matrix(dat_train_10[,-ncol(dat_train)]),dat_train_10[,ncol(dat_train)]==5
            ,family="binomial",alpha = 0,lambda=.7)
pre.res = predict(m1,data.matrix(dat_test[,-ncol(dat_test)]),type = "class")
pred=  pre.res
pred = sapply(pred, ch2int)
mean((pred!=(dat_test[,ncol(dat_train_10)]==5)))