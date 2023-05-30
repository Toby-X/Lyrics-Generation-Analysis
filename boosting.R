#-*-coding:utf-8-*-
library(xgboost)
dat_train = read.csv("data/train_data_all.csv")
dat_train = dat_train[,-c(1,2)]
dat_test = read.csv("data/test_data_all.csv")
dat_test = dat_test[,-c(1,2)]
train_other = read.csv("data/train_other.csv")

dat_train$new_label = 0
dat_test$new_label = 0
dat_train$new_label[dat_train$label.1==3|dat_train$label.1==4|dat_train$label.1==5] = 1
dat_test$new_label[dat_test$label.1==3|dat_test$label.1==4|dat_test$label.1==5] = 1
# for (i in 3:5) {
#   dat_train$new_label[dat_train$label.1==i] = i-2
#   dat_test$new_label[dat_test$label.1==i] = i-2
# }
dat_train = dat_train[,-(ncol(dat_train)-1)]
dat_test = dat_test[,-(ncol(dat_test)-1)]

var_max = max(dat_train[,14:(ncol(dat_train)-1)],axis=1)
dat_train[,14:(ncol(dat_train)-1)] = t(t(dat_train[,14:(ncol(dat_train)-1)])/var_max)
dat_test[,14:(ncol(dat_test)-1)] = t(t(dat_test[,14:(ncol(dat_test)-1)])/var_max)

bstSparse = xgboost(data=data.matrix(dat_train[,-(ncol(dat_train))]),label=data.matrix(dat_train[,ncol(dat_train)])
                    ,max.depth=7,min_child_weight=0.7,eta=.2,nthread=6,nrounds = 2,objective="multi:softmax",num_class=4)
pre = predict(bstSparse,data.matrix(dat_test[,-(ncol(dat_train))]))
mean(pre==dat_test[,ncol(dat_test)])

bstSparse = xgboost(data=data.matrix(dat_train[,-(ncol(dat_train))]),label=data.matrix(dat_train[,ncol(dat_train)])
                    ,max.depth=6,min_child_weight=2,eta=.2,nthread=6,nrounds = 2,objective="multi:softmax",num_class=6)
pre = predict(bstSparse,data.matrix(dat_test[,-(ncol(dat_train))]))
mean(pre==dat_test[,ncol(dat_test)])
