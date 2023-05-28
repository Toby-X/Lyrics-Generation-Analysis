#-*-coding:utf-8-*-
library(xgboost)
dat_train = read.csv("data/train_data_all.csv")
dat_train = dat_train[,-c(1,2)]
dat_test = read.csv("data/test_data_all.csv")
dat_test = dat_test[,-c(1,2)]
train_other = read.csv("data/train_other.csv")