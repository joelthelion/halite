rm(list=ls())

library(caTools)
library(caret)

a <- read.csv("./results.csv")
a$won <- a$player == a$winner

# model <- glm(won~territory+strength+production, data=a)
# model <- glm(won~territory+strength+production+remaining_turns, data=a)
# summary(model)
# a$predicted <- model$fitted.values

sample <- caTools::sample.split(a$X,SplitRatio = 0.5,group=a$filename)
train <- a[sample,]
test  <- a[!sample,]
model <- glm(won~territory+strength+production, data=train)
test$predicted <- predict(model,test)
test$hard <- test$predicted > 0.5
sum(test$hard == test$won) / nrow(test) * 100
