rm(list=ls())

library(caTools)
library(caret)

a <- read.csv("./results.csv")
a$won <- a$player == a$winner

a <- subset(a,turn<50)

# model <- glm(won~territory+strength+production, data=a)
# model <- glm(won~territory+strength+production+remaining_turns, data=a)
# summary(model)
# a$predicted <- model$fitted.values

set.seed(123)
sample <- caTools::sample.split(a$X,SplitRatio = 0.5,group=a$filename)

train <- a[sample,]
test  <- a[!sample,]
# model <- glm(won~territory+strength+production+turn, data=train) # 85.60
#model <- glm(won~territory+strength+production+turn, data=train, family = binomial) # 86.17 with predict type=response 
# model <- glm(won~territory+production+turn, data=train) # 85.63
# model <- glm(won~territory+production+turn, data=train, family = binomial)
model <- glm(won~territory+strength+production+turn, data=train, family = binomial)
# model <- nls(won~1/(1+exp(-(a1*territory+b1*production+c1*strength +d1*turn+e1))),data=train, algorithm="port",
#              lower=c(0,0,0,-1000,-1000)) # 86.08
print(summary(model))
# test$predicted <- predict(model,test,type="response")
test$predicted <- predict(model,test,type="response")
test$hard <- test$predicted > 0.5
sum(test$hard == test$won) / nrow(test) * 100
