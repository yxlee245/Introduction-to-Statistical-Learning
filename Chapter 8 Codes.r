# Fitting Classification Trees
library(tree)

library(ISLR)
attach(Carseats)
High = ifelse(Sales <= 8, "No", "Yes")  # "Yes" if Sales variable exceeds 8, "No" otherwise
Carseats = data.frame(Carseats, High)  # Merge High with rest of Carseats dataframe

tree.carseats = tree(High ~ . - Sales, data=Carseats)
summary(tree.carseats)
plot(tree.carseats)
text(tree.carseats, pretty=0)

tree.carseats

set.seed(2)
train = sample(1:nrow(Carseats), 200)
Carseats.test = Carseats[-train, ]
High.test = High[-train]
tree.carseats = tree(High ~ . - Sales, Carseats, subset=train)
tree.pred = predict(tree.carseats, Carseats.test, type="class")
table(tree.pred, High.test)
(86 + 57) / 200

set.seed(3)
cv.carseats = cv.tree(tree.carseats, FUN=prune.misclass)  # prune.misclass to indicate use of misclassification rate for CV and pruning
names(cv.carseats)
cv.carseats

par(mfrow=c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type="b")
plot(cv.carseats$k, cv.carseats$dev, type="b")

prune.carseats = prune.misclass(tree.carseats, best=9)  # obtain nine-node tree
plot(prune.carseats)
text(prune.carseats, pretty=0)
tree.pred = predict(prune.carseats, Carseats.test, type="class")
table(tree.pred, High.test)
(94 + 60) / 200

prune.carseats = prune.misclass(tree.carseats, best=15)  # obtain larger pruned tree
plot(prune.carseats)
text(prune.carseats, pretty=0)
tree.pred = predict(prune.carseats, Carseats.test, type="class")
table(tree.pred, High.test)
(86 + 62) / 200

# Fitting Regression Trees
library(MASS)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston) / 2)
tree.boston = tree(medv ~ ., data=Boston, subset=train)
summary(tree.boston)
plot(tree.boston)
text(tree.boston, pretty=0)

cv.boston = cv.tree(tree.boston)  # tree size selection by cross-validation
plot(cv.boston$size, cv.boston$dev, type="b")

prune.boston = prune.tree(tree.boston, best=5)
plot(prune.boston)
text(prune.boston, pretty=0)

yhat = predict(tree.boston, newdata=Boston[-train,])
boston.test = Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0, 1)
mean((yhat - boston.test) ^ 2)

# Bagging and Random Forests
library(randomForest)
set.seed(1)
bag.boston = randomForest(medv ~ ., data=Boston, subset=train,
mtry=13, importance=TRUE)  # mtry - number of predictors tro consider for each split of the tree
bag.boston
yhat.bag = predict(bag.boston, newdata=Boston[-train,])
plot(yhat.bag, boston.test)
abline(0, 1)
mean((yhat.bag - boston.test) ^ 2)

bag.boston = randomForest(medv ~ ., data=Boston, subset=train,
mtry=13, ntree=25) # change number of trees grown by randomForest()
yhat.bag = predict(bag.boston, newdata=Boston[-train,])
mean((yhat.bag - boston.test) ^ 2)

set.seed(1)
rf.boston = randomForest(medv ~ ., data=Boston, subset=train,
mtry=6, importance=TRUE)
yhat.rf = predict(rf.boston, newdata=Boston[-train,])
mean((yhat.rf - boston.test) ^ 2)

importance(rf.boston)
varImpPlot(rf.boston)

# Boosting
library(gbm)
set.seed(1)
boost.boston = gbm(medv ~ ., data=Boston[train,], distribution="gaussian",
n.trees=5000, interaction.depth=4)  # distribution="gaussian" for regression problems
summary(boost.boston)

par(mfrow=c(1, 2))
plot(boost.boston, i="rm")  # partial dependence plot for rm
plot(boost.boston, i="lstat")  # partial dependence plot for lstat

yhat.boost = predict(boost.boston, newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost - boston.test) ^ 2)

boost.boston = gbm(medv ~ ., data=Boston[train, ], distribution="gaussian",
n.trees=5000, interaction.depth=4, shrinkage=0.2, verbose=F)  # change shrinkage parameter
yhat.boost = predict(boost.boston, newdata=Boston[-train, ], n.trees=5000)
mean((yhat.boost - boston.test) ^ 2)