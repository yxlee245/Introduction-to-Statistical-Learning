# Principal Components Regression
library(pls)
library(ISLR)

Hitters = na.omit(Hitters)  # remove NAs
x = model.matrix(Salary ~ ., data=Hitters)[, -1]
y = Hitters$Salary

set.seed(2)
pcr.fit = pcr(Salary ~ ., data=Hitters, scale=TRUE,
validation="CV")
# scale=TRUE standardizes every predictor prior to generating principal components
# validation="CV" causes pcr() to compute ten-fold cross-validation error for each possible value of M
summary(pcr.fit)  # RMSE reported by default

validationplot(pcr.fit, val.type="MSEP")

set.seed(1)
train = sample(1:nrow(x), nrow(x) / 2)
test = (-train)
y.test = y[test]

set.seed(1)
pcr.fit = pcr(Salary ~ ., data=Hitters, subset=train, scale=TRUE,
validation="CV")
validationplot(pcr.fit, val.type="MSEP")
pcr.pred = predict(pcr.fit, x[test,], ncomp=7)
mean((pcr.pred - y.test) ^ 2)

pcr.fit = pcr(y ~ x, scale=TRUE, ncomp=7)
summary(pcr.fit)

# Partial Least Squares
set.seed(1)
pls.fit = plsr(Salary ~ ., data=Hitters, subset=train, scale=TRUE,
validation="CV")
summary(pls.fit)
validationplot(pls.fit, val.type="MSEP")
pls.pred = predict(pls.fit, x[test,], ncomp=2)
mean((pls.pred - y.test) ^ 2)

pls.fit = plsr(Salary ~ ., data=Hitters, scale=TRUE, ncomp=2)
summary(pls.fit)