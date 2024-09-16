# Analysis-of-Large-Scale-Data-Sets
What I pretty much did in my analysis is that I did a training set a result set based on college data, then did some linear model, ridge regression model, lasso modeling on the training set then commented on the result that I obtained but this is just a peak of what i did.
this is just part of my code
library(ISLR)
library(caret)
library(tidyverse)
data('College')
set.seed(1)

inTrain <- createDataPartition(College$Apps, p = 0.75, list = FALSE)

training <- College[inTrain,]
testing <- College[-inTrain,]

preObj <- preProcess(training, method = c('center', 'scale'))

training <- predict(preObj, training)
testing <- predict(preObj, testing)

y_train <- training$Apps
y_test <- testing$Apps

one_hot_encoding <- dummyVars(Apps ~ ., data = training)
x_train <- predict(one_hot_encoding, training)
x_test <- predict(one_hot_encoding, testing)

lin_model <- lm(Apps ~ ., data = training)

pred <- predict(lin_model, testing)

(lin_info <- postResample(pred, testing$Apps))

ridge_fit <- train(x = x_train, y = y_train,
                   method = 'glmnet', 
                   trControl = trainControl(method = 'cv', number = 10),
                   tuneGrid = expand.grid(alpha = 0,
                                          lambda = seq(0, 10e2, length.out = 20)))

(ridge_info <- postResample(predict(ridge_fit, x_test), y_test))
coef(ridge_fit$finalModel, ridge_fit$bestTune$lambda)

plot(ridge_fit)
plot(varImp(ridge_fit))


lasso_fit <- train(x = x_train, y = y_train, 
                   method = 'glmnet',
                   trControl = trainControl(method = 'cv', number = 10),
                   tuneGrid = expand.grid(alpha = 1,
                                          lambda = seq(0.0001, 1, length.out = 50)))

(lasso_info <- postResample(predict(lasso_fit, x_test), y_test))

coef(lasso_fit$finalModel, lasso_fit$bestTune$lambda)
plot(lasso_fit)

plot(varImp(lasso_fit))

pcr_model <- train(x = x_train, y = y_train,
                   method = 'pcr',
                   trControl = trainControl(method = 'cv', number = 10),
                   tuneGrid = expand.grid(ncomp = 1:10))
y(pcr_info <- postResample(predict(pcr_model, x_test), y_test))

coef(pcr_model$finalModel)
plot(pcr_model)

plot(varImp(pcr_model))

pls_model <- train(x = x_train, y = y_train,
                   method = 'pls',
                   trControl = trainControl(method = 'cv', number = 10),
                   tuneGrid = expand.grid(ncomp = 1:10))
(pls_info <- postResample(predict(pls_model, x_test), y_test))

coef(pls_model$finalModel)

plot(pls_model)

as_data_frame(rbind(lin_info,
                    ridge_info,
                    lasso_info,
                    pcr_info,
                    pls_info)) %>%
  mutate(model = c('Linear', 'Ridge', 'Lasso', 'PCR', 'PLS')) %>%
  select(model, RMSE, Rsquared)
testing %>%
  summarize(sd = sd(Apps))


library(ggthemes)

residfunc <- function(fit, data) {
  predict(fit, data) - testing$Apps
}

data_frame(Observed = testing$Apps,
           LM = residfunc(lin_model, testing),
           Ridge = residfunc(ridge_fit, x_test),
           Lasso = residfunc(lasso_fit, x_test),
           PCR = residfunc(pcr_model, x_test),
           PLS = residfunc(pls_model, x_test)) %>%
  gather(Model, Residuals, -Observed) %>%
  ggplot(aes(Observed, Residuals, col = Model)) +
  geom_hline(yintercept = 0, lty = 2) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = 'loess', alpha = 0.01, col = 'lightsalmon2') +
  facet_wrap(~ Model, ncol = 5) +
  theme_tufte() +
  theme(legend.position = 'top') +
  coord_flip()

set.seed(1)
library(MASS); library(tidyverse); library(ggplot2); library(ggthemes)
library(broom); library(knitr); library(caret)
theme_set(theme_tufte(base_size = 14) + theme(legend.position = 'top'))
data('Boston')

model <- lm(nox ~ poly(dis, 3), data = Boston)
tidy(model) %>%
  kable(digits = 3)
Boston %>%
  mutate(pred = predict(model, Boston)) %>%
  ggplot() +
  geom_point(aes(dis, nox, col = '1')) +
  geom_line(aes(dis, pred, col = '2'), size = 1.5) +
  scale_color_manual(name = 'Value Type',
                     labels = c('Observed', 'Predicted'),
                     values = c('#56B4E9', '#E69F00'))



errors <- list()
models <- list()
pred_df <- data_frame(V1 = 1:506)
for (i in 1:9) {
  models[[i]] <- lm(nox ~ poly(dis, i), data = Boston)
  preds <- predict(models[[i]])
  pred_df[[i]] <- preds
  errors[[i]] <- sqrt(mean((Boston$nox - preds)^2))
}

errors <- unlist(errors)

names(pred_df) <- paste('Level', 1:9)
data_frame(RMSE = errors) %>%
  mutate(Poly = row_number()) %>%
  ggplot(aes(Poly, RMSE, fill = Poly == which.min(errors))) +
  geom_col() + 
  guides(fill = FALSE) +
  scale_x_continuous(breaks = 1:9) +
  coord_cartesian(ylim = c(min(errors), max(errors))) +
  labs(x = 'Polynomial Degree')

Boston %>%
  cbind(pred_df) %>%
  gather(Polynomial, prediction, -(1:14)) %>%
  mutate(Polynomial = factor(Polynomial, 
                             levels = unique(as.character(Polynomial)))) %>%
  ggplot() + 
  ggtitle('Predicted Values for Each Level of Polynomial') +
  geom_point(aes(dis, nox, col = '1')) + 
  geom_line(aes(dis, prediction, col = '2'), size = 1.5) +
  scale_color_manual(name = 'Value Type',
                     labels = c('Observed', 'Predicted'),
                     values = c('#56B4E9', '#E69F00')) +
  facet_wrap(~ Polynomial, nrow = 3)





errors <- list()

folds <- sample(1:10, 506, replace = TRUE)
errors <- matrix(NA, 10, 9)
for (k in 1:10) {
  for (i in 1:9) {
    model <- lm(nox ~ poly(dis, i), data = Boston[folds != k,])
    pred <- predict(model, Boston[folds == k,])
    errors[k, i] <- sqrt(mean((Boston$nox[folds == k] - pred)^2))
  }
}

errors <- apply(errors, 2, mean)

data_frame(RMSE = errors) %>%
  mutate(Poly = row_number()) %>%
  ggplot(aes(Poly, RMSE, fill = Poly == which.min(errors))) +
  geom_col() + theme_tufte() + guides(fill = FALSE) +
  scale_x_continuous(breaks = 1:9) +
  coord_cartesian(ylim = range(errors))


library(splines)
model <- lm(nox ~ bs(dis, df = 4), data = Boston)

kable(tidy(model), digits = 3)



Boston %>%
  mutate(pred = predict(model)) %>%
  ggplot() +
  geom_point(aes(dis, nox, col = '1')) + 
  geom_line(aes(dis, pred, col = '2'), size = 1.5) +
  scale_color_manual(name = 'Value Type',
                     labels = c('Observed', 'Predicted'),
                     values = c('#56B4E9', '#E69F00')) +
  theme_tufte(base_size = 13)


errors <- list()
models <- list()
pred_df <- data_frame(V1 = 1:506)
for (i in 1:9) {
  models[[i]] <- lm(nox ~ bs(dis, df = i), data = Boston)
  preds <- predict(models[[i]])
  pred_df[[i]] <- preds
  errors[[i]] <- sqrt(mean((Boston$nox - preds)^2))
}

names(pred_df) <- paste(1:9, 'Degrees of Freedom')
data_frame(RMSE = unlist(errors)) %>%
  mutate(df = row_number()) %>%
  ggplot(aes(df, RMSE, fill = df == which.min(errors))) +
  geom_col() + guides(fill = FALSE) + theme_tufte() +
  scale_x_continuous(breaks = 1:9) +
  coord_cartesian(ylim = range(errors))


Boston %>%
  cbind(pred_df) %>%
  gather(df, prediction, -(1:14)) %>%
  mutate(df = factor(df, levels = unique(as.character(df)))) %>%
  ggplot() + ggtitle('Predicted Values for Each Level of Polynomial') +
  geom_point(aes(dis, nox, col = '1')) + 
  geom_line(aes(dis, prediction, col = '2'), size = 1.5) +
  scale_color_manual(name = 'Value Type',
                     labels = c('Observed', 'Predicted'),
                     values = c('#56B4E9', '#E69F00')) +
  facet_wrap(~ df, nrow = 3)



folds <- sample(1:10, size = 506, replace = TRUE)
errors <- matrix(NA, 10, 9)
models <- list()
for (k in 1:10) {
  for (i in 1:9) {
    models[[i]] <- lm(nox ~ bs(nox, df = i), data = Boston[folds != k,])
    pred <- predict(models[[i]], Boston[folds == k,])
    errors[k, i] <- sqrt(mean((Boston$nox[folds == k] - pred)^2))
  }
}

errors <- apply(errors, 2, mean)

data_frame(RMSE = errors) %>%
  mutate(df = row_number()) %>%
  ggplot(aes(df, RMSE, fill = df == which.min(errors))) +
  geom_col() + theme_tufte() + guides(fill = FALSE) +
  scale_x_continuous(breaks = 1:9) +
  coord_cartesian(ylim = range(errors))

warnings()















packages <- c('ISLR', 'caret', 'tidyverse', 'ggthemes', 'rpart', 'rpart.plot', 
              'knitr', 'kableExtra')
sapply(packages, require, character.only = TRUE)
data(OJ)
set.seed(1)
inTrain <- createDataPartition(OJ$Purchase, p = 800/1070, list = FALSE)
training <- OJ[inTrain,]
testing <- OJ[-inTrain,]



rpart_model <- rpart(Purchase ~ ., data = training, method = 'class',
                     control = rpart.control(cp = 0))
summary(rpart_model, cp = 1)

postResample(predict(rpart_model, training, type = 'class'), training$Purchase) %>%
  kable
rpart_model





rpart.plot(rpart_model)



pred <- predict(rpart_model, testing, type = 'class')

caret::confusionMatrix(pred, testing$Purchase)

rpart_cv_model <- train(training[,-1], training[,1],
                        method = 'rpart',
                        trControl = trainControl(method = 'cv', number = 10),
                        tuneGrid = expand.grid(cp = seq(0, 0.5, length.out = 10)))
rpart_cv_model


plot(rpart_cv_model)


rpart_cv_model$bestTune %>% kable
rpart_cv_model$results %>% kable


set.seed(1)
rpart_tuned <- rpart(Purchase ~ ., data = training, method = 'class',
                     control = rpart.control(cp = 0.02))
rpart_tuned



postResample(predict(rpart_model, 
                     training,
                     type = 'class'), training$Purchase) %>% kable

