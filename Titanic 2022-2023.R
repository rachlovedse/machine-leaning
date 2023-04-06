# Lecture 1 ---------------------------------------------------------------

# Open Titanic data
Titanic <- read.csv("C:/Users/Evert de Haan/Dropbox/E-schijf/Groningen aanstelling/Teaching/Data Science Methods for MADS/2021-2022/Titanic example/Data titanic.csv")

# Get descriptives
summary(Titanic)

# Create dummy for age is mising
Titanic$age_missing <- ifelse(is.na(Titanic$age), 1, 0)

# Set missing age to mean value
Titanic$age <- ifelse(is.na(Titanic$age), mean(Titanic$age, na.rm=TRUE), Titanic$age)



# Logistic regression -----------------------------------------------------

#First logistic regression model
Logistic_regression1 <- glm(survived ~ as.factor(sex) + as.factor(pclass) + age, family=binomial, data=Titanic)
summary(Logistic_regression1)


#Get predictions from the logistic regression model
predictions_model1 <- predict(Logistic_regression1, type = "response", newdata=Titanic)



# Fit criteria ------------------------------------------------------------


#Make the basis for the hit rate table
predicted_model1 <- ifelse(predictions_model1>.5,1,0)

hit_rate_model1 <- table(Titanic$survived, predicted_model1, dnn= c("Observed", "Predicted"))

hit_rate_model1

#Get the hit rate
(hit_rate_model1[1,1]+hit_rate_model1[2,2])/sum(hit_rate_model1)


#Top decile lift
library(dplyr) 

decile_predicted_model1 <- ntile(predictions_model1, 10)

decile_model1 <- table(Titanic$survived, decile_predicted_model1, dnn= c("Observed", "Decile"))

decile_model1

#Calculate the TDL
(decile_model1[2,10] / (decile_model1[1,10]+ decile_model1[2,10])) / mean(Titanic$survived)


#Make lift curve
install.packages("ROCR")
library(ROCR)

pred_model1 <- prediction(predictions_model1, Titanic$survived)
perf_model1 <- performance(pred_model1,"tpr","fpr")
plot(perf_model1,xlab="Cumulative % of observations",ylab="Cumulative % of positive cases",xlim=c(0,1),ylim=c(0,1),xaxs="i",yaxs="i")
abline(0,1, col="red")
auc_model1 <- performance(pred_model1,"auc")

#The Gini is related to the "Area under the Curve" (AUC), namely by: Gini = AUC*2 - 1
#So to get the Gini we do:
as.numeric(auc_model1@y.values)*2-1



# Out of sample validation ------------------------------------------------

#Get a 75% estimation sample and 25% validation sample
set.seed(1234)
Titanic$estimation_sample <-rbinom(nrow(Titanic), 1, 0.75)


#Estimate the model using only the estimation sample
Logistic_regression2 <- glm(survived ~ as.factor(sex) + as.factor(pclass) + age, family=binomial, data=Titanic, subset=estimation_sample==1)


#Create a new dataframe with only the validation sample
our_validation_dataset <- Titanic[Titanic$estimation_sample==0,]

#Get predictions for all observations
predictions_model2 <- predict(Logistic_regression2, type = "response", newdata= our_validation_dataset)

### After this you can calculate the fit criteria on this validation sample





# Lecture 2 ---------------------------------------------------------------

# Step wise regression ----------------------------------------------------


library(MASS) 

#Estimate full and null model
Logistic_regression_full <- glm(survived ~ ., data = Titanic, family = binomial, subset=estimation_sample==1)
Logistic_regression_null <- glm(survived ~ 0, data = Titanic, family = binomial, subset=estimation_sample==1)

# Fit the model backward
Logistic_regression_backward <- stepAIC(Logistic_regression_full, direction="backward", trace = TRUE)

# Fit the model forward
Logistic_regression_forward <- stepAIC(Logistic_regression_null, direction="forward", scope=list(lower=Logistic_regression_null, upper=Logistic_regression_full), trace = TRUE)

# Fit the model both directions
Logistic_regression_both <- stepAIC(Logistic_regression_full, direction="both", trace = TRUE)


## To do step-wise regression with the BIC you can add "k = log(n)" (where n is the amount of observations on which the model is estimated) to the stepAIC function, example:

# Fit the model backward using BIC
Logistic_regression_backward_BIC <- stepAIC(Logistic_regression_full, direction="backward", trace = TRUE, k = log(sum(Titanic$estimation_sample)))



# Estimate a CART tree ----------------------------------------------------


# Tree model
library(rpart)
library(partykit)
Cart_tree1 <- rpart(survived ~ as.factor(sex) + as.factor(pclass) + age, data=Titanic, method="class", subset=estimation_sample==1)
Cart_tree1_visual <- as.party(Cart_tree1)
plot(Cart_tree1_visual , type="simple", gp = gpar(fontsize = 10))


# Changing settings
newsettings1 <- rpart.control(minsplit = 100, minbucket = 50, cp = 0.01, maxdepth = 3)

Cart_tree2 <- rpart(survived ~ as.factor(sex) + as.factor(pclass) + age, data=Titanic, method="class", subset=estimation_sample==1, control=newsettings1)
Cart_tree2_visual <- as.party(Cart_tree2)
plot(Cart_tree2_visual , type="simple")


#Save predictions
predictions_cart1 <- predict(Cart_tree1, newdata=our_validation_dataset, type ="prob")



# Bagging -----------------------------------------------------------------
library(ipred)
library(caret)

newsettings2 <- rpart.control(minsplit = 2, cp = 0.0)
#Essentially these two arguments allow the individual trees to grow extremely deep, which leads to trees with high variance but low bias. Then when we apply bagging we're able to reduce the variance of the final model while keeping the bias low.

#estimate model with bagging
Bagging_tree1 <- train(as.factor(survived) ~ sex + pclass + age, data=Titanic, method="treebag", nbagg=500, subset=estimation_sample==1, trControl = trainControl(method = "cv", number = 10), control=newsettings2)

#Save predictions
predictions_bagging1 <- predict(Bagging_tree1, newdata=our_validation_dataset, type ="prob")[,2]


#calculate variable importance...
pred.imp <- varImp(Bagging_tree1)
pred.imp


#You can also plot the results
barplot(pred.imp$importance$Overall, names.arg = row.names(pred.imp$importance))

#For more details, see: https://bradleyboehmke.github.io/HOML/bagging.html



# Boosting ----------------------------------------------------------------

library(gbm)
estimation_sample <- Titanic[Titanic$estimation_sample==1,]

#Estimate the model
boost_tree1 <- gbm(survived ~ as.factor(sex) + as.factor(pclass) + age, data=Titanic, distribution = "bernoulli", n.trees = 10000, shrinkage = 0.01, interaction.depth = 4)

#Get model output (summary also provides a graph)
boost_tree1

best.iter <- gbm.perf(boost_tree1, method = "OOB")
summary(boost_tree1, n.trees = best.iter)

#Save predictions
predictions_boost1 <- predict(boost_tree1, newdata=our_validation_dataset, n.trees = best.iter, type ="response")

#See also: https://datascienceplus.com/gradient-boosting-in-r/ 



# Lecture 3 ---------------------------------------------------------------

# Random forest -----------------------------------------------------------

library(randomForest)

Random_forest1 <- randomForest(as.factor(survived) ~ ., data=Titanic, subset=estimation_sample==1, importance=TRUE)

predictions_forest1 <- predict(Random_forest1, newdata=our_validation_dataset, type ="prob")[,2]

varImpPlot(Random_forest1)

#Some extra setting you can play around with
Random_forest1 <- randomForest(as.factor(survived) ~ ., data=Titanic, subset=estimation_sample==1,
                               ntree=500, mtry=3, nodesize=1, maxnodes=100, importance=TRUE)



# Support Vector Machine --------------------------------------------------

library(e1071)

svm_1 <- svm(survived ~ age + fare, data = Titanic[,1:9], subset=Titanic$estimation_sample==1,
                 type = 'C-classification', probability = TRUE,
                 kernel = 'linear')

plot(svm_1, Titanic, age~fare)

#Get predictions
predictions_svm1 <- predict(svm_1, newdata=our_validation_dataset, probability=TRUE)
predictions_svm1 <- attr(predictions_svm1,"probabilities")[,1]


#Same models, other functions
svm_2 <- svm(survived ~ age + fare, data = Titanic[,1:9], subset=Titanic$estimation_sample==1,
             type = 'C-classification', probability = TRUE,
             kernel = 'polynomial')


#Change the functional form
plot(svm_2, Titanic, age~fare)


svm_3 <- svm(survived ~ age + fare, data = Titanic[,1:9], subset=Titanic$estimation_sample==1,
             type = 'C-classification', probability = TRUE,
             kernel = 'radial')

plot(svm_3, Titanic, age~fare)


svm_4 <- svm(survived ~ age + fare, data = Titanic[,1:9], subset=Titanic$estimation_sample==1,
             type = 'C-classification', probability = TRUE,
             kernel = 'sigmoid')

plot(svm_4, Titanic, age~fare)


# Artificial Neural Networks ----------------------------------------------

library(neuralnet)

# fit neural network
ANN_1 <- neuralnet(survived ~ age + fare + pclass, data = estimation_sample, hidden=c(3,2), stepmax=1e+06, act.fct = "logistic",
             linear.output = FALSE)

plot(ANN_1)

# get model predictions
temp_test <- subset(our_validation_dataset, select = c("survived", "age", "fare", "pclass"))
head(temp_test)
nn.results <- compute(ANN_1, temp_test)
nn.results <- nn.results$net.result
