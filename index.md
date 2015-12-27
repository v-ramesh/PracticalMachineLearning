# Practical Machine Learning Course Project
December 26, 2015  

# Background and Introduction

By using devices such as Jawbone Up, Nike FuelBand, and Fitbit, it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The five ways are: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Only Class A corresponds to correct performance. The goal of this project is to predict the manner in which they did the exercise, i.e., Class A to E. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

# Data Processing

## Load Requisite Libraries


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.0.5 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(rpart)
library(rpart.plot)
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(repmis)
```

## Import the data

We download the training and testing data sets from the given URLs.


```r
# Load Data
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
```
The outcome variable we are interested in prediciting is `classe`.

## Preprocessing

We delete any columns (predictors) of the training set that contain any missing values. 


```r
# Delete columns containing missing values
training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]
```

We also remove the first seven predictors since by looking up the meaning of these variables, it seems reasonable to assume that these should not be used to predict the outcome `classe`. 


```r
trainData <- training[, -c(1:7)]
testData <- testing[, -c(1:7)]
```


```r
dim(trainData)
```

```
## [1] 19622    53
```

```r
dim(testData)
```

```
## [1] 20 53
```

The cleaned data sets `trainData` and `testData` both have 53 columns. While the first 52 variables are the same in both data sets, the last variable in the case of `trainData` is the response variable `classe` while in the case of `testData`, it is `problem_id`. 


## Splitting the Data

We split the cleaned training set `trainData` into a training set (70%) for learning the model, and a cross-validation set (30%) for evaluation of the model.


```r
# Set seed for reproducibility
set.seed(1234) 
inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
train <- trainData[inTrain, ]
valid <- trainData[-inTrain, ]
```

# Prediction Algorithms

We will explore classification trees and random forests for our prediction algorithms.

## Classification Trees

In practice, k is set to 5 or 10 when doing k-fold cross validation. Here we consider 5-fold cross validation (default setting in `trainControl` function is 10) when implementing the algorithm to save a little computing time. Since data transformations may be less important in non-linear models like classification trees, we do not transform any variables.


```r
# Classification Tree
control <- trainControl(method = "cv", number = 5)
fit_rpart <- train(classe ~ ., data = train, method = "rpart", trControl = control)
print(fit_rpart, digits = 4)
```

```
## CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10988, 10991, 10990, 10989 
## Resampling results across tuning parameters:
## 
##   cp       Accuracy  Kappa    Accuracy SD  Kappa SD
##   0.03550  0.5214    0.38010  0.02029      0.03419 
##   0.06093  0.4175    0.21094  0.07043      0.11811 
##   0.11738  0.3333    0.07467  0.04484      0.06843 
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.0355.
```


```r
fancyRpartPlot(fit_rpart$finalModel)
```

![](index_files/figure-html/unnamed-chunk-8-1.png) 


```r
# Predict outcomes for validation set
predict_rpart <- predict(fit_rpart, valid)
# Evaluate prediction results
conf_rpart <- confusionMatrix(valid$classe, predict_rpart); conf_rpart
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1530   35  105    0    4
##          B  486  379  274    0    0
##          C  493   31  502    0    0
##          D  452  164  348    0    0
##          E  168  145  302    0  467
## 
## Overall Statistics
##                                           
##                Accuracy : 0.489           
##                  95% CI : (0.4762, 0.5019)
##     No Information Rate : 0.5317          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3311          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4890   0.5027   0.3279       NA  0.99151
## Specificity            0.9478   0.8519   0.8797   0.8362  0.88641
## Pos Pred Value         0.9140   0.3327   0.4893       NA  0.43161
## Neg Pred Value         0.6203   0.9210   0.7882       NA  0.99917
## Prevalence             0.5317   0.1281   0.2602   0.0000  0.08003
## Detection Rate         0.2600   0.0644   0.0853   0.0000  0.07935
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638  0.18386
## Balanced Accuracy      0.7184   0.6773   0.6038       NA  0.93896
```

```r
error_rpart <- (1 - conf_rpart$overall[1]); error_rpart
```

```
##  Accuracy 
## 0.5109601
```

From the above results, we see that the the accuracy rate is only 0.489, and therefore the out-of-sample error rate is 0.511. Classification trees therefore do not seem like a very effective strategy for predicting the variable `classe`.

## Random Forests

We now consider the random forest method as our predictive learning algorithm.


```r
# Random Forest
fit_rf <- train(classe ~ ., data = train, method = "rf", trControl = control)
print(fit_rf, digits = 4)
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10990, 10988, 10990, 10990 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa   Accuracy SD  Kappa SD
##    2    0.9908    0.9884  0.001950     0.002469
##   27    0.9924    0.9903  0.002185     0.002765
##   52    0.9886    0.9856  0.003081     0.003898
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


```r
# Predict outcomes for validation set
predict_rf <- predict(fit_rf, valid)
# Evaluate prediction results
conf_rf <- confusionMatrix(valid$classe, predict_rf); conf_rf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B   12 1126    1    0    0
##          C    0    5 1017    4    0
##          D    0    2    6  955    1
##          E    0    1    2    3 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9937          
##                  95% CI : (0.9913, 0.9956)
##     No Information Rate : 0.2865          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.992           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9929   0.9929   0.9912   0.9927   0.9991
## Specificity            1.0000   0.9973   0.9981   0.9982   0.9988
## Pos Pred Value         1.0000   0.9886   0.9912   0.9907   0.9945
## Neg Pred Value         0.9972   0.9983   0.9981   0.9986   0.9998
## Prevalence             0.2865   0.1927   0.1743   0.1635   0.1830
## Detection Rate         0.2845   0.1913   0.1728   0.1623   0.1828
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9964   0.9951   0.9947   0.9954   0.9989
```

```r
error_rf <- 1- conf_rf$overall[1]; error_rf
```

```
##    Accuracy 
## 0.006287171
```

From the above results, we see that the accuracy rate is 0.9937, and the out-of-sample error rate is 0.0063. Thus, random forest siginficantly outperforms the classification tree method, in predictive accuracy, for this data set.  This may be due to the fact that many predictors are highly correlated. Random forests chooses a subset of predictors at each split and decorrelates the trees. This leads to high accuracy, although this algorithm is sometimes difficult to interpret and computationally inefficient.

# Prediction on Testing Set

We now use the model constructed using random forests to predict the outcome variable classe for the testing set.


```r
# Predict outcomes for test set
answers <- predict(fit_rf, testData)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

In submitting these results to the course website, we found that all of them were correct.

# Conclusion

In this report, we have used machine learning algorithms to predict how an exercise was performed based on data from activity monitors. We split the training set into a cross-training set (used to build the prediction algorithm) and a validation set (used to evalute out-of-sample errors) with 70% of the data being included in the cross-training set and 30% in the validation set. We considered decision trees and random forests for the machine learning algorithms, and found random forests to be significantly better in their predictive accuracy. Our final machine learning algorithm was therefore chosen to be the one based on random forests. Based on cross-validation with a random 70-30 split, we expect the out-of-sample error rate to be 0.0063. We have run our final algorithm on the 20 test cases provided on which all the predictions turned out to be correct.

