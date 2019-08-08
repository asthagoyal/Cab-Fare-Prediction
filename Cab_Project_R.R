rm(list=ls())

setwd("E:/MY/Project/Cab Fare Prediction")

x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees', "usdm", "party")


install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)


Train = read.csv("train.csv", header = T, na.strings = c(" ", "", "NA"))
str(Train)
dim(Train)

Test= read.csv("test.csv", header = T, na.strings = c(" ", "", "NA"))
str(Test)

#covert factor to numeric
Train$fare_amount = as.numeric(Train$fare_amount)

#Missing Value Analysis
missing_val = data.frame(apply(Train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(Train)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
View(missing_val)

#Median Method
Train$fare_amount[is.na(Train$fare_amount)] = median(Train$fare_amount, na.rm = T)
Train$passenger_count[is.na(Train$passenger_count)] = median(Train$passenger_count, na.rm = T)

#covert Train datetime
Train$pickup_datetime1 = as.Date(Train$pickup_datetime)
Train$year = as.numeric(format(Train$pickup_datetime1, format = "%Y"))
Train$month = as.numeric(format(Train$pickup_datetime1, format = "%m"))
Train$day = as.numeric(format(Train$pickup_datetime1, format = "%d"))
Train$pickup_time <- sapply(strsplit(as.character(Train$pickup_datetime), " "), "[", 2)
Train$Hour <- sapply(strsplit(as.character(Train$pickup_time), ":"), "[", 1)
Train$pickup_datetime = NULL
Train$pickup_datetime1 =NULL
Train$pickup_time =NULL
View(Train)

#covert Test datetime

Test$pickup_datetime1 = as.Date(Test$pickup_datetime)
Test$year = as.numeric(format(Test$pickup_datetime1, format = "%Y"))
Test$month = as.numeric(format(Test$pickup_datetime1, format = "%m"))
Test$day = as.numeric(format(Test$pickup_datetime1, format = "%d"))
Test$pickup_time <- sapply(strsplit(as.character(Test$pickup_datetime), " "), "[", 2)
Test$Hour <- sapply(strsplit(as.character(Test$pickup_time), ":"), "[", 1)
Test$pickup_datetime = NULL
Test$pickup_datetime1 =NULL
Test$pickup_time =NULL

View(Test)

#Missing Value check
Train$Hour = as.numeric(Train$Hour)
Test$Hour = as.numeric(Test$Hour)

Train$Hour[is.na(Train$Hour)] = mean(Train$Hour, na.rm = T)
Train$year[is.na(Train$year)] = mean(Train$year, na.rm = T)
Train$month[is.na(Train$month)] = mean(Train$month, na.rm = T)
Train$day[is.na(Train$day)] = mean(Train$day, na.rm = T)

View(Train)

#Distribution of train attributes

 hist(Train$pickup_longitude, breaks=7)
 hist(Train$dropoff_longitude, breaks=7)
 hist(Train$pickup_latitude, breaks=7)
 hist(Train$dropoff_latitude, breaks=7)
 hist(Train$Hour, breaks=7)
 hist(Train$year, breaks=7)
 hist(Train$passenger_count)
 
 hist(Test$pickup_longitude, breaks=7)
 hist(Test$dropoff_longitude, breaks=7)
 hist(Test$pickup_latitude, breaks=7)
 hist(Test$dropoff_latitude, breaks=7)
 hist(Test$Hour, breaks=7)
 hist(Test$year, breaks=7)
 hist(Test$passenger_count)
 
 
numeric_index = sapply(Train,is.numeric) #selecting only numeric
numeric_data = Train[,numeric_index - 2]
cnames = colnames(numeric_data)
cnames

numeric_index1 = sapply(Test,is.numeric) #selecting only numeric
numeric_data1 = Test[,numeric_index1 ]
cnames1 = colnames(numeric_data1)
cnames1


 
 #remove outlier from test and Train
 for(i in cnames){
   print(i)
   val = Train[,i][Train[,i] %in% boxplot.stats(Train[,i])$out]
   #print(length(val))
   Train = Train[which(!Train[,i] %in% val),]
 }

 

for(i in cnames1){
  print(i)
  val = Test[,i][Test[,i] %in% boxplot.stats(Test[,i])$out]
  #print(length(val))
  Test = Test[which(!Test[,i] %in% val),]
}
 
 qqnorm(Test$pickup_latitude)
 hist(Train$pickup_longitude)
 #for Train
 for(i in cnames){
   print(i)
   Train[,i] = (Train[,i] - min(Train[,i]))/(max(Train[,i] - min(Train[,i])))
 } 
 
 #For Test
 for(i in cnames1){
   print(i)
   Test[,i] = (Test[,i] - min(Test[,i]))/(max(Test[,i] - min(Test[,i])))
 } 
 
 ##################################Feature Selection################################################
 
corrgram(Train[,numeric_index], order = F,
          upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot Train")
 
 
corrgram(Test[,numeric_index1], order = F,
          upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot Test")
 
## Dimension Reduction
Train = subset(Train, select = -c(Hour,day,month, year))
View(Train)
Test = subset(Test, select = -c(Hour,day,month, year))
View(Test)

###################################Model Development#######################################

#############Linear Regression###################3


vif(Train[,-1])

vifcor(Train[,-1], th = 0.9)

train.index = createDataPartition(Train$fare_amount, p = .80, list = FALSE)
train = Train[ train.index,]
test  = Train[-train.index,]

#run regression model
lm_model = lm(fare_amount ~., data = train)

#Summary of the model
summary(lm_model)

#Predict
predictions_LR = predict(lm_model, test[,2:6])

RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}

RMSE(test[,1],predictions_LR)
#17.260

########Decion Tree#########3


fit = rpart(fare_amount ~ ., data = train, method = "anova")
predictions_DT = predict(fit, test[, -1])

#RMSE

RMSE(test[,1], predictions_DT)
#15.324

###################Random Forest####################

rm_model <- randomForest(fare_amount ~., data = train)
print(rm_model) 
predictions_RF = predict(rm_model, test[, -1])
RMSE(test[,1], predictions_RF)

#14.379
