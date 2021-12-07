# -------------------------------------------------------------------------
################## USING NAIVE BAYES CLASSIFIER ################### 
# -------------------------------------------------------------------------

# first way of executing sentiment analysis.

# step 1: load library ----------------------------------------------------

library(readxl)           
library(tm)               
library(e1071)             
library(RTextTools)       
library(caret)            
library(gmodels)          
library(pillar)
library(quanteda)         
library(tidyverse)        

# -------------------------------------------------------------------------
# STEP 1: DATA PREPARATION
# -------------------------------------------------------------------------

# load up the excel data and explore data in R studio----------------------
#1K reviews
revRaw <- read_excel("IMDBDataset_altered/IMDBDataset.xlsx")

#20K reviews
revRaw <- read_excel("IMDBDataset_altered/IMDB20K.xlsx")

#50K reviews
revRaw <- read_excel("IMDBDataset_altered/IMDB50K.xlsx")

#shows the raw data of movie reviews
view(revRaw)

#check data to see if there are missing values
length(which(!complete.cases(revRaw)))

# factorise your dependent variable into categorical object/function/variable
revRaw$sentiment <- factor(revRaw$sentiment)

# Check the counts of positive and negative scores
table(revRaw$sentiment)
#in percentage
prop.table(table(revRaw$sentiment))

# -------------------------------------------------------------------------
# STEP 2: DATA PRE-PROCESSING
# -------------------------------------------------------------------------
# sampling
# Create random samples // #1000,800 / 20k, 16k / 50k, 40k
set.seed(123)
#1k
revIndex <- sample(1000, 800) 

#20k
revIndex <- sample(20000, 16000)

#50k
revIndex <- sample(50000, 40000) 

#split to training and testing set
revTrain <- revRaw[revIndex, ]
revTest  <- revRaw[-revIndex, ]

# review the proportion of class variable
prop.table(table(revTrain$sentiment))
prop.table(table(revTest$sentiment))

#Create a corpus from the sentences
corpTrain <- VCorpus(VectorSource(revTrain$review))
corpTest <- VCorpus(VectorSource(revTest$review))

# create a document-term sparse matrix directly for training set
dtmTrain <- DocumentTermMatrix(corpTrain, control = list(
  language="english",
  tolower = TRUE,
  removeNumbers = TRUE,
  removePunctuation = TRUE,
  stopwords = TRUE,
  stemming = TRUE,
  removeSparseTerms = .998,
  weighting = tm:: weightTfIdf,
  stripWhitespace = TRUE
))

#check sparse terms
dtmTrain
#remove sparse terms
dtmTrain <- removeSparseTerms(dtmTrain, 0.99)
#summary of train dtm
glimpse(dtmTrain)

# create a document-term sparse matrix directly for test set
dtmTest <- DocumentTermMatrix(corpTest, control = list(
  language="english",
  tolower = TRUE,
  removeNumbers = TRUE,
  removePunctuation = TRUE,
  stopwords = TRUE,
  stemming = TRUE,
  removeSparseTerms = .998,
  weighting = tm:: weightTfIdf,
  stripWhitespace = TRUE
))

#check sparse terms
dtmTest
#remove sparse terms
dtmTest <- removeSparseTerms(dtmTest, 0.99)
#summary of test dtm 
glimpse(dtmTest)

# create function to convert counts to a factor
# ifelse(condition, value if condition is true, value if condition is false)
convertCount <- function(x) { x <- ifelse( x> 0, "Positive", "Negative") }

# apply() convert_counts() to columns of train/test data
binaryTrain <- apply(dtmTrain, MARGIN = 2, convertCount)
binaryTest  <- apply(dtmTest, MARGIN = 2, convertCount)

#check term one by one
binaryTrain
binaryTest

#check summary
str(binaryTrain)
str(binaryTest)

# -------------------------------------------------------------------------
# STEP 3: CLASSIFICATION & EVALUATION
# -------------------------------------------------------------------------
# train model on a data -------------------------------------------

# naive bayes -------------------------------------------------------------

#train on training data
revClassifier <- naiveBayes(as.matrix(binaryTrain), revTrain$sentiment)

# test on new data (which is the test data) -------------------------------
revPredict <- predict(revClassifier, as.matrix(binaryTest))

result <- confusionMatrix(revPredict, revTest$sentiment,
                          dnn = c('pred', 'real'))


#to check whole result including accuracy, precision, recall and f-1
result

# using precision aka pos pred value
result[["byClass"]]["Pos Pred Value"]

# using recall aka sensitivity
result[["byClass"]]["Sensitivity"] 

# using f-measure aka F1
result[["byClass"]]["F1"] 


# -------------------------------------------------------------------------
################## USING SVM CLASSIFIER ################### 
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# STEP 1: DATA PREPARATION
# -------------------------------------------------------------------------

#load the package

#set the seed and load/read the data
set.seed(123)

#1k
reviewLabel <- read_excel("IMDBDataset_altered/IMDBDataset.xlsx")

#20K reviews
reviewLabel <- read_excel("IMDBDataset_altered/IMDB20K.xlsx")

#50K reviews
reviewLabel <- read_excel("IMDBDataset_altered/IMDB50K.xlsx")

#factorise dependent var of chr into a factor
reviewLabel$sentiment <- factor(reviewLabel$sentiment)

#change to corpus from sentences
reviewCorpus <- VCorpus(VectorSource(reviewLabel$review))

# -------------------------------------------------------------------------
# STEP 2: DATA PRE-PROCESSING
# -------------------------------------------------------------------------

#document term matrix to separate the sentences into individual word
dtMatrix <- DocumentTermMatrix(reviewCorpus, control = list(
  language="english",
  tolower = TRUE,
  removeNumbers = TRUE,
  removePunctuation = TRUE,
  stopwords = TRUE,
  stemming = TRUE,
  weighting = tm::weightTfIdf,
  stripWhitespace = TRUE
))


#check sparse
dtMatrix
#remove sparse terms
dtMatrix <- removeSparseTerms(dtMatrix, 0.99)
#summary sparse
glimpse(dtMatrix)

#create a container to split the data into train and test set. #1000,800 / 20k, 16k / 50k, 40k
#1000, 800
svmContainer <- create_container(dtMatrix, as.numeric(as.factor(reviewLabel$sentiment[])), 
                              trainSize=1:800, 
                              testSize = 801:1000, 
                              virgin = FALSE)

#20000, 16K
svmContainer <- create_container(dtMatrix, as.numeric(as.factor(reviewLabel$sentiment[])), 
                                 trainSize=1:16000, 
                                 testSize = 16001:20000, 
                                 virgin = FALSE)

#50000, 40K
svmContainer <- create_container(dtMatrix, as.numeric(as.factor(reviewLabel$sentiment[])), 
                                 trainSize=1:40000, 
                                 testSize = 40001:50000, 
                                 virgin = FALSE)



# -------------------------------------------------------------------------
# STEP 3: CLASSIFICATION & EVALUATION
# -------------------------------------------------------------------------

#train the model
svmTrain = train_model(svmContainer, "SVM")

#predict the model
svmPredict = classify_model(svmContainer, svmTrain)

# model summary: precision, recall, fmeasure
svmAnalytics = create_analytics(svmContainer, svmPredict)
summary(svmAnalytics)

head(svmAnalytics@document_summary)
svmAnalytics@ensemble_summary

N=4
set.seed(2014)
#cross validate for svm to check accuracy with 4 fold 
crossVal <- cross_validate(svmContainer,N,"SVM")
crossVal
