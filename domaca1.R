library(caret)
library(readr)
library(dplyr)

##### DATA IMPORT AND PREPROCESSING
myData <- read.csv("ucni_podatki.csv", head=FALSE)
myData2 <- read.csv("ucni_razredi.csv", head=FALSE)
names(myData) <- c('indeks', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6')
names(myData2) <- c('indeks', 'Y')

myData2$Y <-ifelse(myData2$Y=='D', 1,0)
myData2$Y <- as.factor(myData2$Y)

podatki <- merge(myData, myData2, by = "indeks")
View(podatki)

#ce pogledam podatke, vidim, da so tezave v stolpcih X1, X3 in Y (manjkajoce vrednosti, -), "", ...)
str(podatki)
lapply(podatki, typeof)
podatki$X1 <- as.numeric(as.character(podatki$X1))
podatki$X3 <- as.numeric(as.character(podatki$X3))
podatki <- na.omit(podatki)
podatki$indeks <- NULL

levels(podatki$Y)
# levels[1] = 0
# levels[2] = 1

#####F_2

F_beta <- function(data, levels, ...) {
  beta <- 2
  TP <- sum(data$obs==levels[2] & data$pred==levels[2])
  FP <- sum(data$obs==levels[1] & data$pred==levels[2])
  FN <- sum(data$obs==levels[2] & data$pred==levels[1])
  P <- TP + FN
  PPV <- TP / (TP + FP)
  TPR <- TP / P
  f_beta <- (1+beta*beta)*PPV*TPR/(PPV*beta*beta + TPR)
  names(f_beta) <- 'F_beta'
  return(f_beta)
}


#####KNN MODEL
cvtc <- trainControl(method='cv', number=10, savePredictions = TRUE, summaryFunction = F_beta)
knnModel <- train(Y~., data=podatki, method='knn', metric = 'F_beta', tuneGrid=data.frame(k=1:100), 
                  trControl=cvtc)

#najboljsi model je pri k=7, F_2 je za ta model priblizno 0.65
#zakaj dobim vsakic, ko pozenem train knnModel, razlicen odgovor za optimalen k? 
#verjetno posledica precnega preverjanja...

#####LOGISTICNA REGRESIJA

cvtc <- trainControl(method='cv', index = createFolds(podatki$Y, k=10, returnTrain=TRUE), 
                     summaryFunction = F_beta)
logitModel <- train(Y~., data=podatki, method='glm', trControl=cvtc)

#kopirano iz vaje 3.1
cvtc <- trainControl(method='cv', index = createFolds(podatki$Y, k=10, returnTrain=TRUE), 
                     summaryFunction = F_beta)
errors <- data.frame(deg=1:3, err=rep(-1,3), sd=rep(-1,3))
for(k in 1:3){
  tmpData <- data.frame(poly(as.matrix(podatki[, 1:6]), degree=k, raw=TRUE), Y=podatki$Y)
  tmpModel <- train(Y~., data=tmpData, method='glm', trControl=cvtc)
  # Shranimo izraeunane napake
  errors$err[k] <- tmpModel$results$F_beta
  errors$sd[k] <- tmpModel$results$F_betaSD
}
errors

plot(errors$deg, errors$err, type='o', col='green')

#najboljsi model je pri k=1 (torej brez clenov visjih redov), F_2 je za ta model priblizno 0.68



#####RANDOM FOREST
# https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/

cvtc <- trainControl(method='cv', index = createFolds(podatki$Y, k=10, returnTrain=TRUE), 
                     summaryFunction = F_beta)
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:15))
rfModel <- train(Y~., data=podatki, method="rf", metric='F_beta', tuneGrid=tunegrid, trControl=cvtc)
plot(rfModel)

#najboljsi model pri parametru mtry=1, F_beta priblizno 0.68


#####UPORABA NAJBOLJSEGA MODELA NA TESTNIH PODATKIH

testni <- read.csv('testni_podatki.csv', header=FALSE)
View(testni)

names(testni) <- c('indeks', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6')
str(testni)
#vidimo, da je problem s stolpcem X1
testni$X1 <- as.numeric(as.character(testni$X1))
testni$X1[20] = 8.38371346378874 * 10^-5
testni <- na.omit(testni)
testni$indeks <- NULL

#izvedemo napoved
testni$pred <- predict(knnModel, testni)
testni$pred <-ifelse(testni$pred==0, 'N','D')

#zapis rezultatov na csv datoteko
final <- data.frame(indeks=1001:2000, napoved=testni$pred)
View(final)

write.table(final, file = "testni_razredi.csv",row.names = FALSE, col.names = FALSE, sep = ',', quote = FALSE)

