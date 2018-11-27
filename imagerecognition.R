# Install libraries
install.packages('keras')

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("EBImage", version = "3.8")

install.packages("tensorflow")

#load libraries
library(EBImage)
library(keras)
library(tensorflow)
#install_tensorflow()
#install_tensorflow(method = "conda")
#install_tensorflow(version = "nightly")
#install_tensorflow(version = "nightly-gpu")

#Read Images
pics <- c('p1.jpg','p2.jpg','p3.jpg','p4.jpg','p5.jpg','p6.jpg',
          'c1.jpg','c2.jpg','c3.jpg','c4.jpg','c5.jpg','c6.jpg')
mypic <- list()
for(i in 1:12){mypic[[i]] <- readImage(pics[i])}

#Explore
print(mypic[[1]])
display(mypic[[1]])
summary(mypic[[1]])
hist(mypic[[2]])
str(mypic)

#Resize
for(i in 1:12){mypic[[i]] <- resize(mypic[[i]],28,28)}
str(mypic)

#Reshape
for(i in 1:12){mypic[[i]] <- array_reshape(mypic[[i]],c(28,28,3))}
str(mypic)

#Row Bind
trainx <- NULL
for(i in 7:11){trainx <- rbind(trainx,mypic[[i]])}
str(trainx)
testx <- rbind(mypic[[6]],mypic[[12]])
trainy <- c(0,0,0,0,0,1,1,1,1,1)
testy <- c(0,1)

#one hot encoding
trainLables <- to_categorical(trainy)
testLabels <- to_categorical(testy)

#Model
model <- keras_model_sequential()
model %>% 
         layer_dense(units = 256, activation = 'relu', input_shape = c(2352)) %>%
         layer_dense(units = 128, activation = 'relu') %>%
         layer_dense(units = 2, activation = 'softmax')
summary(model)

#compile
model %>%
         compile(loss = 'binary_crossentropy',
                 optimizer = optimizer_rmsprop(),
                 metrics = c('accuracy'))

#Fit Model
history <- model %>%
          fit(trainx,
              trainLables,
              batch_size=32,
              validation_split=0.2)

#Evaluation & Prediction - train data
model %>% evaluate(trainx, trainLables)
pred <- model %>% predict_classes(trainx)
table(predicted = pred, Actual = trainy)
prob <- model %>% predict_proba(trainx)
prob
cbind(prob, predicted = pred, Actual = trainy)

#Evaluation & Prediction - test data
model %>% evaluate(testx, testLabels)
pred1 <- model %>% predict_classes(testx) 
table(predicted = pred1, Actual = testy)
