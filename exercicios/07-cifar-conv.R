# Ajuste uma rede neural para prever as classes do CIFAR10.
library(keras)

base <- dataset_cifar10()
x <- base$train$x/256
y <- to_categorical(base$train$y)

dim(x)
dim(y)

# Model definition ---------------------------------------------

# Model
input <- layer_input(shape = c(32, 32, 3))


output <- input %>%   
  # CONV -> RELU -> CONV -> RELU -> POOL -> DROPOUT
  layer_conv_2d(kernel_size = c(3,3), filters = 32, activation = "relu") %>%
  layer_conv_2d(kernel_size = c(3,3), filters = 64, activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_dropout(0.25) %>%
  # CONV -> RELU -> CONV -> RELU -> POOL -> DROPOUT
  layer_conv_2d(kernel_size = c(3,3), filters = 128, activation = "relu", padding = "same") %>% 
  layer_conv_2d(kernel_size = c(3,3), filters = 256, activation = "relu", padding = "same") %>% 
  layer_max_pooling_2d(pool_size = c(2,2), strides = c(2,2)) %>% 
  layer_dropout(0.2) %>%
  # FLATTERN -> DENSE -> RELU -> DROPOUT
  layer_flatten() %>% 
  layer_dense(128, activation = "relu") %>% 
  layer_dropout(0.2) %>%
  # Softmax Classifier
  layer_dense(10, activation = "softmax") 

model <- keras_model(input, output)
summary(model)

# Compile model
model %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

# Model fitting 
history <- 
  model %>% 
  fit(x, y, batch_size = 32, epochs = 5, validation_split = 0.2)  

history

plot(history)

# Model predict

y_pred <- predict(model, x_test)
classes <- 0:9
y_pred_class <- classes[apply(y_pred, 1, which.max)]

# Submit
cbind(imageId = 1:length(y_pred_class), Label = y_pred_class) %>% 
  write.csv("submission.csv", row.names = F)