# Ajuste uma rede neural para prever as classes do CIFAR10.
# Use batch_norm e dropouts para melhorar o seu modelo.
# Lembre-se que usamos batch_norm anets da ativação então escreva algo
# assim:
#
# input %>% 
#   layer_conv2d(...) %>% 
#   layer_batch_norm() %>% 
#   layer_activation()
# 

base <- dataset_cifar10()
x <- base$train$x/256
y <- to_categorical(base$train$y)

dim(x)
dim(y)

# Model definition ---------------------------------------------

# Model
input <- layer_input(shape = c(32, 32, 3))

output <- input %>%   
  # CONV -> RELU -> CONV -> BATCH NORM -> RELU -> POOL -> DROPOUT
  layer_conv_2d(kernel_size = c(3,3), filters = 32) %>% 
  layer_activation_relu() %>% 
  layer_conv_2d(kernel_size = c(3,3), filters = 32) %>% 
  layer_batch_normalization() %>% 
  layer_activation_relu() %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_dropout(0.25) %>%
  # CONV -> BATCH NORM -> RELU -> POOL -> DROPOUT
  layer_conv_2d(kernel_size = c(3,3), filters = 64) %>% 
  layer_batch_normalization() %>% 
  layer_activation_relu() %>% 
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
