# Ajuste uma MLP para a base do MNIST com uantas layers você preferir.
# O mais importante é lembrar da loss e da ativação da última camada.
library(keras)

base <- dataset_mnist()
x <- array_reshape(base$train$x/256, dim = c(60000, 784))
y <- to_categorical(base$train$y)

dim(x)
dim(y)

# Model definition ---------------------------------------------
input <- layer_input(shape = 28*28)

output <- input %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

model <- keras_model(input, output)

model %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

# Model fitting ------------------------------------------------ 

model %>% 
  fit(x, y, batch_size = 32, epochs = 10, validation_split = 0.2)

# Model predict -----------------------------------------------------------

y_pred <- predict(model, x)
classes <- 0:9
y_pred_class <- classes[apply(y_pred, 1, which.max)]

# Matriz de confusao
caret::confusionMatrix(factor(base$train$y), factor(y_pred_class))
