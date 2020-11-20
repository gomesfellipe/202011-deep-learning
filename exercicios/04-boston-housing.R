# Agora vamos usar bases de dados de verdade!
# Ajuste um MLP com quantas layers e unidades escondidas
# você preferir.
library(keras)

base <- dataset_boston_housing()
x <- base$train$x
x <- apply(x, 2, function(x){ (x - min(x)) / (max(x)-min(x)) }) # min-max transformation

y <- base$train$y
y <- (y - min(y)) / (max(y)-min(y))

# Model definition ---------------------------------------------
input <- layer_input(shape = ncol(x))

output <- input %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  # layer_dropout(0.5) %>%
  layer_dense(units = 64, activation = "relu") %>% 
  # layer_dropout(0.25) %>%
  layer_dense(units = 1)

model <- keras_model(input, output)

summary(model)

model %>% 
  compile(
    loss = "mse",
    optimizer = "sgd",
    metrics = "mape"
  )

# Model fitting ------------------------------------------------ 

# The patience parameter is the amount of epochs to check for improvement.
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 10)


history <- model %>% 
  fit(x, y, 
      batch_size = 16,
      epochs = 500,
      validation_split = 0.2,
      callbacks = list(early_stop))

y_pred <- predict(model, x)
plot(y_pred, y)
abline(0, 1, col = "red")


yardstick::mape_vec(as.numeric(y), as.numeric(y_pred))
yardstick::rmse_vec(as.numeric(y), as.numeric(y_pred))


# ref: https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_regression/


# Comparar com lm ---------------------------------------------------------

model <- lm(y~x)
par(mfrow = c(2,2))
plot(model, which = 1:4)
par(mfrow = c(1,1))

yardstick::mape_vec(as.numeric(y), as.numeric(predict(model, as.data.frame(x))))
yardstick::rmse_vec(as.numeric(y), as.numeric(predict(model, as.data.frame(x))))
