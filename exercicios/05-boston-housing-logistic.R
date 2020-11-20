# Agora vamos usar bases de dados de verdade!
# Ajuste um MLP com quantas layers e unidades escondidas
# você preferir para prever se uma casa será vendida por
# mais de 25k.
library(keras)

base <- dataset_boston_housing()
x <- scale(base$train$x)
y <- base$train$y
y <- as.numeric(y > 25)

# Model definition ---------------------------------------------

input <- layer_input(shape = ncol(x))

output <- input %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model_keras <- keras_model(input, output)

model_keras %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

# Model fitting ------------------------------------------------

history <- model_keras %>% 
  fit(x, y, batch_size = 32, epochs = 100, validation_split = 0.2)

caret::confusionMatrix(
  ifelse(predict(model_keras, x) > 0.5, 1, 0) %>% factor(levels = c(1, 0)),
  y %>% factor(levels = c(1, 0)),
)
# Accuracy : 0.9752 

# Some memory clean-up
k_clear_session()


# GLM ---------------------------------------------------------------------

model_glm <- glm(y~x, family=binomial(link="logit"))

caret::confusionMatrix(
  y %>% factor(levels = c(1, 0)),
  predict(model_glm, as.data.frame(x), type = "response") %>% {ifelse(.>0.5, 1, 0)}%>% factor(levels = c(1, 0))
)


# Cross validation para comparar com glm ----------------------------------

build_model <- function(){
  input <- layer_input(shape = ncol(x))
  
  output <- input %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1, activation = "sigmoid")
  
  model <- keras_model(input, output)
  
  model %>% 
    compile(
      loss = "binary_crossentropy",
      optimizer = "adam",
      metrics = "accuracy"
    )
}

k <- 5
indices <- sample(1:nrow(x))
folds <- cut(1:length(indices), breaks = k, labels = FALSE) 

all_acc_histories <- NULL
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- x[val_indices,]
  val_targets <- y[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- x[-val_indices,]
  partial_train_targets <- y[-val_indices]
  
  # Build the Keras model (already compiled)
  model <- build_model()
  
  # Train the model (in silent mode, verbose=0)
  history <- model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = 100, batch_size = 32
  )
  
  # print(paste0("Validation MAE: ", history$metrics$val_accuracy))
  mae_history <- history$metrics$val_accuracy
  all_acc_histories <- rbind(all_acc_histories, mae_history)
}

average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_acc_histories)),
  validation_mae = apply(all_acc_histories, 2, mean)
)

mean(average_mae_history$validation_mae)

# Comparar com cv de regressao logistica ----------------------------------------

all_acc_histories_glm <- NULL
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- x[val_indices,]
  val_targets <- y[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- x[-val_indices,]
  partial_train_targets <- y[-val_indices]
  
  # Build the Keras model (already compiled)
  model <- glm(partial_train_targets~partial_train_data, family=binomial(link="logit"))

  mae_history <- yardstick::accuracy_vec(partial_train_targets %>% factor(levels = c(1, 0)), 
                                         ifelse(predict(model, as.data.frame(partial_train_data)) > 0.5, 1, 0) %>% factor(levels = c(1, 0)) )
  all_acc_histories_glm <- rbind(all_acc_histories_glm, mae_history)
}

mean(all_acc_histories_glm)

