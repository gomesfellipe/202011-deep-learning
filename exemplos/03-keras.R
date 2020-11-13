library(keras)

# Data generation ----------------------------------------------

n <- 1000

x <- runif(n)
W <- 0.9
B <- 0.1

y <- W * x + B

# Model definition ---------------------------------------------

input <- layer_input(shape = 1) # vai ter uma coluna no meu modelo (a entrada eh uma coluna)
output <- layer_dense(input, units = 1, use_bias = TRUE) # pega o input e faz a multiplicacao de matrizes, quantas unidades na camada oculta, quantidade de outputs o padrao eh usar a activation = 'linear' ou seja nao vai fazer nada na camada
model <- keras_model(input, output) # 

# obter informacoes do modelo declarado
summary(model)

# compila esse modelo definido
# a funcao modifica o modelo entao nao precisa fazer model <- compile(model)
model %>% 
  compile(
    loss = loss_mean_squared_error,
    optimizer = optimizer_sgd(lr = 0.01) # optmizador sgd com learning rate 0.01
  )

# Model fitting ---------------------------------------------------

model %>% 
  fit(
    x = x,
    y = y,
    batch_size = 2,  # 
    epochs = 5 # 
  )

get_weights(model)
y_hat <- predict(model, x)

plot(y, y_hat)
