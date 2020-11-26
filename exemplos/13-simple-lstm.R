# Pacotes ---------------------------------------------------------

library(keras)

# Data ------------------------------------------------------------

n <- 10000
l <- 10

cresc <- sample(c(1,0), size = n, replace = TRUE)
x <- array(dim = c(n, l, 1))
for(i in 1:n) {
  v <- runif(2, min = -1, max = 1)
  if (cresc[i] == 1)
    x[i,,1] <- seq(from = min(v), to = max(v), length.out = l)
  else
    x[i,,1] <- seq(from = max(v), to = min(v), length.out = l)
}

# A sequencia que ou esta aumentando ou esta diminuindo

# Model ------------------------------------------------------------

# use float64 for comparison with R
tensorflow::tf$keras$backend$set_floatx("float64") 

input <- layer_input(shape = c(l,1))

output <- input %>% 
  layer_lstm(
    units = 1, 
    # input_shape = c(5,1),  # sobra de aulas anteriores, nao é necessario
    use_bias = FALSE, # removeu bias para simplificas
    unit_forget_bias = FALSE, # removeu bias para simplificas
    recurrent_activation = "tanh",
    activation = "tanh"
  ) %>% 
  layer_activation("sigmoid")

model <- keras_model(input, output)

model %>% compile(
  loss = "binary_crossentropy", 
  optimizer = "sgd",
  metrics = "accuracy"
)

model %>% fit(x = x, y = cresc, epochs = 50, batch_size = 100)

# Manual calc ------------------------------------------------------

sigm <- function(x) {
  1/(1 + exp(-x))
}

w <- get_weights(model) # duas matrizes de pesos
# a primeira 

s <- 0
c <- 0
x_ <- x[1,,]


for (t in 1:l) {
  
  i     <- tanh(w[[2]][1,1]*s + w[[1]][1,1]*x_[t]) # peso vezes estado anterior + o peso vezes o valor
  f     <- tanh(w[[2]][1,2]*s + w[[1]][1,2]*x_[t]) # forget 
  c_hat <- tanh(w[[2]][1,3]*s + w[[1]][1,3]*x_[t]) # candidato: 
  
  c <- f*c + i*c_hat 
  o <- tanh(s*w[[2]][1,4] + w[[1]][1,4]*x_[t]) # é o que sai na parte de cima do neuronio da lstm
  s <- o*tanh(c)# é o que sai na parte de baixo do neuronio da lstm
  
}
sigm(s)

model(x[1,,,drop=FALSE])

# Results ----------------------------------------------------------

ggplot2::qplot(predict(model, x), cresc, geom = "jitter")
model(x[1:10,,,drop=FALSE])



