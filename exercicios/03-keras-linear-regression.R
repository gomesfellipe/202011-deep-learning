# Agora, ao invés de termos somente uma variável `x`, temos 2.
# Escreva o código usando keras para estimar os parâmetros do modelo.


# Data generation ----------------------------------------------

n <- 1000

x <- matrix(runif(2*n), ncol = 2)
W <- matrix(c(0.2, 0.7), nrow = 2)
B <- 0.1

y <- x %*% W + B

# Model definition ---------------------------------------------

library(keras)

input <- layer_input(shape = ncol(x))
output <- input %>% 
  layer_dense(units = 1)

model <- keras_model(input, output)

summary(model)

model %>% 
  compile(
    loss = "mse",
    optimizer = optimizer_sgd(lr = 0.01)
  )

# Model fitting ------------------------------------------------
grid <-
  expand.grid(
    epochs = c(10, 20, 30, 50, 100),
    batch_size = c(10, 20, 30, 50, 100)
  )

results <-
  purrr::map2(
    grid$epochs,
    grid$batch_size,
    ~{
      input  <- layer_input(shape = 2)
      output <- layer_dense(input, units = 1, use_bias = TRUE)
      model  <- keras_model(input, output)
      
      model %>%
        compile(
          loss = loss_mean_squared_error,
          optimizer = optimizer_sgd(lr = 0.01)
        )
      
      model %>%
        fit(
          x= x, 
          y = y,
          epochs = .x,
          batch_size = .y
        )
      
      dplyr::lst(
        weights = get_weights(model)
        ,
        results = dplyr::tibble(
          parameters = rep(paste0('epochs:',.x,' \nbatch_size:',.y), n),
          fit = predict(model, x) %>% as.numeric(),
          true = as.numeric(y)
        )
        ,
        mape = yardstick::mape_vec(results$true, results$fit)
      )
    }
  )

# avaliar resultados ------------------------------------------------------

library(dplyr)
library(dplyr)
library(ggplot2)
library(purrr)
theme_set(theme_bw())


lvls <- 
  arrange(grid, batch_size, epochs) %>% 
  transmute(parameters = paste0('epochs:',epochs,' \nbatch_size:',batch_size)) %>% 
  pull(parameters)

mape_labels <-
  tibble(mape = map_dbl(results, ~.x$mape)) %>%
  mutate(true = 1, fit = 0) %>%
  bind_cols(transmute(grid, parameters = paste0('epochs:',epochs,' \nbatch_size:',batch_size))) %>% 
  mutate(parameters = factor(parameters, levels = lvls))

results %>%
  purrr::map_dfr(~.x$results) %>%
  mutate(parameters = factor(parameters, levels = lvls)) %>% 
  ggplot(aes(x = true, y = fit))+
  geom_point(alpha = 0.6)+
  scale_y_continuous(breaks = seq(0,1,0.1))+
  geom_label(data = mape_labels,
             aes(label = round(mape, 2)),
             vjust="inward",hjust="inward")+
  facet_wrap(~parameters, nrow = 5)+
  labs(caption = "Valor no à esquerdo corresponde ao MAPE")


