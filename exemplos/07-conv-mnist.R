# Dataset

library(keras)

base <- dataset_mnist()
x <- array_reshape(base$train$x/256, dim = c(60000, 28, 28, 1)) # 60000 imagens, 28 por 28 com 1 canal (cor)
y <- to_categorical(base$train$y)# hot encoding

dim(x)
dim(y)

# Model definition ---------------------------------------------

input <- layer_input(shape = c(28, 28, 1)) # tnato faz o batch size pois vamos passar na funcao fit.. com 28 linhas 28 colunas e 1 canal

# Objetivo do kernel é identificar padroes locais na imagem
# Bias eh bom pq faz as coisas ficarem centralizadas

output <- input %>% 
  layer_conv_2d(kernel_size = c(3,3), filters = 32, activation = "relu", # convolucao com kernel size 3x3 com 32 filtros 
                padding = "same", use_bias = F) %>% # completa com zero para nao perder nenhuma borda
  layer_max_pooling_2d(pool_size = c(2,2)) %>% # diminuir a dimensao com max polling andando com uma janelinha 2x2
  
  layer_conv_2d(kernel_size = c(3,3), filters = 64, activation = "relu", # pega o de 32 filtro e joga para 64
                padding = "same", use_bias = F) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% # camada extra so para mostrar que se continuar fazendo vira um vetor
  
  layer_conv_2d(kernel_size = c(3,3), filters = 128, activation = "relu",
                padding = "same", use_bias = F) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  layer_conv_2d(kernel_size = c(3,3), filters = 256, activation = "relu", # aumentando o numero de canais e tirando altura e largura ate a imagem virar um vetor
                padding = "same", use_bias = F) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% # faz o max pooling para dimunuir a imagem pela metade de novo
  
  layer_flatten() %>% 
  layer_dense(128, activation = "relu") %>% 
  layer_dense(10, activation = "softmax") 
  

# TODO: Testar depois sem usar tantas camadas se o flatten ja esta transformando em um vetorzin

model <- keras_model(input, output)


summary(model)

#Model: "model"
#________________________________________________________________
#Layer (type)                Output Shape              Param #   
#================================================================
#input_2 (InputLayer)        [(None, 28, 28, 1)]       0          # entrada dos dados
#________________________________________________________________
#conv2d_13 (Conv2D)          (None, 28, 28, 32)        320        # 3*3*32 + 1*32 (pesos do kernel + bias) quando use_bias = TRUE (default)
#________________________________________________________________
#max_pooling2d_12 (MaxPoolin (None, 14, 14, 32)        0          # (3*3*32) * 64 filtros Obs.: Numero de pessos nao é afetado pela imagem mas sim pelo tamanho do kernel
#________________________________________________________________
#conv2d_14 (Conv2D)          (None, 14, 14, 64)        18496      # (3*3*64) * 128 e por ai vai 
#________________________________________________________________
#max_pooling2d_13 (MaxPoolin (None, 7, 7, 64)          0         
#________________________________________________________________
#conv2d_15 (Conv2D)          (None, 7, 7, 128)         73856     
#________________________________________________________________
#max_pooling2d_14 (MaxPoolin (None, 3, 3, 128)         0         
#________________________________________________________________
#conv2d_16 (Conv2D)          (None, 3, 3, 256)         295168    
#________________________________________________________________
#max_pooling2d_15 (MaxPoolin (None, 1, 1, 256)         0         
#________________________________________________________________
#flatten_1 (Flatten)         (None, 256)               0         
#________________________________________________________________
#dense (Dense)               (None, 128)               32896     
#________________________________________________________________
#dense_1 (Dense)             (None, 10)                1290      
#================================================================
#Total params: 422,026
#Trainable params: 422,026
#Non-trainable params: 0
#________________________________________________________________



model %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = "sgd",
    metrics = "accuracy"
  )

# Model fitting ------------------------------------------------

model %>% 
  fit(x, y, batch_size = 32, epochs = 3, validation_split = 0.2)  


# predict -----------------------------------------------------------------

y_pred <- predict(model, x)
classes <- 0:9
y_pred_class <- classes[apply(y_pred, 1, which.max)]

# Matriz de confusao
caret::confusionMatrix(factor(base$train$y), factor(y_pred_class))


# saveRDS() # Errado!
save_model_tf(model, "modelo-mnist/") # salva em uma pasta

# Tipo um plumber para tensorflow: https://www.tensorflow.org/tfx/guide/serving

