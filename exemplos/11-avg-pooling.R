library(keras)
library(tidyverse)

# Dados ------------

df <- readr::read_csv(
  pins::pin("https://storage.googleapis.com/deep-learning-com-r/toxic-comments.csv")
)

x <- df$comment_text
y <- as.matrix(df %>% select(-id, -comment_text))

n_palavras <- stringr::str_count(x, pattern = " +") + 1
quantile(n_palavras, c(0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1)) # 90% dos textos tem menos que 150 palavras

# Layer para vetorizacao --------

vectorize <- layer_text_vectorization(
  max_tokens = 10000, # numero de palavras distintas no vocabulario
  output_mode = "int", # se vai ser numerico inteiro ou tfidf ou count ou one-hot
  pad_to_max_tokens = TRUE,  # ja esta no default. so seria util se usasse os outros modos
  output_sequence_length = 150 # truncar os 10% maiores
)

vectorize %>% 
  adapt(x)

vocab <- get_vocabulary(vectorize)

# Definição do modelo -------------

input <- layer_input(shape = 1, dtype = "string") # uma coluna de strings de entrada
output <-  input %>%
  vectorize() %>% # cada texto eh uma matrix com 150 colunas e cada valor eh o int (id) da palavra do texto
  layer_embedding(input_dim = length(vocab) + 2, output_dim = 32) %>% # camada de embeeding. dim eh o tamanho do vocabulario + padding e outvocabulary, e outputdim eh a quantidade de colunas para saida
  # layer_max_pooling_1d(pool_size) # seria para calcular uma medida movel
  layer_global_average_pooling_1d() %>% 
  layer_dense(units = ncol(y), activation = "sigmoid") # sigmoid e nao softmax pq nao queremos que a coluna some 1. o texto pode ser mais que uma categoria

model <- keras_model(input, output)
summary(model)

auc <- tensorflow::tf$keras$metrics$AUC()

model %>% 
  compile(
    loss = "binary_crossentropy", # pq eh como se tivesse ajustando varias regressoes logisticas
    optimizer = "sgd",
    metrics = list("accuracy", auc)
  )

# ajuste

model %>% 
  fit(matrix(x, ncol = 1), y, validation_split = 0.2, batch_size = 32)


