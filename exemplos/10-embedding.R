library(keras)

layer <- layer_text_vectorization(max_tokens = 10, output_mode = "int", # poderia binary que converte para one-hot ou tfidf mas em geral tudo do curso vai ser int que representa cada palavra como inteiro diferente
                                  pad_to_max_tokens = TRUE) # default: preenche o vetor ate o max_tokens

frases <- c(
  "eu gosto de gatos",
  "eu gosto de cachorros",
  "eu gosto de gatos e cachorros"
)

layer %>% 
  adapt(frases)

layer(matrix(frases, ncol = 1))

vocab <- get_vocabulary(layer)

input <- layer_input(shape = 1, dtype = "string")
output <- input %>%
  layer() %>% 
  layer_embedding(input_dim = length(vocab) + 2, output_dim = 2) # +2 pq tem o padding e as palavras fora do vocabulario

model <- keras_model(input, output)
out <- predict(model, matrix(frases, ncol = 1))

dim(out) # 3 frases, 6 tamanho do vocabulario, 2 pq sao o n de outputs
# ou ainda: 3 textos com 6 linhas e 2 colunas


out[1,,]


get_weights(model)[[3]] # 2 linhas a mais por conta do padding e das palavras fora do vocabulario
