library(keras)
install_keras(tensorflow = 'gpu')
library(tm)
library(qdap)
library(plyr)


setwd("~/R/Project/ClearNews")

train <- read.csv("train_data.csv")
test <- read.csv("test.csv")
baseline <- read.csv("baseline.csv")





#word vector
maxlen <- 1000
max_words <- 10000
embedding_dim <- 100

#pre-processing text
corpus <- Corpus(VectorSource(train$TITLE))
corpus <- tm_map(corpus, content_transformer(removePunctuation))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, content_transformer(removeWords), stopwords('english'))
#removeStrange <- function(x) gsub("(<.+>)+", "", x)
#corpus <- tm_map(corpus, content_transformer(removeStrange))
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, content_transformer(stripWhitespace))
corpus <- tm_map(corpus, content_transformer(removeNumbers))
corpus <- tm_map(corpus, content_transformer(stripWhitespace))
cleaned.docs = data.frame(text=sapply(corpus,identity), stringsAsFactors = F)
head(cleaned.docs)

texts <- cleaned.docs
tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(texts)
#gather the sequences
sequences <- texts_to_sequences(tokenizer, texts)
#pad sequences to datafarme
x_full <- pad_sequences(sequences, maxlen = maxlen)

head(x_full)


indices <- sample(1:nrow(cleaned.docs))
training_samples = 80000
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):nrow(cleaned.docs)]

x_train <- x_full[training_indices,]
y_train <- train$CATEGORY[training_indices]
y_train <- gsub('b',0, y_train)
y_train <- gsub('e',1, y_train)
y_train <- gsub('m',2, y_train)
y_train <- gsub('t',3, y_train)


y_train = to_categorical(y_train,num_classes = length(unique(y_train))+1)

x_val <- data[validation_indices,]
y_val <- labels[validation_indices]
y_val = to_categorical(y_val)



y_train_h <-y_train 
y_train <- to_categorical(y_train)


texts <- test
tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(texts)

#gather the sequences
sequences2 <- texts_to_sequences(tokenizer, texts)

x_test <- pad_sequences(sequences, maxlen = maxlen)
y_test<-test$CATEGORY
y_test <- gsub('b',0, y_test)
y_test <- gsub('e',1, y_test)
y_test <- gsub('m',2, y_test)
y_test <- gsub('t',3, y_test)
y_test_h <- y_test 
y_test <- to_categorical(y_test)


########### Model 1 ##############
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_cudnn_lstm(units = 500, return_sequences = TRUE) %>%
  layer_dropout(0.25) %>%
  layer_cudnn_lstm(units = 250, return_sequences = TRUE) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 4, activation = "softmax")
summary(model)



# Compile model
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)


# Early stopping
callback <- callback_early_stopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')

# Fitting model
history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_split = (0.2),
  callbacks = callback
)

history

predicted1 <- predict_classes(model, x = test$TITLE)
cm_table <- table(predicted1, y_test_h)
cm_table
mean(predicted1 == y_test_h)



#pre-processing text
corpus2 <- Corpus(VectorSource(test$TITLE))
corpus2 <- tm_map(corpus2, content_transformer(removePunctuation))
corpus2 <- tm_map(corpus2, content_transformer(tolower))
corpus2 <- tm_map(corpus2, content_transformer(removeWords), stopwords('english'))
#removeStrange <- function(x) gsub("(<.+>)+", "", x)
#corpus <- tm_map(corpus, content_transformer(removeStrange))
corpus2 <- tm_map(corpus2, stemDocument)
corpus2 <- tm_map(corpus2, content_transformer(stripWhitespace))
corpus2 <- tm_map(corpus2, content_transformer(removeNumbers))
corpus2 <- tm_map(corpus2, content_transformer(stripWhitespace))
cleaned.docs2 = data.frame(text=sapply(corpus2,identity), stringsAsFactors = F)
head(cleaned.docs2)

head(cleaned.docs2)
head(cleaned.docs)


texts2 <- cleaned.docs2$text
tokenizer2 <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(texts2)
#gather the sequences
sequences2 <- texts_to_sequences(tokenizer2, texts2)
#gather word index
word_index2 = tokenizer2$word_index
#pad sequences to datafarme
x_train2 <- pad_sequences(sequences2, maxlen = maxlen)
head(predicted1)
predicted <- predict_classes(model, x = x_train2)
head(predicted)
baseline_new <- cbind(baseline, predicted)
baseline_new$predicted <- gsub(0, 'b', baseline_new$predicted)
baseline_new$predicted <- gsub(1, 'e', baseline_new$predicted)
baseline_new$predicted <- gsub(2, 'm', baseline_new$predicted)
baseline_new$predicted <- gsub(3, 't', baseline_new$predicted)

head(baseline_new)
head(test$TITLE)

write.csv(baseline_new, file = "baseline_new.csv")

mean(baseline_new$CATEGORY == baseline_new$predicted)
