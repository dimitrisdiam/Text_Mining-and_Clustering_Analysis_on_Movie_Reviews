:::{.callout-note}
---
  title: "Assignment 4"
author: 
  
  - Diamantidis Adam
- Diamantidis Dimitrios

date: 03-11-2023
format:
  html:
  toc: true
self-contained: true
code-fold: true
df-print: kable
---
  
  :::
  
  ```{r}
#| label: R packages
#| echo: false
#| warning: false
#| message: false

library(ggplot2)   # plots
library(tidyverse) # data
library(tidytext)  # text mining
library(tm)        # text mining
library(wordcloud) # wordclouds
library(wordcloud2) # wordclouds
library(SnowballC) # Porter's stemming algorithm
library(text2vec)  # data
library(textdata)  # lexicons
library(cluster)   # clustering algorithms
library(clustertend) # clustering
library(factoextra)# clustering algorithms & visualization
library(dplyr)     # data manipulation
library(gridExtra) # grids
library(dendextend) # dendrogram (hierarchical clustering)
library(textmineR) # text operations
library(dbscan)    # density-based clustering
```

```{r}
#| label: data loading

data("movie_review")
movie_review <- as_tibble(movie_review)
head(movie_review)
```

# Data description & exploration

## Words and characters
```{r}
# Wordcloud
wordcloud(
  words = movie_review$review,
  min.freq = 30,
  max.words = 25,
  random.order = FALSE,
  colors = brewer.pal(8, "Dark2")
)

# Review with the most characters
max_char_review <- movie_review[which.max(nchar(movie_review$review)), ]
max_char_count <- nchar(max_char_review$review)
cat("Review with the most characters (", max_char_count, " characters):\n", max_char_review$review, "\n\n")

# Review with the least characters
min_char_review <- movie_review[which.min(nchar(movie_review$review)), ]
min_char_count <- nchar(min_char_review$review)
cat("Review with the least characters (", min_char_count, " characters):\n", min_char_review$review, "\n\n")

# Average words per review
average_words_per_review <- mean(sapply(strsplit(movie_review$review, "\\s+"), length))
cat("Average words per review: ", average_words_per_review, "\n")
```

##### Before processing our data, we can see that the reviews contain primarily words such as "the", "movie", "film", "one", "good" and "like". The reviews differ significantly in length: most characters: 13708 vs least characters: 70, with average amount of words per review being 237. Additionally, we can observe that unprocessed reviews contain some HTML scraping remains "br line breaks" which will need to be removed.

## Text mining

```{r}
# tokenization
comp_words <- movie_review |> 
  unnest_tokens(word, review)

# check the resulting tokens
head(comp_words)

# custom stop words
df_stopwords <- 
  stop_words |> 
  rbind(tibble(word = c("br"), lexicon = c("Me", "Me")))

#  remove stop words
comp_words_no_stop <- 
  comp_words |> 
  anti_join(df_stopwords)

comp_words_no_stop |> 
  # count the frequency of each word
  count(word) |> 
  # arrange the words by its frequency in descending order
  arrange(desc(n)) |> 
  # select the top 30 most frequent words
  head(30) |> 
  # make a bar plot (reorder words by their frequencies)
  ggplot(aes(x = n, y = reorder(word, n))) + 
  geom_col(fill="gray") +
  labs(x = "frequency", y="words") + 
  theme_minimal()
```

##### After initial data processing, we observe that the most frequent words are "movie" and "film" (around 8000 occurrences), followed by "time", "story", "people" and "bad" (about 2000 occurrences). 

```{r}
# stemming
comp_words_stemming <- 
  comp_words_no_stop |> 
  mutate(stem = wordStem(word))

comp_words_stemming |> 
  # count the frequency of each word
  count(stem) |> 
  # arrange the words by its frequency in descending order
  arrange(desc(n)) |> 
  # select the top 30 most frequent words
  head(30) |> 
  # make a bar plot (reorder words by their frequencies)
  ggplot(aes(x = n, y = reorder(stem, n))) + 
  geom_col(fill="gray") +
  labs(x = "frequency", y="words") + 
  theme_minimal()
```

##### Applying stemming slightly changes the chart of the most frequent words (stems in this case). At the top, we can still see "movi" and "film" (increased to around 10000 occurrences), followed by "time", "charact", "watch" and "stori" (increased to about 2500 occurrences). These higher counts make sense because a lot of words have been merged to common stems.

```{r}
## Regex

# reviews containing "Movie" or "movie"
reviews_mov <- movie_review |> 
  filter(str_detect(review, "[Mm]ovie"))

# Reviews containing "Actor," "actor," "Actress," or "actress"
reviews_act <- movie_review |> 
  filter(str_detect(review, "[Aa]ctor|[Aa]ctress"))

# reviews containing "Music or "music"
reviews_mus <- movie_review |> 
  filter(str_detect(review, "[Mm]usic"))

# compare the occurrence of each pair of words
kwords    <- c("Movie", "Actor/Actress", "Music")
nr_kwords <- c(nrow(reviews_mov), nrow(reviews_act), nrow(reviews_mus))

tibble(kwords, nr_kwords)


# Detect
head(str_detect(movie_review$review, "(\\w+) ([Mm]ovie)"))
# Extract
head(str_extract(movie_review$review, "(\\w+) ([Mm]ovie)"))
# Subset
head(str_subset(movie_review$review, "(\\w+) ([Mm]ovie)"))
# Match
head(str_match(movie_review$review, "(\\w+) ([Mm]ovie)"))
```


```{r}
## Vocabulary

# 1. word counts

# make it a list (iterator)
words_ls <- list(comp_words_no_stop$word)
# create index-tokens
it <- itoken(words_ls, progressbar = FALSE) 
# collects unique terms 
vocab <- create_vocabulary(it)
# filters the infrequent terms (number of occurrence is less than 50)
vocab <- prune_vocabulary(vocab, term_count_min = 50)
# show the resulting vocabulary object (formatting it with datatable)
# vocab  # commented out to save output space

# 2. Token co-occurrence matrix

# maps words to indices
vectorizer <- vocab_vectorizer(vocab)
# use window of 5 for context words
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5)

# 3. Vectors

glove      <- GlobalVectors$new(rank = 50, x_max = 10)
wv_main <- glove$fit_transform(tcm, n_iter = 20, convergence_tol = 0.001)

# extract context word vector
wv_context <- glove$components

# check the dimension for both matrices
dim(wv_main); dim(wv_context) 

word_vectors <- wv_main + t(wv_context) # transpose one matrix to perform matrix addition

# 4. Similar words

# extract the row of "movie"
movie <- word_vectors["movie", , drop = FALSE]

# calculates pairwise similarities between "movie" and the rest of words
cos_sim_movie <- sim2(x = word_vectors, y = movie, method = "cosine", norm = "l2")

# the top 10 words with the highest similarities
head(sort(cos_sim_movie[,1], decreasing = T), 10)

# extract the row of "actor"
actor <- word_vectors["actor", , drop = FALSE]

# calculates pairwise similarities between "actor" and the rest of words
cos_sim_actor <- sim2(x = word_vectors, y = actor, method = "cosine", norm = "l2")

# the top 10 words with the highest similarities
head(sort(cos_sim_actor[,1], decreasing = T), 10)

# extract the row of "music"
music <- word_vectors["music", , drop = FALSE]

# calculates pairwise similarities between "music" and the rest of words
cos_sim_music <- sim2(x = word_vectors, y = music, method = "cosine", norm = "l2")

# the top 10 words with the highest similarities
head(sort(cos_sim_music[,1], decreasing = T), 10)
```
##### We create a vocabulary of unique terms (appearing at least 50 times). Then, we create a token co-occurrence matrix with the window of 5 context words and fit word vectors on our reviews data set.In the last step we find the most similar 10 words for each of selected words: "movie", "actor", "music".

```{r}
# afinn sentiment
sentiments_afinn <- get_sentiments("afinn")

review_sentiment <- 
  comp_words_no_stop |>
  inner_join(sentiments_afinn) |>
  select(-sentiment)

head(review_sentiment)
```

##### We use afinn lexicon to obtain detailed sentiment (ranging from -5 to 5) for each word.

# Text pre-processing and representation

```{r, warning=FALSE}
movie_review$review <- gsub("<br />", " ", movie_review$review) # removing HTML line breaks remaining from text
corpus <- Corpus(VectorSource(movie_review$review)) 
#corpus.cleaned <- tm::tm_map(corpus, function(x) iconv(x, to='UTF-8-MAC', sub = 'byte')) #for mac
corpus.cleaned <- tm::tm_map(corpus, tm::removeWords, tm::stopwords('english')) 
corpus.cleaned <- tm::tm_map(corpus.cleaned, tm::stemDocument, language = "english") 
corpus.cleaned <- tm::tm_map(corpus.cleaned, tm::stripWhitespace)
corpus.cleaned <- tm::tm_map(corpus.cleaned, removeWords, c(stopwords("en"), "<br /><br />"))
tdm <- tm::DocumentTermMatrix(corpus.cleaned) 
```

##### A document-term matrix counts the occurrence of words in different documents. It is one-document-per-row and one-term-per-column. The function 'CreateDtm' automatically transforms to lowercase and removes punctuations, stopwords and numbers. However, we also excluded 'br' from the document-term matrix, since it only is used for line break.  


# Text clustering

## 1. KMeans, Hierarchical, Density-based and Stacked clustering
```{r}
# Reproduce the results
set.seed(42)
# Building the feature matrices
tdm.tfidf <- tm::weightTfIdf(tdm)
# We remove A LOT of features. R is natively very weak with high dimensional matrix
tdm.tfidf <- tm::removeSparseTerms(tdm.tfidf, 0.999)
# There is the memory-problem part
# - Native matrix isn't "sparse-compliant" in the memory
# - Sparse implementations aren't necessary compatible with clustering algorithms
tfidf.matrix <- as.matrix(tdm.tfidf)
# Cosine distance matrix (useful for specific clustering algorithms)

dist.matrix = proxy::dist(tfidf.matrix, method = "cosine")

truth.K <- 10
truth.K_5 <- 5 
clustering.kmeans_5 <- kmeans(tfidf.matrix, truth.K_5)
summary(silhouette(clustering.kmeans_5$cluster,dist.matrix))

# KMeans
clustering.kmeans <- kmeans(tfidf.matrix, truth.K)

# Hierarchical
clustering.hierarchical <- hclust(dist.matrix, method = "ward.D2")

# Density-based
clustering.dbscan <- dbscan::hdbscan(dist.matrix, minPts = 10)



master.cluster <- clustering.kmeans$cluster
slave.hierarchical <- cutree(clustering.hierarchical, k = truth.K)
slave.dbscan <- clustering.dbscan$cluster
# Preparing the stacked clustering
stacked.clustering <- rep(NA, length(master.cluster)) 
names(stacked.clustering) <- 1:length(master.cluster)

##############################

for (cluster in unique(master.cluster)) {
  indexes = which(master.cluster == cluster, arr.ind = TRUE)
  slave1.votes <- table(slave.hierarchical[indexes])
  slave1.maxcount <- names(slave1.votes)[which.max(slave1.votes)]
  
  slave1.indexes = which(slave.hierarchical == slave1.maxcount, arr.ind = TRUE)
  slave2.votes <- table(slave.dbscan[indexes])
  slave2.maxcount <- names(slave2.votes)[which.max(slave2.votes)]
  
  stacked.clustering[indexes] <- slave2.maxcount
}


points <- cmdscale(dist.matrix, k = 5) # Running the PCA
palette <- colorspace::diverge_hcl(truth.K) # Creating a color palette
previous.par <- par(mfrow=c(2,2), mar = rep(1.5, 4)) # partitionning the plot space

plot(points,
     main = 'K-Means clustering',
     col = as.factor(master.cluster),
     mai = c(0, 0, 0, 0),
     mar = c(0, 0, 0, 0),
     xaxt = 'n', yaxt = 'n',
     xlab = '', ylab = '')

plot(points,
     main = 'Hierarchical clustering',
     col = as.factor(slave.hierarchical),
     mai = c(0, 0, 0, 0),
     mar = c(0, 0, 0, 0),
     xaxt = 'n', yaxt = 'n',
     xlab = '', ylab = '')

plot(points,
     main = 'Density-based clustering',
     col = as.factor(slave.dbscan),
     mai = c(0, 0, 0, 0),
     mar = c(0, 0, 0, 0),
     xaxt = 'n', yaxt = 'n',
     xlab = '', ylab = '')
plot(points,
     main = 'Stacked clustering',
     col = as.factor(stacked.clustering),
     mai = c(0, 0, 0, 0),
     mar = c(0, 0, 0, 0),
     xaxt = 'n', yaxt = 'n',
     xlab = '', ylab = '')

summary(silhouette(clustering.dbscan$cluster,dist.matrix)) # density-based
summary(silhouette(clustering.kmeans$cluster,dist.matrix)) # 10 clusters
summary(silhouette(clustering.kmeans_5$cluster,dist.matrix)) # 5 clusters
```

##### Firstly, creating corpus takes first place of preprocessing, which is followed by tm_map function to remove stop words. After then, stemming and trimming white space is applied via tm library as well. In order to present the preprocessed review column, DocumentTermMatrix is used, adding to removeSparseTerms to our tfidf variable to remove some sparse term to make calculation quicker. Representation part is completed by applying dist function of proxy library with cosine argument as a method. Then, assigning number of cluster of and plugging in into functions take place. K means and density based clustering of hdbscan are used for clustering.

##### While K-means partitions data into clusters based on distance or similarity, DBSCAN is a density-based clustering algorithm that identifies clusters of arbitrary structure. K-Means vs DBSCAN K-Means struggles with non-globular clusters and clusters of multiple sizes, while DBSCAN handles them well. Data Types K-Means works best with data that has a clear centroid, while DBSCAN is not influenced by noise or outliers. Even though there are some methodological differences between k-means and density based, there is no remarkable differences among them in terms of silhouette scores.

## 2. Latent Dirichlet Allocation

We also decided to do Latent Dirichlet Allocation (LDA) on the movie review data set. LDA is a kind of topic modelling and the assumption behind this approach is that each document consists of a mixture of latent, i.e. unknown, underlying topics and each topic consists of a distribution of words (Blei et al., 2003). The main application of LDA is the automated processing of text collections. The documents in a document collection consist of words and the goal is to assign a latent topic to the documents (Jagadeeswaran, 2019). In our case, the documents are the different reviews, the topics are the clusters of the reviews and the words are the words of the reviews.

```{r}
#| label: sparse matrix

# create a sparse (dgCMatrix) for the LDA
m <-  Matrix::sparseMatrix(
  i = tdm$i,
  j = tdm$j,
  x = tdm$v,
  dims = c(tdm$nrow, tdm$ncol),
  dimnames = tdm$dimnames
)
```

## LDA for 5 topics

```{r}
#| label: fitting the LDA for 5 topics

set.seed(42)

lda5 <- FitLdaModel(
  dtm = m,
  # 5 topics
  k = 5,
  # number of Gibbs samples
  iterations = 500,
  # number of burnin iterations
  burnin = 180
)
```

```{r}
#| label: LDA 5 inspection

# show the 10 best words for each topic
data.frame(GetTopTerms(phi = lda5$phi, M = 10))
```

In the above table, the 10 most important word-stemps for each of the 5 topics (i.e., clusters) can be seen. For example, one can tell that cluster three is maybe about horror movies or thrillers (consider the word-temps 'kill', 'run' and 'horror'), so the title of this topic could be 'horror/thriller'. However, other topics are not that easy to distinguish and there can't be found a uniform title. Also note that the word-stemps 'movi' and 'film' are in multiple clusters.

```{r}
#| label: coherence LDA 5

lda5$coherence
```

```{r}
mean(lda5$coherence)
```

The coherence is a measure of interpretability of the different clusters, where by default, the 5 'best' words of each topics are used for the calculation. For using 5 clusters, we get very low scores.

## Assigning each document to the most likely topic

```{r}
#| label: topic assignment LDA 5

# topic proportions for each document
topic_proportions <- posterior(lda5)[, 1:5]

# most likely topic for each document
top_topic_LDA5 <- apply(topic_proportions, 1, which.max)

# Add the top_topic to movie_review
movie_review <- cbind(movie_review, data.frame(top_topic_LDA5))
```

We then assigned each document to the most likely document using the posteriors and added it to the original data frame, as can be seen above.

```{r}
#| label: topic distribution LDA 5

table(movie_review$top_topic_LDA5)
```

Cluster one contains the most reviews, whereas cluster two and three the fewest. 

```{r}
#| label: topic inspection LDA 5

cl_sent1 <- movie_review$top_topic_LDA5[movie_review$sentiment == 1]
cl_sent0 <- movie_review$top_topic_LDA5[movie_review$sentiment == 0]

table(cl_sent0); table(cl_sent1)
```

For another quality assessment, besides computing the coherence, we checked to which clusters the reviews are assigned given the provided sentiments scores. Our assumption is, that reviews having a good sentiment score, are more similar in types of wording and consequently, are more likely in the same cluster. We would therefore expect to see, e.g., for sentiment score 0, that one topic stands out and has by far the most reviews, whereas in the same cluster, reviews with sentiment score 1 have only very few observations. However, this can't be seen in our analysis. For both sentiment scores, the clusters each hold very similar amounts of reviews.
But note, that this only holds on our assumption that we made and therefore can't be used for complete certainty. We think, that the LDA not classified for sentiment, but rather on specific topics, like thriller/horror genre or sports shows as observed before.\
We also checked some reviews for specific clusters by hand (fore example, as assumed before, cluster 3 is about horror/thriller) but could not find much equality within these reviews.

## LDA for 10 topics

```{r}
#| label: fitting the LDA for 10 topics

set.seed(42)

lda10 <- FitLdaModel(
  dtm = m,
  # 10 topics
  k = 10,
  # number of Gibbs samples
  iterations = 500,
  # number of burnin iterations
  burnin = 180
)
```

```{r}
#| label: LDA 10 inspection

# show the 10 best words for each topic
data.frame(GetTopTerms(phi = lda10$phi, M = 10))
```

Again, in the above table, the 10 most important word-stemps for each of the 10 topics (i.e., clusters) can be seen. Also, the topics are not that easy to distinguish. Topic eight maybe about a sports game show considerung the word-stems 'game', 'fight' 'show', 'match ' and 'action'.

```{r}
#| label: coherence LDA 10

lda10$coherence
```

```{r}
mean(lda10$coherence) 
```

Again, we get low coherence scores which is in union with our findings from the table above.

## Assigning each document to the most likely topic

```{r}
#| label: topic assignment LDA 10

# topic proportions for each document
topic_proportions <- posterior(lda10)[, 1:10]

# most likely topic for each document
top_topic_LDA10 <- apply(topic_proportions, 1, which.max)

# Add the top_topic to movie_review
movie_review <- cbind(movie_review, data.frame(top_topic_LDA10))
```

We then assigned each document to the most likely document using the posteriors and added it to the original data frame, as can be seen above.

```{r}
#| label: topic distribution LDA 10

table(movie_review$top_topic_LDA10)
```

As can be seen in the above table, many reviews are assigned to cluster ten, whereas fewer reviews are assigned to cluster one and two.

```{r}
#| label: LDA 10

cl_sent1 <- movie_review$top_topic_LDA10[movie_review$sentiment == 1]
cl_sent0 <- movie_review$top_topic_LDA10[movie_review$sentiment == 0]

table(cl_sent0); table(cl_sent1)
```

Again, we checked to which clusters the reviews are assigned given the provided sentiments scores. And as already seen with 5 clusters, for both sentiment scores, the clusters each hold very similar amounts of reviews.\

To conclude, we found that LDA didn't give us good clustering for the movie reviews data set neither for 5,nor for 10 clusters. In further analysis, one can try to use different amount of topics, compare the perplexity and choose the best model. However, it is not guaranteed, that this will give us a better clustering. This may be due to the fact that maybe LDA is not the best method for this kind of analysis or we just can't get better results because of the nature of the data.



# Evaluation & model comparison


```{r}
#| label: table example
data.frame(
  model       = c("Density-based", "KMeans_5", "KMeans_10", "LDA_5", "LDA_10"),
  performance = c(-0.009103, 0.001412, -0.0051896 , 0.03873127 , 0.04020071),
  comments    = c("Avg Silhouette score", "Avg Silhouette score", "Avg Silhouette score", "Avg coherence score", "Avg coherence score")
)
```

As the final assessment of our clustering methods, we decided to select the following 5 models: Density-based, KMeans for 5 clusters, KMeans for 10 clusters, LDA for 5 clusters and LDA for clusters. It is worth noticing that due to the difference between applied methods we couldn't use a single performance evaluation method for all the models. Therefore the first 3 models scores' have been evaluated based on the silhouette score while the last 2 based on the average coherence score. While the silhouette score measures the quality of clustering by quantifying how close each data point in one cluster is to the data points in the same cluster (intra-cluster similarity) and how far it is from the data points in the nearest neighboring cluster (inter-cluster dissimilarity) ranging between -1 and 1 (stating higher value means higher), the coherence scores are based on the idea that a good topic should contain words that are closely related in meaning. A higher score indicates that the words within each cluster are more semantically related, making the cluster more coherent and interpretable. If the words in a cluster are unrelated or too diverse, the coherence score will be lower.

The silhouettes scores of both two K-means and density based are around 0. Based on that, there can't be seen any significant differences between applied methods and cluster sizes. Additionally, this kind of neutral score can be based on the possibility that review entries might not have structure to be clustered properly.

Taking both the coherence scores and the explicit evaluation of specific cluster-review assignment, we found that LDA should not be the model of choice in this case. In further analysis, one can try to use different amounts of topics, compare the perplexity and choose the best model. However, it is not guaranteed, that this will give us a better clustering for LDA.

In conclusion, for this dataset and the requirement of 5 or 10 clusters, the scores are quite poor indicating that it is not meant for clustering and/or different amount of clusters should be chosen. However, the assignment gave us a great impression about the text-mining and word-based clustering.


