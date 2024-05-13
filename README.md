# Text Mining and Clustering Analysis on Movie Reviews

## Authors
- Adam Diamantidis
- Dimitrios Diamantidis

## Date
03-11-2023

## Project Description
This project explores text mining and clustering techniques applied to a dataset of movie reviews. We utilize various R packages to process, analyze, and cluster text data to uncover underlying patterns in movie review sentiments and content.

## Data Description
The dataset contains reviews of movies collected and formatted as a tibble. Reviews are analyzed for word frequency, character count, and sentiment, providing insights into common themes and opinions expressed in the reviews.

## Methodologies Employed
1. **Text Mining**: We applied tokenization, stop word removal, and stemming to preprocess the data. Techniques like word clouds and frequency bar plots were used to visualize the most common words and themes.
2. **Clustering Techniques**:
    - **KMeans Clustering**: Implemented to partition the reviews into clusters based on textual similarity.
    - **Hierarchical Clustering**: Used to build a dendrogram of reviews based on their thematic closeness.
    - **Density-based Clustering (DBSCAN)**: Applied to identify clusters with varying densities.
    - **Latent Dirichlet Allocation (LDA)**: Utilized to model topics within the reviews, aiming to find common themes across documents.
3. **Evaluation and Model Comparison**: Various clustering outputs were evaluated using silhouette scores and coherence measures to determine the effectiveness of each clustering approach.

## Technologies Used
- R Programming Language
- Libraries: ggplot2, tidyverse, tidytext, tm, wordcloud, SnowballC, text2vec, textdata, cluster, clustertend, factoextra, dplyr, gridExtra, dendextend, textmineR, dbscan

