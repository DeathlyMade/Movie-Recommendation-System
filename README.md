# Movie Recommendation System

## Overview
The Movie Recommendation System is a project designed to suggest movies to users based on their preferences or viewing history. This system leverages data analysis and machine learning techniques to recommend movies using collaborative filtering and content-based filtering approaches.

## Features
- Personalized movie recommendations.
- Support for collaborative filtering based on user ratings.
- Content-based filtering using movie features such as genres, directors, or cast.
- Interactive user interface for exploring recommendations.

## Dataset
The system uses a dataset containing movie details, user ratings, and other relevant metadata. Link to the dataset:
- [Movie Recommender System](https://grouplens.org/datasets/movielens/) of which we have used the MovieLens 1M dataset

### Data Fields
The dataset includes:
- Movie ID
- Title
- Genre
- User ID
- Rating

## Installation
### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Required Python libraries:
  - numpy
  - pandas
  - sklearn
  - matplotlib
  - seaborn
  - collections
  - tqdm

## Methodology
The recommendation system employs the following techniques:

### 1. SVD and KMEANS
Null values are replaced with means, and then SVD (Singular Value Decomposition) is used and matrix is reconstructed with first 20 singular values.
Optimal number of clusters is taken as 3, and kmeans++ is applied on the user-rating train matrix.
For every user from the test matrix, appropriate cluster is calculated by seeing minimum norm from each cluster, and cluster is assigned, and cluster means are assigned to each new user.
Top 10 movies that new user has not seen, are recommended in a descending order of cluster mean rating.

#### Results:
RMSE for SVD + KMEANS = **0.0644**
MAE = **0.0270**

### 2. Collaborative Filtering
Collaborative filtering suggests movies based on user-item interactions. It uses:
- **User-based filtering**: Recommends movies liked by similar users. This is achieved by calculating user similarity using metrics like cosine similarity or Pearson correlation. For example, if User A and User B have similar rating patterns, movies liked by User B are recommended to User A.

- **Item-based filtering**: Recommends movies similar to those a user has rated highly. This approach computes similarity between items (movies) based on user ratings. If a user highly rated Movie X, other movies similar to X in terms of ratings are suggested.

#### Results where SVD was not applied on the user-rating matrix before applying the following algorithm:
- User-based Collaborative filtering
  - RMSE = **0.1969**
  - MAE = **0.0436**.
- Item-based Collaborative filtering
  - RMSE = **0.1923**
  - MAE = **0.0418**.


#### Results where SVD was applied on the user-rating matrix before applying the following algorithm:
- User-based Collaborative filtering
  - RMSE = **0.0671**
  - MAE = **0.0288**.
- Item-based Collaborative filtering
  - RMSE = **0.0592**
  - MAE = **0.0251**.
  

### 2. Content-Based Filtering

Content-based filtering recommends movies by analyzing their features, such as titles and genres. The process involves:  
- **Data Preprocessing:** Standardizing `Title` and `Genres` by converting them to lowercase and removing spaces, then combining them into a "soup" feature.  
- **Vectorization:** Transforming the "soup" into a numerical representation using **CountVectorizer**.  
- **Similarity Calculation:** Computing pairwise **cosine similarity** between movies to identify similar ones.  
- **Recommendation Generation:** Recommending the top 10 movies most similar to those rated by the user, weighted by user ratings.  

This ensures tailored recommendations based on movie content.

#### Results:
 Handling a New Data Point

To demonstrate the recommender system's ability, a new user (`UserID = 10000`) was created with manually assigned ratings for specific movies. These ratings reflected a preference for certain genres, namely sci-fi and action, and a dislike against comedy. The system then generated recommendations based on the user's ratings. The results were analyzed to verify that the recommended movies aligned with the user's genre preferences, highlighting the system's effectiveness in adapting to new data, as the predicted movies belonged to the genres preferred, and not the genre disliked.


### 3. Reinforcement Learning

Reinforcement learning (RL) is used to dynamically optimize movie recommendations by learning from user feedback. The system incorporates several bandit algorithms to maximize cumulative rewards:

- **A/B Testing**: Provides a baseline for comparison.
- **Epsilon-Greedy**: Balances exploration and exploitation.
- **Thompson Sampling**: Uses Bayesian inference for better exploration.
- **UCB (Upper Confidence Bound)**: Focuses on high-confidence recommendations.
- **Gradient Bandit**: Adjusts preferences based on feedback.
- **LinUCB**: Leverages contextual information for personalized suggestions.

### Reward Mechanism
Rewards are assigned when user ratings exceed a set threshold (e.g., 3/5). Algorithms adapt over time to prioritize movies yielding higher rewards.

### Results
- **Thompson Sampling** excels in balancing exploration and exploitation.
- Visualizations show cumulative rewards, highlighting algorithm performance over time.

### Performance Visualization
The accompanying graph illustrates the **total reward at Visit 20,000** for different bandit algorithms:

- **Thompson Sampling** achieves the highest cumulative rewards, demonstrating its effectiveness.
- **Gradient Bandit** shows the lowest reward performance, indicating room for improvement.
- Static strategies like **A/B Testing** underperform compared to adaptive algorithms, showcasing the importance of reinforcement learning in dynamic environments.

This visualization emphasizes how algorithms such as LinUCB and Thompson Sampling adaptively optimize recommendations, ensuring they align with evolving user preferences.

Reinforcement learning ensures adaptive, personalized recommendations as user preferences evolve.


## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- GroupLens Research for MovieLens datasets.
- IMDb for movie metadata.

---
Feel free to reach out for questions or collaboration opportunities!

