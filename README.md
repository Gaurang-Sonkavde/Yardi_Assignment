# Yardi_Assignment
The dataset includes summaries of 9 different disasters, meticulously concatenated for analysis. The project utilizes several preprocessing techniques to enhance data quality.

Preprocessing for Clean Data
Timestamp formats are standardized, and a powerful Preprocessor Class handles text preprocessing. Punctuation is removed, sentences normalized, and Twitter handles removed. Stopwords are eliminated, and lemmatization is performed, resulting in a new feature called "clean_tweet."

Insightful Visualizations
Data visualization sheds light on the dataset's characteristics. The variation of data is examined, and word frequency analysis helps identify key patterns. Moreover, word clouds using Twitter image masks represent the most frequent positive and negative words.

Effective Data Splitting
To set the stage for analysis, data is thoughtfully split into features (X) and labels (y). X represents text data, and y represents sentiment labels (1 for positive and 0 for negative).

Robust Sentiment Analysis Models
Three models are tested for sentiment analysis: NBNeuralNetwork, LSTM, and BERT. PyTorch and Scikit-learn are both leveraged, with PyTorch versions saved as pickle files for predictions due to computational constraints. GridSearchCV is attempted for hyperparameter tuning, ensuring optimized model performance.

Challenges and Solutions
While the BERT model was considered, building it proved time-consuming and raised RAM overwarning issues. Nonetheless, the NBNeuralNetwork and LSTM classifiers, armed with powerful pickle files, pave the way for efficient sentiment predictions.

Future-Ready Prediction Models
The trained classifiers serve as the foundation for creating robust prediction models. The saved pickle files enable sentiment predictions for new text data. With comprehensive documentation and code, this repository offers valuable insights for analyzing tweet sentiments during natural disasters.
