This is an ML project built entirely using Python. The project tries to unravel users' personalities from Twitter tweets, helping in areas from targeted marketing strategies to providing tailored mental health support and contributing to nuanced public opinion analyses. The goal is to explore the potential enhancements in user experience and engagement on the platform. By categorizing users, the project aims to offer personalized content, targeted advertising, and foster community building in the digital space.

Myers-Briggs Type Indicator (MBTI) is a widely used tool for categorizing individuals into 16 distinct personality types. The project uses machine learning models to classify Twitter users based on their MBTI personality types. 

The project used a single (BERT) model multiclass approach and a multimodel binary classification strategy, where we used four distinct machine learning models to predict binary personality traits based on the MBTI dichotomies: Introversion/Extroversion, Intuition/Sensing, Thinking/Feeling, and Judging/Perceiving. In the first approach, a BERT model categorizes tweets into one of the 16 MBTI personality types. The second approach breaks down the personality
classification task into four separate models, each specializing in one dichotomy. We used an ensemble of both these approaches to predict the final MBTI output. 

The Twitter personality detection system successfully utilized diverse machine learning models, demonstrating efficient accuracy across various personality dichotomies.

The project utilized Nitter, which is a discontinued free and open-source viewer for Twitter, for tweet retrieval. The front end was built for desktops using CustomTkinter.


Objective

This project aims to build a machine learning model to classify MBTI (Myers-Briggs Type Indicator) personality types based on users' text data, specifically their tweets.
Dataset

The dataset consists of two columns:

    type: The MBTI personality type of the user (e.g., INFJ, INTJ).
    posts: The last 50 tweets posted by the user, separated by "|||".

Steps Undertaken

    Data Loading and Cleaning
        Loaded the dataset using pandas.
        Created a function clean_text to preprocess the tweets by removing URLs, and punctuations, and converting text to lowercase.

    Text Tokenization
        Used Hugging Face's BERT tokenizer (bert-base-uncased) to tokenize the cleaned text data. Tokenized sequences were padded to a maximum length of 1500 tokens.

    Data Splitting
        Split the data into training, validation, and test sets using train_test_split from scikit-learn.

    Model Creation
        Defined a function create_model to build a BERT-based classification model using TensorFlow and Hugging Face's TFBertModel.
        The model architecture included:
            Input layer for token IDs.
            BERT layer to extract contextual embeddings.
            Dense layer with softmax activation for classification into 16 MBTI personality types.

    Model Training
        Encoded the MBTI personality types into one-hot labels.
        Trained the model using the training data and validated it using the validation data.
        Implemented early stopping to prevent overfitting.

    Model Saving
        Saved the trained model weights using TensorFlow's save_weights method.

    Model Loading and Evaluation
        Recreated the model architecture.
        Loaded the saved weights into the new model instance.
        Evaluated the model's performance on the test set to ensure its effectiveness.
