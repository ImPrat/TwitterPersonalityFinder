import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def get_cv_tfidf():

    # Define the file name for the CSV file
    csv_file = "Data\preprocessed_data.csv"

    # Initialize empty lists to store data from CSV
    list_posts = []
    list_personality = []

    # Read data from CSV file
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # Skip header
        next(reader)
        
        # Read each row and append data to lists
        for row in reader:
            list_posts.append(row[0])
            # Convert string values to integers and split them properly
            personality_values = [int(val) for val in row[1].strip('[]').split()]
            list_personality.append(personality_values)

    # Convert lists to NumPy arrays
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    print("Data loaded successfully.")

    try:
        cntizer = CountVectorizer(analyzer="word",max_features=1000,max_df=0.7,min_df=0.1)
        X_cnt = cntizer.fit_transform(list_posts)
        print("Count Vectorizer Ready")
    except:
        print("Count Vectorizer NOT Ready")

    tfizer = TfidfTransformer()
    try:
        X_tfidf =  tfizer.fit_transform(X_cnt).toarray()
        print("Tfidf Transformer Ready")
    except:
        print("Tfidf Transformer NOT Ready")

    return (cntizer, tfizer)