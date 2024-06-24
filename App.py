import tkinter as tk
from Twitter_Scraper2 import retrieve_tweets
#from ntscraper import Nitter
from nltk.corpus import stopwords
import joblib
import numpy as np

from Vectorizer import get_cv_tfidf
from ML_Prediction2 import preprocess_text, predict_personality

# Initialize Nitter
try:
    print("Initializing Nitter.....")
    scraper = Nitter(log_level=1, skip_instance_check=False)
    print("Scraper initialized")
except:
    print("Scraper NOT initialized")

# Load CountVectorizer and TfidfTransformer
try:
    cntizer, tfizer = get_cv_tfidf()
    print("cntizer and tfizer loaded")
except:
    print("cntizer and tfizer NOT loaded")

# Load trained models
try:
    lr0 = joblib.load(r"Models\lr0_model.joblib")
    lr1 = joblib.load(r"Models\lr1_model.joblib")
    lr2 = joblib.load(r"Models\lr2_model.joblib")
    lr3 = joblib.load(r"Models\lr3_model.joblib")
    knn0 = joblib.load(r"Models\knn0_model.joblib")
    knn1 = joblib.load(r"Models\knn1_model.joblib")
    knn2 = joblib.load(r"Models\knn2_model.joblib")
    knn3 = joblib.load(r"Models\knn3_model.joblib")
    rf0 = joblib.load(r"Models\rf0_model.joblib")
    rf1 = joblib.load(r"Models\rf1_model.joblib")
    rf2 = joblib.load(r"Models\rf2_model.joblib")
    rf3 = joblib.load(r"Models\rf3_model.joblib")
    sgd0 = joblib.load(r"Models\sgd0_model.joblib")
    sgd1 = joblib.load(r"Models\sgd1_model.joblib")
    sgd2 = joblib.load(r"Models\sgd2_model.joblib")
    sgd3 = joblib.load(r"Models\sgd3_model.joblib")
    svm0 = joblib.load(r"Models\svm0_model.joblib")
    svm1 = joblib.load(r"Models\svm1_model.joblib")
    svm2 = joblib.load(r"Models\svm2_model.joblib")
    svm3 = joblib.load(r"Models\svm3_model.joblib")
    xgb0 = joblib.load(r"Models\xgb0_model.joblib")
    xgb1 = joblib.load(r"Models\xgb1_model.joblib")
    xgb2 = joblib.load(r"Models\xgb2_model.joblib")
    xgb3 = joblib.load(r"Models\xgb3_model.joblib")
    print("Models loaded")
except Exception as e:
    print("Model loading failed:", e)


# Create the main application window
root = tk.Tk()
root.title("Twitter Tweet Retriever")

# Function placeholders for predicting MBTI personality types
def predict_personality_1():
    retrieved_tweets = tweets_display.get("3.0", tk.END)
    preprocessed_text = preprocess_text(retrieved_tweets, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True)
    personality = predict_personality(preprocessed_text, lr0, lr1, lr2, lr3)
    result_box1.config(state=tk.NORMAL)
    result_box1.delete('1.0', tk.END)
    result_box1.insert(tk.END, personality)
    result_box1.config(state=tk.DISABLED)

def predict_personality_2():
    retrieved_tweets = tweets_display.get("3.0", tk.END)
    preprocessed_text = preprocess_text(retrieved_tweets, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True)
    personality = predict_personality(preprocessed_text, knn0, knn1, knn2, knn3)
    result_box2.config(state=tk.NORMAL)
    result_box2.delete('1.0', tk.END)
    result_box2.insert(tk.END, personality)
    result_box2.config(state=tk.DISABLED)

def predict_personality_3():
    retrieved_tweets = tweets_display.get("3.0", tk.END)
    preprocessed_text = preprocess_text(retrieved_tweets, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True)
    personality = predict_personality(preprocessed_text, rf0, rf1, rf2, rf3)
    result_box3.config(state=tk.NORMAL)
    result_box3.delete('1.0', tk.END)
    result_box3.insert(tk.END, personality)
    result_box3.config(state=tk.DISABLED)

def predict_personality_4():
    retrieved_tweets = tweets_display.get("3.0", tk.END)
    preprocessed_text = preprocess_text(retrieved_tweets, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True)
    personality = predict_personality(preprocessed_text, sgd0, sgd1, sgd2, sgd3)
    result_box4.config(state=tk.NORMAL)
    result_box4.delete('1.0', tk.END)
    result_box4.insert(tk.END, personality)
    result_box4.config(state=tk.DISABLED)

def predict_personality_5():
    retrieved_tweets = tweets_display.get("3.0", tk.END)
    preprocessed_text = preprocess_text(retrieved_tweets, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True)
    personality = predict_personality(preprocessed_text, svm0, svm1, svm2, svm3)
    result_box5.config(state=tk.NORMAL)
    result_box5.delete('1.0', tk.END)
    result_box5.insert(tk.END, personality)
    result_box5.config(state=tk.DISABLED)

def predict_personality_6():
    retrieved_tweets = tweets_display.get("3.0", tk.END)
    preprocessed_text = preprocess_text(retrieved_tweets, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True)
    personality = predict_personality(preprocessed_text, xgb0, xgb1, xgb2, xgb3)
    result_box6.config(state=tk.NORMAL)
    result_box6.delete('1.0', tk.END)
    result_box6.insert(tk.END, personality)
    result_box6.config(state=tk.DISABLED)

def search_tweets():
    username = username_entry.get()
    try:
        global retrieved_tweets
        retrieved_tweets = retrieve_tweets(username, scraper)
        # Display preprocessed text in tweet_display
        tweets_display.config(state=tk.NORMAL)
        tweets_display.delete('1.0', tk.END)  # Clear previous content
        tweets_display.insert(tk.END, retrieved_tweets)
        tweets_display.config(state=tk.DISABLED)

    except:
        tweets_display.config(state=tk.NORMAL)
        tweets_display.delete('1.0', tk.END)
        tweets_display.insert(tk.END, "Error retrieving tweets. Please try again later.")
        tweets_display.config(state=tk.DISABLED)

# Function to clear all text entry boxes and result boxes
def clear_boxes():
    username_entry.delete(0, tk.END)
    tweets_display.config(state=tk.NORMAL)
    tweets_display.delete('1.0', tk.END)
    tweets_display.config(state=tk.DISABLED)
    result_box1.config(state=tk.NORMAL)
    result_box1.delete('1.0', tk.END)
    result_box1.config(state=tk.DISABLED)
    result_box2.config(state=tk.NORMAL)
    result_box2.delete('1.0', tk.END)
    result_box2.config(state=tk.DISABLED)
    result_box3.config(state=tk.NORMAL)
    result_box3.delete('1.0', tk.END)
    result_box3.config(state=tk.DISABLED)
    result_box4.config(state=tk.NORMAL)
    result_box4.delete('1.0', tk.END)
    result_box4.config(state=tk.DISABLED)
    result_box5.config(state=tk.NORMAL)
    result_box5.delete('1.0', tk.END)
    result_box5.config(state=tk.DISABLED)
    result_box6.config(state=tk.NORMAL)
    result_box6.delete('1.0', tk.END)
    result_box6.config(state=tk.DISABLED)

# Function to perform ensemble prediction based on predictions from six algorithms
def ensemble_prediction():
    predictions = [result_box1.get("1.0", tk.END).strip(),
                   result_box2.get("1.0", tk.END).strip(),
                   result_box3.get("1.0", tk.END).strip(),
                   result_box4.get("1.0", tk.END).strip(),
                   result_box5.get("1.0", tk.END).strip(),
                   result_box6.get("1.0", tk.END).strip()]

    # Perform ensemble prediction (You may replace this with your specific ensemble method)
    # Here, we are simply taking the most common prediction among the six algorithms
    final_prediction = max(set(predictions), key=predictions.count)

    # Display the final ensemble prediction in the ensemble_result_box
    ensemble_result_box.config(state=tk.NORMAL)
    ensemble_result_box.delete("1.0", tk.END)
    ensemble_result_box.insert(tk.END, final_prediction)
    ensemble_result_box.config(state=tk.DISABLED)

# Create left frame for tweet retrieval
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create username entry and search button in left frame
username_label = tk.Label(left_frame, text="Enter Twitter Username:")
username_label.grid(row=0, column=0, padx=5, pady=5)

username_entry = tk.Entry(left_frame, width=30)
username_entry.grid(row=0, column=1, padx=5, pady=5)

search_button = tk.Button(left_frame, text="Search Tweets", command=search_tweets)
search_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

# Create display area for retrieved tweets in left frame
tweets_display_label = tk.Label(left_frame, text="Retrieved Tweets:")
tweets_display_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

tweets_display = tk.Text(left_frame, width=50, height=11, state=tk.DISABLED)
tweets_display.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

# Create clear button
clear_button = tk.Button(left_frame, text="Clear", command=clear_boxes)
clear_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

# Create right frame for MBTI personality prediction
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Create buttons and text boxes for predicting MBTI personality types in right frame
predict_button_1 = tk.Button(right_frame, text="Predict LR", command=predict_personality_1)
predict_button_1.grid(row=0, column=0, padx=5, pady=5)

result_box1 = tk.Text(right_frame, width=50, height=2, state=tk.DISABLED)
result_box1.grid(row=0, column=1, padx=5, pady=5)

predict_button_2 = tk.Button(right_frame, text="Predict KNN", command=predict_personality_2)
predict_button_2.grid(row=1, column=0, padx=5, pady=5)

result_box2 = tk.Text(right_frame, width=50, height=2, state=tk.DISABLED)
result_box2.grid(row=1, column=1, padx=5, pady=5)

predict_button_3 = tk.Button(right_frame, text="Predict RF", command=predict_personality_3)
predict_button_3.grid(row=2, column=0, padx=5, pady=5)

result_box3 = tk.Text(right_frame, width=50, height=2, state=tk.DISABLED)
result_box3.grid(row=2, column=1, padx=5, pady=5)

predict_button_4 = tk.Button(right_frame, text="Predict SGD", command=predict_personality_4)
predict_button_4.grid(row=3, column=0, padx=5, pady=5)

result_box4 = tk.Text(right_frame, width=50, height=2, state=tk.DISABLED)
result_box4.grid(row=3, column=1, padx=5, pady=5)

predict_button_5 = tk.Button(right_frame, text="Predict SVM", command=predict_personality_5)
predict_button_5.grid(row=4, column=0, padx=5, pady=5)

result_box5 = tk.Text(right_frame, width=50, height=2, state=tk.DISABLED)
result_box5.grid(row=4, column=1, padx=5, pady=5)

predict_button_6 = tk.Button(right_frame, text="Predict XGB", command=predict_personality_6)
predict_button_6.grid(row=5, column=0, padx=5, pady=5)

result_box6 = tk.Text(right_frame, width=50, height=2, state=tk.DISABLED)
result_box6.grid(row=5, column=1, padx=5, pady=5)

# Create ensemble button
ensemble_button = tk.Button(right_frame, text="Ensemble", command=ensemble_prediction)
ensemble_button.grid(row=6, column=0, columnspan=1, padx=5, pady=5, sticky='ew')

# Create ensemble result box
ensemble_result_box = tk.Text(right_frame, width=50, height=2, state=tk.DISABLED)
ensemble_result_box.grid(row=6, column=1, columnspan=2, padx=5, pady=5, sticky='ew')


# Run the Tkinter event loop
root.mainloop()