import joblib
import numpy as np
import customtkinter as ctk
from ntscraper import Nitter
from nltk.corpus import stopwords

from Vectorizer import get_cv_tfidf
from Twitter_Scraper2 import retrieve_tweets
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

def predict_personality_1():
    retrieved_tweets = tweets_display.get("1.0", "end-1c")
    preprocessed_text = preprocess_text(retrieved_tweets, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True)
    personality = predict_personality(preprocessed_text, lr0, lr1, lr2, lr3)
    result_box1.configure(state="normal")
    result_box1.delete("1.0", "end")
    result_box1.insert("end", personality)
    result_box1.configure(state="disabled")

def predict_personality_2():
    retrieved_tweets = tweets_display.get("1.0", "end-1c")
    preprocessed_text = preprocess_text(retrieved_tweets, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True)
    personality = predict_personality(preprocessed_text, knn0, knn1, knn2, knn3)
    result_box2.configure(state="normal")
    result_box2.delete("1.0", "end")
    result_box2.insert("end", personality)
    result_box2.configure(state="disabled")

def predict_personality_3():
    retrieved_tweets = tweets_display.get("1.0", "end-1c")
    preprocessed_text = preprocess_text(retrieved_tweets, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True)
    personality = predict_personality(preprocessed_text, rf0, rf1, rf2, rf3)
    result_box3.configure(state="normal")
    result_box3.delete("1.0", "end")
    result_box3.insert("end", personality)
    result_box3.configure(state="disabled")

def predict_personality_4():
    retrieved_tweets = tweets_display.get("1.0", "end-1c")
    preprocessed_text = preprocess_text(retrieved_tweets, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True)
    personality = predict_personality(preprocessed_text, sgd0, sgd1, sgd2, sgd3)
    result_box4.configure(state="normal")
    result_box4.delete("1.0", "end")
    result_box4.insert("end", personality)
    result_box4.configure(state="disabled")

def predict_personality_5():
    retrieved_tweets = tweets_display.get("1.0", "end-1c")
    preprocessed_text = preprocess_text(retrieved_tweets, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True)
    personality = predict_personality(preprocessed_text, svm0, svm1, svm2, svm3)
    result_box5.configure(state="normal")
    result_box5.delete("1.0", "end")
    result_box5.insert("end", personality)
    result_box5.configure(state="disabled")

def predict_personality_6():
    retrieved_tweets = tweets_display.get("1.0", "end-1c")
    preprocessed_text = preprocess_text(retrieved_tweets, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True)
    personality = predict_personality(preprocessed_text, xgb0, xgb1, xgb2, xgb3)
    result_box6.configure(state="normal")
    result_box6.delete("1.0", "end")
    result_box6.insert("end", personality)
    result_box6.configure(state="disabled")

def search_tweets():
    username = username_entry.get()
    try:
        global retrieved_tweets
        retrieved_tweets = retrieve_tweets(username, scraper)
        tweets_display.delete("1.0", "end")
        tweets_display.insert("end", retrieved_tweets)
    except:
        tweets_display.delete("1.0", "end")
        tweets_display.insert("end", "Error retrieving tweets. Please try again later.")

# Function to clear all text entry boxes and result boxes
def clear_boxes():
    username_entry.delete(0, "end")
    tweets_display.delete("1.0", "end")
    result_box1.configure(state="normal")
    result_box1.delete("1.0", "end")
    result_box1.configure(state="disabled")
    result_box2.configure(state="normal")
    result_box2.delete("1.0", "end")
    result_box2.configure(state="disabled")
    result_box3.configure(state="normal")
    result_box3.delete("1.0", "end")
    result_box3.configure(state="disabled")
    result_box4.configure(state="normal")
    result_box4.delete("1.0", "end")
    result_box4.configure(state="disabled")
    result_box5.configure(state="normal")
    result_box5.delete("1.0", "end")
    result_box5.configure(state="disabled")
    result_box6.configure(state="normal")
    result_box6.delete("1.0", "end")
    result_box6.configure(state="disabled")
    ensemble_result_box.configure(state="normal")
    ensemble_result_box.delete("1.0", "end")
    ensemble_result_box.configure(state="disabled")

# Function to perform ensemble prediction based on predictions from six algorithms
def ensemble_prediction():
    predictions = [
        result_box1.get("1.0", "end-1c").strip(),
        result_box2.get("1.0", "end-1c").strip(),
        result_box3.get("1.0", "end-1c").strip(),
        result_box4.get("1.0", "end-1c").strip(),
        result_box5.get("1.0", "end-1c").strip(),
        result_box6.get("1.0", "end-1c").strip(),
    ]

    final_prediction = max(set(predictions), key=predictions.count)
    # Display the final ensemble prediction in the ensemble_result_box
    ensemble_result_box.configure(state="normal")
    ensemble_result_box.delete("1.0", "end")
    ensemble_result_box.insert("end", final_prediction)
    ensemble_result_box.configure(state="disabled")


# Create the main application window
root = ctk.CTk()
root.title("Twitter Tweet Retriever")

root.geometry("700x520")

# Set Dark theme and Blue color scheme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Create a heading label
heading_label = ctk.CTkLabel(root, text="Twitter MBTI Personality Predictor", font=("Arial", 30, "bold"))
heading_label.pack(pady=(20,5))

# Create a description label
description_label = ctk.CTkLabel(root, text="*** This application uses the MBTI personality system to predict your personality type based on your Twitter tweets ***", font=("Roboto", 10), wraplength=600, justify="center")
description_label.pack(pady=(10,0))
description_label = ctk.CTkLabel(root, text="*** Enter your Twitter username below to get started ***", font=("Roboto", 10), wraplength=600, justify="center")
description_label.pack(pady=(0,10))

# Create left frame for tweet retrieval
left_frame = ctk.CTkFrame(root)
left_frame.pack(side=ctk.LEFT, padx=5, pady=10, expand=True)

# Username label and entry in left frame
username_label = ctk.CTkLabel(left_frame, text="Enter Twitter Username:")
username_label.grid(row=0, column=0, padx=5, pady=10)

username_entry = ctk.CTkEntry(left_frame, width=100)
username_entry.grid(row=0, column=1, padx=5, pady=10)

# Search button
search_button = ctk.CTkButton(left_frame, text="Search Tweets",command=search_tweets)
search_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# Tweets display label and textbox
tweets_display_label = ctk.CTkLabel(left_frame, text="Retrieved Tweets:")
tweets_display_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

tweets_display = ctk.CTkTextbox(left_frame, width=350, height=180)
tweets_display.grid(row=4, column=0, columnspan=2, padx=(5,5), pady=5)

# Clear button
clear_button = ctk.CTkButton(left_frame, text="Clear",command=clear_boxes)
clear_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

# Create right frame for MBTI personality prediction
right_frame = ctk.CTkFrame(root)
right_frame.pack(side=ctk.RIGHT, padx=10, pady=10, expand=True)

# Prediction buttons
predict_button_1 = ctk.CTkButton(right_frame, text="Predict LR", command=predict_personality_1)
predict_button_1.grid(row=0, column=0, padx=10, pady=10)
result_box1 = ctk.CTkTextbox(right_frame, width=80, height=1, state="disabled")
result_box1.grid(row=0, column=1, padx=10, pady=(5,5))

predict_button_2 = ctk.CTkButton(right_frame, text="Predict KNN", command=predict_personality_2)
predict_button_2.grid(row=1, column=0, padx=5, pady=10)
result_box2 = ctk.CTkTextbox(right_frame, width=80, height=1, state="disabled")
result_box2.grid(row=1, column=1, padx=5, pady=5)

predict_button_3 = ctk.CTkButton(right_frame, text="Predict RF", command=predict_personality_3)
predict_button_3.grid(row=2, column=0, padx=5, pady=10)
result_box3 = ctk.CTkTextbox(right_frame, width=80, height=1, state="disabled")
result_box3.grid(row=2, column=1, padx=5, pady=5)

predict_button_4 = ctk.CTkButton(right_frame, text="Predict SGD", command=predict_personality_4)
predict_button_4.grid(row=3, column=0, padx=5, pady=10)
result_box4 = ctk.CTkTextbox(right_frame, width=80, height=1, state="disabled")
result_box4.grid(row=3, column=1, padx=5, pady=5)

predict_button_5 = ctk.CTkButton(right_frame, text="Predict SVM", command=predict_personality_5)
predict_button_5.grid(row=4, column=0, padx=5, pady=10)
result_box5 = ctk.CTkTextbox(right_frame, width=80, height=1, state="disabled")
result_box5.grid(row=4, column=1, padx=5, pady=5)

predict_button_6 = ctk.CTkButton(right_frame, text="Predict XGB", command=predict_personality_6)
predict_button_6.grid(row=5, column=0, padx=5, pady=10)
result_box6 = ctk.CTkTextbox(right_frame, width=80, height=1, state="disabled")
result_box6.grid(row=5, column=1, padx=5, pady=5)

# Create ensemble button
ensemble_button = ctk.CTkButton(right_frame, text="Ensemble",command=ensemble_prediction)
ensemble_button.grid(row=6, column=0, padx=5, pady=10)

ensemble_result_box = ctk.CTkTextbox(right_frame, width=80, height=10, fg_color="#2e2e2e", text_color="white", border_color="#535454", border_width=2, font=("Arial", 14, "bold"))
ensemble_result_box.grid(row=6, column=1, padx=5, pady=10)


# Run the Tkinter event loop
root.mainloop()