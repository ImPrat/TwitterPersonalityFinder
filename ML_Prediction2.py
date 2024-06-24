import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Text preprocessing function
def preprocess_text(text, cntizer, tfizer, remove_stop_words=True, remove_mbti_profiles=True):

    useless_words = stopwords.words("english")
    lemmatiser = WordNetLemmatizer()

    unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                   'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    unique_type_list = [x.lower() for x in unique_type_list]

    temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    temp = re.sub("[^a-zA-Z]", " ", temp)
    temp = re.sub(' +', ' ', temp).lower()
    temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

    if remove_stop_words:
        temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
    else:
        temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

    if remove_mbti_profiles:
        for t in unique_type_list:
            temp = temp.replace(t, "")

    vectorized_text = cntizer.transform([temp])
    tfidf_text = tfizer.transform(vectorized_text).toarray()

    return tfidf_text


def translate_back(personality):
    b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s


def predict_personality(preprocessed_text, m1, m2, m3, m4):
    pred = [m1.predict(preprocessed_text)[0], m2.predict(preprocessed_text)[0], 
            m3.predict(preprocessed_text)[0], m4.predict(preprocessed_text)[0]]
    personality = translate_back(pred)
    return personality