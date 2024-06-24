import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from xgboost import XGBClassifier

# Function to create and retrieve the Count Vectorizer and the TF-IDF Transformer
from Vectorizer import get_cv_tfidf
cntizer,tfizer = get_cv_tfidf()

# Initialize NLTK lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
useless_words = stopwords.words("english")

# Load trained LR models
try:
    lr0 = joblib.load(r"Models\lr0_model.joblib")
    lr1 = joblib.load(r"Models\lr1_model.joblib")
    lr2 = joblib.load(r"Models\lr2_model.joblib")
    lr3 = joblib.load(r"Models\lr3_model.joblib")
    print("LR Models loaded")
except:
    print("LR Model loading failed")

# Load trained KNN models
try:
    knn0 = joblib.load(r"Models\knn0_model.joblib")
    knn1 = joblib.load(r"Models\knn1_model.joblib")
    knn2 = joblib.load(r"Models\knn2_model.joblib")
    knn3 = joblib.load(r"Models\knn3_model.joblib")
    print("KNN models loaded")
except:
    print("KNN model loading failed")

# Load trained RF models
try:
    rf0 = joblib.load(r"Models/rf0_model.joblib")
    rf1 = joblib.load(r"Models/rf1_model.joblib")
    rf2 = joblib.load(r"Models/rf2_model.joblib")
    rf3 = joblib.load(r"Models/rf3_model.joblib")
    print("RF models loaded")
except:
    print("RF model loading failed")

# Load trained SGD models
try:
    sgd0 = joblib.load(r"Models\sgd0_model.joblib")
    sgd1 = joblib.load(r"Models\sgd1_model.joblib")
    sgd2 = joblib.load(r"Models\sgd2_model.joblib")
    sgd3 = joblib.load(r"Models\sgd3_model.joblib")
    print("SGD models loaded")
except:
    print("SGD model loading failed")

# Load trained SVM models
try:
    svm0 = joblib.load(r"Models\svm0_model.joblib")
    svm1 = joblib.load(r"Models\svm1_model.joblib")
    svm2 = joblib.load(r"Models\svm2_model.joblib")
    svm3 = joblib.load(r"Models\svm3_model.joblib")
    print("SVM models loaded")
except:
    print("SVM model loading failed")

try:
    xgb0 = joblib.load(r"Models/xgb0_model.joblib")
    xgb1 = joblib.load(r"Models/xgb1_model.joblib")
    xgb2 = joblib.load(r"Models/xgb2_model.joblib")
    xgb3 = joblib.load(r"Models/xgb3_model.joblib")
    print("XGB models loaded")
except:
    print("XGB model loading failed")

# List of MBTI personality types
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                   'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]

# Text preprocessing function
def clean_text(text, remove_stop_words=True, remove_mbti_profiles=True):
    # Remove and clean comments
    temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    temp = re.sub("[^a-zA-Z]", " ", temp)
    temp = re.sub(' +', ' ', temp).lower()

    # Remove multiple letter repeating words
    temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

    # Remove stop words
    if remove_stop_words:
        temp = " ".join([w for w in temp.split(' ') if w not in useless_words])

    # Remove MBTI personality words from posts
    if remove_mbti_profiles:
        for t in unique_type_list:
            temp = temp.replace(t, "")

    return temp

# Translate binary vector to MBTI personality type
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]
def translate_back(personality):
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s

# Example function to predict MBTI personality type from tweets
def predict_personality(tfidf_text):
    
    lr_pred = [lr0.predict(tfidf_text)[0], lr1.predict(tfidf_text)[0], lr2.predict(tfidf_text)[0], lr3.predict(tfidf_text)[0]]
    lr_personality = 'LR: ' + translate_back(lr_pred)

    knn_pred = [knn0.predict(tfidf_text)[0], knn1.predict(tfidf_text)[0], knn2.predict(tfidf_text)[0], knn3.predict(tfidf_text)[0]]
    knn_personality = 'KNN: ' + translate_back(knn_pred)

    rf_pred = [rf0.predict(tfidf_text)[0], rf1.predict(tfidf_text)[0], rf2.predict(tfidf_text)[0], rf3.predict(tfidf_text)[0]]
    rf_personality = 'RF: ' + translate_back(rf_pred)

    sgd_pred = [sgd0.predict(tfidf_text)[0], sgd1.predict(tfidf_text)[0], sgd2.predict(tfidf_text)[0], sgd3.predict(tfidf_text)[0]]
    sgd_personality = 'SGD: ' + translate_back(sgd_pred)

    svm_pred = [svm0.predict(tfidf_text)[0], svm1.predict(tfidf_text)[0], svm2.predict(tfidf_text)[0], svm3.predict(tfidf_text)[0]]
    svm_personality = 'SVM: ' + translate_back(svm_pred)

    xgb_pred = [xgb0.predict(tfidf_text)[0], xgb1.predict(tfidf_text)[0], xgb2.predict(tfidf_text)[0], xgb3.predict(tfidf_text)[0]]
    xgb_personality = 'XGB: ' + translate_back(xgb_pred)

    result = [lr_personality, knn_personality, rf_personality, sgd_personality, svm_personality, xgb_personality]
    return result

tweets = "But seriously the only way to guarantee meaningful positive change in the next 365 days around the sun is to exit your comfort zone and relentlessly attack your goals. That’s my plan, anyway. Good luck and Happy New Year 🙌🏾 ||| Thank you for this evolution @dbrand. I've known ya'll long enough to know no malice was intended, but some reflection goes a long way. Love to see it, good to have you back 🤝 ||| I have…… many thoughts ||| Solar eclipse from space 🤓  https://www.forbes.com/sites/jamiecartereurope/2024/04/08/total-solar-eclipse-photos-nasa-astronauts-take-historic-images-from-space/?sh=5ed68f0565aa ||| The Empire is ONE win away from breaking the most consecutive wins in league history (we are currently tied with @TorontoRush at 30)… Join us on April 27th at home as we try and break the UFA record with 31 STRAIGHT wins! #KeepTheStreak 🗽 https://shopempireultimate.com/products/single-game-tickets-1 ||| 10/10 that was the coolest thing I've ever seen ||| I'm going with Galaxy S24 Ultra ||| We’re at a full PacMan. In the totality zone. Could not be more pumped for this eclipse ||| Theory: the moon is accidentally the most photographed object in human history, and it massively extends its lead today ||| Just pointed my phone camera directly at the midday sun for 5 minute straight. It got a little warm, but... no damage. I'm gonna go ahead and say point your phones at the eclipse tomorrow, it'll be fine 🌞🌚 ||| NEW VIDEO - Is the iPhone Illegal?  https://piped.video/qcH2wgRLiV8 ||| Twitter is definitely still fun for everyone feeling an earthquake at the same time ||| Is getting surgery the worst place to be during a rare east coast earthquake? Because that's where I was, getting my wisdom teeth taken out 😭 |||  ||| I cannot for the life of me find a definitive answer to whether or not pointing a smartphone at the solar eclipse will fry the sensor  Tempted to just take a phone I don't need and point it at the sun for 5 minutes to find out the real answer myself. In the name of science ||| 5 Seasons with New York 🗽 5 Championship Weekend appearances 🏟️ 3 Championships 🏆  & @MKBHD is running it back for year six with The Empire 👏👏👏 ||| First big Vision Pro update: Shared experiences. Spatial personas  This was at the top of my list of what was missing from Vision Pro - the floating head is a little odd looking, but the principle is there: Interact with the same content/game/object as someone else in realtime ||| Studio_Tour_Outtake_DONOTPUBLISH.mp4 ||| Would you trust an AI assistant to plan an entire vacation for you with a single voice command? ||| NEW VIDEO - 1 month and 2000 miles into Tesla Cybertruck ownership... The Full Review!  I’ve put this thing through its paces, contemplated its ICON status, and.. maybe even set a world record?  Full video: https://piped.video/O0cs8aIXgkc ||| Uploading... ||| Fairly confident this next upload is the best video we've ever made. It's so sick. I can't wait to publish this ||| When I was in high school we got in trouble for playing games on TI-84 calculators ||| A(bsolutely) I(ncredible) WWDC  (I'll believe it when I see it) ||| There's two types of people: To-to list app Alarm clock app  Shoutout to @pierce for helping us pit them against each other!  https://piped.video/watch?v=dUMY3Xhmzjc ||| I wish I could screen record my eyeballs sometimes ||| Best mistake I ever made  Just locked down booking a really fun video shoot  in the middle of the country in 2 weeks  Then I looked on the calendar and there's a TOTAL SOLAR ECLIPSE going through the middle of the country on those exact dates  And sure enough (total accident) the video shoot is IN THE MIDDLE of the totality, at the exact hours of the eclipse 🌕🌚  I could not be more excited ||| I get at least 1 email per day spelling MKBHD wrong in the subject line and multiple times throughout the email. I never quite know how to respond. This one does it in bold 😅 ||| OUT NOW: Formula 1 cars, explained for rookies (with Max Verstappen)   Watch our new episode of Huge If True here: https://piped.video/VJgdOMXhEj0?si=Tzdfd08R2xr4gR6P  I love Formula 1. This isnt just a car race, its a massive group science competition. Ten teams are all fighting to be first, and on every team there are hundreds of people, spending millions of dollars, all working together to push technology to its limit.  In this video, we take you on an adventure to see what it takes to build and drive some of the fastest and most expensive cars in the world. We got rare access into a factory in England where they build these cars and into a garage in Bahrain where they race them...  Im so proud of the Huge If True team: animator and producer @Justin_Poore, editor @LogenKershaw and science producer @NMenkart . We have so much more coming for you soon… ||| bruh ||| 16 years 👴🏾 ||| “Starting at” price is always a tricky thing…  NEW VIDEO: https://piped.video/HN-WH7C4K0Q ||| Alright - congrats to @_iamFreddyp on winning a custom PS5 slim - Look out for a DM! ||| Two-faced PS5, anyone? 🔴⚫️  @ColorWare will give 1 random person who follows and RTs this tweet a PS5 slim in whatever colors they want for free! Good luck  😈 ||| That one post that shows up at the top of the timeline for 0.6 seconds before it refreshes and disappears forever ||| Damn, Walmart is selling a Macbook Air M1 for the first time... $699  https://corporate.walmart.com/news/2024/03/15/walmart-brings-the-popular-macbook-air-with-the-m1-chip-to-its-shelves ||| Small phones are dead and we killed them ||| Either a review calling the product bad killed the company  OR  The product was so bad it killed the company ||| Happy pi day 🥧 ||| In case anyone was wondering, yes you can get more than 24 stand hours if you have a long enough flight east 🤓 ||| The new electric performance king? Model S Plaid and Lucid Air Sapphire are on notice (I gotta drive this thing)  Porsche Taycan Turbo GT 1092 Horsepower 0-60mph in 2.2 seconds 0-124mph in 6.4 seconds Nurburgring lap in 07:07! ||| Its been an incredible few days in Japan - believe it or not, for ultimate frisbee. Dream Cup 2024 is a huge tournament here and PoNY was invited from New York and we ended up winning!  Already pumped to get back to the studio with some new video ideas 🙌🏾  Shot on iPhone 15 Pro ||| Good morning to everyone except whoever invented Daylight Savings time ||| Shot on Samsung Galaxy S24 Ultra ||| Here's a look inside the new Rivian R2 (starts at $45,000)... and R3 and R3X!  https://piped.video/k0Gt_PUyldc?si=4kSfHw0Ly5y4WAhP ||| How is it that I can be on the absolute worst WiFi on planet earth, where literal Google searches take minutes to load but somehow a preroll video ad streams in perfect quality with no interruptions ☠️ ||| So you know how @ColorWare takes everything apart and paints it and puts it back together? They do the Apple Pencil now and…… its so sick ||| NEW VIDEO - The Nothing Phone 2A is $350 and that's a damn good deal!  Full review: https://piped.video/VdOlqcg9uMQ ||| FYI the new M3 MacBook Air only supports 2 external displays with the lid CLOSED. Weirdly, if you open the laptop to use the keyboard/trackpad, it turns off one of the external displays  So if you plan on plugging into 2 screens youll also need to bring your own mouse + keyboard ||| Macbook Air M3 refresh is live  - Faster chip - Faster Wifi - Support for 2 external displays - Still 8/256GB base - Same design/battery life  - M2 Macbook Air drops by $100 - M1 Macbook Air discontinued  https://www.apple.com/newsroom/2024/03/apple-unveils-the-new-13-and-15-inch-macbook-air-with-the-powerful-m3-chip/"
cleaned_text = clean_text(tweets, remove_stop_words=True, remove_mbti_profiles=True)
print("Cleaned Text:\n",cleaned_text,'\n')
vectorized_text = cntizer.transform([cleaned_text])
#print("Vectorized Text:\n", vectorized_text,'\n')
tfidf_text = tfizer.transform(vectorized_text).toarray()
#print("Transformed Text:\n",tfidf_text,'\n')
predicted_personalities = predict_personality(tfidf_text)
print("Predicted MBTI personality type:", predicted_personalities)