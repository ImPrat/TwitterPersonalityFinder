This is an ML project built entirely using python. The project tries to unravel user's personalities from Twitter tweets, helping in areas from targeted marketing strategies to providing tailored mental health support and contributing to nuanced public opinion analyses. The goal is to explore the potential enhancements in user experience and engagement on the platform. By categorizing users, the project aims to offer personalized content, targeted advertising, and foster community building in the digital space.

Myers-Briggs Type Indicator (MBTI) is a widely used tool for categorizing individuals into 16 distinct personality types. The project uses machine learning models to classify Twitter users based on their MBTI personality types. 

The project used a single (BERT) model mulitclass approach and a multimodel binary classification strategy, where we use four distinct machine learning models to predict binary personality traits based on the MBTI dichotomies: Introversion/Extroversion, Intuition/Sensing, Thinking/Feeling, and Judging/Perceiving. In the first approach a BERT model is tasked with
categorizing tweets into one of the 16 MBTI personality types.The second approach breaks down the personality
classification task into four separate models, each specializing in one dichotomy. We used an ensemble of both these approaches to predict the final MBTI output. 

The Twitter personality detection system successfully utilized the diverse machine learning models, demonstrating efficient accuracy across various personality dichotomies.

The project utilized Nitter, which is a discontinued free and open source alternative viewer for Twitter, for tweet retrieval. The frontend was built for desktop using CustomTkinter.
