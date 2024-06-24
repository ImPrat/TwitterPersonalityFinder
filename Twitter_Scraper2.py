# Post Retreival
def retrieve_tweets(name, scraper):
    try:
        posts = scraper.get_tweets(name, mode='user', number=50)
        tweets = ''
        for i in range(50):
            if i == 0:
                tweets += posts['tweets'][i]['text']
            else:
                tweets += ' ||| ' + posts['tweets'][i]['text']
        return tweets
    except:
        return "Error retrieving Tweets! Check username or Try again"