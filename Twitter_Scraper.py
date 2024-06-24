import os
import csv
from ntscraper import Nitter

print("Initializing Nitter.....")
scraper = Nitter(log_level=1, skip_instance_check=False)
print("Scraper Ready")

# Post Retreival
def retrieve_tweets(name):
    posts = scraper.get_tweets(name, mode='user', number=50)
    tweets = ''
    for i in range(50):
        if i == 0:
            tweets += posts['tweets'][i]['text']
        else:
            tweets += ' ||| ' + posts['tweets'][i]['text']
    return tweets

username ="mkbhd"
try:
    retrieved_tweets = retrieve_tweets(username)
    print(retrieved_tweets)
except:
    print('Scraper down, please try again')

# CSV  Writing
def save_csv(username, tweets):
    # Check if CSV file exists. If not, create it and write headings
    if not os.path.exists('tweets_database.csv'):
        with open('tweets_database.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['No.', 'Name', 'Tweets']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Open CSV file in append mode
    with open('tweets_database.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['No.', 'Name', 'Tweets']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Get the last number in the existing file
        last_number = 0
        try:
            with open('tweets_database.csv', 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    last_number = int(row['No.'])
        except FileNotFoundError:
            pass
        writer.writerow({'No.': last_number + 1, 'Name': username, 'Tweets': tweets})

save_csv(username, retrieved_tweets)