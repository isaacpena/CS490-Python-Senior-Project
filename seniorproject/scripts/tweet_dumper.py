#!/usr/bin/env python

import codecs
import tweepy
import csv

import sys

consumer_key = ""
consumer_secret = ""
token_key = ""
token_secret = ""

def get_tweets(username):

	authorization = tweepy.OAuthHandler(consumer_key, consumer_secret)
	authorization.set_access_token(token_key, token_secret)
	

	twitterapi = tweepy.API(authorization)
	apitweets = []

	new_tweets = twitterapi.user_timeline(screen_name = username,count=200)

	apitweets.extend(new_tweets)
	oldest = apitweets[-1].id - 1

	while len(new_tweets) > 0:
		new_tweets = api.user_timeline(screen_name = username,count=200,max_id=oldest)
		apitweets.extend(new_tweets)
		oldest = apitweets[-1].id - 1

		print "...{0} tweets downloaded\n".format(len(apitweets))

	outtweets = []
	for tweet in apitweets:
		if not tweet.text.startswith("RT "):	
			outtweets.append(tweet.text)

	
	f = codecs.open('%s_tweets.txt' % username, 'wb', 'utf-8')
	for tweet in outtweets:
		f.write(tweet)
		f.write('\n\t')
	f.close()


if __name__ == '__main__':
	get_tweets(sys.argv[1])
