#!/usr/bin/python

import os
import sys

def print_tweets(filename):
	with open(filename) as f:
		content = f.read()
	tweets = content.split('\n\t')
	print len(tweets)
	for tweet in tweets:
		print tweet
		print "\n--------------------------\n"

if __name__ == "__main__":
	#pass filename
	print_tweets(sys.argv[1])

