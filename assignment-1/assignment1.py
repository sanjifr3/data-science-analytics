#!/usr/bin/python2.7
# MIE1615H Assignment #1 - Sanjif Rajaratnam
import numpy as np  # Libraries

# My Assumptions:
# I made some assumptions when compelting the assignment and this is my explanation for this:

# My understanding of the assignment is that the functions needed to be self-contained. My assumption was that if you
# called the functions in my code, you would get the intended results. I included my file readins within my function
# to ensure this works but this was at the compromise at runtime. Ideally, declaring the read in sets globally would
# reduce run-time by quite a bit.

# I also read that the objective was to analyze the sentiment of the tweets towards the various parties and one of the
# learning objectives was to analyze the data. However, the assignment only asked for the 5 functions. I did not know
# if this was required or not but I did some analysis just in case. However, printing to screen wasn't allowed so I
# included the results as strings with this parameter to allow printing:

# Printing control
Print = True  # If set to true the results will print to screen

def clean_data(tw):
    ''' (str) -> str\n
    Input: a string tw (a tweet line)\n
    Output: a string whose content is that of tw with punctuation removed\n

    Usage: clean_data('living the dream.tommulcair instagram.com/[/8up9qepkxw/')\n
        Returns: "living the dream tommulcair instagramcom8up9qepkxw"\n

    This function cleans the tweet that is passed to it. It loops through the characters in the tweet and keeps only
    alphanumeric (A-Z, a-z, 0-9) characters. It also replaces hashtags with spaces. It then returns the cleaned tweet
    '''

    cleaned_tw = ''  # String to return

    for i in range(len(tw)):
        if tw[i] >= 'A' and tw[i] <= 'z':  # Keep letters
            cleaned_tw += tw[i]
        elif tw[i] >= '0' and tw[i] <= '9':  # Keep numbers
            cleaned_tw += tw[i]
        elif tw[i] == " ":  # Keep spaces
            cleaned_tw += tw[i]
        elif tw[i] == "_":
            cleaned_tw += tw[i]
        elif tw[i] == '#':  # replace hashtags with spaces
            cleaned_tw += ' '

    return cleaned_tw


def remove_stop_words(tw):
    ''' (str) -> str\n
    Input: a string tw (a tweet line)\n
    Output: a string whose content is tw with stop words removed\n

    Usage: remove_stop_words('living the dream.tommulcair instagram.com/p/8up9qepkxw/')\n
        Returns: "living dream tommulcair instagramcomp8up9qepkxw"\n

    This function removes stop words from the tweet. It first reads in the stop_words from stop_word.txt into a set.
    This should ideally be done globally and passed to the function because it would be more computationally efficient
    since the stop_word set never changes from when it is read in and it doesn't need to be read in every time this
    function is called. Next it cleans the tweet using clean_data(tw) and splits it at spaces into a list. It then
    checks and keeps the words that aren't in the stop words set. It then recompiles the remaining words into a string
    which gets returned.
    '''

    # Read in stop words for 'remove_stop_words' function
    stop_words = set(open('stop_words.txt', 'r').read().split('\n'))
    # Ideally this should be done outside of the function to limit the amount of read ins

    # Clean and split the tweet into a list of words.
    # FOR loop to check if words in stop_words, and storing the words that aren't in a
    #  new list
    tw_words_list = [i for i in clean_data(tw).split() if i not in stop_words]

    tw_wo_block_words = ""  # String to return
    first_word = True  # Boolean to check if first word

    # Loop to build tw_wo_block_words string from tw_words_list
    for word in tw_words_list:
        if first_word == True:  # If first word of the list
            tw_wo_block_words = word  # start filling return string
            first_word = False  # Set boolean to false
        else:
            tw_wo_block_words += " " + word  # Concatenate return string

    return tw_wo_block_words


def tokenize_unigram(tw):
    ''' (str) -> [str]\n
    Input: a string tw (a tweet line)\n
    Output: a list whose content which is broken down into unigrams\n

    Usage: tokenize_unigram('living the dream.tommulcair instagram.com/p/8up9qepkxw/')\n
        Return: ['living','dream','tommulcair','instagramcomp8up9qepkxw']\n

    This function removes the stop words and cleans the tweet using remove_stop_words(tw) since remove_stop_words(tw)
    already cleans the tweet using clean_data. It then splits the tweet at spaces and returns a list of words.
    '''
    # Split the tweet into a list after cleaning and removing stop words
    return remove_stop_words(tw).split()


def bag_of_words(tw):
    ''' (str) -> {str,int}\n
    Input: a string tw (a tweet line)\n
    Output: a python dictionary\n

    Usage: bag_of_words('living the dream.tommulcair instagram.com/p/8up9qepkxw/')\n
        Return: {'living':1,'dream':1,'tommulcair':1, 'instagramcomp8up9qepkxw':1}\n

    This cleans the tweet, removes stop words, and splits the tw into unigrams using tokenize_unigram(tw). It then loops
    through the words in the list and either adds them to the dictionary with value 1, or if they already appeared
    earlier in the tw, it increments the value by 1. Then then returns the dictionary of words with counters of how
    much the apppeared.
    '''

    # Get list of unigrams
    unigrams = tokenize_unigram(tw)

    # Create word bag dictionary to return
    word_bag = {}

    # Loop through unigrams
    for word in unigrams:
        if word_bag.has_key(word.lower()):  # Check if dictionary already has the 'word' key
            word_bag[word.lower()] += 1  # Increment 'counter' value
        else:  # If not already in dictionary
            # Add to dictionary with 'unigram' as key, and 1 as 'counter' value
            word_bag[word.lower()] = 1

    return word_bag


def party(tw):
    ''' (str) -> str\n
    Input: a string tw (a tweet line)\n
    Output: a string representing the party\n

    Usage: party('living the dream.tommulcair instagram.com/p/8up9qepkxw/')\n
        Returns: 'NDP'\n

    The party hashtag set was found by using twitter.com and starting a search with
    the party name. From there, the most used party related hashtags were added to the set. Then the
    new hashtags were searched. This continued for a couple cycles till no more new hashtags could be
    found. The 'Unknown' set was classified by hashtags that didn't fall into any of the 3 party
    categories. This could be defined globally to decrease computation time since this list doesn't change during
    runtime.\n

    This function first defines hashtags into 3 sets: NDP, Liberal, and Conservative. It then cleans the tweet, removes
    the stop words, and splits the tweet into a word list using tokenize_unigram(tw). It then creates counters for the
    number of times party related words appear. It loops through the words in the tweet and increments each party's
    respective counter if a word in that respective party's set appears. If no party related words are found or if an
    even amount of party related words are found, the function returns 'Unknown' since the specific party cannot be
    determined. If one party has more related words than the other parties, it will be returned as the party the
    tweet is related to.
    '''

    # Define the set of party words hashtags
    NDP = set(['tommulcair', 'ndp', 'pj17', 'ndpldr', 'ndpol', 'ndleg', 'mulcair'])
    Liberal = set(['realchange', 'justin', 'trudeau', 'justintrudeau', 'liberal', 'lpc'])
    Conservative = set(['conservative', 'cpcldr', 'cpc', 'm103'])
    # These could be defined globally to decrease computation time

    # Split the tweet into unigrams
    unigrams = tokenize_unigram(tw)

    # Start counters for the number of words that fall into the party word sets
    ndp_ctr = 0
    liberal_ctr = 0
    conservative_ctr = 0

    # Loop through unigrams
    for word in unigrams:
        if word.lower() in Liberal:  # Check if word in liberal set
            liberal_ctr += 1  # Increment Liberal hashtag counter
        elif word.lower() in NDP:  # Check if word in NDP set
            ndp_ctr += 1  # Increment NPD hashtag counter
        elif word.lower() in Conservative:  # Check if word in Conservative set
            conservative_ctr += 1  # Increment Conservative hashtag counter

    party = ''  # String to return

    if (liberal_ctr == 0 and conservative_ctr == 0 and ndp_ctr == 0):  # If no party related words found
        party = 'Unknown'  # Return 'Unknown'
    elif (liberal_ctr > conservative_ctr and liberal_ctr > ndp_ctr):  # If mostly Liberal words found
        party = 'Liberal'  # Return 'Liberal'
    elif (ndp_ctr > liberal_ctr and ndp_ctr > conservative_ctr):  # If mosty NDP words found
        party = 'NDP'  # Return 'NDP'
    elif (conservative_ctr > liberal_ctr and conservative_ctr > ndp_ctr):  # If mostly conservative words found
        party = 'Conservative'  # Return 'Conservative'
    else:  # If an even number of words for multiple parties found
        party = 'Unknown'  # Not enough info to determine which party the tweet relates to

    return party

def tweet_score(tw):
    ''' (str) -> double\n
    Input: a string tw (a tweet line)\n
    Output: a floating point number between 0 and 1, or -1\n

    Usage: party('living the dream.tommulcair instagram.com/p/8up9qepkxw/')\n
        Returns: 0.491666666667

    First the corpus words are read in, cleaned, and put into a dictionary with 'key' = word and 'value' = score. This
    should ideally be done globally and passed to the function since it is more computationally efficient to do this
    once instead of every time the function is called. \n

    Then the modding parameters were set. These were found using the learn function and then manually set in this
    function. These numbers were chosen to maximize the performance of the algorithm on the classified tweets set. The
    split_num is the number at which the sentiment shifts from being positive and negative. The max_abs_score is the
    max score achieved by any tweet in the classified set.\n

    This algorithm works by first converting the tweet into a bag of words using the bag_of_words(tw) function. The
    function then loops through each word and checks to see if it is in the corpus dictionary. If the word exists,
    the score in incremented by corpus_word_score * number of times word appears in thw tweet. If no corpus words are
    found in the tweet, the algorithm returns -1 as it cannot classify the tweet. Otherwise the score is modded to [0,1]
    using the following formula: 0.5 + 0.5*((score-split_num)/max_abs_score). The max_abs_score was for the classified
    set, so just in case, the score is corrected 0 or 1 if either limits are exceeded in the unclassifed set. The
    modded score is then returned.
    '''
    # Read in corpus words for 'tweet_score' function
    # Read into list and split by new line
    corpus_words_list = open('corpus.txt', 'r').read().split('\n')
    corpus_words = {}  # Create dictionary to store corpus words and rating

    # Loop through lines in corpus_words_list
    for line in corpus_words_list:
        word = line.split('\t')  # split the line at the tabs between word and score
        corpus_words[word[0]] = int(word[1])  # Add to dictionary: 'key':'word', 'value':'score'

    # The above should be done globally as it is in only read in and never changed. It would be more computationally
    # efficient.

    # Set split_num and max_abs_score
    split_num = 1.5  # number at which to decide if tweet is positive or negative
    max_abs_score = 33.0  # manually found by looking at the results of the score prior to modding 0-1
    # This can be made into a global variable to do live learning but this was against the assignment instructions.
    # The code is now ran in Learning mode to find the desirable split_num and then manually set after it is found.

    word_bag = bag_of_words(tw)  # Create bag of words from tweet

    score = 0.0  # Initialize score to return
    no_corpus_word_found = True  # Boolean to check if no corpus words found in tweet

    # Initially the score is made by summing the corpus words score of the tweet:

    # Loop through word_bag
    for word, count in word_bag.items():
        # Check if word exists in corpus_words dictionary and has a score != 0
        if corpus_words.has_key(word.lower()) and corpus_words[word.lower()] != 0:
            # Increment score by the corpus score of the word * the times it appears in the tweet
            score += corpus_words[word.lower()] * count
            no_corpus_word_found = False  # set no_corpus_word_found to false

    # The score is then modded to [0,1] or -1:

    if no_corpus_word_found == True:  # If no corpus founds were found,
        score = -1  # The tweet cannot be classified
    else:
        score = score - split_num  # Shift to correct for the split_num

        # use max score to make score into an average
        score = 0.5 + 0.5 * (score / max_abs_score)  # mod to [0,1]

        # Corrections to score if max_abs_score guess is wrong
        if score < 0:  # score < 0
            score = 0  # Fix to 0 as it is highly negative

        elif score > 1:  # score > 1
            score = 1  # Fix to 1 as it is highly positive

    return score


def tweet_classifier(tw):
    ''' (str) -> int\n
    Input: a string tw (a tweet line)\n
    Output: a integer: -1, 0, or 1\n

    Usage: party('living the dream.tommulcair instagram.com/p/8up9qepkxw/')\n
        Returns: 0\n

    This function then classifies the tweet with a value of -1, 0, or 1. This function first gets a modded score
    for the tweet using tweet_score(tw). Then if the score returned is -1, this function returns -1 as the tweet cannot
    be classified. If the score returned is less than 0.5, this function returns 0 as the tweet is classified as
    negative. If the score returned is 0.5, this function returns -1 as the tweet cannot be classified as either
    positive or negative. If the score returned is greater than 0.5, this function returns 1 as the tweet is classified
    as positive.
    '''

    score = tweet_score(tw)  # Get continuous score

    # Discretize score
    if score == -1:  # if unknown,
        score = -1  # return -1
    elif score < 0.5:  # If negative,
        score = 0  # return 0
    elif score == 0.5:  # If neither positive or negative,
        score = -1  # return -1
    elif score > 0.5:  # If positive,
        score = 1  # return 1

    return score


def learn(tweets, truth):
    ''' [str],[int] -> double, double, double, double \n
    Input: a list of tweets and truth scores\n
    Output: correct_percentage, unclassifiable_percentage, best_split_num, max_abs_score\n

    This function is used to find the split_num and max_abs_score to be used in the tweet_score function. It also
    splits the cleaned tweet and creates a word dictionary internally to optimize run times. It seeks to
    maximize the accuracy of the algorithm on the known training set. It returns the percentage accuracy of the
    algorithm against the truth and the percentage of the tweets the algorithm was not able to classify. It also returns
    the ideal split_num and max_abs_score to be used with tweet_score(tw):
    '''

    num_tweets = len(tweets)  # Determine the number of tweets
    scores = np.zeros(len(tweets))  # Temp scores matrix

    # Initialize variables to store best values
    best_split_num = 0.0
    best_accuracy = 0.0
    best_unclassifiable_accuracy = 0.0
    max_abs_score = 0.0

    # test_max_abs_score to be used with this function
    test_max_abs_score = 30.0

    # Read in stop words
    stop_words = set(open('stop_words.txt', 'r').read().split('\n'))

    # Read in Corpus words and split by new line
    corpus_words_list = open('corpus.txt', 'r').read().split('\n')
    corpus_words = {}  # Create dictionary to store corpus words and rating

    # Loop through lines in corpus_words_list
    for line in corpus_words_list:
        word = line.split('\t')  # split the line at the tabs between word and score
        corpus_words[word[0]] = int(word[1])  # Add to dictionary: 'key':'word', 'value':'score'

    # Loop through split_num values to test - max and min picked randomly
    for split_test_num in np.arange(-2, 2, 0.5):
        pos_ctr = 0  # Start positive counter
        unknown_ctr = 0  # Start undeterminable counter
        split_num = split_test_num  # Set split_num to test_num

        # Loop through the classified tweets
        for i in range(num_tweets):
            # Create word bag from tweet
            tw_words_list = [j for j in clean_data(tweets[i]).split() if j not in stop_words]
            word_bag = {}  # Create word bag dictionary to return
            for word in tw_words_list:
                if word_bag.has_key(word.lower()):  # Check if dictionary already has the 'word' key
                    word_bag[word.lower()] += 1  # Increment 'counter' value
                else:
                    # Add dictionary to word as key ans 1 as counter value
                    word_bag[word.lower()] = 1

            no_corpus_word_found = True  # Boolean to check if no corpus words found in tweet

            # Classify tweets
            # Loop through word_bag
            for unigram, count in word_bag.items():
                # Check if word exists in corpus_words dictionary and has a score != 0
                if corpus_words.has_key(unigram.lower()) and corpus_words[unigram.lower()] != 0:
                    # Increment score by the corpus score of the word * the times it appears in the tweet
                    scores[i] += corpus_words[unigram.lower()] * count
                    no_corpus_word_found = False  # set no_corpus_word_found to false

            # Find max_abs_score
            if abs(scores[i]) >= max_abs_score:  # If new score is greater,
                max_abs_score = abs(scores[i])  # change max_abs_score to new score

            # The score is then modded to [0,1] or -1:
            if no_corpus_word_found == True:  # If no corpus founds were found,
                scores[i] = -1  # The tweet cannot be classified, so
                unknown_ctr += 1  # increment unknown counter
            else:
                scores[i] = scores[i] - split_num  # Shift to correct for the split_num

                # use max score to make score into an average
                scores[i] = 0.5 + 0.5 * (scores[i] / test_max_abs_score)  # mod to [0,1]

                if scores[i] < 0.5:  # negative case
                    scores[i] = 0
                elif scores[i] == 0.5:  # cannot classify case
                    scores[i] = -1  # unknown,
                    unknown_ctr += 1  # increment unknown counter
                else:  # positive case
                    scores[i] = 1

            if scores[i] == truths[i]:  # If correct,
                pos_ctr += 1  # increment positive counter

        accuracy = float(pos_ctr) / num_tweets  # Determine accuracy

        # If new accuracy is better, update the best variables
        if accuracy > best_accuracy:
            best_split_num = split_test_num
            best_accuracy = accuracy
            best_unclassifiable_accuracy = float(unknown_ctr) / num_tweets

    return best_accuracy, best_unclassifiable_accuracy, best_split_num, max_abs_score


def getBestParty(scores):
    ''' {str,double}: str, double\n
    Input: a dictionary with parties as keys and scores as values\n
    Output: the party with the best score, and the score\n

    This function is used to find the party with the best score and the best score
    given a dictionary with scores.
    '''
    # Initialize variables
    best_party = 'Liberal'
    max_score = scores['Liberal']

    # Loop through items in dictionary and determine which is the best one
    for party, score in scores.items():
        if score > max_score:  # if current items score is better, update best score
            max_score = score
            best_party = party

    return best_party, max_score


def getWorstParty(scores):
    ''' {str,double}: str, double\n
    Input: a dictionary with parties as keys and scores as values\n
    Output: the party with the worst score, and value of the score\n

    This function is used to find the party with the worst score and value of the worst
    score given a dictionary with scores.
    '''
    # Initialize variables
    worst_party = 'Liberal'
    min_score = scores['Liberal']

    # Loop through items in dictionary and determine which is the best one
    for party, score in scores.items():
        if score < min_score:  # if current items score is worse, update worst score
            min_score = score
            worst_party = party

    return worst_party, min_score

## Main ##
if __name__ == "__main__":
    # Classified Set #

    # Read in classified tweets and split by new line
    classified_tweets_list = open('classified_tweets.txt', 'r').read().split('\n')
    del classified_tweets_list[-1]  # Delete last line as it doesn't contain any data

    # Initialize variables to store values
    classified_tweets = [0] * len(classified_tweets_list)  # List to store classified tweets
    truths = np.zeros(len(classified_tweets_list))  # Vector to store truth scores
    i = 0  # index counter

    # Split the classified tweets list into two lists that match index-wise for 'tweet'
    # and 'score'

    # For each line
    for tweet in classified_tweets_list:
        classified_tweets[i] = tweet[2:]  # Populate tweets list
        truths[i] = int(tweet[0]) / 4  # Populate truths matrix - since 1 appears as 4 in file
        i += 1

    # Learn from the training set
    classified_correct_percentage, classified_unknown_percentage, best_split_num, max_abs_score = learn(classified_tweets,
                                                                                                        truths)
    # Strings with results of classified set, use Print = True at top to print
    parameters_for_tweet_score = "The algorithm determined the ideal split_num and max_abs_score to be {} and {}, respectively.".format(
        best_split_num, max_abs_score)
    if Print == True: print parameters_for_tweet_score
    classified_set_results = "The algorithm was able to correctly identify {correct}% of the classified set, and unable to identify {unknown}% of the classified set".format(
        correct=round(classified_correct_percentage * 100, 2), unknown=round(classified_unknown_percentage * 100, 2))
    if Print == True: print classified_set_results  # Print the above statement if Print is set to true
    correct_of_known_results = "The algorithm had an accuracy of {correct}% in the part of the set it was able to identify\n".format(
        correct=round(classified_correct_percentage * (i - 1) / ((1 - classified_unknown_percentage) * (i - 1)) * 100, 2))
    if Print == True: print correct_of_known_results  # Print the above statemnt if Print is set to true

    # Unclassified Set #

    # Read in the unclassified tweets and split by new line
    unclassified_tweets_list = open('unclassified_tweets.txt', 'r').read().split('\n')
    del unclassified_tweets_list[-1]  # Delete last line as it doesn't contain any data

    # Initialize variables to store values
    unclassified_scores = np.zeros(len(unclassified_tweets_list))  # Vector to store unclassifed scores
    unclassified_parties = [0] * len(unclassified_tweets_list)  # List to store parties
    unclassified_unknown_percentage = 0.0  # variable for percentage of set that is unclassifiable
    keys = ['NDP', 'Liberal', 'Conservative']  # keys for score dictionaries below
    buzz = dict.fromkeys(keys, 0)  # Dictionary to store count of tweets of parties
    positive_buzz = dict.fromkeys(keys, 0)  # Dictionary to store count of positive tweets of parties
    negative_buzz = dict.fromkeys(keys, 0)  # Dictionary to store count of negative tweets of parties
    pos_neg_ratio = dict.fromkeys(keys, 0)  # Dictionary to store the postiive to negative ratio of tweets of parties
    percentage_tweets = dict.fromkeys(keys, 0)  # Dictionary to store the percentage of a party tweets over all tweets

    # Loop through the index of all tweets in the unclassified tweets list
    for i in range(len(unclassified_tweets_list)):
        unclassified_scores[i] = tweet_classifier(unclassified_tweets_list[i])  # Classify the tweet sentiment
        unclassified_parties[i] = party(unclassified_tweets_list[i])  # Get the tweet party

        if unclassified_scores[i] == -1:  # If the tweet cannot be classifed,
            unclassified_unknown_percentage += 1.0  # Increment the unknown counter

        if unclassified_parties[i] == 'NDP':  # If the tweet is classified as NDP
            buzz['NDP'] += 1  # Increase the buzz counter
            if unclassified_scores[i] == 1:  # if the sentiment value is positive,
                positive_buzz['NDP'] += 1  # Increase the positive buzz counter
            elif unclassified_scores[i] == 0:  # if the sentiment value is negative,
                negative_buzz['NDP'] += 1  # increase the negative buzz counter

        elif unclassified_parties[i] == 'Liberal':  # if the tweet is classified as Liberal
            buzz['Liberal'] += 1  # increase the buzz counter
            if unclassified_scores[i] == 1:  # if the sentiment value is positive
                positive_buzz['Liberal'] += 1  # increase the positive buzz counter
            elif unclassified_scores[i] == 0:  # if the sentiment value is negative
                negative_buzz['Liberal'] += 1  # increase the negative buzz counter

        elif unclassified_parties[i] == 'Conservative':  # if the tweet is classified as Conservative
            buzz['Conservative'] += 1  # Increase the buzz counter
            if unclassified_scores[i] == 1:  # If the sentiment value is positive,
                positive_buzz['Conservative'] += 1  # Increase the positive buzz counter
            elif unclassified_scores[i] == 0:  # if the sentiment value is negative,
                negative_buzz['Conservative'] += 1  # Increase the negative buzz ccounter

    unclassified_unknown_percentage /= len(unclassified_tweets_list)  # Determine the average over all tweets

    # Determine the percentage of the tweets that were positive
    unclassified_positive_percentage = (sum(1 for value in unclassified_scores if value == 1)) / (
        (1 - unclassified_unknown_percentage) * len(unclassified_scores))

    # String with results of how much of the unclassified set the algorithm was able to classify, use Print = True at top to print
    algorithm_accuracy = "The algorithm was able to apply sentiment analysis to {}% of the unclassified set".format(
        round((1 - unclassified_unknown_percentage) * 100, 2))
    if Print == True: print algorithm_accuracy  # Print the above statement if Print is set to true

    # String to analyze the amount of positive tweets among the classified tweets, use Print = True at top to print
    sentiment_analysis = "About {}% of the classifiable tweets were positive, the rest were negative\n".format(
        round(unclassified_positive_percentage, 3))
    if Print == True: print sentiment_analysis  # Print the above statement if Print is set to true

    # Loop through keys
    for party in keys:
        pos_neg_ratio[party] = float(positive_buzz[party]) / negative_buzz[
            party]  # determine the positive to negative ratio per party
        percentage_tweets[party] = float(buzz[party]) / (
            sum(buzz.values())) * 100  # determine the percentage of party buzz of all party buzz

    name, hits = getBestParty(buzz)  # Determine the party with the most tweets
    most_tweets = "The {} party had the most buzz with {} tweets".format(name, hits)
    if Print == True: print most_tweets  # Print the above statement if Print is set to true

    name, hits = getWorstParty(buzz)  # Determine the party with the least tweets
    least_tweets = "The {} party had the least buzz with {} tweets".format(name, hits)
    if Print == True: print least_tweets  # Print the above statement if Print is set to true

    name, hits = getBestParty(pos_neg_ratio)  # Determine the highest positive to negative tweet ratio
    highest_pos_neg_ratio = "The {} party was the most liked with a positive-negative tweet ratio of {}".format(name,
                                                                                                                round(hits,
                                                                                                                      2))
    if Print == True: print highest_pos_neg_ratio  # Print the above statement if Print is set to true

    name, hits = getWorstParty(pos_neg_ratio)  # Determine the negative positive to negative tweet ratio
    lowest_pos_neg_ratio = "The {} party was the least liked with a positive-negative tweet ratio of {}".format(name,
                                                                                                                round(hits,
                                                                                                                      2))
    if Print == True: print lowest_pos_neg_ratio  # Print the above statement if Print is set to true

    name, hits = getBestParty(positive_buzz)  # Determine the negative positive to negative tweet ratio
    most_positive_tweets = "The {} party had the most positive buzz with {} tweets".format(name, hits)
    if Print == True: print most_positive_tweets  # Print the above statement if Print is set to true

    name, hits = getBestParty(negative_buzz)  # Determine the negative positive to negative tweet ratio
    least_positive_tweets = "The {} party had the most negative buzz with {} tweets".format(name, hits)
    if Print == True: print least_positive_tweets  # Print the above statement if Print is set to true

    # Determine the percentage of the party related tweets that each party had
    for name, value in percentage_tweets.items():
        party_related_tweets = "The {} party had {}% of the party related tweets".format(name, round(value, 2))
        if Print == True: print party_related_tweets  # Print the above statement if Print is set to true
