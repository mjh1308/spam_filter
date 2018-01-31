# Building a Naive-Bayes Classifier for Spam Filtering
# Matthew Jeremy

import collections
import email
import math
import os

"""
This function is responsible for reading the input data/e-mail, parsing through it to extract words from its content, 
and then returning all of these words in a list. 
"""
def load_tokens(email_path):
    f = open(email_path)
    msg = email.message_from_file(f)
    tokens = [word for line in email.iterators.body_line_iterator(msg) for word in line.split()]
    f.close()
    return tokens

"""
This function is responsible for creating a dictionary that maps the words (also referred as tokens) from the data/e-mail 
to their respective Laplace-smoothed log probabilities. Tokens that couldn't be identified as words (i.e. emojis or even
punctuation) will be represented as "<UNK>". 
"""
def log_probs(email_paths, smoothing):
    prob = {}  # list of Laplace smoothed log probabilities for each word
    occurrences = collections.Counter()  # records frequency of a particular word encountered in the emails
    v = set()  # set of vocabulary of words in the emails
    word_total = 0  # total word count of all words in the emails datasets
    for path in email_paths:
        tokens = load_tokens(path)
        word_total += len(tokens)
        occurrences.update(tokens)
        v.update(tokens)
    for w in v:
        prob[w] = math.log((occurrences[w] + smoothing) / (word_total + smoothing * (len(v) + 1)))
    prob["<UNK>"] = math.log(smoothing / (word_total + smoothing * (len(v) + 1)))
    return prob


class SpamFilter(object):
    """
    Initialization method responsible for calculating probabilities of spam and probabilities of ham given the
    input datasets/emails.
    """
    def __init__(self, spam_dir, ham_dir, smoothing):
        spam_paths = [spam_dir + "/" + spam_file for spam_file in os.listdir(spam_dir)]  # spam dataset directory
        ham_paths = [ham_dir + "/" + ham_file for ham_file in os.listdir(ham_dir)]  # ham dataset directory
        self.spam = log_probs(spam_paths, smoothing)  # log probability dictionary of words in spam directory
        self.ham = log_probs(ham_paths, smoothing)  # log probability dictionary of words in ham directory
        self.spam_prob = math.log(len(spam_paths) * 1.0 / (len(spam_paths) + len(ham_paths)))
        self.ham_prob = math.log(len(ham_paths) * 1.0 / (len(spam_paths) + len(ham_paths)))

    """
    This function is responsible for predicting whether a particular e-mail is spam or otherwise (i.e. ham). 
    Words that were not previously identified in the training dataset will be classified as <"UNK">.
    """
    def is_spam(self, email_path):
        spam_p = self.spam_prob  # probability of being spam
        ham_p = self.ham_prob  # probability of NOT being spam
        occurrences = collections.Counter(load_tokens(email_path))
        for w, c in occurrences.most_common():
            if w in self.spam:
                spam_p += c * self.spam[w]
            else:  # not encountered in training process
                spam_p += c * self.spam["<UNK>"]
            if w in self.ham:
                ham_p += c * self.ham[w]
            else:
                ham_p += c * self.ham["<UNK>"]
        if spam_p > ham_p:
            return True  # indicates the email at the given path to be spam
        else:
            return False  # indicates the email at the given path to be NOT spam

    """
    This function is responsible for identifying and returning n words that have the highest indication of spam 
    sorted in descending order with respect to their indication values. The indication value calculation is shown below:
        indication value = log(P(w | spam) / P(w)) = log(P(w | spam)) - log(P(w))
        P(w) = P(w | spam) + P(w | ham)
    """
    def most_indicative_spam(self, n):
        most_spam = {}
        for word, val in self.spam.items():
            indication_val = 0
            if word in self.ham:
                indication_val += math.exp(self.spam_prob) * math.exp(self.spam[word]) + \
                                  math.exp(self.ham_prob) * math.exp(self.ham[word])
            else:
                continue  # word NOT in ham email, thus excluded
            most_spam[word] = val - math.log(indication_val)  # 'val' already in log space
        return sorted(most_spam, key=most_spam.get, reverse=True)[0:n]

    """
    This function is responsible for identifying and returning n words that have the highest indication of ham (i.e. not spam) 
    sorted in descending order with respect to their indication values. The indication value calculation is shown below:
    indication value = log(P(w | ham) / P(w)) = log(P(w | ham)) - log(P(w))
    P(w) = P(w | ham) + P(w | spam)
    """
    def most_indicative_ham(self, n):
        most_ham = {}
        for word, val in self.ham.items():
            indication_val = 0
            if word in self.spam:
                indication_val += math.exp(self.ham_prob) * math.exp(self.ham[word]) + \
                                  math.exp(self.spam_prob) * math.exp(self.spam[word])
            else:
                continue  # word NOT in ham email, thus excluded
            most_ham[word] = val - math.log(indication_val)
        return sorted(most_ham, key=most_ham.get, reverse=True)[0:n]