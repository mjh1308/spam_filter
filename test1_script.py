# Test input file 1

from nb_classifier import SpamFilter

# Trains classifier using training dataset in "datasets" directory with a smoothing factor of 1e-5
test1 = SpamFilter("datasets/train/spam", "datasets/train/ham", 1e-5)

# Classify/predict whether a particular email or text file is spam or otherwise using the model above
print "EXPECTED RESULT: True (SPAM)"
print "ACTUAL RESULT: " + str(test1.is_spam("datasets/train/spam/spam1")) # from training dataset - EXPECTED CLASS: SPAM
print "EXPECTED RESULT: True (SPAM)"
print "ACTUAL RESULT: " + str(test1.is_spam("datasets/test/spam/dev201")) # from test dataset - EXPECTED CLASS: SPAM

print "EXPECTED RESULT: False (HAM)"
print "ACTUAL RESULT: " + str(test1.is_spam("datasets/train/ham/ham1")) # from training dataset - EXPECTED CLASS: HAM
print "EXPECTED RESULT: False (HAM)"
print "ACTUAL RESULT: " + str(test1.is_spam("datasets/test/ham/dev1")) # from test dataset - EXPECTED CLASS: HAM


