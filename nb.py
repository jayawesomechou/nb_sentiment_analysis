from nltk import FreqDist
import glob
from nltk.corpus import stopwords
import math
import re
import nltk
from textblob import TextBlob

###########################
## READ IN TRAINING DATA ##
###########################

stops = stopwords.words('english')
stops.extend([",", ".", "!", "?", "'", '"', "I", "i", "n't", "'ve", "'d", "'s"])

## Read in all positive reviews
## We create a set of unique words for each review. No duplicates.
## We then add that set of words as a list to the master list of positive words.
poswords = []
allpos = glob.glob("pos/*")
for filename in allpos:
    f = open(filename)
    toextend = []
    for line in f:
        words = line.rstrip().split()
        toextend.extend(list(set([w for w in words if not w in stops])))
    f.close()
    poswords.extend(list(set(toextend)))

## Read in all negative reviews
## We create a set of unique words for each review. No duplicates.
## We then add that set of words as a list to the master list of positive words.
negwords = []
allneg = glob.glob("neg/*")
for filename in allneg:
    f = open(filename)
    toextend = []
    for line in f:
        words = line.rstrip().split()
        toextend.extend(list(set([w for w in words if not w in stops])))
    f.close()
    negwords.extend(list(set(toextend)))


###########################################################
## GET NAIVE BAYES PROBABILITIES FOR POS AND NEG CLASSES ##
###########################################################


   #########################################################
   ################# YOUR CODE BEGINS HERE #################
   #########################################################


## GOAL: Populate these two dicts, where each
##      key = word from the pos or neg word list
##      value = NB probability for that word in that class
## You will refer to these dicts in your function
## definition, below, for naive_bayes()
poswordprobs = {}
negwordprobs = {}

## You might need to know the number of tokens and types
## in each of the two classes.
postok = len(poswords)
negtok = len(negwords)
postype = len(set(poswords))
negtype = len(set(negwords))

## And you might need a list of all the words in both sets.
allwords = list(set(negwords)) + list(set(poswords))


## Start by creating FreqDists for poswords and for negwords below
fdistpos = nltk.FreqDist(list(set(poswords)))
fdistneg = nltk.FreqDist(list(set(negwords)))



## Loop through your poswords FreqDist, and calculate the
## probability of each word in the positive class, like this:
## P(word|pos) = count(word) / postok
## Store the results in poswordprobs
## USE LOGS!!!



for item in fdistpos:
#    print (item)
  #  print (fdistpos[item])
    k=item
    v=math.log(float(fdistpos[item])/postok)
    poswordprobs[k]=v



## Now, loop through your negwords FreqDist, and calculate the
## probability of each word in the negative class, like this:
## P(word|neg) = count(neg) / postok
## Store the results in negwordprobs
## USE LOGS!!!

for item in fdistneg:
    k=item
    v=math.log(float(fdistneg[item])/negtok)
    negwordprobs[k]=v
    

    #########################################################
    ################# YOUR CODE ENDS HERE ###################
    #########################################################



######################################
### FUNCTIONS TO PREDICT SENTIMENT ###
######################################


## FUNCTION USING USER DEFINED WORDS TO PREDICT SENTIMENT
def user_defined_keywords(reviewwords):

    ## ENTER YOUR KEYWORDS HERE FOR PART A
    positive_keywords = ['success','amazing','excellent','funny','nice','pleasant','best','interesting','good','great']
    negative_keywords = ['boring','tired','bland','bloody','silly','violent','distasteful','flawed','mistake','confused']

    sentiment = 0
    for w in reviewwords:
        if w in positive_keywords:
            sentiment += 1
        if w in negative_keywords:
            sentiment -=1

    if sentiment > 0:
        return "pos"

    return "neg"

## FUNCTION USING NAIVE BAYES PROBS TO PREDICT SENTIMENT
def naive_bayes(reviewwords):

    defaultprob = math.log(0.0000000000001)
    
    ### POSITIVE SCORE
    posscore = poswordprobs.get(reviewwords[0], defaultprob)
    for i in range(1, len(reviewwords)):
        posscore += poswordprobs.get(reviewwords[i], defaultprob)

    ### CALCULATE NEGATIVE SCORE
    negscore = negwordprobs.get(reviewwords[0], defaultprob)
    for i in range(1, len(reviewwords)):
        negscore += negwordprobs.get(reviewwords[i], defaultprob)

    if (posscore - negscore) >  0:
        return "pos"

    return "neg"

def textblob(reviewwords):
    testimonial = TextBlob(reviewwords)
    if testimonial.sentiment.polarity>0:
        return "pos"
    else:
        return "neg"
    


#################################################
### PREDICT THE SENTIMENT OF THE TEST REVIEWS ###
#################################################

keywordscorrect = 0
nbcorrect = 0
tblobcorrect = 0

testdata = glob.glob("test/*")
for filename in testdata:
    rw = []
    wholereview = ""
    f = open(filename)
    for line in f:
        wholereview += line.rstrip()
        words = line.rstrip().split()
        rw.extend(list(set([w for w in words if not w in stops])))
    f.close()
    reviewwords = list(set(rw))

    filepolarity = re.sub(r"^.*?(pos|neg)-.*?$", r"\1", filename)
    
    with open(filename, 'r') as myfile:
        ftext=myfile.read().replace('\n', '')
        

    if filepolarity == user_defined_keywords(reviewwords):
        keywordscorrect += 1

    if filepolarity == naive_bayes(reviewwords):
        nbcorrect += 1
    
    if filepolarity == textblob(ftext):
        tblobcorrect += 1
        
print("User keyword accuracy: ", (keywordscorrect/200))
print("Naive Bayes accuracy: ", (nbcorrect/200))
print("Textblob Sentiment accuracy: ", (tblobcorrect/200))
