# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

"""

python gridworld.py -a value -i 100 -g BridgeGrid --discount 0.9 --noise 0.2

python gridworld.py -a asynchvalue -i 1000 -k 10

python gridworld.py -a priosweepvalue -i 1000

python gridworld.py -a q -k 5 -m

python gridworld.py -a q -k 100 

python gridworld.py -a q -k 100 --noise 0.0 -e 0.1
python gridworld.py -a q -k 100 --noise 0.0 -e 0.9

python crawler.py

vvvvv       Question # 8      vvvvvv

python gridworld.py -a q -k 50 -n 0 -g BridgeGrid -e 1
python gridworld.py -a q -k 50 -n 0 -g BridgeGrid -e 0


python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10

python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid 


python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid

python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic


"""

# 2 :  Agent Will Never End Up in Unintended Successor State if Noise  =  0.00

"""

 Epsilon         Avg Returns  

       0                     -2.74
       0                     -2.74
       
       1                   -66.76
       2                   -64.07
       5                   -50.80

"""

##############################################################


def question2():
    answerDiscount = 0.9
    answerNoise = 0.00
    return answerDiscount, answerNoise

##############################################################

def question3a():
    answerDiscount = 0.5
    answerNoise = 0.0
    answerLivingReward = -5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

##############################################################

def question3b():
    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

##############################################################

def question3c():
    answerDiscount = 0.8
    answerNoise = 0.00
    answerLivingReward = 0.001
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

##############################################################

def question3d():
    answerDiscount =  0.8
    answerNoise = 0.1
    answerLivingReward = 1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

##############################################################

def question3e():
    answerDiscount = 0.7
    answerNoise = 0.1
    answerLivingReward = 8
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

##############################################################

def question8():
    answerEpsilon = None
    answerLearningRate = None
##    return answerEpsilon, answerLearningRate
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
