# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autogrding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math , operator

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.Q_Values   =   util.Counter (  )


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return    self.Q_Values [  (  state   ,   action  )    ]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        ListOfValues    ,   LegalActions   =     [   ]    ,   self.getLegalActions(state)

        for action in  LegalActions   :
            ListOfValues.append (  self.getQValue(state,action)   )
        if  ( len ( ListOfValues ) >  0  )  :
            return max(ListOfValues)
        else  :
            return  0.0
            

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        
        ListOfActions   ,   LegalActions   =     [   ]    ,   self.getLegalActions(state)
        if len( LegalActions  ) == 0:
            return None
        else :
            for CurrentAction in  LegalActions   :
                ListOfActions.append (   [ CurrentAction  ,  self.getQValue(state,CurrentAction )   ]    )
            myRowIndexKey = operator.itemgetter(  1  )                                                                                                       
            ListOfActions.sort(   key= myRowIndexKey ,      reverse  =  True )                                                                      
            CurrentAction  =    ListOfActions [ 0 ] [ 0 ]   #  Max Action from Sorted List w/ operator.itemgetter module
            return CurrentAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """

        LegalActions = self.getLegalActions(state)
        if len( LegalActions  ) == 0:
            return None
        elif util.flipCoin( self.epsilon ):    #  epsilonGreedySelectAction
            return   random.choice( LegalActions )
        else:
            return   self.getPolicy(state)
            
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
                                   
        DiscountFactor = self.discount
        Q_Alpha  =   self.alpha
        Current_Q_Value      ,      Next_Q_Value     =       self.getQValue(state, action)    ,   self.getValue(nextState)    
        self.Q_Values[ (state, action) ]   =  (   (1 - Q_Alpha) *  Current_Q_Value   +   Q_Alpha * ( reward +  ( DiscountFactor *  Next_Q_Value   )    )   )


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

##########################################################################





##########################################################################

class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action









class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        ####################################################################
        
        features    ,     weights     =     self.featExtractor.getFeatures(state, action)     ,     self.getWeights()
        
        ApproximateQ_Value  = 0.0
        for   CoordinateDirectionTupleKey     in   features   :
##            print "   CoordinateDirectionTupleKey    is  :   ",   i   ,    "   features [ i ]    ",   features [ i ]    ,   "   weights  [  i  ]   ",    weights  [  i  ]
            ApproximateQ_Value =   ApproximateQ_Value  +    (    weights  [  CoordinateDirectionTupleKey  ]   *    features [ CoordinateDirectionTupleKey ]    )
        return ApproximateQ_Value
            
        ####################################################################


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        ####################################################################

        DiscountFactor   =   self.discount
        Q_Alpha  =   self.alpha
        
        Current_Q_Value      ,      Next_Q_Value     =       self.getQValue(state, action)    ,   self.getValue(nextState)    
        
        AdjustmentToUpdateBy   =    (   ( reward +   ( DiscountFactor   *    Next_Q_Value   )   )    -    Current_Q_Value    )
        
        features    ,     weights     =     self.featExtractor.getFeatures(state, action)     ,     self.getWeights()
        
        for   CoordinateDirectionTupleKey     in   features   :
            PriorWeightValue    =    weights  [ CoordinateDirectionTupleKey  ]
            UpdatedWeightForKey   =   (    PriorWeightValue    +    (    Q_Alpha      *   AdjustmentToUpdateBy     *     features[CoordinateDirectionTupleKey ]    )    )
            if  (   UpdatedWeightForKey  !=   PriorWeightValue   )   :  
                weights[CoordinateDirectionTupleKey]   =   UpdatedWeightForKey

        ####################################################################
        
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            weights  =   self.getWeights ( )
            print  "  weight  is    ",   weights
            # you might want to print your weights here for debugging

            pass










