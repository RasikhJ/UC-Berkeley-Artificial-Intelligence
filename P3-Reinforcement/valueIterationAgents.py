# valueIterationAgents.py
# -----------------------
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

##########################################################################

import mdp, util  ,   operator
from learningAgents import ValueEstimationAgent
import collections              # A ValueIterationAgent takes a Markov decision process (see mdp.py) on initialization and runs value iteration for a given number of iterations using the supplied discount factor.

##########################################################################

class ValueIterationAgent(ValueEstimationAgent):    #  * Please read learningAgents.py before reading this.*

    def __init__(self, mdp, discount = 0.9, iterations = 100):      #  Your value iteration agent should take an mdp on construction, run the indicated number of iterations and then act according to the resulting policy.
        
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()                # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        ListOfStates    ,    ListOf_Iterations      =    self.mdp.getStates()   ,       self.iterations
        for Current_Iteration  in range(  ListOf_Iterations  )   :                  ##            print  "Current_Iteration  :   ",   Current_Iteration
            values  =   util.Counter ( )
            for  EachState  in ListOfStates    :
                ListOfValues  =    [  ]
                CurrentStateTerminalBOOL  =   self.mdp.isTerminal( EachState  ) 
                if  (  CurrentStateTerminalBOOL  !=   True   )    :
                    ListOfActions_AtState    =     self.mdp.getPossibleActions(   EachState   )

                    for action in ListOfActions_AtState  :
                        ListOfValues.append (  self.getQValue  ( EachState  ,  action   ))
                    values  [  EachState   ] =   max  (  ListOfValues  )

            self.values = values

    ##########################################################################

    def getValue(self, state):                                                          #          Return the value of the state (computed in __init__).
        return self.values[state]

    ##########################################################################

    def computeQValueFromValues(self, state, action) :      # Compute the Q-value of action in state from the value function stored in self.values.

        Q_Value   ,  DiscountFactor     =      0    ,    self.discount
        ListOfTransitionStates_Probabilities   =    self.mdp.getTransitionStatesAndProbs(state, action)

        for    SuccessorState   ,   Probability    in   ListOfTransitionStates_Probabilities    :
            RegularValue     ,    RewardAmount     =    self.getValue(   SuccessorState   )    ,     self.mdp.getReward  (  state  , action  ,   SuccessorState   )
            FormulaAddedValue  =   (   Probability *  (   RewardAmount  +   (   DiscountFactor   *    RegularValue  )   )   )
            Q_Value   =     (   Q_Value   +    FormulaAddedValue   )

        return  Q_Value 

    ##########################################################################

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        ListOfValues   =    [  ]
        LegalActions  =    self.mdp.getPossibleActions(   state  )
        
        if  (  len  (  LegalActions  )   ==   0   )   :
            return  None
        else : 
            values = util.Counter()
            for action in   LegalActions   :
                ListOfValues.append  (   [   action  ,    (   self.getQValue  (  state  ,   action )   )    ]    )
                values  [ action  ]  =  self.getQValue  (  state  ,   action )
            myRowIndexKey = operator.itemgetter(  1  )                                                                                                       
            ListOfValues.sort(   key= myRowIndexKey ,      reverse  =  True )
            return ListOfValues [ 0 ] [ 0 ]

    ##########################################################################

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

############################################################################

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """

        ValueIterationAgent.__init__(self, mdp, discount, iterations)               ##        self.mdp = mdp                ##        self.discount = discount              ##        self.iterations = iterations
        self.values = util.Counter()                # A Counter is a dict with default 0
        self.runValueIteration()

    ##########################################################################

    def runValueIteration( self ):

        ListOfStates    ,    ListOf_Iterations      =    self.mdp.getStates()   ,       self.iterations

        for EachState in ListOfStates    :
            self.values [  EachState  ]  =   0

        for Current_Iteration  in range(  ListOf_Iterations  )   :
            EachState = ListOfStates  [   Current_Iteration   % len(ListOfStates)]
            ListOfValues  =    [  ]
            CurrentStateTerminalBOOL  =   self.mdp.isTerminal( EachState  ) 
            if  (  CurrentStateTerminalBOOL  !=   True   )    :
                for CurrentAction in self.mdp.getPossibleActions(EachState)  : 
                    ListOfValues.append  (    self.getQValue  ( EachState  ,  CurrentAction   )  )
                self.values[ EachState ]   =   (  max (  ListOfValues )   )

############################################################################

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration

        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        self.values = util.Counter()                # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):

        theta   =   self.theta          #  theta, which is passed in as a parameter, will represent our tolerance for error when deciding whether to update the value of a state.

        ListOfStates    ,    ListOf_Iterations      =    self.mdp.getStates()   ,       self.iterations
        for EachState in ListOfStates    :
            self.values [  EachState  ]  =   0

        predecessors  =  { }                                  #    predecessors of a state =  as all states that have a nonzero probability of reaching s by taking some action.
        PriorityQueue = util.PriorityQueue ( )         #    Initialize an empty priority queue.                   #   Please use util.PriorityQueue in your implementation. The update method in this class will likely be useful; look at its documentation.

        for EachState in ListOfStates:
            CurrentStateTerminalBOOL  =   self.mdp.isTerminal( EachState  ) 
            if  (  CurrentStateTerminalBOOL  !=   True   )    :
                LegalActions  =    self.mdp.getPossibleActions(   EachState )
                for action in LegalActions  :
                    for SuccessorState, Probability   in self.mdp.getTransitionStatesAndProbs(EachState, action):
                        if  (  SuccessorState  not  in   predecessors   ) :
                            predecessors [ SuccessorState ]   =   set  (  [  EachState  ]   )
                        else  :
                            predecessors [ SuccessorState ].add ( EachState )

                ListOfQ_Values   =   [  ]
                for a in  LegalActions :
                    ListOfQ_Values.append ( self.getQValue ( EachState   ,  a   )  )
                
                diff  =  abs(self.values[  EachState  ] -  (  max (  ListOfQ_Values ))   )           #     Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
                PriorityQueue.push (  EachState  ,   -diff  )

        for i in range  (ListOf_Iterations )   :
            if PriorityQueue.isEmpty():
                break
            else :
                EachState = PriorityQueue.pop()

                newListOfQ_Values   =   [  ]
                newLegalActions  =  self.mdp.getPossibleActions( EachState )
                for a in  newLegalActions :
                    newListOfQ_Values.append ( self.getQValue ( EachState   ,  a   )  )
                    
                self.values[  EachState  ] = max ( newListOfQ_Values )

                for CurrentPredecessor  in  predecessors [ EachState ] :
                    FinalListOfQ_Values   =   [  ]
                    FinalLegalActions  =  self.mdp.getPossibleActions( CurrentPredecessor )
                    for a in  FinalLegalActions :
                        FinalListOfQ_Values.append ( self.getQValue (  CurrentPredecessor   ,  a   )  )

                    diff =   (  abs(self.values[ CurrentPredecessor  ]   -   ( max ( FinalListOfQ_Values )  )  )  )
                    if diff   >   theta:
                        PriorityQueue.update (CurrentPredecessor, -diff)

############################################################################


