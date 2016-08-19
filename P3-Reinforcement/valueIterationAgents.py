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

##PseudoCode
##def Value_Iteration ( MDP   ,   TransitionFunction   ,    RewardFunction    ,   DiscountFactor  )   :
##    states  ,   actions   =   MDP.states   ,   MDP.rewards
####    TransitionStates   =
####    Probabilities          =


"""

python autograder.py
python autograder.py -q q1

python gridworld.py -m 
python gridworld.py -h
python gridworld.py -g MazeGrid

python autograder.py -t test_cases/q2/1-bridge-gird

:"""


"""

python gridworld.py -a value -i 100 -g BridgeGrid --discount 0.9 --noise 0.2

python gridworld.py -a asynchvalue -i 1000 -k 10

python gridworld.py -a priosweepvalue -i 1000

python gridworld.py -a q -k 5 -m

python gridworld.py -a q -k 100 

python gridworld.py -a q -k 100 --noise 0.0 -e 0.1
python gridworld.py -a q -k 100 --noise 0.0 -e 0.9

python crawler.py

python gridworld.py -a q -k 50 -n 0 -g BridgeGrid -e 1


python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10

python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid 


python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid

python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic


"""

##########################################################################

import mdp, util  ,   operator
from learningAgents import ValueEstimationAgent
import collections              # A ValueIterationAgent takes a Markov decision process (see mdp.py) on initialization and runs value iteration for a given number of iterations using the supplied discount factor.

##########################################################################

class ValueIterationAgent(ValueEstimationAgent):    #  * Please read learningAgents.py before reading this.*

   ##########################################################################
     
    def __init__(self, mdp, discount = 0.9, iterations = 100):      #  Your value iteration agent should take an mdp on construction, run the indicated number of iterations and then act according to the resulting policy.
        
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()                # A Counter is a dict with default 0
        self.runValueIteration()
        
    ##########################################################################

    def runValueIteration(self):
        # Write value iteration code here

##        print "\n\n\n\n   :             self is...  ",    self 
##        print "\n\n\n\n   :             mdp is...  ",    self.mdp  
##        print  "\n\n\n\n   :            States are...  ",    self.mdp.getStates()
##        print  "\n\n\n\n"
        
##        for state in ListOfStates  :
##            ListOfActions_AtState    =     self.mdp.getPossibleActions(state)
##            for action in ListOfActions_AtState  :
##                print  "   :    ",     state    ,   "           ",   action,   "      "  ,     self.mdp.getTransitionStatesAndProbs(state, action)
####                print  "\n   :    ",     state    ,   "           ",       self.mdp.getReward(state, action, nextState)
##                print  "   :    ",     state    ,   "           ",   action,   "      "  ,        self.mdp.isTerminal(state)
        
        "*** YOUR CODE HERE ***"

        ListOfStates    ,    ListOf_Iterations      =    self.mdp.getStates()   ,       self.iterations
        
        ##########################################################################

        for Current_Iteration  in range(  ListOf_Iterations  )   :                  ##            print  "Current_Iteration  :   ",   Current_Iteration
            values  =   util.Counter ( )
            for  EachState  in ListOfStates    :
                ListOfValues  =    [  ]
                CurrentStateTerminalBOOL  =   self.mdp.isTerminal( EachState  ) 
                if  (  CurrentStateTerminalBOOL  !=   True   )    :
                    ListOfActions_AtState    =     self.mdp.getPossibleActions(   EachState   )

                    for action in ListOfActions_AtState  :
                        ListOfValues.append (  self.getQValue  ( EachState  ,  action   ))
##                    print  "state   " ,   EachState  , "   values  [ EachState   ]      ",    values  [  EachState   ] 
                    values  [  EachState   ] =   max  (  ListOfValues  )

            self.values = values

          ##########################################################################


          ##########################################################################


          ##########################################################################


          ##########################################################################


    ##########################################################################


    ##########################################################################

    def getValue(self, state):   #          Return the value of the state (computed in __init__).
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


        ##########################################################################


        ##########################################################################


        ##########################################################################


    ##########################################################################

##        util.raiseNotDefined()

    ##########################################################################

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        
        ##########################################################################

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


        ##########################################################################


        ##########################################################################


    ##########################################################################

##        util.raiseNotDefined()

    ##########################################################################

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

    ##########################################################################





############################################################################










############################################################################

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    ##########################################################################


    ##########################################################################

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

        ##########################################################################

        ValueIterationAgent.__init__(self, mdp, discount, iterations)               ##        self.mdp = mdp                ##        self.discount = discount              ##        self.iterations = iterations
        self.values = util.Counter()                # A Counter is a dict with default 0
        self.runValueIteration()

        ##########################################################################


    ##########################################################################


    ##########################################################################

    def runValueIteration( self ):

        ##########################################################################

        ListOfStates    ,    ListOf_Iterations      =    self.mdp.getStates()   ,       self.iterations
        
        ##########################################################################

        for EachState in ListOfStates    :
            self.values [  EachState  ]  =   0
        
        ##########################################################################

        for Current_Iteration  in range(  ListOf_Iterations  )   :
##            EachState  =   ListOfStates [  Current_Iteration  ]
            EachState = ListOfStates  [   Current_Iteration   % len(ListOfStates)]
            ListOfValues  =    [  ]
            CurrentStateTerminalBOOL  =   self.mdp.isTerminal( EachState  ) 
            if  (  CurrentStateTerminalBOOL  !=   True   )    :

                for CurrentAction in self.mdp.getPossibleActions(EachState)  : 
                    ListOfValues.append  (    self.getQValue  ( EachState  ,  CurrentAction   )  )
                self.values[ EachState ]   =   (  max (  ListOfValues )   )
        
        ##########################################################################








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

        ##########################################################################

        ListOfStates    ,    ListOf_Iterations      =    self.mdp.getStates()   ,       self.iterations
        
        ##########################################################################

        for EachState in ListOfStates    :
            self.values [  EachState  ]  =   0
        
        ##########################################################################


##       ListOfValues   =    [  ]
##        LegalActions  =    self.mdp.getPossibleActions(   state  )
##        
##        if  (  len  (  LegalActions  )   ==   0   )   :
##            return  None
##        else : 
##            values = util.Counter()
##            for action in   LegalActions   :
##                ListOfValues.append  (   [   action  ,    (   self.getQValue  (  state  ,   action )   )    ]    )
##                values  [ action  ]  =  self.getQValue  (  state  ,   action )
##            myRowIndexKey = operator.itemgetter(  1  )                                                                                                       
##            ListOfValues.sort(   key= myRowIndexKey ,      reverse  =  True )
##            return ListOfValues [ 0 ] [ 0 ]




        def get_intended  ( state ) :
            ListOfQ_Values   =   [  ]
            for a in self.mdp.getPossibleActions(state)  :
                ListOfQ_Values.append ( self.getQValue(state, a)  )
            return max( ListOfQ_Values )


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
                               

##                        predecessors[SuccessorState].add(EachState)

##                MaxQValueFromCurrentState  =   

                diff  =  abs(self.values[EachState] - get_intended  (EachState) )             #     Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
                PriorityQueue.push(EachState, -diff)


        for i in range  (ListOf_Iterations )   :
            if PriorityQueue.isEmpty():
                break
            else :
                EachState = PriorityQueue.pop()

                self.values[EachState] = get_intended (  EachState)

                for p in predecessors[EachState]:
                    diff = abs(self.values[p] - get_intended(p))
                    if diff   >   theta:
                        PriorityQueue.update (p, -diff)

 




############################################################################


