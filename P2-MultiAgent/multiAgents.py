# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1  )   you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

#################################################################################################################################

from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

#################################################################################################################################


#################################################################################################################################

class ReflexAgent  (  Agent  )  :                     #  A reflex agent chooses an action at each choice point by examining its alternatives via a state evaluation function.

    def getAction(self, gameState  )  :          # Do Not Change Method --- Leave getAction AS IS!!!
        """getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}"""

        legalMoves    =     gameState.getLegalActions  (    )                                                                           # Collect legal moves and successor states
        scores    =     [self.evaluationFunction  (  gameState, action  )   for action in legalMoves]                 # Choose one of the best actions
        bestScore    =     max  (  scores  )  
        bestIndices    =     [index for index in range  (  len  (  scores  )    )   if scores[index] == bestScore]
        chosenIndex    =     random.choice  (  bestIndices  )                                                                          # Pick randomly among the best
        
        return legalMoves[chosenIndex]

    #################################################################################################################################

    def evaluationFunction  (     self  ,   currentGameState   , action     )     :                                                  # The code extracts useful information from the state                     Useful information you can extract from a GameState (pacman.py  )  

        successorGameState       =     currentGameState.generatePacmanSuccessor  (  action  )         #  Successor Game State     # from   pacman.py                                    #  combine these Variables to create a masterful evaluation function.        
        PacmansPositionXY        =     successorGameState.getPacmanPosition  (    )                           #   New  ( x , y )  coordinates  Pacman's Position After Moving
        RemainingFood               =     successorGameState.getFood  (    )                                             #   RemainingFood
        newGhostStates    =     successorGameState.getGhostStates  (    )
        newScaredTimes    =     [ghostState.scaredTimer for ghostState in newGhostStates]                          #    Equals   [ 0 ]    Because No Pellets?       $Number of Time Step Moves ( t )   which EQUALS....     # of Time Steps that Ghost Will Remain Scared B/c of Power Pellet Eaten           #   holds the number of moves that each ghost will remain scared because of Pacman having eaten a power pellet

        if len( RemainingFood.asList())==0   :
            DistanceToFOOD_Reciprocal = 1
        else  :
            for foodPos in RemainingFood.asList()   :
                DistanceToFOOD_Reciprocal  =  1/float(min([10000000000] + [  manhattanDistance( PacmansPositionXY , foodPos) ]   )  )

        Ghost_XY_SuccessorState   =    successorGameState.getGhostPositions()
        
        ghostScore   ,   DistanceLIST   =    0    ,    [    ]

        for ghostPos in Ghost_XY_SuccessorState  :
            CurrentDistance   =   manhattanDistance( PacmansPositionXY , ghostPos)
            if  (  CurrentDistance   not  in  DistanceLIST  )    :
                DistanceLIST.append  (   manhattanDistance( PacmansPositionXY , ghostPos)   )

        closestGhost   =  min  (  DistanceLIST   ) 
        if closestGhost   <     3   :
            ghostScore = -300
            
        return (successorGameState.getScore() - currentGameState.getScore()) + DistanceToFOOD_Reciprocal + ghostScore

################################################################################

def scoreEvaluationFunction  (  currentGameState  )  :
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents  )  .
    """
    return currentGameState.getScore  (    )  

################################################################################

class MultiAgentSearchAgent  (  Agent  )  :
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent   (  game.py  )  
      is another abstract class.
    """

    def __init__  (  self, evalFn = 'scoreEvaluationFunction', depth = '2'  )  :
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup  (  evalFn, globals  (    )    )  
        self.depth = int  (  depth  )  

################################################################################


################################################################################

class MinimaxAgent  (  MultiAgentSearchAgent  )  :                                  #   Your minimax agent   (  question 2  )  

    def getAction  (  self, gameState  )  :                                                      #    Returns the minimax action from the current gameState using self.depth and self.evaluationFunction.
        TotalNumberOfAgents     =     gameState.getNumAgents  (    )                                                  #     Returns the total number of agents in the game
        
        def   Minimax_MAX   (    yourGameState    ,    yourDepth   )    :
            LegalMoves_Pacman   =    yourGameState.getLegalActions(  0  )
            if   (  yourDepth   > self.depth    ) or    (   yourGameState.isWin()  ==  True  )    or      (  len  (  LegalMoves_Pacman   )   ==   0   )       :
                return self.evaluationFunction(yourGameState), None                     #  Defaults to the Function  :   def  scoreEvaluationFunction  
            MaxSuccessor_COST  =   0
            for action in LegalMoves_Pacman :
                successor    =   ( Minimax_MIN (   yourGameState.generateSuccessor(0, action)   , 1,   yourDepth  ), action) 
                if  (  successor    >    MaxSuccessor_COST    )   :
                    MaxSuccessor_COST   =   successor
            return MaxSuccessor_COST

        def  Minimax_MIN  (  yourGameState   , agent_index,   yourDepth    ):
            LegalMoves_Ghosts = yourGameState.getLegalActions(agent_index)
            if not LegalMoves_Ghosts or yourGameState.isLose():
                return self.evaluationFunction(yourGameState), None

            Successor_COST_LIST        =        [  ]
            for action in LegalMoves_Ghosts   :
                SuccessorState      =     (   yourGameState.generateSuccessor(agent_index, action)  )                              ##            MinSuccessor_COST  =   float ( "inf" )
                if agent_index ==   TotalNumberOfAgents   - 1:
                    Successor_COST_LIST.append( Minimax_MAX (  SuccessorState  ,   yourDepth   + 1))
                else:
                    Successor_COST_LIST.append(  Minimax_MIN  (  SuccessorState    ,    agent_index + 1,   yourDepth  ))
            return min(Successor_COST_LIST)

        return Minimax_MAX (gameState, 1)[1]
        
#################################################################################################################################

class AlphaBetaAgent  (  MultiAgentSearchAgent  )  :            #   Your minimax agent with alpha-beta pruning   (  question 3  )  

    def getAction  (  self, gameState  )  :                                    #   Returns the minimax action using self.depth and self.evaluationFunction

        TotalNumberOfAgents     =     gameState.getNumAgents  (    )                                                  #     Returns the total number of agents in the game

        def max_value(  yourGameState  ,depth, agentIndex, alpha, beta):
            depth   =    (   depth    -    1     )
            if depth < 0 or (  yourGameState.isLose() or yourGameState.isWin()  )  ==    True    :

                return (self.evaluationFunction(yourGameState),None)
            v = float("-inf")                                                                                               #   initialize v = Negative Infinity

            for action in yourGameState.getLegalActions(agentIndex):
                successor = yourGameState.generateSuccessor(agentIndex,action)
                score = min_value(successor,depth,agentIndex+1, alpha, beta)[0]
                if score > v:
                    v = score
                    maxAction = action
                if v > beta:
                    return (v,maxAction)
                alpha = max   (  alpha  ,   v  )
            return (v,maxAction)

        #################################################################################################################################

        def min_value(yourGameState,depth,agentIndex, alpha, beta):
            if   (   agentIndex    ==    TotalNumberOfAgents    )   :
                return    (     max_value   (    yourGameState,depth + 1,agentIndex, alpha, beta) )
            if depth < 0 or yourGameState.isLose() or yourGameState.isWin():
                return (self.evaluationFunction(yourGameState),None)
            v = float("inf")
            evalfunc, nextAgent = (min_value, agentIndex+1) if agentIndex < yourGameState.getNumAgents()-1 else (max_value, 0)
            
            for action in yourGameState.getLegalActions(agentIndex):
                successor = yourGameState.generateSuccessor(agentIndex,action)
                score = evalfunc(successor,depth,nextAgent, alpha, beta)[0]
                if score < v:
                    v = score
                    minAction = action
                if v < alpha:
                    return (v,minAction)
                beta = min(beta,v)          
            return (v,minAction)
        
        return max_value(gameState,self.depth,0, float("-inf"), float("inf"))[1]

################################################################################

class ExpectimaxAgent  (  MultiAgentSearchAgent  )  :
    """
      Your expectimax agent   (  question 4  )  
    """

    def getAction  (  self, gameState  )  :
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        "*** YOUR CODE HERE ***"

        def DynamicDepthSearch (  yourGameState  ,   yourDepth   , Agent_IndexNUM):
                                                 
            TotalNumberOfAgents     =     yourGameState.getNumAgents  (    )                                   # Collect legal moves and successor states                #     Returns the total number of agents in the game

            if (  Agent_IndexNUM == TotalNumberOfAgents   and    (     yourDepth  == self.depth   )  )   :
                return self.evaluationFunction(  yourGameState  )
            elif (  Agent_IndexNUM == TotalNumberOfAgents   and    (     yourDepth  != self.depth   )  ) :
                return DynamicDepthSearch (  yourGameState  , (    yourDepth   +  1   ) , 0)
            else  :
                actions = yourGameState.getLegalActions( Agent_IndexNUM )
                if len(actions) == 0  :
                    return self.evaluationFunction(  yourGameState ) 

                next_states  =   [  ]

                for action in actions   :
                    SuccessorState   =    (  yourGameState.generateSuccessor(Agent_IndexNUM, action)    )
                    NextDepth   =      yourDepth
                    NextAgentNum    =    (   Agent_IndexNUM + 1  )
                    next_states.append ( (   DynamicDepthSearch    (    SuccessorState      ,   NextDepth     ,   NextAgentNum   )    )   )

                if Agent_IndexNUM == 0   :
                    return   max  (   next_states   )
                else  :
                    return (     float   (   sum  (   next_states   )  )     /      float   (    len   (   next_states   )     )    )

        legalMoves    =     gameState.getLegalActions  (  0  )                                                                          

        return max(     legalMoves   ,       key = lambda myVar: DynamicDepthSearch  ( gameState.generateSuccessor(0,   myVar   ),      1,     1       ) )

################################################################################

def betterEvaluationFunction  (  currentGameState  )  :
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable  evaluation function   (  question 5  )  .

      DESCRIPTION: < Goal is to Complete The Game in as Few Moves as Capable. Fewer Moves = Higher Score >
    """
    "*** YOUR CODE HERE ***"

    PacmansPositionXY = currentGameState.getPacmanPosition()
    RemainingFood  = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = currentGameState.getScore()               

    DistanceToFoodLIST  =   [   ]
    for   CurrentFoodXY   in    RemainingFood.asList()   :
        RegularDistanceToFood    =     manhattanDistance( PacmansPositionXY , CurrentFoodXY   )
        ReciprocalDistance           =     (   1.0     /    float  (    RegularDistanceToFood    )    )
        DistanceToFoodLIST.append  (  RegularDistanceToFood     )

    if   len  (  DistanceToFoodLIST )   !=   0   :
        score   =    (   score    +   (   10.0   /   min  (  DistanceToFoodLIST  )   )       )

    return  score

################################################################################

# Abbreviation
better = betterEvaluationFunction

################################################################################
