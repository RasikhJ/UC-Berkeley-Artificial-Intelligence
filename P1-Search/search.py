# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions 
from util import Stack
from util import Queue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s   =   Directions.SOUTH
    w   =   Directions.WEST
    return   [s, s, w, s, w, w, s, w] 

#####################################################################################


#####################################################################################

def depthFirstSearch(problem):                                                                                             #  Search the deepest nodes in the search tree first. Your search algorithm needs to return a list of actions that reaches the goal. Make sure to implement a graph search algorithm.

    ClosedLIST     ,     OpenLIST     ,    myPath    =       {  }    ,        Stack(   )    ,      [   ]            # Pseudocode : closed is a set that's Empty ( No items in Set )         # Closed List Stores.... Every Expanding Node ( Goal is to Avoid Repeated States )        # Open List is the Fringe of Unexpanded Node
    OpenLIST.push (   (  problem.getStartState (  ), problem.getStartState (  ), 'Stop'))          #   Push 'item' onto the stack  ( OpenLIST )
  
    while not OpenLIST.isEmpty (  ):
        node   =   OpenLIST.pop (  )                                                                                            #   Pop the most recently pushed item from the stack      #   Return A Tuple with Information 
        if problem.isGoalState (  node [  0  ] ):
            while node [  2  ]  != 'Stop':
                myPath.append (  node [  2  ] )
                node   =   ClosedLIST [  node [  1  ] ]
            myPath.reverse (  )
            return myPath

        ClosedLIST [  node [  0  ] ]   =   node
        children   =   problem.getSuccessors (  node [  0  ]  )             #  x, y, z in 
        for child in children:
            if child [  0] not in ClosedLIST:
                OpenLIST.push (   (  child [  0], node [  0  ] , child [  1  ] ))
    return None

#####################################################################################

def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 74]"

  CurrentStateSpace   =   problem.getStartState (  )
  OpenLIST   =   set (   [  CurrentStateSpace  ] )
  CostFunc_NumberOfActions   =    [    ] 
  successorStack   =   util.Queue (  )

  while not problem.isGoalState (  CurrentStateSpace):
      for successor in problem.getSuccessors (  CurrentStateSpace):          
          if successor [  0  ]  not in OpenLIST:
              OpenLIST.add (  successor [  0  ] )
              successorStack.push (   (  successor [  0  ] , CostFunc_NumberOfActions +  [  successor [  1  ] ]))              
      
      CurrentStateSpace   =   successorStack.pop (  )   
      ( CurrentStateSpace   ,    CostFunc_NumberOfActions   )    =     (   CurrentStateSpace [  0  ]    ,      CurrentStateSpace [  1  ]     )
      
  return CostFunc_NumberOfActions

#####################################################################################

def uniformCostSearch(problem):
    "Search the node of least total cost first. "

    ClosedLIST   =   set (  )
    OpenLIST   =   util.PriorityQueue (  )
    OpenLIST.push (  (   problem.getStartState (  )   ,  [    ] , 0), 0)
    while not OpenLIST.isEmpty (  ):
        myLocation  , myPath, myCost   =   OpenLIST.pop (  )
        if problem.isGoalState (    myLocation  ):
            return myPath
        if   myLocation   not in ClosedLIST:
            ClosedLIST.add (   myLocation   )
            for mySuccessor, myAction, myStepCost   in problem.getSuccessors (    myLocation  ):
                if mySuccessor  not in ClosedLIST:
                    OpenLIST.push (  (mySuccessor,  myPath +  [  myAction  ] , myStepCost  +myCost), myStepCost  +myCost)
    return  None

#####################################################################################

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

#####################################################################################

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."

    ClosedLIST   =   set (  )
    OpenLIST   =   util.PriorityQueue (  )
    OpenLIST.push (   (  problem.getStartState (  ),  [  ], 0), 0)
    while not OpenLIST.isEmpty (  ):
        myLocation  , myPath, myCost   =   OpenLIST.pop (  )
        if problem.isGoalState (    myLocation  ):
            return myPath
        if   myLocation   not in ClosedLIST:
            ClosedLIST.add (    myLocation  )
            for mySuccessor,myAction,myStepCost   in problem.getSuccessors (    myLocation  ):
                if mySuccessor not in ClosedLIST:
                    UniformSearchCost  =   myStepCost   + myCost
                    GreedyAlgorithmHeuristicCost  =   heuristic (  mySuccessor, problem)
                    A_Star_SummationCost   =   UniformSearchCost+GreedyAlgorithmHeuristicCost
                    OpenLIST.push (   (  mySuccessor, myPath +  [  myAction  ] , UniformSearchCost), A_Star_SummationCost)
    return  None

#####################################################################################

# Abbreviations
bfs   =   breadthFirstSearch
dfs   =   depthFirstSearch
astar   =   aStarSearch
ucs   =   uniformCostSearch

  

