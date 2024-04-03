# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #finding nearest ghost
        # ghostdist = 99999099
        # scareddist = 99999999
        # scaredexist = False      
        # for ghost in newGhostStates:
        #     d = util.manhattanDistance(newPos, ghost.getPosition())
        #     if ghost.scaredTimer != 0 : #ghost is scared
        #         if d < scareddist:
        #             scareddist = d
        #             scaredexist = True
        #     else:               
        #         if d < ghostdist:
        #             ghostdist = d
        # if not scaredexist:
        #     scareddist = 1

        fooddist = 99999999
        if len(newFood.asList()) == 0:
            fooddist = 0
        else:
            for food in newFood.asList():
                d = util.manhattanDistance(newPos, food)
                if d < fooddist:
                    fooddist = d

        if fooddist == 0:
            fooddist = 1  
        return successorGameState.getScore() + 1.0 / fooddist

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def Value(state, depth, index):
            if index == state.getNumAgents(): # when all agents visited, go to next depth
                index = 0
                depth += 1

            if depth == self.depth or state.isLose() or state.isWin():
                return [self.evaluationFunction(state), ""]

            if index == 0:
                return MaxValue(state, depth, index)
            return MinValue(state, depth, index)

        def MaxValue(state, depth, index):
            v = float("-inf")
            direction = ""

            actions = state.getLegalActions(0)
            
            for action in actions:
                successor = state.generateSuccessor(0, action)
                val = Value(successor, depth, index + 1)
                if val[0] > v:
                    v = val[0]
                    direction = action

            return [v, direction]

        def MinValue(state, depth, index):
            v = float("inf")
            direction = ""

            actions = state.getLegalActions(index)
            
            for action in actions:
                successor = state.generateSuccessor(index, action)
                val = Value(successor, depth, index + 1)
                if val[0] < v:
                    v = val[0]
                    direction = action

            return [v, direction]

        return Value(gameState, 0, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def Value(state, depth, index, alpha, beta):
            if index == state.getNumAgents(): # when all agents visited, go to next depth
                index = 0
                depth += 1

            if depth == self.depth or state.isLose() or state.isWin():
                return [self.evaluationFunction(state), ""]

            if index == 0:
                return MaxValue(state, depth, index, alpha, beta)
            return MinValue(state, depth, index, alpha, beta)

        def MaxValue(state, depth, index, alpha, beta):
            v = float("-inf")
            direction = ""

            actions = state.getLegalActions(0)
            
            for action in actions:
                successor = state.generateSuccessor(0, action)
                val = Value(successor, depth, index + 1, alpha, beta)
                if val[0] > v:
                    v = val[0]
                    direction = action
                if v > beta:
                    return [v, direction]
                alpha = max(alpha, v)

            return [v, direction]

        def MinValue(state, depth, index, alpha, beta):
            v = float("inf")
            direction = ""

            actions = state.getLegalActions(index)
            
            for action in actions:
                successor = state.generateSuccessor(index, action)
                val = Value(successor, depth, index + 1, alpha, beta)
                if val[0] < v:
                    v = val[0]
                    direction = action
                if v < alpha:
                    return [v, direction]
                beta = min(beta, v)

            return [v, direction]

        return Value(gameState, 0, 0, float("-inf"), float("inf"))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def Value(state, depth, index):
            if index == state.getNumAgents(): # when all agents visited, go to next depth
                index = 0
                depth += 1

            if depth == self.depth or state.isLose() or state.isWin():
                return [self.evaluationFunction(state), ""]

            if index == 0:
                return MaxValue(state, depth, index)
            return ExpValue(state, depth, index)

        def MaxValue(state, depth, index):
            v = float("-inf")
            direction = ""

            actions = state.getLegalActions(0)
            
            for action in actions:
                successor = state.generateSuccessor(0, action)
                val = Value(successor, depth, index + 1)
                if val[0] > v:
                    v = val[0]
                    direction = action

            return [v, direction]

        def ExpValue(state, depth, index):
            v = 0.00
            direction = ""

            actions = state.getLegalActions(index)
            
            for action in actions:
                successor = state.generateSuccessor(index, action)
                v += Value(successor, depth, index + 1)[0]
                
            v /= len(actions)
            return [v, direction]

        return Value(gameState, 0, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()

    # find the nearest food
    foodPos = currentGameState.getFood()
    minfooddist = 0
    if len(foodPos.asList()) != 0:
        fooddist = []
        for food in foodPos.asList():
            fooddist.append(util.manhattanDistance(pacmanPos, food))
        minfooddist = min(fooddist)

    # find the nearest capsule
    capsulePos = currentGameState.getCapsules()
    mincapsuledist = 0
    if len(capsulePos) != 0:        
        capsuledist = []
        for capsule in capsulePos:
            capsuledist.append(util.manhattanDistance(pacmanPos, capsule))
        mincapsuledist = min(capsuledist)
    if mincapsuledist == 0:
        mincapsuledist = 0.1


    # find the nearest ghost
    ghostPos = currentGameState.getGhostStates()
    ghostdist = []
    for ghost in ghostPos:
        ghostdist.append(util.manhattanDistance(pacmanPos, ghost.getPosition()))
    minghostdist = min(ghostdist)
    if minghostdist == 0:
        minghostdist = 0.1

    return -5 * minfooddist +  12/mincapsuledist - 20/minghostdist - 50 * len(foodPos.asList()) - 30 * len(capsulePos)
    
# Abbreviation
better = betterEvaluationFunction
