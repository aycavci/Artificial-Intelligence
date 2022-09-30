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
        chosenIndex = random.choice(bestIndices)
        # Pick randomly among the best

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
        # (x, y)
        newFood = successorGameState.getFood()
        # T, F food 2D grid matrix
        newFoodPos = newFood.asList()
        # positions of foods as a list (x, y) list
        newGhostStates = successorGameState.getGhostStates()
        # you can get position getPosition() and direction getDirection() of ghosts from their states
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        current_food_heuristic = 0
        current_ghost_heuristic = 0

        if successorGameState.isWin():
            return 10000000000000000

        for ghost in newGhostStates:
            if manhattanDistance(newPos, ghost.getPosition()) == 0:
                return -100
            else:
                ghost_heuristic = 1 / manhattanDistance(newPos, ghost.getPosition())
                if ghost_heuristic > current_ghost_heuristic:
                    current_ghost_heuristic = ghost_heuristic
        if min(newScaredTimes) == 0:
            current_ghost_heuristic = -1 * current_ghost_heuristic
        else:
            current_ghost_heuristic = current_ghost_heuristic

        for foodPos in newFoodPos:
            if currentGameState.getNumFood() > successorGameState.getNumFood():
                current_food_heuristic = 100
            else:
                food_heuristic = 1 / manhattanDistance(newPos, foodPos)
                if food_heuristic > current_food_heuristic:
                    current_food_heuristic = food_heuristic

        heuristic = current_food_heuristic + current_ghost_heuristic

        return heuristic

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

        return self.recursiveMinimax(self.index, gameState, self.depth)[1]

    def recursiveMinimax(self, agentIndex, gameState, depth):

        if gameState.isWin() or gameState.isLose():
            action = ""
            evaluationValue = self.evaluationFunction(gameState)
            return [evaluationValue, action]

        if depth == 0:
            action = ""
            evaluationValue = self.evaluationFunction(gameState)
            return [evaluationValue, action]

        if agentIndex == 0:
            v = [float("-inf"), ""]
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                evaluationList = self.recursiveMinimax(agentIndex+1, successor, depth)
                if v[0] < evaluationList[0]:
                    v[0] = evaluationList[0]
                    v[1] = action
            return v
        else:
            if agentIndex == gameState.getNumAgents()-1:
                depth -= 1
                v = [float("inf"), ""]
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    evaluationList = self.recursiveMinimax(0, successor, depth)
                    if v[0] > evaluationList[0]:
                        v[0] = evaluationList[0]
                        v[1] = action
                return v
            else:
                v = [float("inf"), ""]
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    evaluationList = self.recursiveMinimax(agentIndex+1, successor, depth)
                    if v[0] > evaluationList[0]:
                        v[0] = evaluationList[0]
                        v[1] = action
                return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        a = float("-inf")
        b = float("inf")

        return self.recursiveMinimax(self.index, gameState, self.depth, a, b)[1]


    def recursiveMinimax(self, agentIndex, gameState, depth, a, b):

        if gameState.isWin() or gameState.isLose():
            action = ""
            evaluationValue = self.evaluationFunction(gameState)
            return [evaluationValue, action]

        if depth == 0:
            action = ""
            evaluationValue = self.evaluationFunction(gameState)
            return [evaluationValue, action]

        if agentIndex == 0:
            v = [float("-inf"), ""]
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                evaluationList = self.recursiveMinimax(agentIndex + 1, successor, depth, a, b)
                if v[0] < evaluationList[0]:
                    v[0] = evaluationList[0]
                    v[1] = action
                if v[0] > b:
                    return v
                a = max(a, v[0])
            return v
        else:
            if agentIndex == gameState.getNumAgents() - 1:
                depth -= 1
                v = [float("inf"), ""]
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    evaluationList = self.recursiveMinimax(0, successor, depth, a, b)
                    if v[0] > evaluationList[0]:
                        v[0] = evaluationList[0]
                        v[1] = action
                        if v[0] < a:
                            return v
                        b = min(b, v[0])
                return v
            else:
                v = [float("inf"), ""]
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    evaluationList = self.recursiveMinimax(agentIndex + 1, successor, depth, a, b)
                    if v[0] > evaluationList[0]:
                        v[0] = evaluationList[0]
                        v[1] = action
                        if v[0] < a:
                            return v
                        b = min(b, v[0])
                return v


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

        return self.recursiveMinimax(self.index, gameState, self.depth)[1]

    def recursiveMinimax(self, agentIndex, gameState, depth):

        if gameState.isWin() or gameState.isLose():
            action = ""
            evaluationValue = self.evaluationFunction(gameState)
            return [evaluationValue, action]

        if depth == 0:
            action = ""
            evaluationValue = self.evaluationFunction(gameState)
            return [evaluationValue, action]

        if agentIndex == 0:
            v = [float("-inf"), ""]
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                evaluationList = self.recursiveMinimax(agentIndex + 1, successor, depth)
                if v[0] < evaluationList[0]:
                    v[0] = evaluationList[0]
                    v[1] = action
            return v
        else:
            if agentIndex == gameState.getNumAgents() - 1:
                depth -= 1
                v = [0, ""]
                for action in gameState.getLegalActions(agentIndex):
                    p = 1 / len(gameState.getLegalActions(agentIndex))
                    successor = gameState.generateSuccessor(agentIndex, action)
                    evaluationList = self.recursiveMinimax(0, successor, depth)
                    v[0] += p * evaluationList[0]
                    v[1] = action
                return v
            else:
                v = [0, ""]
                for action in gameState.getLegalActions(agentIndex):
                    p = 1 / len(gameState.getLegalActions(agentIndex))
                    successor = gameState.generateSuccessor(agentIndex, action)
                    evaluationList = self.recursiveMinimax(agentIndex+1, successor, depth)
                    v[0] += p * evaluationList[0]
                    v[1] = action
                return v


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    # (x, y)
    newFood = currentGameState.getFood()
    # T, F food 2D grid matrix
    newFoodPos = newFood.asList()
    # positions of foods as a list (x, y) list
    newGhostStates = currentGameState.getGhostStates()
    # you can get position getPosition() and direction getDirection() of ghosts from their states
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    current_food_heuristic = 0
    current_ghost_heuristic = 0
    current_score_heuristic = currentGameState.getScore()
    if currentGameState.isWin():
        return 10000000000000000

    for ghost in newGhostStates:
        if manhattanDistance(newPos, ghost.getPosition()) == 0:
            return -100
        else:
            ghost_heuristic = 1 / manhattanDistance(newPos, ghost.getPosition())
            if ghost_heuristic > current_ghost_heuristic:
                current_ghost_heuristic = ghost_heuristic
    if min(newScaredTimes) == 0:
        current_ghost_heuristic = -1 * current_ghost_heuristic
    else:
        current_ghost_heuristic = current_ghost_heuristic

    for foodPos in newFoodPos:
        if currentGameState.getNumFood() > currentGameState.getNumFood():
            current_food_heuristic = 100
        else:
            food_heuristic = 1 / manhattanDistance(newPos, foodPos)
            if food_heuristic > current_food_heuristic:
                current_food_heuristic = food_heuristic

    heuristic = current_food_heuristic + current_ghost_heuristic + current_score_heuristic

    return heuristic


# Abbreviation
better = betterEvaluationFunction
