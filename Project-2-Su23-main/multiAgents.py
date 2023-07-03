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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPos = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #print(successorGameState)
        #print(newPos)
        #print(newFood)
        #print(newGhostStates)
        #print(newScaredTimes)

        # manhattan dist to closest food, ghost
        #distance_over_food = manhattanDistance(newPos, successorGameState) / newFood
        if newFood:
            min_MH = min(manhattanDistance(newPos, food) for food in newFood)
        else:
            min_MH = 0  # No food left
        min_ghost_MH = min(newGhostPos) if newGhostPos else float('inf')

        is_scared = newScaredTimes[newGhostPos.index(min_ghost_MH)] if newGhostPos else 0

        # prevent divison by 0
        if min_ghost_MH == 0:
            return float('-inf')
        if min_MH == 0:
            return float('inf')

        # if number of squares left as scared is greater than the distance
        if is_scared > min_ghost_MH:
            # take reciprocal because if distance is close, we want larger value
            return successorGameState.getScore() + 1.0 / min_ghost_MH
        else:
            # if distance is far, we want smaller e(x)
            return successorGameState.getScore() - 1.0 / min_ghost_MH + 1.0 / min_MH


def scoreEvaluationFunction(currentGameState: GameState):
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
    # code from lectures
    def max_value(self, gameState, depth, agentIndex):
        """
        Given a state, keeps iterating and increasing until the 
        maximum value is found -> V(s) = max V(s’) for all s’ successors of s
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        v = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self.min_value(successor, depth, agentIndex + 1))
        return v

    def min_value(self, gameState, depth, agentIndex):
        """
        Given a state, keeps iterating and increasing until the 
        minimum value is found -> V(s') = min V(s) for all s successors of s’
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
            
        v = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = min(v, self.max_value(successor, depth + 1, 0) if agentIndex == gameState.getNumAgents() - 1 else self.min_value(successor, depth, agentIndex + 1))
        return v

    def getAction(self, gameState: GameState):
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
        #agentIndex = 0
        #legal_actions = gameState.getLegalActions(agentIndex)
        #successor_gamestate = gameState.generateSuccessor(agentIndex, action)
        #total_agents = gameState.getNumAgents()
        is_winning_state = gameState.isWin()
        is_losing_state = gameState.isLose()

        # base case: win or lose
        if is_winning_state or is_losing_state:
            return self.evaluationFunction(gameState)

        pacmans_actions = gameState.getLegalActions(0)
        return max(pacmans_actions, key=lambda action: 
                self.min_value(gameState.generateSuccessor(0, action), 0, 1))



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_value(self, gameState, depth, agentIndex, alpha, beta):
        """
        Given a state, keeps iterating and increasing until the 
        maximum value is found -> V(s) = max V(s’) for all s’ successors of s
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        v = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            # only called once for each action
            successor = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self.min_value(successor, depth, agentIndex + 1, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, gameState, depth, agentIndex, alpha, beta):
        """
        Given a state, keeps iterating and increasing until the 
        minimum value is found -> V(s') = min V(s) for all s successors of s’
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
            
        v = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            # only called once for each action
            successor = gameState.generateSuccessor(agentIndex, action)
            v = min(v, self.max_value(successor, depth + 1, 0, alpha, beta) 
                if agentIndex == gameState.getNumAgents() - 1 else self.min_value(successor, depth, agentIndex + 1, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-inf')
        beta = float('inf')
        pacmans_actions = gameState.getLegalActions(0)
        v = float('-inf')
        best_action = Directions.STOP

        for action in pacmans_actions:
            # only called once for each action
            successor = gameState.generateSuccessor(0, action)
            temp = self.min_value(successor, 0, 1, alpha, beta)
            # update thresholds
            if temp > v:
                v = temp
                best_action = action
            # prune on equality
            if v >= beta:
                return action
            # maximizer
            alpha = max(alpha, v)
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def max_value(self, gameState, depth):
            """
            Given a state, keeps iterating and increasing until the 
            maximum value is found -> V(s) = max V(s’) for all s’ successors of s
            """
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            v = float('-inf')
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                v = max(v, self.expectimax(successor, depth, 1)) 
            return v

    def expectimax(self, gameState, depth, agentIndex):
        """
        Given a state, calculates the expectimax values
        """
        # same base case as before
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        v = 0
        actions = gameState.getLegalActions(agentIndex)
        utility_weight = 1.0 / len(actions) # uniform distribution probability
        nextDepth = 0
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            # if the current agent is not the last ghost
            if agentIndex < gameState.getNumAgents() - 1:
                nextAgent = agentIndex + 1
            # otherwise it is pacman
            else:
                nextAgent = 0
            # if next agent is a ghost, depth is the same
            # otherwise all ghosts have moved, and we go deeper by 1
            nextDepth = depth if nextAgent else depth + 1
            v += utility_weight * (self.max_value(successor, nextDepth) 
                            if nextAgent == 0 
                            else self.expectimax(successor, nextDepth, nextAgent))
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        pacmans_actions = gameState.getLegalActions(0)
        v = float('-inf')
        best_action = Directions.STOP

        return max(pacmans_actions, key = lambda action: 
                self.expectimax(gameState.generateSuccessor(0, action), 0, 1))

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: A better evaluation function is one that works for reflex agents,
                not just adversarial search agents. In this context, we want pacman
                to perceive how the world could be. Thus this function takes into 
                account how many ghosts are alive/how many are scared, how much 
                food is left, as well as the distances to each. Then it returns a 
                score given the current gamestate
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # MH to the nearest food
    foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
    nearestFoodDist = min(foodDistances) if foodDistances else 0

    # MH to the nearest ghost
    ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    nearestGhostDist = min(ghostDistances) if ghostDistances else 0

    # is Pacman able to eat a ghost? if yes, pos factor else neg
    if newScaredTimes and min(newScaredTimes) > nearestGhostDist:
        ghostFactor = 200
    else:
        ghostFactor = -200

    # Return an evaluation of the current state
    return currentGameState.getScore() + (1.0 / (nearestFoodDist + 1)) 
    + ghostFactor * (1.0 / (nearestGhostDist + 1))

# Abbreviation
better = betterEvaluationFunction
