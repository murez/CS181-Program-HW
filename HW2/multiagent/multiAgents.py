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
import random, util, math, json

from game import Agent


def ghostDistance(ghost, point):
    return manhattanDistance(ghost.getPosition(), point) * -2 * (ghost.isPacman - 0.5)


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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        capsulePositions = currentGameState.getCapsules()
        foodNum = currentGameState.getFood().count()
        foodList = newFood.asList()
        "*** YOUR CODE HERE ***"

        # newGhostPositions = [ghostPos.getPosition() for ghostPos in newGhostStates]
        # newGhostisPacman = [ghostPos.isPacman for ghostPos in newGhostStates]
        value = 0
        if len(foodList) == foodNum:
            value -= min([manhattanDistance(newPos, foodPos) + \
                          min(manhattanDistance(ghost.getPosition(), foodPos) for ghost in newGhostStates)
                          for foodPos in foodList])
        if capsulePositions:
            value -= min(manhattanDistance(newPos, capsulePos) for capsulePos in capsulePositions)
        for ghost in newGhostStates:  # the impact of ghost surges as distance get close
            # dis -= 9 >> int(max((manhattanDistance(ghost.getPosition(), newPos) - 2), 0))
            if ghost.scaredTimer > manhattanDistance(ghost.getPosition(), newPos):
                value += 2 << ghost.scaredTimer
            else:
                ghost_dist = ghostDistance(ghost, newPos)
                if ghost_dist >= 0:
                    value -= 2.71828 ** (2 - ghost_dist)
                else:
                    value += 2.71828 ** (- ghost_dist)
        if action == 'stop':
            value -= random.random() * 0.5 * abs(value)
        return value


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minmax(0, gameState, 0)

    def minmax(self, agent_id, now_game_state, step):
        if agent_id == now_game_state.getNumAgents():
            agent_id = 0
            step += 1

        if not now_game_state.getLegalActions(agent_id) or step == self.depth:
            return self.evaluationFunction(now_game_state)

        next_minmax_result = [
            (self.minmax(agent_id + 1, now_game_state.getNextState(agent_id, next_step), step), next_step)
            for next_step in now_game_state.getLegalActions(agent_id)]
        if agent_id == 0:
            if step == 0:
                return max(next_minmax_result, key=lambda x: x[0])[1]
            else:
                return max(next_minmax_result, key=lambda x: x[0])[0]
        else:
            return min(next_minmax_result, key=lambda x: x[0])[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabeta(0, gameState, 0)

    def alphabeta(self, agent_id, now_game_state, step, alpha=-math.inf, beta=math.inf):
        if agent_id == now_game_state.getNumAgents():
            agent_id = 0
            step += 1

        if not now_game_state.getLegalActions(agent_id) or step == self.depth:
            return self.evaluationFunction(now_game_state)
        best_next_step = Directions.STOP
        if agent_id == 0:
            value = -math.inf
        else:
            value = math.inf
        for next_step in now_game_state.getLegalActions(agent_id):
            next_value = self.alphabeta(agent_id + 1, now_game_state.getNextState(agent_id, next_step), step, alpha,
                                        beta)
            if agent_id == 0:
                if next_value >= value:
                    value, best_next_step = next_value, next_step
                if value > beta:
                    break
                alpha = max(alpha, value)
            else:
                if next_value <= value:
                    value, best_next_step = next_value, next_step
                if value < alpha:
                    break
                beta = min(beta, next_value)
        if step == 0 and agent_id == 0:
            return best_next_step
        else:
            return value


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
        return self.expectimax(0, gameState, 0)

    def expectimax(self, agent_id, now_game_state, step):
        if agent_id == now_game_state.getNumAgents():
            agent_id = 0
            step += 1

        if not now_game_state.getLegalActions(agent_id) or step == self.depth:
            return self.evaluationFunction(now_game_state)

        next_minmax_result = [
            (self.expectimax(agent_id + 1, now_game_state.getNextState(agent_id, next_step), step), next_step)
            for next_step in now_game_state.getLegalActions(agent_id)]
        if agent_id == 0:
            if step == 0:
                return max(next_minmax_result, key=lambda x: x[0])[1]
            else:
                return max(next_minmax_result, key=lambda x: x[0])[0]
        else:
            return sum([x[0] for x in next_minmax_result]) / len(next_minmax_result)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()

    value = currentGameState.getScore()
    value -= 2.71828 ** (
            2 - min(manhattanDistance(newPos, ghost.getPosition()) for ghost in currentGameState.getGhostStates()))
    value += sum((ghost.scaredTimer - manhattanDistance(newPos, ghost.getPosition()))
                 for ghost in currentGameState.getGhostStates()
                 if ghost.scaredTimer > manhattanDistance(newPos, ghost.getPosition()))
    if currentGameState.getNumFood():
        value -= 5 * currentGameState.getNumFood()
        value -= 2 * min(manhattanDistance(newPos, foodPos) + \
                         min(manhattanDistance(ghost.getPosition(), foodPos) for ghost in
                             currentGameState.getGhostStates())
                         for foodPos in currentGameState.getFood().asList())

    if currentGameState.getCapsules():
        value -= min(manhattanDistance(newPos, capsulePos) for capsulePos in currentGameState.getCapsules())

    return value


# Abbreviation
better = betterEvaluationFunction


# class ContestAgent(MultiAgentSearchAgent):
#     """
#       Your agent for the mini-contest
#     """
#
#     def getAction(self, gameState):
#         """
#           Returns an action.  You can use any method you want and search to any depth you want.
#           Just remember that the mini-contest is timed, so you have to trade off speed and computation.
#
#           Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
#           just make a beeline straight towards Pacman (or away from him if they're scared!)
#         """
#         "*** YOUR CODE HERE ***"
#         util.raiseNotDefined()

def bestEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()

    value = currentGameState.getScore()
    value -= 2.71828 ** (
            2 - min(manhattanDistance(newPos, ghost.getPosition()) for ghost in currentGameState.getGhostStates()))
    value += sum((ghost.scaredTimer - manhattanDistance(newPos, ghost.getPosition()))
                 for ghost in currentGameState.getGhostStates()
                 if ghost.scaredTimer > manhattanDistance(newPos, ghost.getPosition()))
    if currentGameState.getNumFood():
        value -= 5 * currentGameState.getNumFood()
        value -= 2 * min(manhattanDistance(newPos, foodPos)
                         for foodPos in currentGameState.getFood().asList())

    if currentGameState.getCapsules():
        value -= min(manhattanDistance(newPos, capsulePos) for capsulePos in currentGameState.getCapsules())

    return value


class ContestAgent(ExpectimaxAgent):
    def __init__(self, evalFn='bestEvaluationFunction', depth='3'):
        self.index = 0  # Pacman is always agent index 0
        # self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

        self.ghost_state_base = 2.71828
        self.ghost_state_zero = 2
        self.ghost_value = 1
        self.food_num_value = 5
        self.food_dis_value = 2
        self.capsule_dis_value = 1
        self.capsule_dis_num = 0
        with open("./data_2.json", 'rb') as f:
            data = json.load(f)
            self.score_value = data["score_value"]
            self.ghost_state_base = data["ghost_state_base"]
            self.ghost_state_zero = data["ghost_state_zero"]
            self.ghost_value = data["ghost_value"]
            self.food_num_value = data["food_num_value"]
            self.food_dis_value = data["food_dis_value"]
            self.capsule_dis_value = data["capsule_dis_value"]
            self.capsule_dis_num = data["capsule_dis_num"]

    def evaluationFunction(self, currentGameState):
        """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).

        DESCRIPTION: <write something here so we know what you did>
        """
        "*** YOUR CODE HERE ***"
        newPos = currentGameState.getPacmanPosition()

        value = self.score_value * currentGameState.getScore()
        # value -= self.ghost_state_base ** (self.ghost_state_zero - min(manhattanDistance(newPos, ghost.getPosition())
        #                                                                for ghost in currentGameState.getGhostStates()))

        for ghost in currentGameState.getGhostStates():
            if ghost.scaredTimer <= 0:
                value -= self.ghost_state_base ** (
                            self.ghost_state_zero - manhattanDistance(newPos, ghost.getPosition()))

        value += self.ghost_value * sum((ghost.scaredTimer - manhattanDistance(newPos, ghost.getPosition()))
                     for ghost in currentGameState.getGhostStates()
                     if ghost.scaredTimer > manhattanDistance(newPos, ghost.getPosition()))
        if currentGameState.getNumFood():
            value -= self.food_num_value * currentGameState.getNumFood()
            value -= self.food_dis_value * min(manhattanDistance(newPos, foodPos)
                             for foodPos in currentGameState.getFood().asList())

        if currentGameState.getCapsules():
            value -= self.capsule_dis_value * min(manhattanDistance(newPos, capsulePos) for capsulePos in currentGameState.getCapsules())
        value += self.capsule_dis_num * len(currentGameState.getCapsules())
        return value
