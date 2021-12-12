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
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


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

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        legal_actions = self.getLegalActions(state)
        max_q_val = max([self.getQValue(state, action) for action in legal_actions], default=0)
        return max_q_val

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)
        get_q_val = self.getQValue
        max_q_action = max(legal_actions, key=lambda x: get_q_val(state, x), default=None)
        return max_q_action
        # util.raiseNotDefined()

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
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if len(legalActions) == 0:
            return None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        self.values[(state, action)] = ((1 - self.alpha) * self.getQValue(state, action)) + (
                self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState)))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
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
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
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
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        weights = self.getWeights()
        features = self.featExtractor.getFeatures(state, action)
        partial_q = util.Counter({key: features[key] * weights[key] for key in weights.keys()})
        return partial_q.totalCount()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        weights = self.getWeights()
        features = self.featExtractor.getFeatures(state, action)

        if nextState:
            reward += self.discount * self.getValue(nextState)
        diff = reward - self.getQValue(state, action)
        for key in features.keys():
            weights[key] += self.alpha * diff * features[key]
        self.weights = weights

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


get_weak_ghost_states = lambda y: list(filter(lambda x: x.scaredTimer > 0, y))


def has_weak_ghosts(ghostStates):
    for ghostState in ghostStates:
        if ghostState.scaredTimer > 0:
            return True
    return False


get_strong_ghost_states = lambda y: list(filter(lambda x: x.scaredTimer == 0, y))

is_weak_ghost = lambda x: x.scaredTimer > 0

is_strong_ghost_neighbor = lambda x: x['#-of-strong-ghost-1-step-away'] != 0

get_ghost_positions = lambda x: [getGhostPosition(ghostState) for ghostState in x]


def getGhostPosition(ghostState):
    x, y = ghostState.getPosition()
    dx, dy = Actions.directionToVector(ghostState.getDirection())
    return int(x + dx), int(y + dy)


def distanceObject(pos, obj, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if (pos_x, pos_y) in obj:
            return dist / (walls.width * walls.height)
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    return None


def closestCapsule(pos, capsules, walls):
    if not capsules:
        return None
    else:
        return distanceObject(pos, capsules, walls)


distanceGhost = lambda pos, ghost, walls: distanceObject(pos, [ghost], walls)

distanceGhostState = lambda pos, ghostState, walls: distanceGhost(pos, getGhostPosition(ghostState), walls)


class BetterExtractor(FeatureExtractor):
    def __init__(self):
        self.param = {
            'bias': -1.1869386151859338,
            'closest_weak_ghost_0': -0.4649615270560723,
            'closest_weak_ghost_1': -4.850390180600743,
            'strong_ghost_1_step_0': -6.569637113088273,
            'strong_ghost_1_step_1': -7.795684250866552,
            'strong_ghost_2_step_0': -6.785999786918713,
            'strong_ghost_2_step_1': 2.555640948401763,
            'weak_ghost_0': -3.400446220585069,
            'weak_ghost_1': -2.0120989246248673,
            'eats_capsule': 4.8657842860880525,
            'eats_food': 9.077065896989659,
            'closest_capsule_0': 5.98444945759584,
            'closest_capsule_1': 4.2062442789363,
            'closest_food_0': 4.791562262169061,
            'closest_food_1': -1.959221289065276
        }

    def getFeatures(self, state, action):
        from pacman import PacmanRules
        capsules = state.getCapsules()
        food = state.getFood()
        walls = state.getWalls()
        ghostStates = state.getGhostStates()
        ghosts = get_ghost_positions(ghostStates)

        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        features = util.Counter()

        features["bias"] = 1.0 * self.param['bias']

        weakGhostStates = list(filter(
            lambda x: distanceGhostState((next_x, next_y), x, walls) is not None,
            get_weak_ghost_states(ghostStates)
        ))

        strong_ghost_away = lambda next_x, next_y: sum(
            (next_x, next_y) in Actions.getLegalNeighbors(ghostState.getPosition(), walls) for ghostState in
            get_strong_ghost_states(ghostStates))

        if weakGhostStates:
            closestGhostState = min(weakGhostStates, key=lambda x: distanceGhostState((next_x, next_y), x, walls))
            features['closest-weak-ghost-distance'] = self.param['closest_weak_ghost_0'] - self.param[
                'closest_weak_ghost_1'] * distanceGhostState((next_x, next_y), closestGhostState, walls)

        features['#-of-strong-ghost-1-step-away'] = self.param['strong_ghost_1_step_0'] + self.param[
            'strong_ghost_1_step_1'] * strong_ghost_away(next_x, next_y)

        features['#-of-strong-ghost-2-step-away'] = self.param['strong_ghost_2_step_0'] + self.param[
            'strong_ghost_2_step_1'] * (sum(strong_ghost_away(nx, ny) for nx, ny in
                                            [(next_x + 1, next_y), (next_x - 1, next_y),
                                             (next_x, next_y + 1), (next_x, next_y - 1)]))

        features['weak-ghost'] = self.param['weak_ghost_0'] + self.param['weak_ghost_1'] * float(
            len(get_weak_ghost_states(ghostStates)))

        if not ((has_weak_ghosts(ghostStates) or is_strong_ghost_neighbor(features)) or not (
                (next_x, next_y) in capsules)):
            features['eats-capsule'] = 10.0 * self.param['eats_capsule']

        if not ((((has_weak_ghosts(ghostStates) or is_strong_ghost_neighbor(features)) or capsules) or (
                next_x, next_y) in capsules) or not food[next_x][next_y]):
            features['eats-food'] = self.param['eats_food']

        capsule_dist = closestCapsule((next_x, next_y), capsules, walls)
        if not (has_weak_ghosts(ghostStates) or not (capsule_dist is not None)):
            features['closest-capsule'] = self.param['closest_capsule_0'] + self.param['closest_capsule_1'] * float(
                capsule_dist)

        food_dist = closestFood((next_x, next_y), food, walls)
        if not ((has_weak_ghosts(ghostStates) or capsules) or not (food_dist is not None)):
            features['closest-food'] = self.param['closest_food_0'] + self.param['closest_food_1'] * float(
                food_dist) / (walls.width * walls.height)

        features.divideAll(10.0)
        return features
