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


class BetterExtractor(FeatureExtractor):
    "Your extractor entry goes here.  Add features for capsuleClassic."
    def __init__(self):
        import json
        self.param = json.load(open('parameters.json'))

    def hasWeakGhosts(self, ghostStates):
        for ghostState in ghostStates:
            if ghostState.scaredTimer > 0:
                return True
        return False

    def isWeakGhost(self, ghostState):
        return ghostState.scaredTimer > 0

    def getWeakGhostStates(self, ghostStates):
        return list(filter(lambda ghostState: ghostState.scaredTimer > 0, ghostStates))

    def getStrongGhostStates(self, ghostStates):
        return list(filter(lambda ghostState: ghostState.scaredTimer == 0, ghostStates))

    def isStrongGhostAtNeighbor(self, ghosts, features):
        return features['#-of-strong-ghost-1-step-away'] != 0

    """
    Generate your own feature
    """

    def getFeatures(self, state, action):
        from pacman import PacmanRules
        # extract the grid of food and wall locations and get the ghost locations
        capsules = state.getCapsules()
        food = state.getFood()
        walls = state.getWalls()
        ghostStates = state.getGhostStates()
        ghosts = getGhostPositions(ghostStates)

        features = util.Counter()



        features["bias"] = 1.0 * self.param['bias']

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        weakGhostStates = list(filter(
            lambda ghostState: distanceGhostState((next_x, next_y), ghostState, walls) is not None,
            self.getWeakGhostStates(ghostStates)
        ))

        if weakGhostStates:
            closestGhostState = min(
                weakGhostStates,
                key=lambda ghostState: distanceGhostState((next_x, next_y), ghostState, walls))
            features['closest-weak-ghost-distance'] = 1 * self.param['closest_weak_ghost_0'] - self.param[
                'closest_weak_ghost_1'] * distanceGhostState((next_x, next_y), closestGhostState, walls)
        # print('closest-weak-ghost-distance:', features['closest-weak-ghost-distance'])

        # Check strong ghosts are 1-step away
        features['#-of-strong-ghost-1-step-away'] = self.param['strong_ghost_1_step_0'] + self.param[
            'strong_ghost_1_step_1'] * sum(
            (next_x, next_y) in Actions.getLegalNeighbors(ghostState.getPosition(), walls) for ghostState in
            self.getStrongGhostStates(ghostStates))

        features['#-of-strong-ghost-2-step-away'] = self.param['strong_ghost_2_step_0'] + self.param['strong_ghost_2_step_1'] * (
                sum(
                    (next_x + 1, next_y) in Actions.getLegalNeighbors(ghostState.getPosition(), walls) for
                    ghostState in
                    self.getStrongGhostStates(ghostStates)) + sum(
            (next_x - 1, next_y) in Actions.getLegalNeighbors(ghostState.getPosition(), walls) for ghostState in
            self.getStrongGhostStates(ghostStates)) + sum(
            (next_x, next_y + 1) in Actions.getLegalNeighbors(ghostState.getPosition(), walls) for ghostState in
            self.getStrongGhostStates(ghostStates)) + sum(
            (next_x, next_y - 1) in Actions.getLegalNeighbors(ghostState.getPosition(), walls) for ghostState in
            self.getStrongGhostStates(ghostStates)))

        # print('#-of-strong-ghost-1-step-away:', features['#-of-strong-ghost-1-step-away'])
        features['weak-ghost'] = self.param['weak_ghost_0'] + self.param['weak_ghost_1'] * float(
            len(self.getWeakGhostStates(ghostStates)))

        # If there is no danger of ghosts or weak ghosts, then chase a capsule.
        if not self.hasWeakGhosts(ghostStates) and \
                not self.isStrongGhostAtNeighbor(ghosts, features) and \
                (next_x, next_y) in capsules:
            features['eats-capsule'] = 10.0 * self.param['eats_capsule']
            # print('eats-capsule:', features['eats-capsule'])

        # If there is no danger of ghosts or weak ghosts and not capsule nearby, then eat a food.
        if not self.hasWeakGhosts(ghostStates) and \
                not self.isStrongGhostAtNeighbor(ghosts, features) and \
                not capsules and \
                not (next_x, next_y) in capsules and \
                food[next_x][next_y]:
            features['eats-food'] = 1.0 * self.param['eats_food']
            # print('eats-food:', features['eats-food'])

        capsuleDist = closestCapsule((next_x, next_y), capsules, walls)
        if not self.hasWeakGhosts(ghostStates) and \
                capsuleDist is not None:
            features['closest-capsule'] = self.param['closest_capsule_0'] + self.param['closest_capsule_1'] * float(
                capsuleDist) / (walls.width * walls.height)
            # print('closest-capsule:', features['closest-capsule'])

        foodDist = closestFood((next_x, next_y), food, walls)
        if not self.hasWeakGhosts(ghostStates) and \
                not capsules and \
                foodDist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features['closest-food'] = self.param['closest_food_0'] + self.param['closest_food_1'] * float(foodDist) / (
                        walls.width * walls.height)
            # print('closest-food:', features['closest-food'])

        # features["Tunnel"] = float(len(PacmanRules.getLegalActions(state)) <= 2)
        features.divideAll(10.0)
        return features


def getGhostPositions(ghostStates):
    positions = []
    for ghostState in ghostStates:
        x, y = ghostState.getPosition()
        dx, dy = Actions.directionToVector(ghostState.getDirection())
        positions.append((int(x + dx), int(y + dy)))
    return positions


def getGhostPosition(ghostState):
    x, y = ghostState.getPosition()
    dx, dy = Actions.directionToVector(ghostState.getDirection())
    return (int(x + dx), int(y + dy))


def distanceGhostState(pos, ghostState, walls):
    ghost = getGhostPosition(ghostState)
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # If we find a ghost at this location then exit
        if ghost == (pos_x, pos_y):
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            return float(dist) / (walls.width * walls.height)
        # Otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no ghost found
    return None


def distanceGhost(pos, ghost, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # If we find a ghost at this location then exit
        if ghost == (pos_x, pos_y):
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            return float(dist) / (walls.width * walls.height)
        # Otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no ghost found
    return None


def closestCapsule(pos, capsules, walls):
    if not capsules:
        return None

    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if (pos_x, pos_y) in capsules:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None


def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None

    # def getFeatures(self, state, action):
    #     features = SimpleExtractor().getFeatures(state, action)
    #     # Add more features here
    #     "*** YOUR CODE HERE ***"
    #
    #     def closestItem(pos, itemlist, walls):
    #         fringe = [(pos[0], pos[1], 0)]
    #         expanded = set()
    #         while fringe:
    #             pos_x, pos_y, dist = fringe.pop(0)
    #             if (pos_x, pos_y) in expanded:
    #                 continue
    #             expanded.add((pos_x, pos_y))
    #             # if we find a food at this location then exit
    #             if (pos_x, pos_y) in itemlist:
    #                 return dist
    #             # otherwise spread out from the location to its neighbours
    #             nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
    #             for nbr_x, nbr_y in nbrs:
    #                 fringe.append((nbr_x, nbr_y, dist + 1))
    #         # no food found
    #         return None
    #
    #     from pacman import PacmanRules
    #
    #     pacmanPostion = state.getPacmanPosition()
    #     capsules = state.getCapsules()
    #     food = state.getFood()
    #     walls = state.getWalls()
    #     ghosts = state.getGhostPositions()
    #     ghostStates = state.getGhostStates()
    #     # distanceToGhost = closestItem(pacmanPostion, ghosts)
    #     # Location after Pacman takes the action
    #     x, y = state.getPacmanPosition()
    #     dx, dy = Actions.directionToVector(action)
    #     next_x, next_y = int(x + dx), int(y + dy)
    #     distanceFood = closestFood((next_x, next_y), food, walls)
    #     distanceCapsule = closestItem((next_x, next_y), capsules, walls) + 0.01
    #     distanceToGhost = closestItem((next_x, next_y), ghosts, walls) + 0.01
    #     features["Bias"] = 1.0
    #
    #     for ghost in ghostStates:
    #         if ghost.scaredTimer > 0:
    #             features["DistanceToClostestGhost"] = float(distanceToGhost) / (walls.width * walls.height)
    #             features["ScaredGhost1StepAway"] = sum(
    #                 (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
    #             if food[next_x][next_y]:
    #                 features["Food"] = 1.0
    #             if distanceFood is not None:
    #                 features["ClosestFood"] = float(distanceFood) / (walls.width * walls.height)
    #         else:
    #             features["Ghost1StepAway"] = sum(
    #                 (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
    #
    #             if distanceCapsule is not None:
    #                 features["DistanceToCapsule"] = 1.0 / (float(distanceCapsule))
    #
    #             if not features["Ghost1StepAway"] and food[next_x][next_y]:
    #                 features["Food"] = 1.0
    #
    #             if distanceFood is not None:
    #                 features["ClosestFood"] = float(distanceFood) / (walls.width * walls.height)
    #
    #             if len(PacmanRules.getLegalActions(state)) < 4:
    #                 features["Tunnel"] = 1.0
    #
    #     features.divideAll(10.0)
    #     return features
