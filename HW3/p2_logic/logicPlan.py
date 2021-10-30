# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game
from itertools import combinations

pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'


class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()


def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """

    "*** YOUR CODE HERE ***"
    A = logic.Expr("A")
    B = logic.Expr("B")
    C = logic.Expr("C")
    clause1 = A | B
    clause2 = ~A % (~B | C)
    clause3 = logic.disjoin(~A, ~B, C)
    return logic.conjoin(clause1, clause2, clause3)


def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr("A")
    B = logic.Expr("B")
    C = logic.Expr("C")
    D = logic.Expr("D")
    clause1 = C % (B | D)
    clause2 = A >> logic.conjoin(~B, ~D)
    clause3 = ~logic.conjoin(B, ~C) >> A
    clause4 = ~D >> C
    return logic.conjoin(clause1, clause2, clause3, clause4)


def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    WumpusAlive0 = logic.PropSymbolExpr("WumpusAlive", 0)
    WumpusAlive1 = logic.PropSymbolExpr("WumpusAlive", 1)
    WumpusBorn = logic.PropSymbolExpr("WumpusBorn", 0)
    WumpusKilled = logic.PropSymbolExpr("WumpusKilled", 0)
    clause1 = WumpusAlive1 % logic.disjoin(logic.conjoin(WumpusAlive0, ~WumpusKilled),
                                           logic.conjoin(~WumpusAlive0, WumpusBorn))
    clause2 = ~logic.conjoin(WumpusAlive0, WumpusBorn)
    clause3 = WumpusBorn
    return logic.conjoin(clause1, clause2, clause3)


def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"
    return logic.pycoSAT(logic.to_cnf(sentence))


def atLeastOne(literals):
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single 
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    for literal in literals:
        if literal:
            return logic.disjoin(literals)


def atMostOne(literals):
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    return logic.conjoin([logic.disjoin(~c[0], ~c[1]) for c in combinations(literals, 2)])


def exactlyOne(literals):
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    return logic.conjoin(atLeastOne(literals), atMostOne(literals))


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    parsed_models = map(lambda x: (logic.PropSymbolExpr.parseExpr(x[0]), x[1]), model.items())
    clean_parsed_models = filter(lambda x: type(x[0]) is tuple and x[0][0] in actions and x[1], parsed_models)
    plan = {int(p[1]): p[0] for p, v in clean_parsed_models}
    return [plan[k] for k in sorted(plan.keys())]


def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    "*** YOUR CODE HERE ***"
    possibilities = [logic.PropSymbolExpr(pacman_str, n_x, n_y, t - 1) & logic.PropSymbolExpr(action, t - 1)
                     for n_x, n_y, action in
                     [(x, y + 1, 'South'), (x, y - 1, 'North'), (x + 1, y, 'West'), (x - 1, y, 'East')]
                     if not walls_grid[n_x][n_y]]
    return logic.PropSymbolExpr(pacman_str, x, y, t) % logic.disjoin(possibilities)


def get_initial(width, height, initial_state):
    expression = None
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            if (x, y) == initial_state:
                if expression:
                    expression = logic.conjoin(expression, logic.PropSymbolExpr("P", x, y, 0))
                else:
                    expression = logic.Expr(logic.PropSymbolExpr("P", x, y, 0))
            else:
                if expression:
                    expression = logic.conjoin(expression, logic.Expr("~", logic.PropSymbolExpr("P", x, y, 0)))
                else:
                    expression = logic.Expr("~", logic.PropSymbolExpr("P", x, y, 0))
    return expression


def update_success_exclusion(success, exclusion, t, walls, actions, width, height):
    # get success
    suc = logic.conjoin([pacmanSuccessorStateAxioms(x, y, t, walls)
                         for x in range(1, width + 1) for y in range(1, height + 1)
                         if (x, y) not in walls.asList()])
    if success:
        success = logic.conjoin(suc, success)
    else:
        success = suc
    # get exclusion
    exc = exactlyOne([logic.PropSymbolExpr(action, t - 1) for action in actions])
    if exclusion:
        exclusion = logic.conjoin(exclusion, exc)
    else:
        exclusion = exc
    return success, exclusion


def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    actions = ['North', 'East', 'South', 'West']
    init_state = problem.getStartState()
    goal_state = problem.getGoalState()
    success = None
    exclusion = None
    initial = get_initial(width, height, init_state)

    goal = logic.conjoin(logic.PropSymbolExpr("P", goal_state[0], goal_state[1], 1),
                         pacmanSuccessorStateAxioms(goal_state[0], goal_state[1], 1, walls))
    founded_model = findModel(logic.conjoin(initial, goal))
    if founded_model:
        return extractActionSequence(founded_model, actions)
    t = 0
    while True:
        t += 1
        # update success exclusion
        success, exclusion = update_success_exclusion(success, exclusion, t, walls, actions, width, height)
        # get goal
        goal = logic.conjoin(logic.PropSymbolExpr("P", goal_state[0], goal_state[1], t + 1),
                             pacmanSuccessorStateAxioms(goal_state[0], goal_state[1], t + 1, walls))
        # find model and return
        founded_model = findModel(logic.conjoin(initial, goal, exclusion, success))
        if founded_model:
            return extractActionSequence(founded_model, actions)


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    "*** YOUR CODE HERE ***"
    actions = ['North', 'East', 'South', 'West']

    pacman_init_location, food_locations_grid = problem.getStartState()
    food_locations = food_locations_grid.asList()

    initial = get_initial(width, height, pacman_init_location)

    food_eaten = logic.conjoin([logic.PropSymbolExpr("P", x, y, 0)
                                for x, y in food_locations])
    founded_model = findModel(logic.conjoin(initial, food_eaten))
    if founded_model:
        return extractActionSequence(founded_model, actions)
    success = None
    exclusion = None
    t = 1
    while True:
        t += 1
        # update success, exclusion
        success, exclusion = update_success_exclusion(success, exclusion, t, walls, actions, width, height)
        # get food_eaten
        food_eaten = logic.conjoin(
            [logic.disjoin(
                [logic.PropSymbolExpr("P", x, y, i)
                 for i in range(0, t)])
                for x, y in food_locations])
        founded_model = findModel(logic.conjoin(initial, food_eaten, exclusion, success))
        if founded_model:
            return extractActionSequence(founded_model, actions)


# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
