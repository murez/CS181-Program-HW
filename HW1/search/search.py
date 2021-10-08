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
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    visiting_stack = util.Stack()
    visited = set()
    now_state = problem.getStartState()
    result_action = []
    visiting_stack.push((now_state, []))
    while (not visiting_stack.isEmpty()) and (not problem.isGoalState(now_state)):
        now_state, now_actions = visiting_stack.pop()
        if problem.isGoalState(now_state):
            return now_actions
        visited.add(now_state)
        successors = problem.getSuccessors(now_state)
        for next_state, action, cost in successors:
            if next_state not in visited:
                # visiting_stack.push((now_state, now_actions))
                result_action = now_actions + [action]
                visiting_stack.push((next_state, result_action))
                # break



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from collections import deque  # beat that util
    now_state = problem.getStartState()
    result_action = []
    visited = set()
    visiting_queue = deque([(now_state, [])])
    while visiting_queue:
        now_state, now_actions = visiting_queue.popleft()
        if problem.isGoalState(now_state):
            return now_actions
        if now_state not in visited:
            visited.add(now_state)
            successors = problem.getSuccessors(now_state)
            for next_state, action, cost in successors:
                if next_state not in visited:
                    result_action = now_actions + [action]
                    visiting_queue.append((next_state, result_action))
    return result_action


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    now_state = problem.getStartState()
    result_action = []
    visited = set()
    visiting_pq = util.PriorityQueue()
    visiting_pq.push((now_state, []), 0)
    while not visiting_pq.isEmpty():
        now_state, now_actions = visiting_pq.pop()
        if problem.isGoalState(now_state):
            return now_actions
        if now_state not in visited:
            visited.add(now_state)
            successors = problem.getSuccessors(now_state)
            for next_state, action, cost in successors:
                if next_state not in visited:
                    result_action = now_actions + [action]
                    visiting_pq.push((next_state, result_action), problem.getCostOfActions(result_action))
    return result_action


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    now_state = problem.getStartState()
    result_action = []
    visited = set()
    visiting_pq = util.PriorityQueue()
    visiting_pq.push((now_state, []), nullHeuristic(now_state))
    while not visiting_pq.isEmpty():
        now_state, now_actions = visiting_pq.pop()
        if problem.isGoalState(now_state):
            return now_actions
        if now_state not in visited:
            visited.add(now_state)
            successors = problem.getSuccessors(now_state)
            for next_state, action, cost in successors:
                if next_state not in visited:
                    result_action = now_actions + [action]
                    visiting_pq.push((next_state, result_action), problem.getCostOfActions(result_action)+heuristic(next_state, problem))
    return result_action


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
