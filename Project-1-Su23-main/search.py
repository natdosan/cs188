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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
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
    start_state = problem.getStartState()
    stack = util.Stack()
    stack.push((start_state, [])) # (state, actions)

    # we keep a set so we avoid duplications
    visited = set()
    while not stack.isEmpty():
        # get node from top of stack
        state, actions = stack.pop()
        # if valid goal state return the path
        if problem.isGoalState(state):
            return actions
        # avoid expanding already visited states
        if state not in visited:
            visited.add(state)
            # get succeeding states
            successors = problem.getSuccessors(state)
            for successor in successors:
                next_state, action = successor[0], successor[1]
                next_actions = actions + [action]
                # add next set of (state, action) by expanding greedily
                stack.push((next_state, next_actions))

    # If no goal state found
    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    queue = util.Queue()
    queue.push((start_state, [])) # (state, actions)

    # we keep a set so we avoid duplications
    visited = set()
    while not queue.isEmpty():
        # get node from top of stack
        state, actions = queue.pop()
        # if valid goal state return the path
        if problem.isGoalState(state):
            return actions
        # avoid expanding already visited states
        if state not in visited:
            visited.add(state)
            # get succeeding states
            successors = problem.getSuccessors(state)
            for successor in successors:
                next_state, action = successor[0], successor[1]
                next_actions = actions + [action]
                # add next set of (state, action) by expanding greedily
                queue.push((next_state, next_actions))

    # If no goal state found
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    priority_queue = util.PriorityQueue()
    # (state, actions), 0 initial cost
    priority_queue.push((start_state, []), 0) 

    # we keep a set so we avoid duplications
    visited = set()
    while not priority_queue.isEmpty():
        # get node from top of stack
        state, actions = priority_queue.pop()
        # if valid goal state return the path
        if problem.isGoalState(state):
            return actions
        # avoid expanding already visited states
        if state not in visited:
            visited.add(state)
            # get succeeding states
            successors = problem.getSuccessors(state)
            for successor in successors:
                next_state, action = successor[0], successor[1]
                next_actions = actions + [action]
                curr_cost = problem.getCostOfActions(next_actions)
                priority_queue.push((next_state, next_actions), curr_cost)

    # If no goal state found
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    priority_queue = util.PriorityQueue()
    # (state, actions), 0 initial cost
    priority_queue.push((start_state, [], 0), 0) 

    # instead we use a dictionary to keep track of cost
    visited = {}
    while not priority_queue.isEmpty():
        # get node from top of stack
        state, actions, cost = priority_queue.pop()
        # if valid goal state
        if problem.isGoalState(state):
            return actions
        # avoid expanding already visited states
        if state not in visited or cost < visited[state]:
            # instead of adding to set, we create a state : cost pair
            visited[state] = cost

            # get succeeding states
            successors = problem.getSuccessors(state)
            for successor in successors:
                next_state, action = successor[0], successor[1]
                next_actions = actions + [action]
                # get the cost and then estimate cost from next_state to 
                # the goal state
                next_cost = problem.getCostOfActions(next_actions)
                priority = next_cost + heuristic(next_state, problem)
                priority_queue.push((next_state, next_actions, next_cost), priority)

    # If no goal state found
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
