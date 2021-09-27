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


import math

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        cur_pos = currentGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        current_ghost_states = currentGameState.getGhostStates()
        current_scared_times = [ghostState.scaredTimer for ghostState in current_ghost_states]

        # base score
        score = successorGameState.getScore()

        # ghost factors: near is good if scared, bad otherwise
        ghost_vars = zip(currentGameState.getGhostPositions(), current_scared_times)
        ghost_proximity_scale = 2
        for g in ghost_vars:
            if g[1] > 5:
                score += 100 / max(0.1, util.manhattanDistance(newPos, g[0]))
        ghost_factors = [(util.manhattanDistance(newPos, g[0]) * (ghost_proximity_scale * (-1 if g[1] > 0 else 1)))
                         for g in ghost_vars]
        score += 25 * math.log(max(1, sum(ghost_factors)), 2)

        # discourage staying in same place
        if newPos == currentGameState.getPacmanPosition():
            score -= 10

        # eating is good
        if newFood[newPos[0]][newPos[1]]:
            score += 20
        else:  # otherwise moving towards food is also good
            new_food_dists = [util.manhattanDistance(newPos, f) for f in newFood.asList()]
            if len(new_food_dists) == 0:
                new_food_dists = [0]
            current_food_dists = [util.manhattanDistance(cur_pos, f) for f in newFood.asList()]
            if len(current_food_dists) == 0:
                current_food_dists = [0]
            score += random.randint(1, 15) * (min(current_food_dists) - min(
                new_food_dists))  # randomness to avoid getting stuck between comparable states

        # eat capsule if near it anyways
        if sum([g[1] for g in ghost_vars]) == 0:
            capsule_vars = zip(currentGameState.getCapsules(),
                               [util.manhattanDistance(cur_pos, c) for c in currentGameState.getCapsules()])

            for c in capsule_vars:
                if c[1] < 5:
                    if util.manhattanDistance(newPos, c[0]) < util.manhattanDistance(cur_pos, c[0]):
                        score += 50

        return score

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
        """
        actions = gameState.getLegalActions(0) #pacman actions
        for i in range(len(actions)):
            action = actions[i]
            val = self.Minimax(gameState.generateSuccessor(0, action),0,1)
            if i is 0:
                bestAction = actions[0]
                bestVal = val
                #print "bestVal = " + str(bestVal)
                #print "bestAction = " + str(i)
            if val>bestVal:
                bestVal = val
                bestAction = action
                #print "bestVal = " + str(bestVal)
                #print "bestAction = " + str(i)
        return bestAction


    def Minimax(self, gameState, currentDepth, currentAgent):
        if currentDepth >= self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if currentAgent is 0: #pacman's turn
            nextActions = gameState.getLegalActions(currentAgent)
            values = []
            argMax=0
            if currentAgent >= gameState.getNumAgents()-1: #on the last ghost
                nextAgent = 0
                nextDepth = currentDepth + 1
            else:
                nextAgent = currentAgent + 1
                nextDepth = currentDepth
            for i in range(len(nextActions)):
                nextAction = nextActions[i]
                values.append(self.Minimax(gameState.generateSuccessor(currentAgent, nextAction),nextDepth, nextAgent))
                if i is 0:
                    valMax = values[0]
                if values[i]>valMax:
                    argMax = i
                    valMax = values[i]
            return valMax
        if currentAgent > 0:
            nextActions = gameState.getLegalActions(currentAgent)
            values = []
            argMin = 0
            #updating and tracking currentAgent
            if currentAgent >= gameState.getNumAgents()-1: #on the last ghost
                nextAgent = 0
                nextDepth = currentDepth + 1
            else:
                nextAgent = currentAgent + 1
                nextDepth = currentDepth
            for i in range(len(nextActions)):
                nextAction = nextActions[i]
                values.append(self.Minimax(gameState.generateSuccessor(currentAgent, nextAction),nextDepth, nextAgent))
                if i is 0:
                    valMin = values[0]
                if values[i]<valMin:
                    argMin = i
                    valMin = values[i]
            return valMin


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.AlphaBetaSearch(gameState)
    def AlphaBetaSearch(self, gameState):
        nextActions=gameState.getLegalActions(0)
        v, actionIndex = self.MaxValue(gameState,-999999,999999,0,0)
        return nextActions[actionIndex]
    def MaxValue(self, gameState, alpha, beta, currentDepth, currentAgent):
        if currentDepth>=self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = -999999
        nextActions = gameState.getLegalActions(currentAgent)
        if currentAgent >= gameState.getNumAgents() - 1:  # on the last ghost
            nextAgent = 0
            nextDepth = currentDepth
        else:
            nextAgent = currentAgent + 1
            nextDepth = currentDepth
        for i in range(len(nextActions)):
            nextAction = nextActions[i]
            possibleV = self.MinValue(gameState.generateSuccessor(currentAgent,nextAction), alpha, beta, nextDepth, nextAgent)
            if possibleV > v:
                v = possibleV
                actionIndex = i
            if v > beta:
                if currentDepth is 0:
                    return v, actionIndex
                return v
            alpha = max(alpha, v)
        if currentDepth is 0:
            return v, actionIndex
        return v
    def MinValue(self, gameState, alpha, beta, currentDepth, currentAgent):
        if currentDepth >= self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = 999999
        nextActions = gameState.getLegalActions(currentAgent)
        if currentAgent >= gameState.getNumAgents() - 1:  # on the last ghost
            nextAgent = 0
            nextDepth = currentDepth + 1
        else:
            nextAgent = currentAgent + 1
            nextDepth = currentDepth
        for i in range(len(nextActions)):
            nextAction = nextActions[i]
            if currentAgent >= gameState.getNumAgents() - 1:  # on the last ghost
                v = min(v, self.MaxValue(gameState.generateSuccessor(currentAgent, nextAction), alpha, beta, nextDepth, nextAgent))
            else:
                v = min(v, self.MinValue(gameState.generateSuccessor(currentAgent, nextAction), alpha, beta, nextDepth,
                                         nextAgent))
            if v < alpha:
                return v
            beta = min(beta,v)
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
        actions = gameState.getLegalActions(0) #pacman actions
        for i in range(len(actions)):
            action = actions[i]
            val = self.Expectimax(gameState.generateSuccessor(0, action),0,1)
            if i is 0:
                bestAction = actions[0]
                bestVal = val
            if val>bestVal:
                bestVal = val
                bestAction = action
        return bestAction

    def Expectimax(self, gameState, currentDepth, currentAgent):
        if currentDepth >= self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if currentAgent is 0: #pacman's turn
            nextActions = gameState.getLegalActions(currentAgent)
            values = []
            argMax=0
            '''if currentAgent >= gameState.getNumAgents()-1: #on the last ghost
                nextAgent = 0
                nextDepth = currentDepth + 1
            else:
                nextAgent = currentAgent + 1
                nextDepth = currentDepth'''  # know you're on Pacman's turn
            nextAgent = currentAgent + 1
            nextDepth = currentDepth
            for i in range(len(nextActions)):
                nextAction = nextActions[i]
                values.append(self.Expectimax(gameState.generateSuccessor(currentAgent, nextAction),nextDepth, nextAgent))
                if i is 0:
                    valMax = values[0]
                elif values[i]>valMax:
                    argMax = i
                    valMax = values[i]
            return valMax

        if currentAgent > 0:
            nextActions = gameState.getLegalActions(currentAgent)
            values = []
            arg_expected = 0
            #updating and tracking currentAgent
            if currentAgent >= gameState.getNumAgents()-1: #on the last ghost
                nextAgent = 0
                nextDepth = currentDepth + 1
            else:
                nextAgent = currentAgent + 1
                nextDepth = currentDepth
            for i in range(len(nextActions)):
                nextAction = nextActions[i]
                values.append(self.Expectimax(gameState.generateSuccessor(currentAgent, nextAction),nextDepth, nextAgent))
            avg = 0
            for v in values:
                avg += float(v)/len(values)
            return avg

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

