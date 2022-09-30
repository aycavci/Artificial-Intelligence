# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for iteration in range(self.iterations):
            x = util.Counter()
            for state in self.mdp.getStates():
                x[state] = self.computeMaxValueFromValues(state)
            self.values = x

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        v = 0
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            v += transition[1] * (self.mdp.getReward(state, action, transition[0]) + self.discount * self.values[transition[0]])
        return v

    def computeMaxValueFromValues(self, state):
        if self.mdp.isTerminal(state):
            return 0
        else:
            v = float("-inf")
            for action in self.mdp.getPossibleActions(state):
                if v < self.computeQValueFromValues(state, action):
                    v = self.computeQValueFromValues(state, action)
            return v

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        else:
            v = float("-inf")
            best_action = None
            for action in self.mdp.getPossibleActions(state):
                if v < self.computeQValueFromValues(state, action):
                    v = self.computeQValueFromValues(state, action)
                    best_action = action
            return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for iteration in range(self.iterations):
            self.values[states[iteration % len(states)]] = self.computeMaxValueFromValues(states[iteration % len(states)])


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        dictionary = {}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                    if dictionary.get(transition[0]) is None:
                        dictionary[transition[0]] = [state]
                    else:
                        if state not in dictionary[transition[0]]:
                            dictionary[transition[0]].append(state)
        queue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                diff = abs(self.values[state] - self.computeMaxValueFromValues(state))
                queue.push(state, -diff)
        for iteration in range(self.iterations):
            if queue.isEmpty():
                break
            else:
                state = queue.pop()
                if not self.mdp.isTerminal(state):
                    self.values[state] = self.computeMaxValueFromValues(state)
                for predecessor in dictionary[state]:
                    diff = abs(self.values[predecessor] - self.computeMaxValueFromValues(predecessor))
                    if diff > self.theta:
                        queue.update(predecessor, -diff)







