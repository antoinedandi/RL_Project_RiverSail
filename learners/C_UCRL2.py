import numpy as np
import random as rd
import copy as cp
from learners.utils import *


class C_UCRL2:
    def __init__(self, nS, nA, env, classes, delta):
        """
        Class-UCRL2
        """
        self.nS = nS
        self.nA = nA
        self.nC = len(classes)

        self.t = 1
        self.delta = delta

        self.env = env.env
        self.classes = classes
        self.stateToClasses = np.zeros((self.nS, self.nA), dtype=int)
        for i, c in enumerate(classes):
            for s, a in c:
                self.stateToClasses[s, a] = i

        self.observations = [[], [], []]  # list of the observed (states, actions, rewards) ordered by time
        self.vk = np.zeros((self.nS, self.nA))  # the state-action count for the current episode k
        self.Nk = np.zeros((self.nS, self.nA))  # the state-action count prior to episode k

        self.ClassNk = np.zeros(self.nC)
        self.ClassVk = np.zeros(self.nC)

        self.r_distances = np.zeros((self.nS, self.nA))
        self.p_distances = np.zeros((self.nS, self.nA))

        self.class_r_distances = np.zeros((self.nC))
        self.class_p_distances = np.zeros((self.nC))

        self.Pk = np.zeros((self.nS, self.nA, self.nS))
        self.Rk = np.zeros((self.nS, self.nA))

        self.u = np.zeros(self.nS)
        self.span = []
        self.policy = np.zeros((self.nS, self.nA)) # policy, seen as a stochastic policy here.
        for s in range(self.nS):
            for a in range(self.nA):
                self.policy[s, a] = 1. / self.nA

    def name(self):
        return "C_UCRL2"

    # Auxiliary function to update N the current state-action count.
    def updateN(self):
        for s in range(self.nS):
            for a in range(self.nA):
                self.Nk[s, a] += self.vk[s, a]

        # Auxiliary function to update N the current state-action count.
    def updateClassN(self):
        for c in range(self.nC):
            self.ClassNk[c] += self.ClassVk[c]

    # Auxiliary function to update R the accumulated reward.
    def updateR(self):
        self.Rk[self.observations[0][-2], self.observations[1][-1]] += self.observations[2][-1]

    # Auxiliary function to update P the transitions count.
    def updateP(self):
        self.Pk[self.observations[0][-2], self.observations[1][-1], self.observations[0][-1]] += 1

    # Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
    def class_distances(self):
        for c in range(self.nC):
            n = max(self.ClassNk[c], 1)

            self.class_r_distances[c] = np.sqrt(((1 + 1 / n) * np.log(4 * np.sqrt(n + 1) * self.nC / self.delta) )/ n)
            self.class_p_distances[c] = np.sqrt((2 * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) * (2**2 - 2) * self.nC / self.delta) )/ n)
            # self.class_p_distances[c] = np.sqrt((2 * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) * (2 ** self.nC - 2) * self.nC / self.delta)) / n)

    # Computing the maximum proba in the Extended Value Iteration for given vlass c.
    def class_max_proba(self, p_estimate, sorted_indices, c):
        min1 = min([1, p_estimate[sorted_indices[-1]] + (self.class_p_distances[c] / 2)])
        max_p = np.zeros(self.nS)
        if min1 == 1:
            max_p[sorted_indices[-1]] = 1
        else:
            max_p = cp.deepcopy(p_estimate)
            max_p[sorted_indices[-1]] += self.class_p_distances[c] / 2
            l = 0
            while sum(max_p) > 1:
                max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])  # Error?
                l += 1
        return max_p

    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def class_EVI(self, r_estimate, p_estimate, epsilon=0.01, max_iter=1000):
        u0 = self.u - min(self.u)  # sligthly boost the computation and doesn't seems to change the results
        u1 = np.zeros(self.nS)
        sorted_indices = np.arange(self.nS)
        niter = 0
        while True:
            niter += 1

            for s in range(self.nS):
                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    c = self.stateToClasses[s, a]
                    invSigma = np.argsort(self.env.sigma[s, a])
                    max_p = self.class_max_proba(p_estimate[c, invSigma], sorted_indices, c)
                    temp[a] = min((1, r_estimate[c] + self.class_r_distances[c])) + sum(
                        [u * p for (u, p) in zip(u0, max_p)])
                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] for a in arg]
                (nmax, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.nA)]

            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1 - min(u1)
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.nS)
                sorted_indices = np.argsort(u0)
            if niter > max_iter:
                self.u = u1 - min(u1)
                print("No convergence in EVI")
                break

    # To start a new episode (init var, computes estmates and run EVI).
    def new_episode(self):
        self.updateN()
        self.updateClassN()
        self.vk = np.zeros((self.nS, self.nA))
        self.ClassVk = np.zeros(self.nC)

        class_p_estimate = np.zeros((self.nC, self.nS))
        class_r_estimate = np.zeros(self.nC)

        for c in range(self.nC):
            for s, a in self.classes[c]:
                class_r_estimate[c] += self.Rk[s, a] / max(1, self.ClassNk[c])
                class_p_estimate[c] += self.Pk[s, a, self.env.sigma[s, a]]  / max(1, self.ClassNk[c])
        self.class_distances()
        self.class_EVI(class_r_estimate, class_p_estimate, epsilon=1. / max(1, self.t))

    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.ClassVk = np.zeros(self.nC)
        self.Nk = np.zeros((self.nS, self.nA))
        self.ClassNk = np.zeros(self.nC)
        self.u = np.zeros(self.nS)
        self.Pk = np.zeros((self.nS, self.nA, self.nS))
        self.Rk = np.zeros((self.nS, self.nA))
        self.span = [0]
        for s in range(self.nS):
            for a in range(self.nA):
                self.policy[s, a] = 1. / self.nA
        self.new_episode()

    # To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
    def play(self, state):

        action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        c = self.stateToClasses[state, action]
        if self.ClassVk[c] >= max([1, self.ClassNk[c]]):  # Stopping criterion
            self.new_episode()
            action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        return action

    # To update the learner after one step of the current policy.
    def update(self, state, action, reward, observation):
        c = self.stateToClasses[state, action]
        self.vk[state, action] += 1
        self.ClassVk[c] += 1
        self.observations[0].append(observation)
        self.observations[1].append(action)
        self.observations[2].append(reward)
        self.updateP()
        self.updateR()
        self.t += 1
