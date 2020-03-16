import numpy as np
import sys
from six import StringIO, b
import scipy.stats as stat
import matplotlib.pyplot as plt
import random

from gym import utils
import environments.discreteMDP
from gym import Env, spaces
from environments.discreteMDP import Dirac


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


def riverSailMap(X):
    maze = np.full((3, X), 1.)
    return maze


class RiverSail(environments.discreteMDP.DiscreteMDP):
    metadata = {'render.modes': ['maze']}

    def __init__(self, sizeX, wind=0.3, seed=None):
        """

        :param sizeX: length of the 2-d grid
        :param wind: real-value in [0,1], wind force - makes transitions more (1) or less (0) stochastic.
        :param rewardStd: standard deviation of rewards.
        :param initialSingleStateDistribution: True: the initial distribution is a Dirac at one state, chosen uniformly randomly amongts valid non-goal states; False: initial Distribution is uniform random amongst non-goal states.
        :param seed: seed for reproducibillity purpose
        """

        # desc = maps[map_name]
        self.sizeX, self.sizeY = sizeX, 3
        self.wind = wind
        self.reward_range = (0, 1)
        self.rewardStd = 0.  # TODO : is this useful ??
        self.map_name = 'riversail'

        self.nA = 4
        self.nS_all = self.sizeY * self.sizeX
        self.nameActions = ["up", "right", "down", "left"]

        self.seed(seed)
        self.initializedRender = False

        # Wind is chosen randomly among the 8 possible directions
        self.directions = ['up', 'up_right', 'right', 'down_right', 'down', 'down_left', 'left', 'up_left', 'no_wind']

        # define stochastic transitions of the wind effect
        self.wind_direction = random.choice(range(len(self.directions)))
        self.massmap = [0., 0., 0., 0., 0., 0., 0., 0., 1 - self.wind]  # u_ ur _r dr d_ dl _l ul stay
        self.massmap[self.wind_direction] += self.wind

        self.maze = riverSailMap(self.sizeX)

        self.mapping = []
        self.revmapping = []  # np.zeros(sizeX*sizeY)
        cpt = 0
        for x in range(self.sizeY):
            for y in range(self.sizeX):
                xy = self.to_s((x, y))
                if self.maze[x, y] >= 1:
                    self.mapping.append(xy)
                    self.revmapping.append((int)(cpt))
                    cpt = cpt + 1
                else:
                    self.revmapping.append((int)(-1))

        self.classes = self.getClasses()

        self.nS = len(self.mapping)

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        # Define the state that will lead to a reward
        self.goalstates = self.makeGoalStates()

        self.isd = self.makeInitialSingleStateDistribution()

        self.P = self.makeTransition(self.isd)
        self.R = self.makeRewards()
        self.sigma = self.getSigma()
        self.lastaction = None  # for rendering
        self.lastreward = 0.  # for rendering

        self.states = range(0, self.nS)
        self.actions = range(0, self.nA)

        self.reward_range = (0, 1)
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed(None)
        self.initializedRender = False
        self.reset()

    def to_s(self, rowcol):
        return rowcol[0] * self.sizeX + rowcol[1]

    def from_s(self, s):
        return s // self.sizeX, s % self.sizeX

    def step(self, a):
        # Update river and wind direction if we reach one end of the river
        X = self.sizeX
        y, x = self.from_s(self.mapping[self.s])
        if x == 0 or x == X - 1:
            self.change_river_and_wind()
            # print(self.sigma[7, 1])  # testing
        # Perform step
        transitions = self.P[self.s][a]
        rewarddis = self.R[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, d = transitions[i]
        r = rewarddis.rvs()
        m = rewarddis.mean()
        self.s = s
        self.lastaction = a
        self.lastreward = r
        return (s, r, d, m)

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s

    def getClasses(self):

        # There are 20 equivalence classes for RiverSail env as shown on the report

        classes = []

        state_0 = [self.to_s([0, 0]), self.to_s([1, 0]), self.to_s([2, 0]),
                   self.to_s([0, self.sizeX-1]), self.to_s([2, self.sizeX-1])]

        state_1 = [self.to_s([1, self.sizeX-1])]

        states_top = [self.to_s([0, i]) for i in range(self.sizeX - 1) if (0 < i < self.sizeX - 1)]
        states_mid = [self.to_s([1, i]) for i in range(self.sizeX - 1) if (0 < i < self.sizeX - 1)]
        states_low = [self.to_s([2, i]) for i in range(self.sizeX - 1) if (0 < i < self.sizeX - 1)]

        # extremities classes
        classes.append([(s, a) for s in state_0 for a in range(self.nA)])
        classes.append([(s, a) for s in state_1 for a in range(self.nA)])
        # top row classes
        classes.append([(s, 0) for s in states_top])
        classes.append([(s, 1) for s in states_top if (self.from_s(s)[1] < self.sizeX - 2)])
        classes.append([(s, 2) for s in states_top])
        classes.append([(s, 3) for s in states_top if (self.from_s(s)[1] > 1)])
        # middle row classes
        classes.append([(s, 0) for s in states_mid])
        classes.append([(s, 1) for s in states_mid if (self.from_s(s)[1] < self.sizeX - 2)])
        classes.append([(s, 2) for s in states_mid])
        classes.append([(s, 3) for s in states_mid if (self.from_s(s)[1] > 1)])
        # bottom row classes
        classes.append([(s, 0) for s in states_low])
        classes.append([(s, 1) for s in states_low if (self.from_s(s)[1] < self.sizeX - 2)])
        classes.append([(s, 2) for s in states_low])
        classes.append([(s, 3) for s in states_low if (self.from_s(s)[1] > 1)])
        # other classes
        classes.append([(s, 3) for s in states_top if (self.from_s(s)[1] == 1)])
        classes.append([(s, 3) for s in states_mid if (self.from_s(s)[1] == 1)])
        classes.append([(s, 3) for s in states_low if (self.from_s(s)[1] == 1)])
        classes.append([(s, 1) for s in states_top if (self.from_s(s)[1] == self.sizeX - 2)])
        classes.append([(s, 1) for s in states_mid if (self.from_s(s)[1] == self.sizeX - 2)])
        classes.append([(s, 1) for s in states_low if (self.from_s(s)[1] == self.sizeX - 2)])

        return classes

    def getSigma(self):
        # shape of sigma : nS * nA * nS
        P = self.P
        sigma = np.zeros((3 * self.sizeX, self.nA, 3 * self.sizeX), dtype=int)

        for s in range(self.nS):
            for a in range(self.nA):
                temp_1 = [y[1] for x, y in sorted(enumerate(P[s][a]), key=lambda x: x[1][0], reverse=True) if y[0] > 0.0]
                temp_2 = []
                for v in temp_1:
                    if v not in temp_2:
                        temp_2.append(v)
                for i in range(self.nS):
                    if i not in temp_2:
                        temp_2.append(i)
                sigma[s, a] = np.array(temp_2)

        return sigma

    def makeGoalStates(self):
        goalstates = []
        s = [1, self.sizeX - 1]
        goalstates.append(self.revmapping[self.to_s(s)])
        self.maze[s[0]][s[1]] = 2.
        return goalstates

    def makeInitialSingleStateDistribution(self):
        # put the boat at the entrance of the river
        yx = [1, 3]
        isd = np.zeros(self.nS)
        isd[self.revmapping[self.to_s(yx)]] = 1.
        return isd

    def change_river_and_wind(self):
        # new river means new wind direction
        self.wind_direction = random.choice(range(len(self.directions)))
        self.massmap = [0., 0., 0., 0., 0., 0., 0., 0., 1 - self.wind]  # u_ ur _r dr d_ dl _l ul stay
        self.massmap[self.wind_direction] += self.wind
        self.P = self.makeTransition(self.isd)
        self.R = self.makeRewards()
        self.sigma = self.getSigma()

    def compose_state_action(self, s, a):
        X = self.sizeX
        Y = self.sizeY
        y, x = s
        # get neighbouring states corresponding to the actions
        us = [max((y - 1), 0), x % X]         # up
        rs = [y % Y, min(x + 1, X - 1)]       # right
        ds = [min((y + 1), Y - 1), x % X]     # down
        ls = [y % Y, max(x - 1, 0)]           # left
        states = [us, rs, ds, ls]
        return states[a]

    def makeTransition(self, initialstatedistribution):
        X = self.sizeX
        Y = self.sizeY
        # P: Transitions (P : (S x A x S) -> [0,1])
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for s in range(self.nS):
            y, x = self.from_s(self.mapping[s])

            if (self.maze[y][x] == 2.):  # reached goal stage
                for a in range(self.nA):
                    li = P[s][a]
                    for ns in range(self.nS):
                        if (initialstatedistribution[ns] > 0):
                            li.append((initialstatedistribution[ns], ns, False))
            if (x == 0 or x == X - 1):  # reached one end of the river
                for a in range(self.nA):
                    li = P[s][a]
                    for ns in range(self.nS):
                        if (initialstatedistribution[ns] > 0):
                            li.append((initialstatedistribution[ns], ns, False))
            else:
                # Get neighbouring states
                u_ = [max((y - 1), 0), x % X]                    # up
                ur = [max((y - 1), 0), min(x + 1, X - 1)]        # up right
                _r = [y % Y, min(x + 1, X - 1)]                  # right
                dr = [min((y + 1), Y - 1), min(x + 1, X - 1)]    # down right
                d_ = [min((y + 1), Y - 1), x % X]                # down
                dl = [min((y + 1), Y - 1), max(x - 1, 0)]        # down left
                _l = [y % Y, max(x - 1, 0)]                      # left
                ul = [max((y - 1), 0), max(x - 1, 0)]            # up left
                ss = [y % Y, x % X]                              # stay
                for a in range(self.nA):
                    li = P[s][a]
                    li.append((self.massmap[0], self.revmapping[self.to_s(self.compose_state_action(u_, a))], False))
                    li.append((self.massmap[1], self.revmapping[self.to_s(self.compose_state_action(ur, a))], False))
                    li.append((self.massmap[2], self.revmapping[self.to_s(self.compose_state_action(_r, a))], False))
                    li.append((self.massmap[3], self.revmapping[self.to_s(self.compose_state_action(dr, a))], False))
                    li.append((self.massmap[4], self.revmapping[self.to_s(self.compose_state_action(d_, a))], False))
                    li.append((self.massmap[5], self.revmapping[self.to_s(self.compose_state_action(dl, a))], False))
                    li.append((self.massmap[6], self.revmapping[self.to_s(self.compose_state_action(_l, a))], False))
                    li.append((self.massmap[7], self.revmapping[self.to_s(self.compose_state_action(ul, a))], False))
                    li.append((self.massmap[8], self.revmapping[self.to_s(self.compose_state_action(ss, a))], False))

        return P

    def makeRewards(self):
        R = {s: {a: Dirac(0.) for a in range(self.nA)} for s in range(self.nS)}

        for s in range(self.nS):
            x, y = self.from_s(self.mapping[s])
            if (self.maze[x][y] == 2.):
                for a in range(self.nA):
                    if (self.rewardStd > 0):
                        ma, mb = (0 - 1.) / self.rewardStd, (1 - 1.) / self.rewardStd
                        R[s][a] = stat.truncnorm(ma, mb, loc=1., scale=self.rewardStd)
                    else:
                        R[s][a] = Dirac(1.)
        return R

    def getTransition(self, s, a):
        transition = np.zeros(self.nS)
        for c in self.P[s][a]:
            p, ss, isA = c
            transition[ss] += p
        return transition

    def render(self, mode='text'):
        self.rendermode = mode
        if (mode == 'maze'):
            if (not self.initializedRender):
                self.initRender()
                self.initializedRender = True

            plt.figure(self.numFigure)
            row, col = self.from_s(self.mapping[self.s])
            v = self.maze[row][col]
            self.maze[row][col] = 1.5
            print('wind : {}'.format(self.directions[self.wind_direction]))
            plt.imshow(self.maze, cmap='coolwarm', interpolation='nearest')
            plt.title('wind : {}'.format(self.directions[self.wind_direction]), loc='left')
            plt.axis('off')
            self.maze[row][col] = v
            plt.show(block=False)
            plt.pause(0.01)
        elif (mode == 'text') or (mode == 'ansi'):
            outfile = StringIO() if mode == 'ansi' else sys.stdout

            symbols = {0.: 'X', 1.: '.', 2.: 'G'}
            desc = [[symbols[c] for c in line] for line in self.maze]
            row, col = self.from_s(self.mapping[self.s])
            desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
            if self.lastaction is not None:
                outfile.write("  ({})\n".format(self.nameActions[self.lastaction]))
            else:
                outfile.write("\n")
            outfile.write("\n".join(''.join(line) for line in desc) + "\n")

            if mode != 'text':
                return outfile
        else:
            super(RiverSail, self).render(mode)

    def initRender(self):
        if (self.rendermode == 'maze'):
            self.numFigure = plt.gcf().number
            plt.figure(self.numFigure)
            plt.imshow(self.maze, cmap='hot', interpolation='nearest')
            plt.savefig('MDP-gridworld-' + self.map_name + '.png')
            plt.show(block=False)
            plt.pause(0.5)
        else:
            super(RiverSail, self).initRender()
