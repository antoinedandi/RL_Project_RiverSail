import numpy as np
import sys
from six import StringIO, b
import scipy.stats as stat
import matplotlib.pyplot as plt

from gym import utils
from gym.envs.toy_text import discrete
import environments.discreteMDP
from gym import Env, spaces
import string
from environments.discreteMDP import Dirac


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


def riverSailMap(Y):
    maze = np.full((3, Y), 1.)
    return maze


# Upgrade of the previous class, walls are no longer visible as unaccessible states for the learner (they're no longer exisiting for the learner).
class riverSail(environments.discreteMDP.DiscreteMDP):
    metadata = {'render.modes': ['text', 'ansi', 'pylab', 'maze'], 'maps': ['random', '2-room', '4-room']}

    def __init__(self, sizeX, slippery=0.1, initialSingleStateDistribution=False, seed=None):
        """

        :param sizeX: length of the 2-d grid
        :param sizeY: height of the 2-d grid
        :param map_name: random, 2-room or 4-room
        :param slippery: real-value in [0,1], makes transitions more (1) or less (0) stochastic.
        :param nbGoals: number og goal states to be generated
        :param rewardStd: standard deviation of rewards.
        :param density: density of walls (for random map)
        :param lengthofwalks: average lengh of walls (for random map)
        :param initialSingleStateDistribution: True: the initial distribution is a Dirac at one state, chosen uniformly randomly amongts valid non-goal states; False: initial Distribution is uniform random amongst non-goal states.
        :param seed:
        """

        # desc = maps[map_name]
        self.sizeY, self.sizeX = sizeX, 3
        self.reward_range = (0, 1)
        self.rewardStd = 0.
        self.map_name = 'riversail'

        self.nA = 4
        self.nS_all = self.sizeX * self.sizeY
        self.nameActions = ["Up", "Down", "Left", "Right"]

        self.seed(seed)
        self.initializedRender = False

        """ A modifier pour faire le VENT """
        # stochastic transitions
        slip = min(slippery, 1. / 3.)
        self.massmap = [[slip, 1. - 3 * slip, slip, 0., slip],  # up : up down left right stay
                        [slip, 0., slip, 1. - 3 * slip, slip],  # down
                        [1. - 3 * slip, slip, 0., slip, slip],  # left
                        [0., slip, 1. - 3 * slip, slip, slip]]  # right

        self.maze = riverSailMap(self.sizeY)

        self.mapping = []
        self.revmapping = []  # np.zeros(sizeX*sizeY)
        cpt = 0
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                xy = self.to_s((x, y))
                if self.maze[x, y] >= 1:
                    self.mapping.append(xy)
                    self.revmapping.append((int)(cpt))
                    cpt = cpt + 1
                else:
                    self.revmapping.append((int)(-1))

        # print(self.revmapping)
        self.nS = len(self.mapping)

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.goalstates = self.makeGoalStates()

        if (initialSingleStateDistribution):
            isd = self.makeInitialSingleStateDistribution(self.maze)
        else:
            isd = self.makeInitialDistribution(self.maze)
        P = self.makeTransition(isd)
        R = self.makeRewards()

        self.P = P
        self.R = R
        self.isd = isd
        self.lastaction = None  # for rendering
        self.lastreward = 0.  # for rendering

        self.states = range(0, self.nS)
        self.actions = range(0, self.nA)
        self.nameActions = list(string.ascii_uppercase)[0:min(self.nA, 26)]

        self.reward_range = (0, 1)
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed(None)
        self.initializedRender = False
        self.reset()

    def to_s(self, rowcol):
        return rowcol[0] * self.sizeY + rowcol[1]

    def from_s(self, s):
        return s // self.sizeY, s % self.sizeY

    def step(self, a):
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

    def makeGoalStates(self):
        goalstates = []
        s = [1, self.sizeY - 1]
        goalstates.append(self.revmapping[self.to_s(s)])
        self.maze[s[0]][s[1]] = 2.
        return goalstates

    def makeInitialSingleStateDistribution(self, maze):
        xy = [1, 1]  # [np.random.randint(self.sizeX), np.random.randint(self.sizeY)]
        while (self.maze[xy[0]][xy[1]] != 1):
            xy = [self.np_random.randint(self.sizeX), self.np_random.randint(self.sizeY)]
        isd = np.zeros(self.nS)
        isd[self.revmapping[self.to_s(xy)]] = 1.
        return isd

    def makeInitialDistribution(self, maze):
        isd = np.ones(self.nS)
        for g in self.goalstates:
            isd[g] = 0
            # isd = np.array(maze == 1.).astype('float64').ravel()
        isd /= isd.sum()
        return isd

    def makeTransition(self, initialstatedistribution):
        X = self.sizeX
        Y = self.sizeY
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for s in range(self.nS):
            x, y = self.from_s(self.mapping[s])
            if (self.maze[x][y] == 2.):
                for a in range(self.nA):
                    li = P[s][a]
                    for ns in range(self.nS):
                        if (initialstatedistribution[ns] > 0):
                            li.append((initialstatedistribution[ns], ns, False))
            else:
                us = [(x - 1) % X, y % Y]
                ds = [(x + 1) % X, y % Y]
                ls = [x % X, (y - 1) % Y]
                rs = [x % X, (y + 1) % Y]
                ss = [x, y]
                if (self.maze[us[0]][us[1]] <= 0 or self.maze[x][y] <= 0): us = ss
                if (self.maze[ds[0]][ds[1]] <= 0 or self.maze[x][y] <= 0): ds = ss
                if (self.maze[ls[0]][ls[1]] <= 0 or self.maze[x][y] <= 0): ls = ss
                if (self.maze[rs[0]][rs[1]] <= 0 or self.maze[x][y] <= 0): rs = ss
                for a in range(self.nA):
                    li = P[s][a]
                    li.append((self.massmap[a][0], self.revmapping[self.to_s(ls)], False))
                    li.append((self.massmap[a][1], self.revmapping[self.to_s(us)], False))
                    li.append((self.massmap[a][2], self.revmapping[self.to_s(rs)], False))
                    li.append((self.massmap[a][3], self.revmapping[self.to_s(ds)], False))
                    li.append((self.massmap[a][4], self.revmapping[self.to_s(ss)], False))

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
            plt.imshow(self.maze, cmap='hot', interpolation='nearest')
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
            super(GridWorld, self).render(mode)

    def initRender(self):
        if (self.rendermode == 'maze'):
            self.numFigure = plt.gcf().number
            plt.figure(self.numFigure)
            plt.imshow(self.maze, cmap='hot', interpolation='nearest')
            plt.savefig('MDP-gridworld-' + self.map_name + '.png')
            plt.show(block=False)
            plt.pause(0.5)
        else:
            super(GridWorld, self).initRender()