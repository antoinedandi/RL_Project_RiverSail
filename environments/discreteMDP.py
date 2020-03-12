import numpy as np
import sys
from six import StringIO
from gym import Env, spaces
from gym.utils import seeding
from gym import utils

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, Circle
import string


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class DiscreteMDP(Env):

    """
    Parameters
    - nS: number of states
    - nA: number of actions
    - P: transition distributions (*)
    - R: reward distributions (*)
    - isd: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, done), ...]
      R[s][a] == distribution(mean,param)
       One can sample R[s][a] using R[s][a].rvs()
    (**) list or array of length nS


    """

    metadata = {'render.modes': ['text', 'ansi', 'pylab', 'maze']}

    def __init__(self, nS, nA, P, R, isd,nameActions=[],seed=None):
        self.nS = nS
        self.nA = nA
        self.P = P
        self.R = R
        self.isd = isd
        self.reward_range = (0, 1)

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)


        # Rendering parameters and variables:
        self.lastaction=None
        self.lastreward=0.
        if(len(nameActions)==0):
            self.nameActions = list(string.ascii_uppercase)[0:min(self.nA,26)]
        self.states = range(0,self.nS)
        self.actions = range(0,self.nA)
        self.initializedRender=False
        self.rendermode = ''


        # Initialization
        self.seed(seed)
        self.reset()



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        return self.s

    def step(self, a):
        """

        :param a: action
        :return:  (state, reward, IsDone?, meanreward)
        The meanreward is returned for information, it should not be given to the learner.
        """
        transitions = self.P[self.s][a]
        rewarddis = self.R[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, d= transitions[i]
        r =  rewarddis.rvs()
        m = rewarddis.mean()
        self.s = s
        self.lastaction=a
        self.lastreward=r
        return (s, r, d, m)

    def getTransition(self,s,a):
        transition = np.zeros(self.nS)
        for c in self.P[s][a]:
            transition[c[1]]=c[0]
        return transition
    

    def getMeanReward(self, s, a):
        rewarddis = self.R[s][a]
        r =  rewarddis.mean()
        return r


    def initRender(self):
        if (self.rendermode=='pylab'):
            self.numFigure = plt.gcf().number
            plt.figure(self.numFigure)
            scalearrow = self.nS#(int) (np.sqrt(self.nS))
            scalepos =  10*self.nS#*self.nS
            G = nx.MultiDiGraph(action=0, rw=0.)
            for s in self.states:
                for a in self.actions:
                    for ssl in self.P[s][a]: #ssl = (p(s),s, 'done')
                            G.add_edge(s, ssl[1], action=a, weight=ssl[0], rw=self.R[s][a].mean())

            # Other possible laoyouts:
            #pos = nx.shell_layout(G)
            #pos = nx.spectral_layout(G)
            #pos = nx.fruchterman_reingold_layout(G)
            #pos = nx.kamada_kawai_layout(G)
            pos=nx.spring_layout(G)


            for x in self.states:
                pos[x] = [pos[x][0] * scalepos, pos[x][1] * scalepos]
            self.G = G
            self.pos = pos

            colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                      'tab:olive', 'tab:cyan']
            plt.clf()
            ax = plt.gca()

            nx.draw_networkx_nodes(G, pos, node_size=400,
                                   node_color=['tab:gray' if s != self.s else 'tab:orange' for s in self.G.nodes()])
            nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

            for n in G:
                c = Circle(pos[n], radius=0.2, alpha=0.)
                ax.add_patch(c)
                G.node[n]['patch'] = c
            counts = np.zeros((self.nS, self.nS))
            countsR =  np.zeros((self.nS, self.nS))
            seen = {}
            for u, v, d in G.edges(data=True):
                n1 = G.node[u]['patch']
                n2 = G.node[v]['patch']
                rad = 0.1
                if (u, v) in seen:
                    rad = seen.get((u, v))
                    rad = (rad + np.sign(rad) * 0.1) * -1
                alpha = d['weight']
                color = colors[d['action']]
                if alpha > 0:
                    counts[u][v] = counts[u][v] + 1
                    if (u != v):
                        e = FancyArrowPatch(n1.center, n2.center, patchA=n1, patchB=n2,
                                            arrowstyle='-|>',
                                            connectionstyle='arc3,rad=%s' % rad,
                                            mutation_scale=15.0 + scalearrow,
                                            lw=2,
                                            alpha=alpha,
                                            color=color)
                        seen[(u, v)] = rad
                        ax.add_patch(e)
                        if (d['rw'] > 0):
                            countsR[u][v] = countsR[u][v] + 1
                            nx.draw_networkx_edge_labels([u, v, d], pos,
                                                         edge_labels=dict([((u, v), str(np.ceil(d['rw'] * 100) / 100))]),
                                                         label_pos=0.5 + 0.1 * countsR[u][v], font_color=color, alpha=alpha,
                                                         font_size=8)

                    else:
                        n1c = [n1.center[0] + 0.1 * (2 * counts[u][v] + scalearrow),
                               n1.center[1] + 0.1 * (2 * counts[u][v] + scalearrow)]
                        e1 = FancyArrowPatch(n1.center, n1c,
                                             arrowstyle='-|>',
                                             connectionstyle='arc3,rad=1.',
                                             mutation_scale=15.0 + scalearrow,
                                             lw=2,
                                             alpha=alpha,
                                             color=color)
                        e2 = FancyArrowPatch(n1c, n1.center,
                                             arrowstyle='-|>',
                                             connectionstyle='arc3,rad=1.',
                                             mutation_scale=15.0 + scalearrow,
                                             lw=2,
                                             alpha=alpha,
                                             color=color
                                             )
                        ax.add_patch(e1)
                        ax.add_patch(e2)
                        if (d['rw'] > 0):
                            countsR[u][v] = countsR[u][v] + 1
                            pos[u] = [pos[u][0] + 0.1 * (2 * countsR[u][v] + scalearrow),
                                      pos[u][1] + 0.1 * (2 * countsR[u][v] + scalearrow)]
                            nx.draw_networkx_edge_labels([u, v, d], pos,
                                                         edge_labels=dict(
                                                             [((u, v), str(np.ceil(d['rw'] * 100) / 100))]),
                                                         label_pos=0.5, font_color=color,
                                                         alpha=alpha, font_size=8)
                            pos[u] = [pos[u][0] - 0.1 * (2 * countsR[u][v] + scalearrow),
                                      pos[u][1] - 0.1 * (2 * countsR[u][v] + scalearrow)]

            ax.autoscale()
            plt.axis('equal')
            plt.axis('off')
            plt.savefig('MDP-discrete.png')
            plt.show(block=False)
            plt.pause(0.5)
        if (self.rendermode == 'pydot'):

            G = pydot.Dot(graph_type='digraph')
            #colors = ['green', 'red', 'blue', 'orange', 'purple']
            colors = ['#FF0000', '#00FF00','#0000FF','#888800','#008888', '#880088', '#555555']



            nodes= []
            for s in self.states:
                fillcolor="#AAAAAA"
                if(s==self.s):
                    fillcolor="#FFAA00"
                nodes.append(pydot.Node(str(s),style="filled",fillcolor=fillcolor))
                G.add_node(nodes[-1])
            for s in self.states:
                for a in self.actions:
                    for ssl in self.P[s][a]: #ssl = (p(s),s, 'done')
                        label =""
                        if (self.R[s][a].mean()>0):
                            label = str("{:01.2f}".format(self.R[s][a].mean()))
                        col = colors[a% len(colors)]
                        r1=int(col[1:3],16)
                        g1=int(col[3:5],16)
                        b1=int(col[5:7],16)
                        #print("a:",a,r1,g1,b1,ssl[0])
                        r2 = (hex((int) (r1*(ssl[0]))))[2:]
                        g2 = (hex((int) (g1*(ssl[0]))))[2:]
                        b2 = (hex((int) (b1*(ssl[0]))))[2:]
                        if (len(r2)==1):
                            r2='0'+r2[0]
                        if (len(g2)==1):
                            g2='0'+g2[0]
                        if (len(b2)==1):
                            b2='0'+b2[0]
                        col = col[0]+r2+g2+b2
                        #print("a:",a,col)
                        e= pydot.Edge(nodes[s],nodes[ssl[1]], label=label, color=col, weight=ssl[0])
                        G.add_edge(e)
            self.G=G
            #G_str=G.create_png(prog='dot')
            #G_im = IPython.display.Image(G_str)
            #IPython.display(G_im)
            #display(pl)
            #print(pydot.EDGE_ATTRIBUTES)
            G.write_png('MDP-pydot-rendering.png')







    def render(self, mode='pylab'):
        self.rendermode=mode
        if (mode=="text"):
            #Print the MDp in text mode.
            # Red  = current state
            # Blue = all states accessible from current state (by playing some action)
            outfile = StringIO() if mode == 'ansi' else sys.stdout

            desc = [str(s)  for s in self.states]


            desc[self.s] = utils.colorize(desc[self.s], "red", highlight=True)
            for a in self.actions:
                for ssl in self.P[self.s][a]:
                    if (ssl[0]>0):
                        desc[ssl[1]] = utils.colorize(desc[ssl[1]], "blue", highlight=True)

            desc.append(" \t\tr="+str(self.lastreward))

            if self.lastaction is not None:
                outfile.write("  ({})\t".format(self.nameActions[self.lastaction % 26]))
            else:
                outfile.write("\n")
            outfile.write("".join(''.join(line) for line in desc) + "\n")

            if mode != 'text':
                return outfile
        elif(mode=="pydot"):
            if (not self.initializedRender):
                self.initRender()
                self.initializedRender=True
            for s in self.states:
                fillcolor="#AAAAAA"
                if(s==self.s):
                    fillcolor="#FFAA00"
                n=(self.G.get_node(str(s)))[0]
                #print(n, "(before) ")
                n.set("fillcolor",fillcolor)
                #print(n,"(after)")
            self.G.write_png('MDP-pydot-rendering.png')
            #plt.pause(3)
        else:
            """            
            # Print the MDP in an image MDP.png, MDP.pdf
            # Node colors : orange = current state, gray = other states
            # Edge colors : the color indicates the corresponding action (e.g. blue= action 0, red = action 1, etc)
            # Edge transparency: indicates the probability with which we transit to that state.
            # Edge label: A label indicates a positive reward, with mean value given by the labal (color of the label = action)
            # Print also the MDP only shoinwg the rewards in MDPonlytherewards.pdg, MDPonlytherewards.pdf
            # Node colors : orange = current state, gray = other states
            # Edge colors : the color indicates the corresponding action (e.g. blue= action 0, red = action 1, etc)
            # Edge transparency: indicates the value of the mean reward.
            # Edge label: A label indicates a positive reward, with mean value given by the labal (color of the label = action)
            """
            if (not self.initializedRender):
                self.initRender()
                self.initializedRender=True
            G = self.G
            pos = self.pos

            plt.figure(self.numFigure)
            nx.draw_networkx_nodes(G, pos, node_size=400,
                                   node_color=['tab:gray' if s != self.s else 'tab:orange' for s in self.G.nodes()])
            plt.show(block=False)
            plt.pause(0.01)


import scipy.stats as stat

class Dirac:
    def __init__(self,value):
        self.v = value
    def rvs(self):
        return self.v
    def mean(self):
        return self.v


class RandomMDP(DiscreteMDP):
    def __init__(self, nbStates,nbActions, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.1, maxProportionSupportStart=0.2, minNonZeroProbability=0.2, minNonZeroReward=0.3, rewardStd=0.5, seed=None):
        self.nS = nbStates
        self.nA = nbActions
        self.states = range(0,self.nS)
        self.actions = range(0,self.nA)

        self.seed(seed)

        self.startdistribution = np.zeros((self.nS))
        self.rewards = {}
        self.transitions = {}
        self.P = {}
        # Initialize a randomly generated MDP
        for s in self.states:
            self.P[s]={}
            self.transitions[s]={}
            self.rewards[s]={}
            for a in self.actions:
                self.P[s][a]=[]
                self.transitions[s][a]={}
                my_mean = self.sparserand(p=maxProportionSupportReward, min=minNonZeroReward)
                if (rewardStd>0 and my_mean>0 and my_mean<1):
                    ma, mb = (0 - my_mean) / rewardStd, (1 - my_mean) / rewardStd
                    self.rewards[s][a] = stat.truncnorm(ma,mb,loc=my_mean,scale=rewardStd)
                else:
                    self.rewards[s][a] = Dirac(my_mean)
                transitionssa = np.zeros((self.nS))
                for s2 in self.states:
                    transitionssa[s2] = self.sparserand(p=maxProportionSupportTransition,min=minNonZeroProbability)
                mass = sum(transitionssa)
                if (mass > 0):
                    transitionssa = transitionssa / sum(transitionssa)
                    transitionssa = self.reshapeDistribution(transitionssa, minNonZeroProbability)
                else:
                    transitionssa[self.np_random.randint(self.nS)] = 1.
                li= self.P[s][a]
                [li.append((transitionssa[s], s, False)) for s in self.states if transitionssa[s]>0]
                self.transitions[s][a]= {ss:transitionssa[ss] for ss in self.states}

            self.startdistribution[s] = self.sparserand(p=maxProportionSupportStart,min=minNonZeroProbability)
        mass = sum(self.startdistribution)
        if (mass > 0):
            self.startdistribution = self.startdistribution / sum(self.startdistribution)
            self.startdistribution = self.reshapeDistribution(self.startdistribution, minNonZeroProbability)
        else:
            self.startdistribution[self.np_random.randint(self.nS)] = 1.

        checkRewards = sum([sum([self.rewards[s][a].mean() for a in self.actions]) for s in self.states])
        if(checkRewards==0):
            s = self.np_random.randint(0,self.nS)
            a = self.np_random.randint(0,self.nA)
            my_mean = minNonZeroReward + self.np_random.rand() * (1. - minNonZeroReward)
            if (rewardStd > 0 and my_mean > 0 and my_mean < 1):
                ma, mb = (0 - my_mean) / rewardStd, (1 - my_mean) / rewardStd
                self.rewards[s][a] = stat.truncnorm(ma, mb, loc=my_mean, scale=rewardStd)
            else:
                self.rewards[s][a] = Dirac(my_mean)
        #print("Random MDP is generated")
        #print("initial:",self.startdistribution)
        #print("rewards:",self.rewards)
        #print("transitions:",self.P)

        # Now that the Random MDP is generated with a given seed, we finalize its generation with an empty seed (seed=None) so that transitions/rewards are indeed stochastic:
        super(RandomMDP, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution,seed=None)

    def sparserand(self,p=0.5, min=0., max=1.):
        u = self.np_random.rand()
        if (u <= p):
            return min + self.np_random.rand() * (max - min)
        return 0.

    def reshapeDistribution(self,distribution, p):
        mdistribution = [0 if x < p else x for x in distribution]
        mass= sum(mdistribution)
        while(mass<0.99999999):
            i = self.np_random.randint(0, len(distribution))
            if(mdistribution[i]<p):
                newp = min(p, 1.-mass)
                if (newp==p):
                    mass = mass - mdistribution[i]+p
                    mdistribution[i]=p
            if (mdistribution[i] >= p):
                newp = min(1.,mdistribution[i] + 1.-mass)
                mass = mass- mdistribution[i]+newp
                mdistribution[i] = newp
        mass = sum(mdistribution)
        mdistribution = [x/mass for x in mdistribution]
        return mdistribution


class RiverSwim(DiscreteMDP):
    def __init__(self, nbStates, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):#, ergodic=False): # TODO ergordic option
        self.nS = nbStates
        self.nA = 2
        self.states = range(0,self.nS)
        self.actions = range(0,self.nA)
        self.nameActions = ["R", "L"]


        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] =1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a RiverSwim MDP
        for s in self.states:
            self.P[s]={}
            self.transitions[s]={}
            # GOING RIGHT
            self.transitions[s][0]={}
            self.P[s][0]= [] #0=right", 1=left
            li = self.P[s][0]
            prr=0.
            if (s<self.nS-1):
                li.append((rightProbaright, s+1, False))
                self.transitions[s][0][s+1]=rightProbaright
                prr=rightProbaright
            prl = 0.
            if (s>0):
                li.append((rightProbaLeft, s-1, False))
                self.transitions[s][0][s-1]=rightProbaLeft
                prl=rightProbaLeft
            li.append((1.-prr-prl, s, False))
            self.transitions[s][0][s ] = 1.-prr-prl

            self.P[s][1] = []  # 0=right", 1=left
            self.transitions[s][1]={}
            li = self.P[s][1]
            if (s > 0):
                li.append((1., s - 1, False))
                self.transitions[s][1][s-1]=1.
            else:
                li.append((1., s, False))
                self.transitions[s][1][s]=1.

            self.rewards[s]={}
            if (s==self.nS-1):
                self.rewards[s][0] = Dirac(rewardR)
            else:
                self.rewards[s][0] = Dirac(0.)
            if (s==0):
                self.rewards[s][1] = Dirac(rewardL)
            else:
                self.rewards[s][1] = Dirac(0.)
                
        #print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)


        super(RiverSwim, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution)
    





class ThreeState(DiscreteMDP):
    def __init__(self, delta = 0.005, fixed_reward = True):
        self.nS = 3
        self.nA = 2
        self.states = range(0,self.nS)
        self.actions = range(0,self.nA)
        self.nameActions = ["R", "L"]


        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] =1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a 3-state MDP

        s = 0
        self.P[s]={}
        self.transitions[s]={}
        # Action 0
        self.transitions[s][0]={}
        self.P[s][0]= [] #0=right", 1=left
        self.P[s][0].append((delta, 1, False))
        self.transitions[s][0][1] = delta
        self.P[s][0].append((1.- delta, 2, False))
        self.transitions[s][0][2] = 1. - delta
        # Action 1 is just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s]={}
        if fixed_reward:
            self.rewards[s][0] = Dirac(0.)
            self.rewards[s][1] = Dirac(0.)
        else:
            self.rewards[s][0] = Dirac(0.)
            self.rewards[s][1] = Dirac(0.)
        
        s = 1
        self.P[s]={}
        self.transitions[s]={}
        # Action 0
        self.transitions[s][0]={}
        self.P[s][0]= [] #0=right", 1=left
        self.P[s][0].append((1., 0, False))
        self.transitions[s][0][0] = 1.
        # Action 1 which just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s]={}
        if fixed_reward:
            self.rewards[s][0] = Dirac(1./3.)
            self.rewards[s][1] = Dirac(1./3.)
        else:
            self.rewards[s][0] = stat.bernoulli(1./3.)
            self.rewards[s][1] = stat.bernoulli(1./3.)
        
        s = 2
        self.P[s]={}
        self.transitions[s]={}
        # Action 0
        self.transitions[s][0]={}
        self.P[s][0]= [] #0=right", 1=left
        self.P[s][0].append((1., 2, False))
        self.transitions[s][0][2] = 1.
        # Action 1 which just the same for s = 0 and s = 1
        self.transitions[s][1]={}
        self.P[s][1]= [] #0=right", 1=left
        self.P[s][1].append((delta, 1, False))
        self.transitions[s][1][1] = delta
        self.P[s][1].append((1.- delta, 0, False))
        self.transitions[s][1][0] = 1. - delta
        # reward
        self.rewards[s]={}
        if fixed_reward:
            self.rewards[s][0] = Dirac(2./3.)
            self.rewards[s][1] = Dirac(2./3.)
        else:
            self.rewards[s][0] = stat.bernoulli(2./3.)
            self.rewards[s][1] = stat.bernoulli(2./3.)
         
        #print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)
        super(ThreeState, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution)


