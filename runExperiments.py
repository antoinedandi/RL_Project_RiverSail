import engine.buildWorlds as bW
import learners.OptimalControl  as opt
import learners.Random as lr
import learners.Human as lh
import learners.UCRL2 as ucrl
import learners.C_UCRL2 as c_ucrl

import time
import numpy as np
import pylab as pl
import copy
import pickle

from joblib import Parallel, delayed
import multiprocessing


#################################
# Running a single experiment:
#################################

def animate(env, learner, timeHorizon, rendermode='pylab'):
    observation = env.reset()
    learner.reset(observation)
    cumreward = 0.
    cumrewards = []
    print("New initialization of ", learner.name())
    print("Initial state:" + str(observation))
    for t in range(timeHorizon):
        state = observation
        env.render(rendermode)
        action = learner.play(state)  # Get action
        observation, reward, done, info = env.step(action)
        print("S:"+str(state)+" A:"+str(action) + " R:"+str(reward)+" S:"+str(observation) +" done:"+str(done) +"\n")
        learner.update(state, action, reward, observation)  # Update learners
        cumreward += reward
        cumrewards.append(cumreward)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break


def oneXpNoRender(env,envname,learner,timeHorizon):
    observation = env.reset()
    learner.reset(observation)
    cumreward = 0.
    cumrewards = []
    cummean = 0.
    cummeans = []
    print("New initialization of ", learner.name())
    print("Initial state:" + str(observation))
    for t in range(timeHorizon):
        state = observation
        action = learner.play(state)  # Get action
        observation, reward, done, info = env.step(action)
        learner.env = env
        if ('Taxi' in envname):
            reward = (reward+10.)/30.
        learner.update(state, action, reward, observation)  # Update learners
        #print("info:",info, "reward:", reward)
        cumreward += reward
        try:
            cummean += info
        except TypeError:
            cummean += reward
        cumrewards.append(cumreward)
        cummeans.append(cummean)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            observation = env.reset()
            # break

    print("Cumreward: " + str(cumreward))
    print("Cummean: " + str(cummean))
    return cummeans  # cumrewards,cummeans


def oneXpNoRenderWithDump(env, envname, learner, timeHorizon):
    observation = env.reset()
    learner.reset(observation)
    cumreward = 0.
    cumrewards = []
    cummean = 0.
    cummeans = []
    print("New initialization of ", learner.name(), ' for environment ', envname)
    print("Initial state:" + str(observation))
    for t in range(timeHorizon):
        state = observation
        action = learner.play(state)  # Get action
        observation, reward, done, info = env.step(action)
        learner.env = env
        if ('Taxi' in envname): # Need to rescale the rewards
            reward = (reward+10.)/30.
        learner.update(state, action, reward, observation)  # Update learners
        #print("info:",info, "reward:", reward)
        cumreward += reward
        try:
            cummean += info
        except TypeError:
            cummean +=reward
        cumrewards.append(cumreward)
        cummeans.append(cummean)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            observation = env.reset()  # converts an episodic MDP into an infinite time horizon MDP
            # break

    filename = "results/cumMeans_" + envname + "_" + learner.name() + "_" + str(timeHorizon) +"_" + str(time.time())
    file =  open(filename,'wb')
    pickle.dump(cummeans, file)
    file.close()
    return filename



#################################
# Running multiple experiments:
#################################

def run_large_exp(envName = "riverSwim", timeHorizon=1000, nbReplicates=100):

    if (envName in bW.registerWorlds.keys()):
        regName = (bW.registerWorlds[envName])(0)
        print("Environment " + envName + " registered as " + regName)
        envName= regName

    envOpt = bW.makeWorld(envName)

    print("Computing an estimate of the optimal policy (for regret)...")
    opti_learner = opt.Opti_controller(envOpt.env, envOpt.observation_space.n, envOpt.action_space.n)
    print(opti_learner.policy)


    print("*********************************************")

    dump_cumRewardsAlgos = []
    names = []
    meanelapsedtimes = []

    learners = []

    learners.append(ucrl.UCRL2(envOpt.observation_space.n, envOpt.action_space.n, delta=0.05))
    learners.append(c_ucrl.C_UCRL2(envOpt.observation_space.n, envOpt.action_space.n, envOpt, envOpt.env.classes, delta=0.05))


    for learner in learners:
        names.append(learner.name())
        dump_cumRewards, meanelapsedtime = multicoreXpsNoRenderWithDump(envName, learner, nbReplicates, timeHorizon)
        dump_cumRewardsAlgos.append(dump_cumRewards)
        meanelapsedtimes.append(meanelapsedtime)

    opttimeHorizon = min(max((10000, timeHorizon)),10**8)
    cumReward_opti = oneXpNoRender(envOpt,envName,opti_learner,opttimeHorizon)

    gain =  cumReward_opti[-1] / len(cumReward_opti)
    print("Average gain is ", gain)
    opti_reward = [[t * gain for t in range(timeHorizon)]]
    filename = "results/cumMeans_" + envName + "_"+opti_learner.name()+"_" + str(timeHorizon) + "_"+str(time.time())
    file = open(filename, 'wb')
    pickle.dump(opti_reward, file)
    file.close()
    dump_cumRewardsAlgos.append(filename)

    [print(str(names[i]), "average runtime is ", meanelapsedtimes[i] ) for i in range(len(names))]
    median, quantile1,quantile2,times = analyzeResults(names,dump_cumRewardsAlgos, timeHorizon, envName)
    plotCumulativeRegretsFromDump(names,envName, median,quantile1,quantile2, times,timeHorizon)

    print("*********************************************")



def starOneXp(args):
    return oneXpNoRender(*args)


def multicoreXpsNoRender(envName,learner,nbReplicates,timeHorizon):
    num_cores = multiprocessing.cpu_count()
    envs = []
    learners = []
    timeHorizons = []


    for i in range(nbReplicates):
        envs.append(bW.makeWorld(envName))
        learners.append(copy.deepcopy(learner))
        timeHorizons.append(copy.deepcopy(timeHorizon))

    t0 = time.time()

    cumRewards = Parallel(n_jobs=num_cores)(delayed(starOneXp)(i) for i in zip(envs,learners,timeHorizons))

    elapsed = time.time()-t0
    return cumRewards, elapsed / nbReplicates




def starOneXpWithDump(args):
    return oneXpNoRenderWithDump(*args)


def multicoreXpsNoRenderWithDump(envName, learner, nbReplicates, timeHorizon):
    num_cores = multiprocessing.cpu_count()
    envs = []
    envnames = []
    learners = []
    timeHorizons = []


    for i in range(nbReplicates):
        envs.append(bW.makeWorld(envName))
        envnames.append(envName)
        learners.append(copy.deepcopy(learner))
        timeHorizons.append(copy.deepcopy(timeHorizon))

    t0 = time.time()

    dump_cumRewards = Parallel(n_jobs=num_cores)(delayed(starOneXpWithDump)(i) for i in zip(envs,envnames,learners,timeHorizons))

    elapsed = time.time()-t0
    return dump_cumRewards, elapsed / nbReplicates




#################################
# Analyze and plot results
#################################


def analyzeResults(names, dump_cumulativerewards_, timeHorizon, envName =""):
    """

    :param names: get list of algorithm names
    :param dump_cumulativerewards_: list of filenames, each getting cumulative rewards for multiple runs. Last file of the list is cum reward of Oracle.
    :param timeHorizon:
    :param envName:
    :return: vectors median, quantile0.25, quantile0.75, timesteps, where median[i] is median of expreimnts at time timesteps[i]
    """
    median = []
    quantile1 = []
    quantile2 = []
    nbAlgs = len(dump_cumulativerewards_) - 1

    # Downsample the times, espeiclaly incase timeHorizon is huge.
    skip = max(1, (timeHorizon // 1000))
    times = [t for t in range(0, timeHorizon, skip)]

    for j in range(nbAlgs):
        data_j = []
        for i in range(len(dump_cumulativerewards_[j])):
            file_oracle = open(dump_cumulativerewards_[-1], 'rb')
            cum_rewards_oracle = pickle.load(file_oracle)
            cum_rewards_oracle = cum_rewards_oracle[0]
            file = open(dump_cumulativerewards_[j][i], 'rb')
            cum_rewards_ij = pickle.load(file)
            data_j.append([cum_rewards_oracle[t] - cum_rewards_ij[t] for t in range(0,timeHorizon,skip)])
            file_oracle.close()
            file.close()

        filename = "results/cumRegret_" + envName + "_" + names[j] + "_" + str(timeHorizon) + "_" + str(
            j) + "_" + str(
            time.time())
        file = open(filename, 'wb')
        pickle.dump(data_j, file)
        file.close()

        median.append(np.quantile(data_j, 0.5, axis=0))
        quantile1.append(np.quantile(data_j, 0.25, axis=0))
        quantile2.append(np.quantile(data_j, 0.75, axis=0))

    return median,quantile1,quantile2,times



def plot_results_from_dump(envName, tmax,tplot):
    """
    Requires result files to be named "results/cumRegret_" + envName + "_" + learner.name() + "_" + str(tmax)"
    :param envName: name of the environment
    :param tmax: assumes the data have been generated with maximal time horizon tmax.
    :param tplot: the results are plotted only until time horizon tplot.
    :return:
    """
    if (envName in bW.registerWorlds.keys()):
        regName = (bW.registerWorlds[envName])(0)
        #print("Environment " + envName + " registered as " + regName)
        envName= regName

    envOpt = bW.makeWorld(envName)

    median = []
    quantile1 = []
    quantile2 = []
    learners = []
    names = []

    skip = max(1, (tmax // 1000))
    itimes = [t for t in range(0,tmax,skip)]
    times = [itimes[i] for i in range(len(itimes)) if i*skip<tplot]

    #Declare list of algorithms (only to get their names):
    learners.append(ucrl.UCRL2(envOpt.observation_space.n, envOpt.action_space.n, delta=0.05 ))
    learners.append(c_ucrl.C_UCRL2(envOpt.observation_space.n, envOpt.action_space.n, envOpt, envOpt.env.classes, delta=0.05))

    for learner in learners:
        names.append(learner.name())
        filename = "results/cumRegret_" + envName + "_" + learner.name() + "_" + str(tmax)
        file = open(filename, 'rb')
        data_j = pickle.load(file)
        file.close()

        q = np.quantile(data_j, 0.5, axis=0)
        median.append([q[i] for i in range(len(q)) if i*skip<tplot])
        q=np.quantile(data_j, 0.25, axis=0)
        quantile1.append([q[i] for i in range(len(q)) if i*skip<tplot])
        q = np.quantile(data_j, 0.75, axis=0)
        quantile2.append([q[i] for i in range(len(q)) if i*skip<tplot])

    plotCumulativeRegretsFromDump(names, envName, median, quantile1, quantile2, times, tplot)


def plotCumulativeRegretsFromDump(names,envName,median, quantile1, quantile2, times, timeHorizon):
    nbFigure = pl.gcf().number+1
    pl.figure(nbFigure)
    textfile = "results/Regret-"
    colors= ['black', 'blue','gray', 'green', 'red']  # ['black', 'purple', 'blue','cyan','yellow', 'orange', 'red', 'chocolate']

    style = ['o','v','s','d','<']
    for i in range(len(median)):
        pl.plot(times,median[i],style[i],label=names[i],color=colors[i%len(colors)],linewidth=2.0,linestyle='--',markevery=0.05)
        pl.plot(times,quantile1[i], color=colors[i % len(colors)],linestyle=':',linewidth=0.7)
        pl.plot(times,quantile2[i], color=colors[i % len(colors)],linestyle=':',linewidth=0.7)
        textfile += names[i] + "-"
        print(names[i], ' has regret ', median[i][-1], ' after ', timeHorizon, ' time steps with quantiles ',
              quantile1[i][-1], ' and ', quantile2[i][-1])

    textfile+="_"+str(timeHorizon)+"_"+envName
    pl.legend(loc=2)
    pl.xlabel("Time steps", fontsize=13, fontname = "Arial")
    pl.ylabel("Regret", fontsize=13, fontname = "Arial")
    #pl.xticks(times)
    pl.ticklabel_format(axis='both', useMathText = True, useOffset = True, style='sci', scilimits=(0, 0))
    pl.ylim(0)
    pl.savefig(textfile+'.png')
    pl.savefig(textfile+ '.pdf')
    pl.xscale('log')
    pl.savefig(textfile + '_xlog.png')
    pl.savefig(textfile + '_xlog.pdf')
    pl.ylim(100)
    if(timeHorizon>1000):
        pl.xlim(1000,timeHorizon)
    pl.xscale('linear')
    pl.yscale('log')
    pl.savefig(textfile + '_ylog.png')
    pl.savefig(textfile + '_ylog.pdf')
    pl.xscale('log')
    pl.savefig(textfile + '_loglog.png')
    pl.savefig(textfile + '_loglog.pdf')



##############
def demo_animate():
    testName = 'riversail_10'
    envName = (bW.registerWorlds[testName])(0)
    env = bW.makeWorld(envName)
    # -> Choose which learner to use:
    # learner = lr.Random(env)
    # learner = lh.Human(env)
    learner = ucrl.UCRL2(env.observation_space.n, env.action_space.n, delta=0.05)
    # learner = c_ucrl.C_UCRL2(env.observation_space.n, env.action_space.n, env, env.env.classes, delta=0.05)
    animate(env, learner, 50, 'maze')

#######################
# Animate the RiverSail MDPs:
#######################
demo_animate()


#######################
# Running a full example:
#######################
# run_large_exp('riversail_5', timeHorizon=10000, nbReplicates=4)
# run_large_exp('riversail_10', timeHorizon=10000, nbReplicates=4)
# run_large_exp('riversail_20', timeHorizon=10000, nbReplicates=4)