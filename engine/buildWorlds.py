import gym
from gym.envs.registration import  register
import numpy as np



"""
This file contains methods to register several MDP environments into gym

"""
def registerRandomMDP(nbStates=5, nbActions=4, max_steps=np.infty, reward_threshold=np.infty, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.1, maxProportionSupportStart=0.2, minNonZeroProbability=0.2, minNonZeroReward=0.3, rewardStd=0.5,seed=None):
    register(
        id='RandomMDP-'+str(nbStates)+'-v0',
        entry_point='environments.discreteMDP:RandomMDP',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'nbActions': nbActions, 'maxProportionSupportTransition': maxProportionSupportTransition, 'maxProportionSupportReward': maxProportionSupportReward,
                'maxProportionSupportStart': maxProportionSupportStart, 'minNonZeroProbability':minNonZeroProbability, 'minNonZeroReward':minNonZeroReward, 'rewardStd':rewardStd, 'seed':seed }
    )
    return 'RandomMDP-'+str(nbStates)+'-v0'

def registerRiverSwim(nbStates=5, max_steps=np.infty, reward_threshold=np.infty, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):
    register(
        id='RiverSwim-'+str(nbStates)+'-v0',
        entry_point='environments.discreteMDP:RiverSwim',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
                'rewardL': rewardL, 'rewardR':rewardR, }
    )
    return 'RiverSwim-'+str(nbStates)+'-v0'


def registerGridworld(sizeX=10, sizeY=10, map_name="4-room", rewardStd=0., initialSingleStateDistribution=False, max_steps=np.infty, reward_threshold=np.infty):
    register(
        id='Gridworld-'+map_name+'-v0',
        entry_point='environments.gridworld:GridWorld',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'sizeX': sizeX,'sizeY':sizeY,'map_name':map_name,'rewardStd':rewardStd, 'initialSingleStateDistribution':initialSingleStateDistribution}
    )
    return 'Gridworld-'+map_name+'-v0'


def registerThreeState(delta = 0.005, max_steps=np.infty, reward_threshold=np.infty, fixed_reward = True):
    register(
        id='ThreeState-v0',
        entry_point='environments.discreteMDP:ThreeState',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'delta': delta, 'fixed_reward': fixed_reward, }
    )
    return 'ThreeState-v0'


def registerRiverSail(sizeX=10, initialSingleStateDistribution=False, max_steps=np.infty, reward_threshold=np.infty):
    register(
        id='RiverSail-'+'-v0',
        entry_point='environments.riversail:RiverSail',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'sizeX': sizeX}
    )
    return 'RiverSail-'+'-v0'


registerWorlds = {
    "random10" : lambda x: registerRandomMDP(nbStates=10, nbActions=3, maxProportionSupportTransition=0.1, maxProportionSupportReward=0.1, maxProportionSupportStart=0.1, minNonZeroProbability=0.15, minNonZeroReward=0.3, rewardStd=0.1,seed=10),
    "random100" : lambda x: registerRandomMDP(nbStates=100, nbActions=3, maxProportionSupportTransition=0.1, maxProportionSupportReward=0.1, maxProportionSupportStart=0.1, minNonZeroProbability=0.15, minNonZeroReward=0.3, rewardStd=0.1,seed=10),
    "three-state" : lambda x: registerThreeState(delta = 0.005),
    "riverSwim6" : lambda x: registerRiverSwim(nbStates=6, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.),
    "riverSwim25" : lambda x: registerRiverSwim(nbStates=25, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.),
    "random_grid" : lambda x: registerGridworld(sizeX=8, sizeY=5, map_name="random", rewardStd=0.01, initialSingleStateDistribution=True),
    "2-room" : lambda x: registerGridworld(sizeX=9, sizeY=11, map_name="2-room", rewardStd=0.0, initialSingleStateDistribution=True),
    "4-room" : lambda x: registerGridworld(sizeX=7, sizeY=7, map_name="4-room", rewardStd=0.0, initialSingleStateDistribution=True),
    "riversail" : lambda x: registerRiverSail(sizeX=20)
}


def makeWorld(registername):
    """

    :param registername: name of the environment to be registered into gym
    :return:  full name of the registered environment
    """
    return gym.make(registername)
