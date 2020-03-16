import gym
from gym.envs.registration import  register
import numpy as np



"""
This file contains methods to register several MDP environments into gym

"""

def registerRiverSail(sizeX=10, max_steps=np.infty, reward_threshold=np.infty):
    register(
        id='RiverSail-'+'-v0',
        entry_point='environments.riversail:RiverSail',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'sizeX': sizeX}
    )
    return 'RiverSail-'+'-v0'


registerWorlds = {
    "riversail_5"  : lambda x: registerRiverSail(sizeX=5),
    "riversail_10" : lambda x: registerRiverSail(sizeX=10),
    "riversail_20" : lambda x: registerRiverSail(sizeX=20)
}


def makeWorld(registername):
    """

    :param registername: name of the environment to be registered into gym
    :return:  full name of the registered environment
    """
    return gym.make(registername)
