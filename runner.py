'''
Scudstorm runner

Entelect Challenge 2018
Author: Matthew Baas
'''

from common import util
import os
import time
from common.metrics import Stopwatch
from common import metrics
import numpy as np
import random

def fight(env, agent1, agent2, n_fights, max_steps, debug=False):
    '''
    Function to run [agents] in the [env] for [runs] number of times each. 
    i.e performs rollouts of each agent [runs] number of times.

    - agents : list of agents on which to evaluate rollouts
    - env : env to run agents through
    - refbot : agent which will be player B for all agents
    - runs : int number of times each agent should play a rollout
    '''

    queue = list([agent1,])
    queue = n_fights*queue
    init_length = len(queue)
    n_envs = env.num_envs
    print(">> ROLLOUTS >> Running rollout wave with queue length  ", init_length)
    pbar = metrics.ProgressBar(init_length)
    interior_steps = 0
    early_eps = 0
    failed_eps = 0
    agent1Wins = 0
    agent2Wins = 0
    ties = 0

    while len(queue) > 0:
        # KEEP THIS THERE OTHERWISE SHIT BREAKS
        pbar.show(init_length - len(queue))

        if len(queue) >= n_envs:
            cur_playing_agents = [queue.pop() for i in range(n_envs)]
        else:
            cur_playing_agents = [queue.pop() for i in range(len(queue))]

        step = 0
        dummy_actions = [(0, 0, 3,) for _ in range(n_envs - len(cur_playing_agents))]
        suc = env.reset()
        if all(suc) == False:
            print("something fucked out. Could not reset all envs.")
            return
        #obs = env.get_base_obs()
        obs = util.get_initial_obs(n_envs)

        for a in cur_playing_agents:
            a.fitness_score = 0
            a.mask_output = False

        ## TODO: Modify this for loop to be able to end early for games which finish early
        while step < max_steps:
            if debug:
                ss = Stopwatch()
            actions = [agent.step(obs[i][0]) for i, agent in enumerate(cur_playing_agents)]
            ref_actions = [agent2.step(obs[i][1]) for i in range(len(obs))]

            if len(dummy_actions) > 0:
                actions.extend(dummy_actions)
            
            if len(actions) != len(ref_actions):
                print("LEN OF ACTIONS != LEN OF REF ACTIONS!!!!")
                raise ValueError

            if debug:
                print(">> storm >> taking actions: ", actions, ' and ref actions ', ref_actions)

            obs, rews, ep_infos = env.step(actions, p2_actions=ref_actions)
            interior_steps += n_envs
            ## TODO: loop through obs and check which one is a ControlObj, and stop processing the agents for the rest of that episode
            failure = False
            for i, a in enumerate(cur_playing_agents):
                if type(rews[i][0]) == util.ControlObject:
                    if rews[i][0].code == "EARLY":
                        a.mask_output = True
                        if step == max_steps-1:
                            early_eps += 1
                        #a.fitness_score = a.fitness_score + 1
                    elif rews[i][0].code == "FAILURE":
                        # redo this whole fucking batch
                        failed_eps += 1
                        failure = True
                        break
                else:
                    #print(rews)
                    # if rews[i][0] >= 0.95:
                    #     agent1Wins += 1
                    # elif rews[i][1] >= 0.95:
                    #     agent2Wins += 1
                    pass
                    #a.fitness_score = rews[i][0] + gamma*a.fitness_score
                if 'winner' in ep_infos[i].keys():
                    if ep_infos[i]['winner'] == 'A':
                        agent1Wins += 1
                    elif ep_infos[i]['winner'] == 'B':
                        agent2Wins += 1
                    elif ep_infos[i]['winner'] == 'TIE':
                        ties += 1

            if failure:
                curQlen = len(queue)
                queue = cur_playing_agents + queue
                print("Failure detected. Redoing last batch... (len Q before = ", curQlen, ' ; after = ', len(queue))
                break

            if debug:
                print("obs shape = ", obs.shape)
                print("rews shape = ", rews.shape)
                print('>> storm >> just took step {}. Took: {}'.format(step, ss.delta))
            step = step + 1
        
        for a in cur_playing_agents:
            a.fitness_averaging_list.append(a.fitness_score)

    pbar.close()

    return agent1Wins, agent2Wins, early_eps, failed_eps, ties

def run_battle(a1, a2, env):

    checkpoint_names = os.listdir(util.get_savedir('checkpoints'))
    checkpoint_names = sorted(checkpoint_names, reverse=True)

    elite = a1
    elite.load(util.get_savedir('checkpoints'), checkpoint_names[0])

    refbot_names = os.listdir(util.get_savedir('refbots'))
    refbot_names = sorted(refbot_names, reverse=True)

    for agent_name in refbot_names:
        a2.load(util.get_savedir('refbots'), agent_name)

        agent1Wins, agent2Wins, early_eps, failed_eps, ties = fight(env, elite, a2, n_fights=4, max_steps=80, debug=False)
        #print("Agent1Wins: ", agent1Wins)
        #print("Agent2Wins: ", agent2Wins)
        print("Elite (" + checkpoint_names[0] + ") wins: ", agent1Wins)
        print(str(agent_name) + ' wins: ', agent2Wins)
        print("EarlyEps: ", early_eps)
        print("FailedEps: ", failed_eps)
        print("Ties: ", ties)