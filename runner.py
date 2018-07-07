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
from common.mmr_lib import MMRBracket
from scud2 import Scud
import numpy as np
import random
import StarterBotPrime

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

    pbar.close()

    return agent1Wins, agent2Wins, early_eps, failed_eps, ties

def run_battle(a1, a2, env):

    checkpoint_names = os.listdir(util.get_savedir('checkpoints'))
    checkpoint_names = sorted(checkpoint_names, reverse=True)

    elite = a1
    #elite.load(util.get_savedir(), 'elite')
    elite.load(util.get_savedir('checkpoints'), 'gen30elite.h5')

    #refbot_names = os.listdir(util.get_savedir('refbots'))
    #refbot_names = sorted(refbot_names, reverse=True)

    agent1Wins, agent2Wins, early_eps, failed_eps, ties = fight(env, elite, StarterBotPrime, n_fights=32, max_steps=110)
    #print("Agent1Wins: ", agent1Wins)
    #print("Agent2Wins: ", agent2Wins)
    print("Elite (" + 'elite.h5' + ") wins: ", agent1Wins)
    print('StarterBot wins: ', agent2Wins)
    print("EarlyEps: ", early_eps)
    print("FailedEps: ", failed_eps)
    print("Ties: ", ties)


    # for agent_name in checkpoint_names:
    #     a2.load(util.get_savedir('checkpoints'), agent_name)

    #     agent1Wins, agent2Wins, early_eps, failed_eps, ties = fight(env, elite, a2, n_fights=4, max_steps=90, debug=False)
    #     #print("Agent1Wins: ", agent1Wins)
    #     #print("Agent2Wins: ", agent2Wins)
    #     print("Elite (" + 'elite.h5' + ") wins: ", agent1Wins)
    #     print(str(agent_name) + ' wins: ', agent2Wins)
    #     print("EarlyEps: ", early_eps)
    #     print("FailedEps: ", failed_eps)
    #     print("Ties: ", ties)

def calculate_mmr_values(players, env, total_games=10, game_max_steps=175):
    # players should be a list of agents
    bracket = MMRBracket()

    for p in players:
        bracket.addPlayer(p.name)

    from itertools import combinations
    all_possible_matchups = [comb for comb in combinations(players, 2)]

    for i in range(total_games):
        matchups = random.sample(all_possible_matchups, env.num_envs)
        agent1s, agent2s = zip(*matchups)

        matchup_dict, early_eps, failed_eps, ties = parallel_fight(env, agent1s, agent2s, max_steps=game_max_steps)
        print("EarlyEps: ", early_eps)
        print("i = ", i)
        print("Ties: ", ties)

        for i, ag in enumerate(agent1s):
            if ag.name in matchup_dict:
                if matchup_dict[ag.name] >= 1:
                    bracket.recordMatch(ag.name, agent2s[i].name, winner=ag.name)
                elif matchup_dict[ag.name] <= -1:
                    bracket.recordMatch(ag.name, agent2s[i].name, winner=agent2s[i].name)
                elif matchup_dict[ag.name] == 0:
                    bracket.recordMatch(ag.name, agent2s[i].name, draw=True)

    mmrs = bracket.getRatingList()
    mmrs = sorted(mmrs, key = lambda elm : elm[-1], reverse=True)
    print(str('='*60) + '\n' + 'MMR Rankings\n' + str('-'*60))
    for mmr in mmrs:
        print("Name: {:25} | MMR: {:7.2f} | Simulated matches: {:3d}".format(mmr[0], mmr[2], mmr[1]))

    return mmrs

def mmr_from_checkpoints(env):
    checkpoint_names = os.listdir(util.get_savedir('checkpoints'))
    checkpoint_names = sorted(checkpoint_names, reverse=True)

    agents = [Scud(name=str(name)) for name in checkpoint_names]
    
    for agent in agents:
        agent.load(util.get_savedir('checkpoints'), agent.name)

    agents.append(StarterBotPrime)

    mmrs = calculate_mmr_values(agents, env, total_games=30)
    ## Will print out something like
    # Name: StarterBotPrime           | MMR: 1049.76 | Simulated matches:   5
    # Name: gen50elite.h5             | MMR: 1037.71 | Simulated matches:   2
    # Name: gen40elite.h5             | MMR: 1000.00 | Simulated matches:   0
    # Name: gen10elite.h5             | MMR: 1000.00 | Simulated matches:   0
    # Name: gen30elite.h5             | MMR:  997.71 | Simulated matches:   2
    # Name: gen20elite.h5             | MMR:  914.82 | Simulated matches:   5

def parallel_fight(env, agent1s, agent2s, max_steps, debug=False):

    n_envs = env.num_envs
    assert len(agent2s) == n_envs and len(agent1s) == n_envs, "agent lengths must be same as env"

    print(">> PARALLEL FIGHT >> Running rollouts with {} games  ".format(n_envs))
    pbar = metrics.ProgressBar(max_steps)
    early_eps = 0
    failed_eps = 0
    matchup_dict = {}
    ties = 0

    step = 0
    suc = env.reset()
    if all(suc) == False:
        print("something fucked out. Could not reset all envs.")
        return
    #obs = env.get_base_obs()
    obs = util.get_initial_obs(n_envs)
    cur_playing_agents = agent1s

    while step < max_steps:
        pbar.show(step)

        if debug:
            ss = Stopwatch()
        actions = [agent.step(obs[i][0]) for i, agent in enumerate(cur_playing_agents)]
        ref_actions = [agent.step(obs[i][1]) for i, agent in enumerate(agent2s)]
        
        if len(actions) != len(ref_actions):
            print("LEN OF ACTIONS != LEN OF REF ACTIONS!!!!")
            raise ValueError

        if debug:
            print(">> storm >> taking actions: ", actions, ' and ref actions ', ref_actions)

        obs, rews, ep_infos = env.step(actions, p2_actions=ref_actions)
        ## TODO: loop through obs and check which one is a ControlObj, and stop processing the agents for the rest of that episode
        failure = False
        for i, a in enumerate(cur_playing_agents):
            if type(rews[i][0]) == util.ControlObject:
                if rews[i][0].code == "EARLY":
                    a.mask_output = True
                        
                    #a.fitness_score = a.fitness_score + 1
                elif rews[i][0].code == "FAILURE":
                    # redo this whole fucking batch
                    failed_eps += 1
                    failure = True
                    break

            if 'winner' in ep_infos[i].keys():
                early_eps += 1
                if ep_infos[i]['winner'] == 'A':
                    matchup_dict[a.name] = 1
                elif ep_infos[i]['winner'] == 'B':
                    matchup_dict[a.name] = -1
                elif ep_infos[i]['winner'] == 'TIE':
                    matchup_dict[a.name] = 0
                    ties += 1

        if failure:
            print("Failure detected. Skipping batch")
            break

        if debug:
            print("obs shape = ", obs.shape)
            print("rews shape = ", rews.shape)
            print('>> storm >> just took step {}. Took: {}'.format(step, ss.delta))
        step = step + 1

    pbar.close()

    return matchup_dict, early_eps, failed_eps, ties