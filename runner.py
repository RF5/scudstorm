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
from common.mmr_lib import MMRBracket, Game
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
            agent2.mask_output = False

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

    #elite = a1
    #elite.load(util.get_savedir(), 'elite')
    a1.load(util.get_savedir('checkpoints'), 'gen260elite.h5')
    a1.name = 'gen260elite.h5'

    #a2.load(util.get_savedir('checkpoints'), 'gen330elite.h5')
    #a2.name = 'gen330elite.h5'
    a2 = StarterBotPrime

    agent1Wins, agent2Wins, early_eps, failed_eps, ties = fight(env, a1, a2, n_fights=4, max_steps=150)
    #refbot_names = os.listdir(util.get_savedir('refbots'))
    #refbot_names = sorted(refbot_names, reverse=True)
    print("[AGENT1] Elite (" + a1.name + ") wins: ", agent1Wins)
    print('[AGENT2] StarterBot (' + a2.name + ') wins: ', agent2Wins)
    print("EarlyEps: ", early_eps)
    print("FailedEps: ", failed_eps)
    print("Ties: ", ties)
    
    # for j in range(len(checkpoint_names)):
    #     a1.load(util.get_savedir('checkpoints'), checkpoint_names[j])
    #     a1.name = checkpoint_names[j]

    #     agent1Wins, agent2Wins, early_eps, failed_eps, ties = fight(env, a1, a2, n_fights=16, max_steps=150)
    #     #print("Agent1Wins: ", agent1Wins)
    #     #print("Agent2Wins: ", agent2Wins)
    #     print("[AGENT1] Elite (" + 'gen50elite.h5' + ") wins: ", agent1Wins)
    #     print('[AGENT2] StarterBot wins: ', agent2Wins)
    #     print("EarlyEps: ", early_eps)
    #     print("FailedEps: ", failed_eps)
    #     print("Ties: ", ties)

    #     print("{:20} games: {:5} | wins: {:4} | win rate: {:4.2f}%".format(a1.name, early_eps,
    #         agent1Wins, 100*agent1Wins / early_eps))

    #     print("{:20} games: {:5} | wins: {:4} | win rate: {:4.2f}%".format(a2.name, early_eps,
    #         agent2Wins, 100*agent2Wins / early_eps))


def calculate_mmr_values(players, env, fight_fn, total_games=10, game_max_steps=150):
    # players should be a list of agents
    bracket = MMRBracket()

    for p in players:
        bracket.addPlayer(p.name)

    from itertools import combinations
    all_possible_matchups = [comb for comb in combinations(players, 2)]
    all_games = []

    for i in range(total_games):
        matchups = random.sample(all_possible_matchups, env.num_envs)
        # p2 = players[-1]
        # p1 = players[0]
        # p1s = random.choices(players[:-1], k=env.num_envs)
        # matchups = [(p1s[i], p2) for i in range(env.num_envs)]

        games_arr, early_eps, failed_eps, ties = fight_fn(env, matchups, max_steps=game_max_steps)
        
        print("i = ", i, "EarlyEps: ", early_eps, "Ties: ", ties, 'GamesArr len: ', len(games_arr))
        all_games.extend(games_arr)

        for i, game in enumerate(games_arr):
            if game.winner == 'TIE':
                bracket.recordMatch(game.p1, game.p2, draw=True)
            else:
                bracket.recordMatch(game.p1, game.p2, winner=game.winner)

    mmrs = bracket.getRatingList()
    mmrs = sorted(mmrs, key = lambda elm : elm[-1], reverse=True)
    print(str('='*60) + '\n' + 'MMR Rankings\n' + str('-'*60))
    for mmr in mmrs:
        print("Name: {:25} | MMR: {:7.2f} | Simulated matches: {:3d}".format(mmr[0], mmr[2], mmr[1]))

    bins = {}
    for playe in players:
        bins[playe.name] = 0
        bins[playe.name + 'wins'] = 0
    for game in all_games:
        if game.winner == 'TIE':
            continue
        bins[game.winner + 'wins'] += 1
        bins[game.p1] += 1
        bins[game.p2] += 1

    for playe in players:
        print("{:20} games: {:5} | wins: {:4} | win rate: {:4.2f}%".format(playe.name, bins[playe.name],
            bins[playe.name + 'wins'], 100*bins[playe.name + 'wins'] / bins[playe.name]))

    return mmrs

def mmr_from_checkpoints(env):
    checkpoint_names = os.listdir(util.get_savedir('checkpoints'))
    checkpoint_names = sorted(checkpoint_names, reverse=True)

    agents = [Scud(name=str(name)) for name in checkpoint_names]
    
    for agent in agents:
        agent.load(util.get_savedir('checkpoints'), agent.name)
        #agent.load(util.get_savedir('checkpoints'), checkpoint_names[0])

    agents.append(StarterBotPrime)

    print("ranking agents:")
    for a in agents:
        print(str(a))
    mmrs = calculate_mmr_values(agents, env, parallel_fight, total_games=80)
    ## Will print out something like
    # ============================================================
    # MMR Rankings
    # ------------------------------------------------------------
    # Name: StarterBotPrime           | MMR: 1256.71 | Simulated matches:  27
    # Name: gen10elite.h5             | MMR: 1032.77 | Simulated matches:  16

def parallel_fight(env, matchups, max_steps, debug=False):
    a1s = []
    a2s = []
    for a, b in matchups:
        a1s.append(a)
        a2s.append(b)
    #a1s, a2s = [(e, f,) for e, f in zip(*matchups)]
    

    n_envs = env.num_envs
    assert len(matchups) == n_envs, "agent lengths must be same as env"

    print(">> PARALLEL FIGHT >> Running rollouts with {} games  ".format(n_envs))
    pbar = metrics.ProgressBar(max_steps)
    early_eps = 0
    failed_eps = 0
    games = []
    ties = 0

    step = 0
    suc = env.reset()
    if all(suc) == False:
        print("something fucked out. Could not reset all envs.")
        return
    #obs = env.get_base_obs()
    obs = util.get_initial_obs(n_envs)

    for aa, bb in matchups:      
        aa.mask_output = False
        bb.mask_output = False

    while step < max_steps:
        pbar.show(step)

        if debug:
            ss = Stopwatch()

        actions = [agent.step(obs[i][0]) for i, agent in enumerate(a1s)]
        ref_actions = [agen2.step(obs[i][1]) for i, agen2 in enumerate(a2s)]
        
        if len(actions) != len(ref_actions):
            print("LEN OF ACTIONS != LEN OF REF ACTIONS!!!!")
            raise ValueError

        if debug:
            print(">> storm >> taking actions: ", actions, ' and ref actions ', ref_actions)

        obs, rews, ep_infos = env.step(actions, p2_actions=ref_actions)
      
        failure = False
        for i in range(n_envs):
            if type(rews[i][0]) == util.ControlObject:
                if rews[i][0].code == "EARLY":
                    a1s[i].mask_output = True
                    a2s[i].mask_output = True
                elif rews[i][0].code == "FAILURE":
                    # redo this whole fucking batch
                    failed_eps += 1
                    failure = True
                    break

            if 'winner' in ep_infos[i].keys():
                early_eps += 1
                if ep_infos[i]['winner'] == 'A':
                    #matchup_dict[a.name] = 'A'
                    games.append(Game(a1s[i].name, a2s[i].name, winner=a1s[i].name))
                elif ep_infos[i]['winner'] == 'B':
                    #matchup_dict[a.name] = 'B'
                    games.append(Game(a1s[i].name, a2s[i].name, winner=a2s[i].name))
                elif ep_infos[i]['winner'] == 'TIE':
                    #matchup_dict[a.name] = 'TIE'
                    games.append(Game(a1s[i].name, a2s[i].name, winner='TIE'))
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

    return games, early_eps, failed_eps, ties

if __name__ == '__main__':
    print("Testing MMR system...")

    def play_func(env, matchups, max_steps=3):
        games = []
        for a1, a2 in matchups:
            if a1.name.startswith('p') and a2.name.startswith('p') == False:
                win = random.random() < 1 # 70% change of winning
                if win:
                    games.append(Game(a1.name, a2.name, winner=a1.name))
                else:
                    games.append(Game(a1.name, a2.name, winner=a2.name))
            elif a2.name.startswith('p') and a1.name.startswith('p') == False:
                win = random.random() < 1 # 70% change of winning
                if win:
                    games.append(Game(a1.name, a2.name, winner=a2.name))
                else:
                    games.append(Game(a1.name, a2.name, winner=a1.name))
            elif a1.name.startswith('a') and a2.name.startswith('a') == False:
                win = random.random() < 0.7 # 70% change of winning
                if win:
                    games.append(Game(a1.name, a2.name, winner=a1.name))
                else:
                    games.append(Game(a1.name, a2.name, winner=a2.name))
            elif a2.name.startswith('a') and a1.name.startswith('a') == False:
                win = random.random() < 0.7 # 70% change of winning
                if win:
                    games.append(Game(a1.name, a2.name, winner=a2.name))
                else:
                    games.append(Game(a1.name, a2.name, winner=a1.name))

            else:
                win = random.random() < 0.5
                if win:
                    games.append(Game(a1.name, a2.name, winner=a1.name))
                else:
                    games.append(Game(a1.name, a2.name, winner=a2.name))
        
        return games, 0, 0, 0

    class EEnv:
        num_envs = 5

    class PPloyo:
        def __init__(self, name):
            self.name = name

    players = ['a1', 'a2', 'a3', 'hello', 'meme', 'review', 'kappa', 'pacifica', 'seth rich']
    myArr = []
    for kk in players:
        myArr.append(PPloyo(kk))

    mmrs = calculate_mmr_values(myArr, EEnv, play_func, total_games=400)

    print(">>> Advanced tests completed successfully.")
