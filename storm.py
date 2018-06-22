'''
Scudstorm training orchestrator -- storm

Entelect Challenge 2018
Author: Matthew Baas
'''
import tensorflow as tf
from common import util
import os
import argparse
import time
from common.metrics import Stopwatch
from scud2 import Scud
import numpy as np
import random

summary = tf.contrib.summary

##############################
###### TRAINING CONFIG #######
n_steps = 20
n_generations = 5
trunc_size = 2
replace_refbot_every = 5

# the top [n_elite_in_royale] of agents will battle it out over an additional
# [elite_additional_episodes] episodes (averaging rewards over them) to find the
# true elite for the next generation. In paper n_elite_in_royale = 10, 
# elite_additional_episodes = 30. For ideal performance, ensure n_elite_in_royale % n_envs = 0
elite_additional_episodes = 4
n_elite_in_royale = 2

max_steps_per_eval = 25
gamma = 0.99 # reward decay
n_population = 4 #12
sigma = 0.0015 # guassian std scaling

scud_debug = False

##############################

def train(env, n_envs, no_op_vec):
    print(str('='*50) + '\n' + 'Initializing agents\n' + str('='*50) )

    # Setting up logs
    writer = summary.create_file_writer(util.get_logdir('test3'), flush_millis=10000)
    writer.set_as_default()
    global_step = tf.train.get_or_create_global_step()

    ## TODO: batch jobs to allow for like 1k population with only like 10 envs.
    #n_population = n_envs

    ## TODO: change agent layers to use xavier initializer
    agents = [Scud(name=str(i), debug=scud_debug) for i in range(n_population)]
    refbot = Scud(name='refbot', debug=scud_debug)

    next_generation = [Scud(name=str(i) + 'NEXT', debug=scud_debug) for i in range(n_population)]

    print(str('='*50) + '\n' + 'Beginning training\n' + str('='*50) )
    s = Stopwatch()
    total_s = Stopwatch()
    for g in range(n_generations):
        #####################
        ## GA Algorithm
        for i in range(n_population):
            if g == 0:
                break
            else:
                kappa = random.sample(agents[0:trunc_size], 1)
                mutate(kappa[0], next_generation[i], g)

        # swap agents and the next gen's agents. i.e set next gen agents to be current agents to evaluate
        tmp = agents
        agents = next_generation
        next_generation = tmp
        
        # evaluate fitness on each agent in population
        agents = evaluate_fitness(env, agents, refbot, debug=False)
        # sort them based on final discounted reward
        agents = sorted(agents, key = lambda agent : agent.fitness_score, reverse=True)


        #################################
        ## Replacing reference bot
        if g % replace_refbot_every == 0 and g != 0:
            print(">> STORM >> Upgrading refbot now.")
            good_params = agents[trunc_size-1].get_flat_weights()
            refbot.set_flat_weights(good_params)
        
        ##################################
        ## Summary information
        with summary.always_record_summaries(): 
            sc_vec = [a.fitness_score for a in agents]
            summary.scalar('rewards/mean', np.mean(sc_vec))
            summary.scalar('rewards/max', agents[0].fitness_score)
            summary.scalar('rewards/min', agents[-1].fitness_score)
            summary.scalar('rewards/var', np.var(sc_vec))
            summary.scalar('rewards/truc_mean', np.mean(sc_vec[:trunc_size]))
        
        for a in agents:
            print(a.name, " with fitness score: ", a.fitness_score)

        # setup next generation parents / elite agents
        if g == 0:
            elite_candidates = set(agents[0:n_elite_in_royale])
        else:
            elite_candidates = set(agents[0:n_elite_in_royale-1]) | set([elite,])
        # finding next elite by battling proposed elite candidates for some additional rounds
        print("Evaluating elite agent...")
        elo_ags = evaluate_fitness(env, elite_candidates, refbot, runs=elite_additional_episodes)
        elo_ags = sorted(elo_ags, key = lambda agent : agent.fitness_score, reverse=True)
        elite = elo_ags[0]

        try:
            agents.remove(elite)
            agents = [elite,] + agents
        except ValueError:
            agents = [elite,] + agents[:len(agents)-1]

        for a in elo_ags:
            print('Elite stats: ', a.name, " with fitness score: ", a.fitness_score)

        with summary.always_record_summaries(): 
            summary.scalar('rewards/elite_score', elite.fitness_score)
            summary.scalar('time/wall_clock_time', total_s.deltaT())
            summary.scalar('time/single_gen_time', s.deltaT())

        global_step.assign_add(1)
        # for i in range(n_envs):
        #     print("mutating agent ", i)
        #     parent_ind = np.random.randint(n_elite)
        #     offspring = mutate(agents[parent_ind], g)

        print(str('='*50) + '\n' + 'Generation ' + str(g) + '. Took  ' + s.delta +  '\n' + str('='*50) )
        s.reset()
        
    summary.flush()

def mutate(parent, child, g):
    old_params = parent.get_flat_weights()
    new_params = []
    print(">> storm >> mutating agent: ", parent.name)
    for param in old_params:
        new_params.append(param + sigma*np.random.randn(*param.shape))
    child.name = parent.name + "gen" + str(g)
    child.set_flat_weights(new_params)

def evaluate_fitness(env, agents, refbot, runs=1, debug=False):
    '''
    Function to run [agents] in the [env] for [runs] number of times each. 
    i.e performs rollouts of each agent [runs] number of times.

    - agents : list of agents on which to evaluate rollouts
    - env : env to run agents through
    - refbot : agent which will be player B for all agents
    - runs : int number of times each agent should play a rollout
    '''


    queue = list(agents)
    queue = runs*queue
    init_length = len(queue)
    print(">> ROLLOUTS >> Running rollout wave with queue length  ", init_length)

    while len(queue) > 0:
        time.sleep(0.1)
        if len(queue) <= 0.5*init_length:
            print("50% done rollout...")

        if len(queue) >= env.num_envs:
            cur_playing_agents = [queue.pop() for i in range(env.num_envs)]
        else:
            cur_playing_agents = [queue.pop() for i in range(len(queue))]

        step = 0
        dummy_actions = [(0, 0, 3,) for _ in range(env.num_envs - len(cur_playing_agents))]
        suc = env.reset()
        if all(suc) == False:
            print("something fucked out. Could not reset all envs.")
            return
        #obs = env.get_base_obs()
        obs = util.get_initial_obs(env.num_envs)

        for a in cur_playing_agents:
            a.fitness_score = 0

        ## TODO: Modify this for loop to be able to end early for games which finish early
        while step < max_steps_per_eval:
            if debug:
                ss = Stopwatch()
            actions = [agent.step(obs[i][0]) for i, agent in enumerate(cur_playing_agents)]
            ref_actions = [refbot.step(obs[i][1]) for i in range(len(obs))]

            if len(dummy_actions) > 0:
                actions.extend(dummy_actions)
            
            if len(actions) != len(ref_actions):
                print("LEN OF ACTIONS != LEN OF REF ACTIONS!!!!")
                raise ValueError

            if debug:
                print(">> storm >> taking actions: ", actions, ' and ref actions ', ref_actions)

            obs, rews = env.step(actions, p2_actions=ref_actions)

            for i, a in enumerate(cur_playing_agents):
                a.fitness_score = rews[i][0] + gamma*a.fitness_score

            if debug:
                print("obs shape = ", obs.shape)
                print("rews shape = ", rews.shape)
                print('>> storm >> just took step {}. Took: {}'.format(step, ss.delta))
            step = step + 1
        
        for a in cur_playing_agents:
            a.fitness_averaging_list.append(a.fitness_score)
    
    for a in agents:
        a.squash_fitness_scores()

    return agents
