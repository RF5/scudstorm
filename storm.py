'''
Scudstorm training orchestrator -- storm

Entelect Challenge 2018
Author: Matthew Baas
'''
import tensorflow as tf
from common import util
import os
from common.metrics import Stopwatch
from common import metrics
from scud2 import Scud
import numpy as np
import random
import tf_logging_util
import json

summary = tf.contrib.summary

##############################
###### TRAINING CONFIG #######
n_generations = 61#100
trunc_size = 7#4
scoring_method = 'dense' # Possibilities: 'dense' and 'binary'
invalid_act_penalty_dense = -3
invalid_act_penalty_binary = -0.01

## Refbot/opponent upgrading config
replace_refbot_every = 1
refbot_queue_length = 30

# the top [n_elite_in_royale] of agents will battle it out over an additional
# [elite_additional_episodes] episodes (averaging rewards over them) to find the
# true elite for the next generation. In paper n_elite_in_royale = 10, 
# elite_additional_episodes = 30. For ideal performance, ensure n_elite_in_royale % n_envs = 0
elite_additional_episodes = 4#4
n_elite_in_royale = 4

max_episode_length = 140#90
gamma = 0.99 # reward decay. 
gamma_func = lambda x : 0.022*x + 0.97
n_population = 96#100
sigma = 0.002 # guassian std scaling

scud_debug = False
verbose_training = False
elite_score_moving_avg_periods = 4
elite_savename = 'elite'
save_elite_every = 10

tf_logging_util.set_logging_level('error')

def train(env, n_envs, no_op_vec, resume_trianing):
    print(str('='*50) + '\n' + 'Initializing agents\n' + str('='*50) )
    ##############################
    ## Summary buckets
    #failed_episodes = 0
    #early_episodes = 0
    refbot_back_ind = 1
    elite_overthrows = 0
    elite = None
    starting_gen = 0 # default startin generation number. Is overwritten if resuming
    ## Setting up logs
    writer = summary.create_file_writer(util.get_logdir('train12A'), flush_millis=10000)
    writer.set_as_default()
    global_step = tf.train.get_or_create_global_step()

    ## TODO: change agent layers to use xavier initializer
    agents = [Scud(name=str(i), debug=scud_debug) for i in range(n_population)]
    total_steps = 0

    elite_moving_average = metrics.MovingAverage(elite_score_moving_avg_periods)
    next_generation = [Scud(name=str(i) + 'next', debug=scud_debug) for i in range(n_population)]

    refbot_queue = [Scud(name='refbot' + str(i), debug=scud_debug) for i in range(refbot_queue_length)]
    for i, bot in enumerate(refbot_queue):
        bot.refbot_position = i
    refbot = refbot_queue[0]

    ## DOES NOT WORK WITH EAGER EXECUTION
    # with summary.always_record_summaries():
    #     summary.graph(agents[0].model.graph)
    total_s = Stopwatch()
    ########################################
    ## Restoring from last training session
    if resume_trianing:
        # loading up config from last train finish
        print("Restoring progress config from last run...")
        config_path = os.path.join(util.get_savedir(), 'progress.json')
        conf = json.load(open(config_path, 'r'))
        
        starting_gen = conf['gen_at_end'] + 1
        elite_overthrows = conf['elite_overthrows']
        total_steps = conf['total_steps'] 
        total_s.startime = conf['clock_start_time']
        global_step.assign(starting_gen)

        # Loading truncs, elite and refbot
        print(str('='*50) + '\n' + '>> STORM >> Resuming training.\n' + str('='*50))
        trunc_names = os.listdir(util.get_savedir('truncFinals'))
        trunc_names = sorted(trunc_names, reverse=True)

        for j in range(trunc_size):
            agents[j+1].load(util.get_savedir('truncFinals'), trunc_names[j])

        refbot_names = os.listdir(util.get_savedir('refbots'))
        refbot_names = sorted(refbot_names, reverse=False)
        refbot_q_names = refbot_names[-refbot_queue_length:]
        # sec = 0
        # for i in range(5, refbot_queue_length):
        #     refbot_queue[i].load(util.get_savedir('refbots'), refbot_q_names[sec])
        #     refbot_queue[i].refbot_position = i
        #     sec = sec + 1
        for i in range(refbot_queue_length):
            refbot_queue[i].load(util.get_savedir('refbots'), refbot_q_names[i])
            refbot_queue[i].refbot_position = i

        elite = agents[0]
        elite.load(util.get_savedir(), 'elite')

        print(">> STORM >> Successfully restored from last checkpoints")

    print(str('='*50) + '\n' + 'Beginning training (at gen ' + str(starting_gen) + ')\n' + str('='*50))
    s = Stopwatch()

    #partition_stopwatch = Stopwatch()
    for g in range(starting_gen, starting_gen + n_generations):
        #####################
        ## Hyperparameter annealing
        gamma = gamma_func((g+1)/n_generations)

        #####################
        ## GA Algorithm
        for i in range(n_population):
            if g == 0:
                break
            else:
                kappa = random.sample(agents[0:trunc_size], 1)
                mutate(kappa[0], next_generation[i], g)
        #partition_stopwatch.lap('mutation')
        # swap agents and the next gen's agents. i.e set next gen agents to be current agents to evaluate
        tmp = agents
        agents = next_generation
        next_generation = tmp
        
        # evaluate fitness on each agent in population
        try:
            agents, additional_steps, rollout_info = evaluate_fitness(env, agents, refbot, debug=False)
        except KeyboardInterrupt as e:
            print("Received keyboard interrupt {}. Saving and then closing env.".format(e))
            break
        total_steps += additional_steps

        # sort them based on final discounted reward
        agents = sorted(agents, key = lambda agent : agent.fitness_score, reverse=True)

        #partition_stopwatch.lap('fitness evaluation + sorting')
        
        ##################################
        ## Summary information
        with summary.always_record_summaries(): 
            sc_vec = [a.fitness_score for a in agents]
            summary.scalar('rewards/mean', np.mean(sc_vec))
            summary.scalar('rewards/max', agents[0].fitness_score)
            summary.scalar('rewards/min', agents[-1].fitness_score)
            summary.scalar('rewards/var', np.var(sc_vec))
            summary.scalar('rewards/truc_mean', np.mean(sc_vec[:trunc_size]))
            summary.scalar('hyperparameters/gamma', gamma)

            summary.scalar('main_rollout/agentWins', rollout_info['agentWins'])
            summary.scalar('main_rollout/refbotWins', rollout_info['refbotWins'])
            summary.scalar('main_rollout/ties', rollout_info['ties'])
            summary.scalar('main_rollout/early_eps', rollout_info['early_eps'])
            summary.scalar('main_rollout/failed_eps', rollout_info['failed_eps'])

            if len(rollout_info['ep_lengths']) > 0:
                mean_ep_lengg = np.mean(rollout_info['ep_lengths'])
                summary.histogram('main_rollout/ep_lengths', rollout_info['ep_lengths'])
                summary.scalar('main_rollout/mean_ep_length', mean_ep_lengg)
                print("Mean ep length: ", mean_ep_lengg)

            if len(rollout_info['agent_actions']) > 0:
                summary.histogram('main_rollout/agent_a0', rollout_info['agent_actions'])
                summary.histogram('main_rollout/agent_a0_first15steps', rollout_info['agent_early_actions'])
        
        print("Main stats: agent wins - {} | refbot wins - {} | Early - {}".format(rollout_info['agentWins'], rollout_info['refbotWins'], rollout_info['early_eps']))
        for a in agents[:5]:
            print(a.name, " with fitness score: ", a.fitness_score)

        ############################################
        ## Evaluating elite candidates to find elite

        #partition_stopwatch.lap('summaries 1')
        # setup next generation parents / elite agents
        if g == 0:
            if resume_trianing == False:
                elite_candidates = set(agents[0:n_elite_in_royale])
            else:
                elite_candidates = set(agents[0:n_elite_in_royale-1]) | set([elite,])
        else:
            elite_candidates = set(agents[0:n_elite_in_royale-1]) | set([elite,])
        # finding next elite by battling proposed elite candidates for some additional rounds
        #print("Evaluating elite agent...")
        elo_ags, additional_steps, rollout_info = evaluate_fitness(env, elite_candidates, refbot, runs=elite_additional_episodes)
        total_steps += additional_steps
        elo_ags = sorted(elo_ags, key = lambda agent : agent.fitness_score, reverse=True)
        if elite != elo_ags[0]:
            elite_overthrows += 1
        elite = elo_ags[0]

        #partition_stopwatch.lap('elite battle royale')

        try:
            agents.remove(elite)
            agents = [elite,] + agents
        except ValueError:
            agents = [elite,] + agents[:len(agents)-1]

        print("Elite stats: agent wins - {} | refbot wins - {} | Early - {}".format(rollout_info['agentWins'], rollout_info['refbotWins'], rollout_info['early_eps']))
        for i, a in enumerate(elo_ags):   
            print('Elite stats: pos', i, '; name: ', a.name, " ; fitness score: ", a.fitness_score)

        ############################
        ## Summary information 2
        with summary.always_record_summaries(): 
            elite_moving_average.push(elite.fitness_score)
            summary.scalar('rewards/elite_moving_average', elite_moving_average.value())
            summary.scalar('rewards/elite_score', elite.fitness_score)
            summary.scalar('rewards/stable_mean', np.mean([a.fitness_score for a in elo_ags]))
            summary.scalar('time/wall_clock_time', total_s.deltaT())
            summary.scalar('time/single_gen_time', s.deltaT())
            summary.scalar('time/total_game_steps', total_steps)
            summary.scalar('time/elite_overthrows', elite_overthrows)

            summary.scalar('elite_rollout/agentWins', rollout_info['agentWins'])
            summary.scalar('elite_rollout/refbotWins', rollout_info['refbotWins'])
            summary.scalar('elite_rollout/ties', rollout_info['ties'])
            summary.scalar('elite_rollout/early_eps', rollout_info['early_eps'])
            summary.scalar('elite_rollout/failed_eps', rollout_info['failed_eps'])

            if len(rollout_info['ep_lengths']) > 0:
                mean_ep_lengE = np.mean(rollout_info['ep_lengths'])
                summary.histogram('elite_rollout/ep_lengths', rollout_info['ep_lengths'])
                summary.scalar('elite_rollout/mean_ep_length', mean_ep_lengE)
                print("Elite mean ep length: ", mean_ep_lengE)

            if len(rollout_info['agent_actions']) > 0:
                summary.histogram('elite_rollout/agent_a0', rollout_info['agent_actions'])
                summary.histogram('elite_rollout/agent_a0_first15steps', rollout_info['agent_early_actions'])

            summary.scalar('hyperparameters/refbot_back_ind', refbot_back_ind)            

        #################################
        ## Replacing reference bot
        if g % replace_refbot_every == 0:
            toback = refbot
            del refbot_queue[0]
            
            refbot_back_ind = np.random.random_integers(0, refbot_queue_length-1)
            print(str('='*50) + '\n' + '>> STORM >> Upgrading refbot (to pos ' + str(refbot_back_ind) + ') now.\n' + str('='*50) )
            #good_params = agents[trunc_size-1].get_flat_weights()
            good_params = agents[np.random.random_integers(0, trunc_size-1)].get_flat_weights()
            toback.set_flat_weights(good_params)

            refbot_queue.append(toback)
            #refbot = refbot_queue[0]
            ################
            ## Sampling refbot uniformly from past <refbot_queue_length> generation's agents
            refbot = refbot_queue[refbot_back_ind]

            for meme_review, inner_refbot in enumerate(refbot_queue):
                inner_refbot.refbot_position = meme_review
            
            #for bot in refbot_queue:
            #    print("Bot ", bot.name, ' now has refbot pos: ', bot.refbot_position)

        #################################
        ## Saving agents periodically
        if g % save_elite_every == 0 and g != 0:
            elite.save(util.get_savedir('checkpoints'), 'gen' + str(g) + 'elite')
            if refbot_queue_length < 5:
                for refAgent in refbot_queue:
                    refAgent.save(util.get_savedir('refbots'), 'gen' + str(g) + 'pos' + str(refAgent.refbot_position))

            if trunc_size < 5:
                for i, truncAgent in enumerate(agents[:trunc_size]):
                    truncAgent.save(util.get_savedir('truncs'), 'gen' + str(g) + 'agent' + str(i))
            
        global_step.assign_add(1)

        print(str('='*50) + '\n' + 'Generation ' + str(g) + '. Took  ' + s.delta +  '(total: ' + total_s.delta + ')\n' + str('='*50) )
        s.reset()
        #partition_stopwatch.lap('summaries 2 and updates/saves')

    ###############################
    ## Shutdown behavior

    #print("PARTITION STOPWATCH RESULTS:") # last i checked runtime is *dominated*
    #partition_stopwatch.print_results()
    elite.save(util.get_savedir(), elite_savename)
    summary.flush()
    for i, ag in enumerate(agents[:trunc_size]):
        ag.save(util.get_savedir('truncFinals'), 'finalTrunc' + str(i))

    print("End refbot queue: ", len(refbot_queue))
    for identity, refAgent in enumerate(refbot_queue):
        refAgent.save(util.get_savedir('refbots'), 'finalRefbot{:03d}'.format(identity))

    ##########################
    ## Saving progress.config
    conf = {}
    conf['gen_at_end'] = g
    conf['gamma_at_end'] = gamma
    conf['elite_overthrows'] = elite_overthrows
    conf['total_steps'] = total_steps
    conf['clock_start_time'] = total_s.startime
    path = os.path.join(util.get_savedir(), 'progress.json')
    with open(path, 'w') as config_file:
        config_file.write(json.dumps(conf))
    print(">> STORM >> Saved progress.config to: ", path)

def mutate(parent, child, g):
    old_params = parent.get_flat_weights()
    new_params = []
    if verbose_training:
        print(">> storm >> mutating agent: ", parent.name)
    for param in old_params:
        new_params.append(param + sigma*np.random.randn(*param.shape))
    child.tau_lineage = parent.tau_lineage + ["-" + str(g)]
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
    n_envs = env.num_envs
    print(">> ROLLOUTS >> Running rollout wave with queue length  ", init_length)
    pbar = metrics.ProgressBar(init_length)
    interior_steps = 0
    rollout_info = {'early_eps': 0, 
        'failed_eps': 0, 
        'agentWins': 0, 
        'refbotWins': 0, 
        'ties': 0,
        'ep_lengths': [],
        'agent_actions': [],
        'agent_early_actions': []}
    

    while len(queue) > 0:
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

        while step < max_episode_length:
            if debug:
                ss = Stopwatch()
            actions = [agent.step(obs[i][0]) for i, agent in enumerate(cur_playing_agents)]
            
            
            #ref_actions = [refbot.step(obs[i][1]) for i in range(len(obs))]
            ref_actions = refbot.step(obs[:, 1], batch_predict=True)

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
                        if step == max_episode_length-1:
                            rollout_info['early_eps'] += 1

                    elif rews[i][0].code == "FAILURE":
                        # redo this whole fucking batch
                        rollout_info['failed_eps'] += 1
                        failure = True
                        break
                else:
                    inner_rew = rews[i][0]
                    if 'valid' in ep_infos[i].keys():
                        if ep_infos[i]['valid'] == False:
                            if scoring_method == 'binary':
                                inner_rew += invalid_act_penalty_binary
                            else:
                                inner_rew += invalid_act_penalty_dense

                    a.fitness_score = inner_rew + gamma*a.fitness_score
                    _, _, building_act = actions[i]
                    rollout_info['agent_actions'].append(building_act)
                    if step < 15:
                        rollout_info['agent_early_actions'].append(building_act)

                if 'winner' in ep_infos[i].keys():
                    if ep_infos[i]['winner'] == 'A':
                        rollout_info['agentWins'] += 1
                    elif ep_infos[i]['winner'] == 'B':
                        rollout_info['refbotWins'] += 1
                    else:
                        rollout_info['ties'] += 1
                    rollout_info['ep_lengths'].append(ep_infos[i]['n_steps'])

            if failure:
                curQlen = len(queue)
                queue = cur_playing_agents + queue
                print("Failure detected. Redoing last batch... (len Q before = ", curQlen, ' ; after = ', len(queue), ')')
                break

            if debug:
                print("obs shape = ", obs.shape)
                print("rews shape = ", rews.shape)
                print('>> storm >> just took step {}. Took: {}'.format(step, ss.delta))
            step = step + 1
        
        for a in cur_playing_agents:
            a.fitness_averaging_list.append(a.fitness_score)
    
    for a in agents:
        a.squash_fitness_scores()

    pbar.close()

    return agents, interior_steps, rollout_info