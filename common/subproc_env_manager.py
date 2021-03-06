import numpy as np
from multiprocessing import Process, Pipe
import time

verbose = False

def worker(remote, parent_remote, env_fn_wrapper):
	parent_remote.close()
	env = env_fn_wrapper.x()
	while True:
		cmd, data = remote.recv()
		if cmd == 'step':
			action, p2_act = data
			ob, rew, ep_info = env.step(action, p2_act)
			
			remote.send((ob, rew, ep_info))
		elif cmd == 'reset':
			sucess = env.reset()

			remote.send((sucess,))
		elif cmd == 'getobs':
			ob = env.get_obs()
			remote.send(ob)

		elif cmd == 'close':
			success = env.close()
			env.cleanup()
			remote.close()
			break
		# elif cmd == 'action_spec':
		# 	remote.send(env.action_spec())
		else:
			raise NotImplementedError


class CloudpickleWrapper(object):
	"""
	Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
	"""
	def __init__(self, x):
		self.x = x
	def __getstate__(self):
		import cloudpickle
		return cloudpickle.dumps(self.x)
	def __setstate__(self, ob):
		import pickle
		self.x = pickle.loads(ob)


class SubprocEnvManager(object):
	def __init__(self, env_fns):
		"""
		envs: list of gym environments to run in subprocesses
		"""
		self.closed = False
		nenvs = len(env_fns)
		self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
		self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
			for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
		for p in self.ps:
			p.daemon = True # if the main process crashes, we should not cause things to hang
			p.start()
		for remote in self.work_remotes:
			remote.close()

		self.reset_counter = 0

		# The action space is non-stationary, so these lines don't really make sense. 
		#self.remotes[0].send(('get_spaces', None))
		#self.action_space, self.observation_space = self.remotes[0].recv()

	def step(self, actions, p2_actions=None):
		#print("==========\nENTERING STEP FUNCTION\n=============")
		if p2_actions is None:
			p2_actions = [(0, 0, 3) for _ in range(len(actions))]
		for remote, action, p2_act in zip(self.remotes, actions, p2_actions): # so we feed in an array of actions for each agent

			remote.send(('step', (action, p2_act)))
		results = [remote.recv() for remote in self.remotes]
		#obs, rews, dones, infos = zip(*results)
		obs, rews, ep_infos = zip(*results)
		# for ss in results[0]:
		# 	print("SUBPROC ENV: ", ss, 'with length: ' + str(len(ss)))
		#print(len(ep_infos))
		#print(ep_infos)
		# print(obs[0].shape)
		#print('obs shape', len(obs[0]))
		#print("ref_obs shape", len(ref_obs[]))
		return np.stack(obs, axis=0), np.stack(rews, axis=0), np.stack(ep_infos, axis=0)

	def reset(self):
		self.reset_counter += 1
		for remote in self.remotes:
			remote.send(('reset', None))
		
		k = np.stack([remote.recv() for remote in self.remotes])
		if verbose:
			print("============\nCLEARED RESET (Reset counter = {})\n============".format(self.reset_counter))
		#print("\n-----> Reset counter = {}".format(self.reset_counter))
		time.sleep(0.01)
		return k

	# def action_spec(self):
	# 	for remote in self.remotes:
	# 		remote.send(('action_spec', None))
	# 	results = [remote.recv() for remote in self.remotes]
	# 	return results

	def get_base_obs(self):
		for remote in self.remotes:
			remote.send(('getobs', None))
		
		obs = np.stack([remote.recv() for remote in self.remotes], axis=0)
		return obs

	def close(self):
		print("subproc_env_manager: attempting to close")
		if self.closed:
			return

		for remote in self.remotes:
			remote.send(('close', None))
		for p in self.ps:
			p.join()
		self.closed = True

	@property
	def num_envs(self):
		return len(self.remotes)