import numpy as np
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_fn_wrapper):
	parent_remote.close()
	env = env_fn_wrapper.x()
	while True:
		cmd, data = remote.recv()
		if cmd == 'step':
			ob, reward, done, info = env.step(data)
			
			remote.send((ob, reward, done, info))
		elif cmd == 'reset':
			environ_state = env.reset()

			ob = environ_state[0].observation
			info = environ_state[0].discount
			stepType = environ_state[0].step_type
			if stepType == environment.StepType.LAST: # if current step is the terminal step
				done = True
			else:
				done = False
			reward = environ_state[0].reward

			remote.send((ob, reward, done, info))
		elif cmd == 'close':
			remote.close()
			break
		elif cmd == 'get_spaces':
			# Returns the action space and observation space
			remote.send((env.action_spec(), env.observation_spec()))
		elif cmd == 'action_spec':
			remote.send(env.action_spec())
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

		# The action space is non-stationary, so these lines don't really make sense. 
		#self.remotes[0].send(('get_spaces', None))
		#self.action_space, self.observation_space = self.remotes[0].recv()

	def step(self, actions):
		#print("==========\nENTERING STEP FUNCTION\n=============")
		for remote, action in zip(self.remotes, actions): # so we feed in an array of actions for each agent
			action = [action]

			remote.send(('step', action))
		results = [remote.recv() for remote in self.remotes]
		obs, rews, dones, infos = zip(*results)
		return np.stack(obs), np.stack(rews), np.stack(dones), infos

	def reset(self):
		for remote in self.remotes:
			remote.send(('reset', None))
		
		k = np.stack([remote.recv() for remote in self.remotes])
		print("============\nCLEARED RESET\n============")
		return k

	def reset_task(self):
		for remote in self.remotes:
			remote.send(('reset_task', None))
		return np.stack([remote.recv() for remote in self.remotes])

	def action_spec(self):
		for remote in self.remotes:
			remote.send(('action_spec', None))
		results = [remote.recv() for remote in self.remotes]
		return results

	def close(self):
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