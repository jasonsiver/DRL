import gym, threading
import numpy as np
from gym import wrappers
from policy import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
import tensorflow as tf
filename1 = "/home/initial/eclipse-workspace/test/trpo-master/src/log-files/policy_gain/"
filename2 = "/home/initial/eclipse-workspace/test/trpo-master/src/log-files/value_gain/"

env_name='HumanoidasimoMRD3-v1'
#env_name='Humanoid-v1'
num_episodes=200000
gamma=0.995
lam=0.98
kl_targ=0.003
batch_size=32
hid1_mult=10
policy_logvar=-1.0
now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
obs_dim = 266#126#environment._observationDim
obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
act_dim = 17#len(environment._humanoid.motors)
logger = Logger(logname=env_name, now=now)          

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True

       

class Central_agent:
    def __init__(self):
        with tf.name_scope("central_agent"):
            self.val_func = NNValueFunction(obs_dim, hid1_mult)
            self.policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar)
            self.num_tuple = 0
    
    def update_parameter_server(self, episode, trajectories, name):
        self.num_tuple += len(trajectories)
        if len(trajectories) < batch_size:
            return        
        
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = self.build_train_set(trajectories)
        # add various stats to training log:
        self.log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
        self.policy.update(observes, actions, advantages, logger)  # update policy
        self.val_func.fit(observes, disc_sum_rew, logger)  # update value function
        logger.write(display=True)  # write logger results to file and stdout
        print(['thread_name: ' + name + ', episode: ' + str(episode) + ', tuples: ' + str(self.num_tuple)]) 
        if((episode % (batch_size * 3) == 0)):  # & (name == "local_thread3")):
            #print(['stop'])
            self.policy.save(episode, filename1)
            self.val_func.save(episode, filename2)
    def build_train_set(self, trajectories):
        observes = np.concatenate([t['observes'] for t in trajectories])
        actions = np.concatenate([t['actions'] for t in trajectories])
        disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
        advantages = np.concatenate([t['advantages'] for t in trajectories])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        return observes, actions, advantages, disc_sum_rew

    def log_batch_stats(self, observes, actions, advantages, disc_sum_rew, logger, episode):
        logger.log({'_mean_obs': np.mean(observes),
                    '_min_obs': np.min(observes),
                    '_max_obs': np.max(observes),
                    '_std_obs': np.mean(np.var(observes, axis=0)),
                    '_mean_act': np.mean(actions),
                    '_min_act': np.min(actions),
                    '_max_act': np.max(actions),
                    '_std_act': np.mean(np.var(actions, axis=0)),
                    '_mean_adv': np.mean(advantages),
                    '_min_adv': np.min(advantages),
                    '_max_adv': np.max(advantages),
                    '_std_adv': np.var(advantages),
                    '_mean_discrew': np.mean(disc_sum_rew),
                    '_min_discrew': np.min(disc_sum_rew),
                    '_max_discrew': np.max(disc_sum_rew),
                    '_std_discrew': np.var(disc_sum_rew),
                    '_Episode': episode
                    })
        
class Agent:
    def __init__(self, thread_name, thread_type, central_agent):
        self.name = thread_name

        self.episode = 0
        self.central_agent = central_agent

        self.killer = GracefulKiller()
        self.env, obs_dim, act_dim = self.init_gym(env_name)
        obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
        #aigym_path = os.path.join('/home/initial/eclipse-workspace/test/trpo-master/src/result', env_name, now)
        #self.env = wrappers.Monitor(self.env, aigym_path, force=True)
        self.scaler = Scaler(obs_dim)
        #self.central_agent.val_func = NNValueFunction(obs_dim, hid1_mult)
        #self.central_agent.policy = self.central_agent.Policy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar)
        # run a few episodes of untrained policy to initialize scaler:
        self.run_policy(self.env, self.central_agent.policy, self.scaler, logger, episodes=5)
        self.episode = 0  
    def init_gym(self, env_name):
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        return env, obs_dim, act_dim
    def run(self):
        while self.episode < num_episodes:
            trajectories = self.run_policy(self.env, self.central_agent.policy, self.scaler, logger, episodes=batch_size)
            self.episode += len(trajectories)
            self.add_value(trajectories, self.central_agent.val_func)  # add estimated values to episodes
            self.add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
            self.add_gae(trajectories, gamma, lam)  # calculate advantage
            self.central_agent.update_parameter_server(self.episode, trajectories, self.name)
        logger.close()
        self.central_agent.policy.close_sess()
        self.central_agent.val_func.close_sess()            
    def run_episode(self, env, policy, scaler, animate=False):
        obs = env.reset()
        observes, actions, rewards, unscaled_obs = [], [], [], []
        done = False
        step = 0.0
        scale, offset = scaler.get()
        scale[-1] = 1.0  # don't scale time step feature
        offset[-1] = 0.0  # don't offset time step feature
        while not done:
            if animate:
                env.render()
            obs = obs.astype(np.float32).reshape((1, -1))
            obs = np.append(obs, [[step]], axis=1)  # add time step feature
            unscaled_obs.append(obs)
            obs = (obs - offset) * scale  # center and scale observations
            observes.append(obs)
            action = self.central_agent.policy.sample(obs).reshape((1, -1)).astype(np.float32)
            actions.append(action)
            obs, reward, done, _ = self.env.step(np.squeeze(action, axis=0))
            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            rewards.append(reward)
            step += 1e-3  # increment time step feature
            if self.killer.kill_now:
                if input('Terminate training (y/[n])? ') == 'y':
                    break
                self.killer.kill_now = False            

        return (np.concatenate(observes), np.concatenate(actions),
                np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))

    def run_policy(self, env, policy, scaler, logger, episodes):
        total_steps = 0
        trajectories = []
        for e in range(episodes):
            observes, actions, rewards, unscaled_obs = self.run_episode(env, policy, scaler)
            total_steps += observes.shape[0]
            trajectory = {'observes': observes,
                          'actions': actions,
                          'rewards': rewards,
                          'unscaled_obs': unscaled_obs}
            trajectories.append(trajectory)
        unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
        scaler.update(unscaled)  # update running statistics for scaling observations
        logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                    'Steps': total_steps})
        return trajectories
    def discount(self, x, gamma):
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]
    def add_disc_sum_rew(self, trajectories, gamma):
        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            disc_sum_rew = self.discount(rewards, gamma)
            trajectory['disc_sum_rew'] = disc_sum_rew
    def add_value(self, trajectories, val_func):
        for trajectory in trajectories:
            observes = trajectory['observes']
            values = val_func.predict(observes)
            trajectory['values'] = values
    def add_gae(self, trajectories, gamma, lam):
        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            values = trajectory['values']
            # temporal differences
            tds = rewards - values + np.append(values[1:] * gamma, 0)
            advantages = self.discount(tds, gamma * lam)
            trajectory['advantages'] = advantages

    
SESS = tf.Session()
N_WORKERS = 16
def main():
    '''
    env_name='HumanoidasimoMRD3-v1'
    #env_name='Humanoid-v1'
    num_episodes=200000
    gamma=0.995
    lam=0.98
    kl_targ=0.003
    batch_size=32
    hid1_mult=10
    policy_logvar=-1.0
    
    killer = GracefulKiller()
    env, obs_dim, act_dim = init_gym(env_name)
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    logger = Logger(logname=env_name, now=now)
    aigym_path = os.path.join('/home/initial/eclipse-workspace/test/trpo-master/src/result', env_name, now)
    env = wrappers.Monitor(env, aigym_path, force=True)
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, hid1_mult)
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar)
    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, logger, episodes=5)
    episode = 0
    while episode < num_episodes:
        trajectories = run_policy(env, policy, scaler, logger, episodes=batch_size)
        episode += len(trajectories)
        add_value(trajectories, val_func)  # add estimated values to episodes
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lam)  # calculate advantage
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
        policy.update(observes, actions, advantages, logger)  # update policy
        val_func.fit(observes, disc_sum_rew, logger)  # update value function
        logger.write(display=True)  # write logger results to file and stdout
        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False
    logger.close()
    policy.close_sess()
    val_func.close_sess()
    '''

    frames = 0
    isLearned = False
    
    with tf.device("/cpu:0"):
        central_agent = Central_agent()
        threads = []
        for i in range(N_WORKERS):
            thread_name = "local_thread"+str(i+1)
            threads.append(Agent(thread_name, "learning", central_agent))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    
    running_threads = []
    for worker in threads:
        job = lambda: worker.run() 
        t = threading.Thread(target=job)
        t.start()
        
if __name__ == "__main__":
    main()
