import copy, csv, glob, gym, math, os, inspect, random, signal, shutil, time, threading
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import scipy.signal
import tensorflow as tf
from gym.utils import seeding
from sklearn.utils import shuffle

flag_render = True

filename1 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/result/data_policy/data_'
filename2 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/result/data_value_function/data_'
filename3 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/result/data_log/log_train.csv'
filename4 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/character/sphere_blue.urdf'
filename5 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/character/sphere_red.urdf'
filename6 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/character/char.xml'
filename7 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/character/MRchar.urdf'
filename8 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/mocap/biped3d_walk/'
filename9 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/mocap/biped3d_walk.txt'

N_WORKERS = 1
obs_dim = 126#environment._observationDim
obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
act_dim = 17#len(environment._humanoid.motors)
Ts = 0.001
num_episodes = 5000000
gamma = 0.995
lam = 0.98
kl_targ = 0.003
batch_size = 32
hid1_mult = 1#10
policy_logvar = -1.0
RANDOM_SEED = 42

class Policy(object):
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar):
        #self.sess = SESS
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = kl_targ
        self.hid1_mult = hid1_mult
        self.policy_logvar = policy_logvar
        self.epochs = 20
        self.lr = None
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._build_graph()
    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
            self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
            self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
            self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
            self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
            self.lr_ph = tf.placeholder(tf.float32, (), 'eta')
            self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
            self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')
            self._build_model()

            logp = -0.5 * tf.reduce_sum(self.log_vars)
            logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) / tf.exp(self.log_vars), axis=1)
            self.logp = logp
            logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
            logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) / tf.exp(self.old_log_vars_ph), axis=1)
            self.logp_old = logp_old

            log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
            log_det_cov_new = tf.reduce_sum(self.log_vars)
            tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))
            self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new + tf.reduce_sum(tf.square(self.means - self.old_means_ph) / tf.exp(self.log_vars), axis=1) - self.act_dim)
            self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) + tf.reduce_sum(self.log_vars))
            
            self.sampled_act = (self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(self.act_dim,)))

            loss1 = -tf.reduce_mean(self.advantages_ph * tf.exp(self.logp - self.logp_old))
            loss2 = tf.reduce_mean(self.beta_ph * self.kl)
            loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
            self.loss = loss1 + loss2 + loss3
            optimizer = tf.train.AdamOptimizer(self.lr_ph)
            self.train_op = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()  # define a saver for saving and restoring
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)
    def _build_model(self):
        with tf.variable_scope('policy'):
            self.hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
            self.hid3_size = self.act_dim * 10  # 10 empirically determined
            self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))
            self.lr = 9e-4 / np.sqrt(self.hid2_size)  # 9e-4 empirically determined
            out = tf.layers.dense(self.obs_ph, self.hid1_size   , tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out = tf.layers.dense(out        , self.hid2_size   , tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.hid1_size))   , name="h2")
            out = tf.layers.dense(out        , self.hid3_size   , tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.hid2_size))   , name="h3")
            self.means = tf.layers.dense(out        , self.act_dim         , kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.hid3_size))   , name="means")
            logvar_speed = (10 * self.hid3_size) // 48
            log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32, tf.constant_initializer(0.0))
            self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.policy_logvar
            #print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
            #      .format(self.hid1_size, self.hid2_size, self.hid3_size, self.lr, logvar_speed))
    def _act(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.sampled_act, feed_dict=feed_dict)
    def _update(self, observes, actions, advantages, logger):
        feed_dict = {self.obs_ph: observes, self.act_ph: actions, self.advantages_ph: advantages, self.beta_ph: self.beta, self.eta_ph: self.eta, self.lr_ph: self.lr * self.lr_multiplier}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars], feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5
        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier})
        #print('Policy:')
        #print(['PolicyLoss: ' + str(loss)])
        #print(['PolicyEntropy: ' + str(entropy)])
        #print(['KL: ' + str(kl)])
        #print(['Beta: ' + str(self.beta)])
        #print(['_lr_multiplier: ' + str(self.lr_multiplier)])
    def _save(self, episode):
        self.saver.save(self.sess, filename1 + str(episode) + '.ckpt')
    def _restore(self, count):
        self.saver.restore(self.sess, filename1 + str(count) + '.ckpt')
    def _close_sess(self):
        self.sess.close()
    
class Value(object):
    def __init__(self, obs_dim, hid1_mult):
        #self.sess = SESS
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.epochs = 10
        self.lr = None  # learning rate set in _build_graph()
        self._build_graph()
        #self.sess = tf.Session(graph=self.g)
        #self.sess.run(self.init)
    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            self._build_model()
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()  # define a saver for saving and restoring
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)
    def _build_model(self):
        with tf.variable_scope('value'):
            hid1_size = self.obs_dim * self.hid1_mult  # default multipler 10 chosen empirically on 'Hopper-v1'
            hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined
            #print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}' .format(hid1_size, hid2_size, hid3_size, self.lr))
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hid1_size)), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hid2_size)), name="h3")
            out = tf.layers.dense(out, 1                 , kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hid3_size)), name='output')
            self.out = tf.squeeze(out)
    def _update(self, x, y, logger):
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self._predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat) / np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :], self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self._predict(x)
        loss = np.mean(np.square(y_hat - y))  # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func
        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})
        #print('Value')
        #print(['ValFuncLoss: ' + str(loss)])
        #print(['ExplainedVarNew: ' + str(exp_var)])
        #print(['ExplainedVarOld: ' + str(old_exp_var)])
    def _predict(self, x):
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)
        return np.squeeze(y_hat)
    def _save(self, episode):
        self.saver.save(self.sess, filename2 + str(episode) + '.ckpt')
    def _restore(self, count):
        self.saver.restore(self.sess, filename2 + str(count) + '.ckpt')
    def _close_sess(self):
        self.sess.close()
        
class Logger(object):
    def __init__(self):
        path = filename3
        self.write_header = True
        self.log_entry = {}
        self.f = open(path, 'w')
        self.writer = None  # DictWriter created with first call to write() method
    def write(self, display=True):
        if display:
            self.disp(self.log_entry)
        if self.write_header:
            self.writer = csv.DictWriter(self.f, self.log_entry.keys())
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.f.flush()
        self.log_entry = {}
    def log(self, items):
        self.log_entry.update(items)
    def close(self):
        self.f.close()

class MotionReferenceCharacter:
    def __init__(self, dt, tm, rah, qh, curvature, rvh, vh, pos_ini, state):
        self.dt = dt
        self.tm = tm
        self.rah = rah
        self.qh = qh
        self.curvature = curvature
        self.rvh = rvh
        self.vh = vh
        self.MRchar = p.loadURDF(filename7)
        self.foot_step = 0.518  # m/sec
        self.foot_width = 0.17
        x_ini = - self.foot_step / 2.0
        self.pos_MRD = [x_ini, 0.0, 0.01, 0.0]#[pos_ini[0], pos_ini[1], 0.01, pos_ini[3]]  # x,y,z,the
        #self.pos_FP = [pos_ini[0], pos_ini[1]+self.foot_width / 2.0, 0.0, pos_ini[3]]
        #self.old_pos_FP = [pos_ini[0], pos_ini[1]-self.foot_width / 2.0, 0.0, pos_ini[3]]
        #self.pos_FP_right = np.array([0.0, 0.0, 0.0])
        #self.pos_FP_left = np.array([0.0, 0.0, 0.0])
        self.pos_FP_right = [x_ini, -self.foot_width / 2.0, 0.0, 0.0]
        self.pos_FP_left = [x_ini, self.foot_width / 2.0, 0.0, 0.0]
        self.objs_b = p.loadURDF(filename4, [self.pos_FP_right[0], self.pos_FP_right[1], self.pos_FP_right[2]])
        self.objs_r = p.loadURDF(filename5,  [self.pos_FP_left[0], self.pos_FP_left[1], self.pos_FP_left[2]])
        self.foot_pos_left = self.pos_FP_left#self.pos_FP  # []
        self.foot_pos_right = self.pos_FP_right#self.old_pos_FP  # []
        self.dthe = 0.0
        self.total_step = 10
        self.count = 0.0
        self.n = 0
        p.resetBasePositionAndOrientation(self.objs_b, [self.pos_FP_right[0], self.pos_FP_right[1], self.pos_FP_right[2]], [0.0, 0.0, 0.0, 1.0])
        p.resetBasePositionAndOrientation(self.objs_r, [self.pos_FP_left[0], self.pos_FP_left[1], self.pos_FP_left[2]], [0.0, 0.0, 0.0, 1.0])
        #if((-1 * np.power(-1, self.n)) == -1):
        #    p.resetBasePositionAndOrientation(self.objs_b, [self.old_pos_FP[0], self.old_pos_FP[1], self.old_pos_FP[2]], [0.0, 0.0, 0.0, 1.0])
        #    p.resetBasePositionAndOrientation(self.objs_r, [self.pos_FP[0], self.pos_FP[1], self.pos_FP[2]], [0.0, 0.0, 0.0, 1.0])
        #else:
        #    p.resetBasePositionAndOrientation(self.objs_b, [self.pos_FP[0], self.pos_FP[1], self.pos_FP[2]], [0.0, 0.0, 0.0, 1.0])
        #    p.resetBasePositionAndOrientation(self.objs_r, [self.old_pos_FP[0], self.old_pos_FP[1], self.old_pos_FP[2]], [0.0, 0.0, 0.0, 1.0])        
        self.dt_inv = int(1.0 / dt)
        self.i_lis = 0
        self.id_abdomen_x = 2
        self.id_abdomen_y = 3
        self.id_abdomen_z = 4
        self.id_right_hip_x = 5
        self.id_right_hip_y = 6
        self.id_right_hip_z = 7
        self.id_right_knee = 8
        self.id_right_ankle_x = 9
        self.id_right_ankle_y = 10
        self.id_right_ankle_z = 11
        self.id_left_hip_x = 12
        self.id_left_hip_y = 13
        self.id_left_hip_z = 14
        self.id_left_knee = 15
        self.id_left_ankle_x = 16
        self.id_left_ankle_y = 17
        self.id_left_ankle_z = 18
        self.data_abdomen_x = []
        self.data_abdomen_y = []
        self.data_abdomen_z = []
        self.data_right_hip_x = []
        self.data_right_hip_y = []
        self.data_right_hip_z = []
        self.data_right_knee = []
        self.data_right_ankle_x = []
        self.data_right_ankle_y = []
        self.data_right_ankle_z = []
        self.data_left_hip_x = []
        self.data_left_hip_y = []
        self.data_left_hip_z = []
        self.data_left_knee = []
        self.data_left_ankle_x = []
        self.data_left_ankle_y = []
        self.data_left_ankle_z = []
        self.flag_MRD = False
        self.MRD_tm = []
        self.MRD_rah = []
        self.MRD_qh = []
        self.elapsed_time = 0.0
        self.pos_step = np.zeros([7])        
        

        self.root_X_sum = 0.0
        self.root_Y_sum = 0.0
        self.root_Z_sum = 0.0
        self.root_x_sum = 0.0#0.25
        self.root_y_sum = 0.0
        self.root_z_sum = 0.0
        self.dt_mpcap_samp = 0.033332
        self.dt_mocap = 0.633
        self.pos_FP = np.array([0.0,0.0,0.0,0.0])
        self.data_num = 10000
        self.dt_inv = int(self.dt_mocap/self.dt)
        
        ###
        self.total_count = 0
        self.pos_MRD_hist = []


        for i in range(0, self.dt_inv, 1):
            self.count += 1
            #rd = np.random.randn() * 0.05
            dthe = 0.0 / 180.0 * 3.1415# + rd
            #rd2 = np.random.randn() * 0.15
            foot_step2 = self.foot_step/self.dt_mocap# + rd2
            self.old_pos_MRD = self.pos_MRD.copy()
            self.pos_MRD[3] = self.pos_MRD[3] + dthe
            self.pos_MRD[0] = self.pos_MRD[0] + foot_step2 * self.dt * math.cos(self.pos_MRD[3])
            self.pos_MRD[1] = self.pos_MRD[1] + foot_step2 * self.dt * math.sin(self.pos_MRD[3])
            p.addUserDebugLine([self.old_pos_MRD[0], self.old_pos_MRD[1], self.old_pos_MRD[2]], [self.pos_MRD[0], self.pos_MRD[1], self.pos_MRD[2]], [1.0, 0.5, 0.5], 1.0, 100.0)
            if(self.count == self.dt_inv):
                self.pos_MRD_hist.append(self.pos_MRD.copy())
                
                
        idx = 26#, Y = find_nearest(np.asarray(curvature), d_pos_FP)
        self.MRD_tm = self.tm[idx]
        self.MRD_rah = self.rah[idx]
        self.MRD_qh = self.qh[idx]
        self.MRD_rvh = self.rvh[idx]
        self.MRD_vh = self.vh[idx]        
        
        
        self.data_num = self.MRD_qh[:, [7]]
        self.root_X = self.MRD_rah[:, [0]]
        self.root_Y = self.MRD_rah[:, [1]]
        self.root_Z = self.MRD_rah[:, [2]]
        self.root_x = self.MRD_qh[:, [0]]
        self.root_y = self.MRD_qh[:, [1]]
        self.root_z = self.MRD_qh[:, [2]]
        self.root_qw = self.MRD_qh[:, [3]]
        self.root_qx = self.MRD_qh[:, [4]]
        self.root_qy = self.MRD_qh[:, [5]]
        self.root_qz = self.MRD_qh[:, [6]]
        data_abdomen_x_sub = self.MRD_qh[:, [7]]
        data_abdomen_y_sub = self.MRD_qh[:, [8]]
        data_abdomen_z_sub = self.MRD_qh[:, [9]]
        data_right_hip_x_sub = self.MRD_qh[:, [10]]
        data_right_hip_y_sub = self.MRD_qh[:, [11]]
        data_right_hip_z_sub = self.MRD_qh[:, [12]]
        data_right_knee_sub = self.MRD_qh[:, [13]]
        data_right_ankle_x_sub = self.MRD_qh[:, [14]]
        data_right_ankle_y_sub = self.MRD_qh[:, [15]]
        data_right_ankle_z_sub = self.MRD_qh[:, [16]]
        data_left_hip_x_sub = self.MRD_qh[:, [17]]
        data_left_hip_y_sub = self.MRD_qh[:, [18]]
        data_left_hip_z_sub = self.MRD_qh[:, [19]]
        data_left_knee_sub = self.MRD_qh[:, [20]]
        data_left_ankle_x_sub = self.MRD_qh[:, [21]]
        data_left_ankle_y_sub = self.MRD_qh[:, [22]]
        data_left_ankle_z_sub = self.MRD_qh[:, [23]]
        self.root_X_vel = self.MRD_rvh[:, [0]]
        self.root_Y_vel = self.MRD_rvh[:, [1]]
        self.root_Z_vel = self.MRD_rvh[:, [2]]
        self.root_x_vel = self.MRD_vh[:, [0]]
        self.root_y_vel = self.MRD_vh[:, [1]]
        self.root_z_vel = self.MRD_vh[:, [2]]
        self.root_qw_vel = self.MRD_vh[:, [3]]
        self.root_qx_vel = self.MRD_vh[:, [4]]
        self.root_qy_vel = self.MRD_vh[:, [5]]
        self.root_qz_vel = self.MRD_vh[:, [6]]
        data_abdomen_x_vel_sub = self.MRD_vh[:, [7]]
        data_abdomen_y_vel_sub = self.MRD_vh[:, [8]]
        data_abdomen_z_vel_sub = self.MRD_vh[:, [9]]
        data_right_hip_x_vel_sub = self.MRD_vh[:, [10]]
        data_right_hip_y_vel_sub = self.MRD_vh[:, [11]]
        data_right_hip_z_vel_sub = self.MRD_vh[:, [12]]
        data_right_knee_vel_sub = self.MRD_vh[:, [13]]
        data_right_ankle_x_vel_sub = self.MRD_vh[:, [14]]
        data_right_ankle_y_vel_sub = self.MRD_vh[:, [15]]
        data_right_ankle_z_vel_sub = self.MRD_vh[:, [16]]
        data_left_hip_x_vel_sub = self.MRD_vh[:, [17]]
        data_left_hip_y_vel_sub = self.MRD_vh[:, [18]]
        data_left_hip_z_vel_sub = self.MRD_vh[:, [19]]
        data_left_knee_vel_sub = self.MRD_vh[:, [20]]
        data_left_ankle_x_vel_sub = self.MRD_vh[:, [21]]
        data_left_ankle_y_vel_sub = self.MRD_vh[:, [22]]
        data_left_ankle_z_vel_sub = self.MRD_vh[:, [23]]
            
               
        if((-1 * np.power(-1, self.n)) == -1):  # right swing
            self.data_abdomen_x = data_abdomen_x_sub[self.i_lis]
            self.data_abdomen_y = data_abdomen_y_sub[self.i_lis]
            self.data_abdomen_z = data_abdomen_z_sub[self.i_lis]
            self.data_right_hip_x = data_right_hip_x_sub[self.i_lis]
            self.data_right_hip_y = data_right_hip_y_sub[self.i_lis]
            self.data_right_hip_z = data_right_hip_z_sub[self.i_lis]
            self.data_right_knee = data_right_knee_sub[self.i_lis]
            self.data_right_ankle_x = data_right_ankle_x_sub[self.i_lis]
            self.data_right_ankle_y = data_right_ankle_y_sub[self.i_lis]
            self.data_right_ankle_z = data_right_ankle_z_sub[self.i_lis]
            self.data_left_hip_x = data_left_hip_x_sub[self.i_lis]
            self.data_left_hip_y = data_left_hip_y_sub[self.i_lis]
            self.data_left_hip_z = data_left_hip_z_sub[self.i_lis]
            self.data_left_knee = data_left_knee_sub[self.i_lis]
            self.data_left_ankle_x = data_left_ankle_x_sub[self.i_lis]
            self.data_left_ankle_y = data_left_ankle_y_sub[self.i_lis]
            self.data_left_ankle_z = data_left_ankle_z_sub[self.i_lis]  
            self.data_abdomen_x_vel = data_abdomen_x_vel_sub[self.i_lis]
            self.data_abdomen_y_vel = data_abdomen_y_vel_sub[self.i_lis]
            self.data_abdomen_z_vel = data_abdomen_z_vel_sub[self.i_lis]
            self.data_right_hip_x_vel = data_right_hip_x_vel_sub[self.i_lis]
            self.data_right_hip_y_vel = data_right_hip_y_vel_sub[self.i_lis]
            self.data_right_hip_z_vel = data_right_hip_z_vel_sub[self.i_lis]
            self.data_right_knee_vel = data_right_knee_vel_sub[self.i_lis]
            self.data_right_ankle_x_vel = data_right_ankle_x_vel_sub[self.i_lis]
            self.data_right_ankle_y_vel = data_right_ankle_y_vel_sub[self.i_lis]
            self.data_right_ankle_z_vel = data_right_ankle_z_vel_sub[self.i_lis]
            self.data_left_hip_x_vel = data_left_hip_x_vel_sub[self.i_lis]
            self.data_left_hip_y_vel = data_left_hip_y_vel_sub[self.i_lis]
            self.data_left_hip_z_vel = data_left_hip_z_vel_sub[self.i_lis]
            self.data_left_knee_vel = data_left_knee_vel_sub[self.i_lis]
            self.data_left_ankle_x_vel = data_left_ankle_x_vel_sub[self.i_lis]
            self.data_left_ankle_y_vel = data_left_ankle_y_vel_sub[self.i_lis]
            self.data_left_ankle_z_vel = data_left_ankle_z_vel_sub[self.i_lis] 
        else:
            self.data_abdomen_x = data_abdomen_x_sub[self.i_lis]
            self.data_abdomen_y = data_abdomen_y_sub[self.i_lis]
            self.data_abdomen_z = data_abdomen_z_sub[self.i_lis]
            self.data_right_hip_x = data_left_hip_x_sub[self.i_lis]
            self.data_right_hip_y = data_left_hip_y_sub[self.i_lis]
            self.data_right_hip_z = data_left_hip_z_sub[self.i_lis]
            self.data_right_knee = data_left_knee_sub[self.i_lis]
            self.data_right_ankle_x = data_left_ankle_x_sub[self.i_lis]
            self.data_right_ankle_y = data_left_ankle_y_sub[self.i_lis]
            self.data_right_ankle_z = data_left_ankle_z_sub[self.i_lis]
            self.data_left_hip_x = data_right_hip_x_sub[self.i_lis]
            self.data_left_hip_y = data_right_hip_y_sub[self.i_lis]
            self.data_left_hip_z = data_right_hip_z_sub[self.i_lis]
            self.data_left_knee = data_right_knee_sub[self.i_lis]
            self.data_left_ankle_x = data_right_ankle_x_sub[self.i_lis]
            self.data_left_ankle_y = data_right_ankle_y_sub[self.i_lis]
            self.data_left_ankle_z = data_right_ankle_z_sub[self.i_lis]  
            self.data_abdomen_x_vel = data_abdomen_x_vel_sub[self.i_lis]
            self.data_abdomen_y_vel = data_abdomen_y_vel_sub[self.i_lis]
            self.data_abdomen_z_vel = data_abdomen_z_vel_sub[self.i_lis]
            self.data_right_hip_x_vel = data_left_hip_x_vel_sub[self.i_lis]
            self.data_right_hip_y_vel = data_left_hip_y_vel_sub[self.i_lis]
            self.data_right_hip_z_vel = data_left_hip_z_vel_sub[self.i_lis]
            self.data_right_knee_vel = data_left_knee_vel_sub[self.i_lis]
            self.data_right_ankle_x_vel = data_left_ankle_x_vel_sub[self.i_lis]
            self.data_right_ankle_y_vel = data_left_ankle_y_vel_sub[self.i_lis]
            self.data_right_ankle_z_vel = data_left_ankle_z_vel_sub[self.i_lis]
            self.data_left_hip_x_vel = data_right_hip_x_vel_sub[self.i_lis]
            self.data_left_hip_y_vel = data_right_hip_y_vel_sub[self.i_lis]
            self.data_left_hip_z_vel = data_right_hip_z_vel_sub[self.i_lis]
            self.data_left_knee_vel = data_right_knee_vel_sub[self.i_lis]
            self.data_left_ankle_x_vel = data_right_ankle_x_vel_sub[self.i_lis]
            self.data_left_ankle_y_vel = data_right_ankle_y_vel_sub[self.i_lis]
            self.data_left_ankle_z_vel = data_right_ankle_z_vel_sub[self.i_lis]  
            
            
                    
        p.resetJointState(self.MRchar, self.id_abdomen_x       , self.data_abdomen_x)
        p.resetJointState(self.MRchar, self.id_abdomen_y       , self.data_abdomen_y)
        p.resetJointState(self.MRchar, self.id_abdomen_z       , self.data_abdomen_z)
        p.resetJointState(self.MRchar, self.id_right_hip_x     , self.data_right_hip_x)
        p.resetJointState(self.MRchar, self.id_right_hip_y     , self.data_right_hip_y)
        p.resetJointState(self.MRchar, self.id_right_hip_z     , self.data_right_hip_z)
        p.resetJointState(self.MRchar, self.id_right_knee      , self.data_right_knee)
        p.resetJointState(self.MRchar, self.id_right_ankle_x   , self.data_right_ankle_x)
        p.resetJointState(self.MRchar, self.id_right_ankle_y   , self.data_right_ankle_y)
        p.resetJointState(self.MRchar, self.id_right_ankle_z   , self.data_right_ankle_z)
        p.resetJointState(self.MRchar, self.id_left_hip_x      , self.data_left_hip_x)
        p.resetJointState(self.MRchar, self.id_left_hip_y      , self.data_left_hip_y)
        p.resetJointState(self.MRchar, self.id_left_hip_z      , self.data_left_hip_z)
        p.resetJointState(self.MRchar, self.id_left_knee       , self.data_left_knee)
        p.resetJointState(self.MRchar, self.id_left_ankle_x    , self.data_left_ankle_x)
        p.resetJointState(self.MRchar, self.id_left_ankle_y    , self.data_left_ankle_y)
        p.resetJointState(self.MRchar, self.id_left_ankle_z    , self.data_left_ankle_z)                  

        self.pos_FP[0] = self.pos_MRD_hist[-1][0] - math.sin(self.pos_MRD_hist[-1][3]) * self.foot_width/2.0 * (-1.0 * np.power(-1.0, self.n))
        self.pos_FP[1] = self.pos_MRD_hist[-1][1] + math.cos(self.pos_MRD_hist[-1][3]) * self.foot_width/2.0 * (-1.0 * np.power(-1.0, self.n))
        self.pos_FP[3] = self.pos_MRD_hist[-1][3]
        if((-1 * np.power(-1, self.n)) == -1):
            self.pos_FP_right = self.pos_FP.copy()
        else:
            self.pos_FP_left = self.pos_FP.copy()    
        p.resetBasePositionAndOrientation(self.objs_b, [self.pos_FP_right[0], self.pos_FP_right[1], self.pos_FP_right[2]], [0.0, 0.0, 0.0, 1.0]) # left foot
        p.resetBasePositionAndOrientation(self.objs_r, [self.pos_FP_left[0], self.pos_FP_left[1], self.pos_FP_left[2]], [0.0, 0.0, 0.0, 1.0]) # right foot
        qw, qx, qy, qz = _euler_to_quaternion_angle(0.0, 0.0, 0.0)#root_Z[i_lis])
        p.resetBasePositionAndOrientation(self.MRchar, [self.root_x_sum+self.root_x[self.i_lis], self.root_y_sum+self.root_y[self.i_lis], self.root_z_sum+self.root_z[self.i_lis]-0.9], [qx, qy, qz, qw]) # MR character
            
        self.n = 1
        self.count = 0 
        ###

    def _step(self, state, qpos = 0.0, flag_qpos = False, human = 0, flag_next_ok = False):
        if(flag_qpos):
            id_abdomen_x = 2
            id_abdomen_y = 3
            id_abdomen_z = 4
            id_right_hip_x = 6
            id_right_hip_y = 7
            id_right_hip_z = 8
            id_right_knee = 10
            id_right_ankle_x = 12
            id_right_ankle_y = 13
            id_right_ankle_z = 14
            id_left_hip_x = 16
            id_left_hip_y = 17
            id_left_hip_z = 18
            id_left_knee = 20
            id_left_ankle_x = 22
            id_left_ankle_y = 23
            id_left_ankle_z = 24
            self.data_abdomen_x = qpos[7]#0.0#data_abdomen_x_sub[i_lis]
            self.data_abdomen_y = qpos[8]#0.0#data_abdomen_y_sub[i_lis]
            self.data_abdomen_z = qpos[9]#0.0#data_abdomen_z_sub[i_lis]
            self.data_right_hip_x = qpos[10]#0.0#data_right_hip_x_sub[i_lis]
            self.data_right_hip_y = qpos[11]#0.0#data_right_hip_y_sub[i_lis]
            self.data_right_hip_z = qpos[12]#0.0#data_right_hip_z_sub[i_lis]
            self.data_right_knee = qpos[13]#0.0#data_right_knee_sub[i_lis]
            self.data_right_ankle_x = qpos[14]#0.0#data_right_ankle_x_sub[i_lis]
            self.data_right_ankle_y = qpos[15]#0.0#data_right_ankle_y_sub[i_lis]
            self.data_right_ankle_z = qpos[16]#0.0#data_right_ankle_z_sub[i_lis]
            self.data_left_hip_x = qpos[17]#0.0#data_left_hip_x_sub[i_lis]
            self.data_left_hip_y = qpos[18]#0.0#data_left_hip_y_sub[i_lis]
            self.data_left_hip_z = qpos[19]#0.0#data_left_hip_z_sub[i_lis]
            self.data_left_knee = qpos[20]#0.0#data_left_knee_sub[i_lis]
            self.data_left_ankle_x = qpos[21]#0.0#data_left_ankle_x_sub[i_lis]
            self.data_left_ankle_y = qpos[22]#0.0#data_left_ankle_y_sub[i_lis]
            self.data_left_ankle_z = qpos[23]#0.0#data_left_ankle_z_sub[i_lis]
            p.resetJointState(human    , id_abdomen_x       , self.data_abdomen_x)
            p.resetJointState(human    , id_abdomen_y       , self.data_abdomen_y)
            p.resetJointState(human    , id_abdomen_z       , self.data_abdomen_z)
            p.resetJointState(human    , id_right_hip_x     , self.data_right_hip_x)
            p.resetJointState(human    , id_right_hip_y     , self.data_right_hip_y)
            p.resetJointState(human    , id_right_hip_z     , self.data_right_hip_z)
            p.resetJointState(human    , id_right_knee      , self.data_right_knee)
            p.resetJointState(human    , id_right_ankle_x   , self.data_right_ankle_x)
            p.resetJointState(human    , id_right_ankle_y   , self.data_right_ankle_y)
            p.resetJointState(human    , id_right_ankle_z   , self.data_right_ankle_z)
            p.resetJointState(human    , id_left_hip_x      , self.data_left_hip_x)
            p.resetJointState(human    , id_left_hip_y      , self.data_left_hip_y)
            p.resetJointState(human    , id_left_hip_z      , self.data_left_hip_z)
            p.resetJointState(human    , id_left_knee       , self.data_left_knee)
            p.resetJointState(human    , id_left_ankle_x    , self.data_left_ankle_x)
            p.resetJointState(human    , id_left_ankle_y    , self.data_left_ankle_y)
            p.resetJointState(human    , id_left_ankle_z    , self.data_left_ankle_z)
            p.resetBasePositionAndOrientation(human, [qpos[0], qpos[1], qpos[2]], [qpos[4], qpos[5], qpos[6], qpos[3]])
        

        self.total_count += 1
        #print(self.total_count)
        if(self.total_count % self.dt_inv == 0.0):
            flag_next_ok = True
        else:
            flag_next_ok = False
        
        if((self.count < self.dt_inv)) :
            self.count += 1
        
            
            rd = np.random.randn() * 0.05
            dthe = 0.0 / 180.0 * 3.1415# + rd
    
            rd2 = np.random.randn() * 0.15
            foot_step2 = self.foot_step/self.dt_mocap# + rd2
    
            self.old_pos_MRD = self.pos_MRD.copy()
            self.pos_MRD[3] = self.pos_MRD[3] + dthe
            self.pos_MRD[0] = self.pos_MRD[0] + foot_step2 * self.dt * math.cos(self.pos_MRD[3])
            self.pos_MRD[1] = self.pos_MRD[1] + foot_step2 * self.dt * math.sin(self.pos_MRD[3])
            p.addUserDebugLine([self.old_pos_MRD[0], self.old_pos_MRD[1], self.old_pos_MRD[2]], [self.pos_MRD[0], self.pos_MRD[1], self.pos_MRD[2]], [1.0, 0.5, 0.5], 1.0, 10.0)
            if(self.count == self.dt_inv):
                self.pos_MRD_hist.append(self.pos_MRD.copy())
            
            if((self.count == self.dt_inv) & flag_next_ok):
                self.count = 0
                self.pos_FP[0] = self.pos_MRD_hist[-1][0] - math.sin(self.pos_MRD_hist[-1][3]) * self.foot_width/2.0 * (-1.0 * np.power(-1.0, self.n))
                self.pos_FP[1] = self.pos_MRD_hist[-1][1] + math.cos(self.pos_MRD_hist[-1][3]) * self.foot_width/2.0 * (-1.0 * np.power(-1.0, self.n))
                self.pos_FP[3] = self.pos_MRD_hist[-1][3]
                if((-1 * np.power(-1, self.n)) == -1):
                    self.pos_FP_right = self.pos_FP.copy()
                else:
                    self.pos_FP_left = self.pos_FP.copy()        
                # nearest motion in the MRD filenaes
                idx = 26#, Y = find_nearest(np.asarray(curvature), d_pos_FP)
                self.MRD_tm = self.tm[idx]
                self.MRD_rah = self.rah[idx]
                self.MRD_qh = self.qh[idx]
                self.MRD_rvh = self.rvh[idx]
                self.MRD_vh = self.vh[idx]        
                self.flag_MRD=True
                self.dtime = self.MRD_tm[0][0][0]#1.0/len(MRD_tm)
            
            if(self.flag_MRD==True):
                self.data_num = self.MRD_qh[:, [7]]
            
                self.root_X = self.MRD_rah[:, [0]]
                self.root_Y = self.MRD_rah[:, [1]]
                self.root_Z = self.MRD_rah[:, [2]]
                self.root_x = self.MRD_qh[:, [0]]
                self.root_y = self.MRD_qh[:, [1]]
                self.root_z = self.MRD_qh[:, [2]]
                self.root_qw = self.MRD_qh[:, [3]]
                self.root_qx = self.MRD_qh[:, [4]]
                self.root_qy = self.MRD_qh[:, [5]]
                self.root_qz = self.MRD_qh[:, [6]]
                data_abdomen_x_sub = self.MRD_qh[:, [7]]
                data_abdomen_y_sub = self.MRD_qh[:, [8]]
                data_abdomen_z_sub = self.MRD_qh[:, [9]]
                data_right_hip_x_sub = self.MRD_qh[:, [10]]
                data_right_hip_y_sub = self.MRD_qh[:, [11]]
                data_right_hip_z_sub = self.MRD_qh[:, [12]]
                data_right_knee_sub = self.MRD_qh[:, [13]]
                data_right_ankle_x_sub = self.MRD_qh[:, [14]]
                data_right_ankle_y_sub = self.MRD_qh[:, [15]]
                data_right_ankle_z_sub = self.MRD_qh[:, [16]]
                data_left_hip_x_sub = self.MRD_qh[:, [17]]
                data_left_hip_y_sub = self.MRD_qh[:, [18]]
                data_left_hip_z_sub = self.MRD_qh[:, [19]]
                data_left_knee_sub = self.MRD_qh[:, [20]]
                data_left_ankle_x_sub = self.MRD_qh[:, [21]]
                data_left_ankle_y_sub = self.MRD_qh[:, [22]]
                data_left_ankle_z_sub = self.MRD_qh[:, [23]]
                self.root_X_vel = self.MRD_rvh[:, [0]]
                self.root_Y_vel = self.MRD_rvh[:, [1]]
                self.root_Z_vel = self.MRD_rvh[:, [2]]
                self.root_x_vel = self.MRD_vh[:, [0]]
                self.root_y_vel = self.MRD_vh[:, [1]]
                self.root_z_vel = self.MRD_vh[:, [2]]
                self.root_qw_vel = self.MRD_vh[:, [3]]
                self.root_qx_vel = self.MRD_vh[:, [4]]
                self.root_qy_vel = self.MRD_vh[:, [5]]
                self.root_qz_vel = self.MRD_vh[:, [6]]
                data_abdomen_x_vel_sub = self.MRD_vh[:, [7]]
                data_abdomen_y_vel_sub = self.MRD_vh[:, [8]]
                data_abdomen_z_vel_sub = self.MRD_vh[:, [9]]
                data_right_hip_x_vel_sub = self.MRD_vh[:, [10]]
                data_right_hip_y_vel_sub = self.MRD_vh[:, [11]]
                data_right_hip_z_vel_sub = self.MRD_vh[:, [12]]
                data_right_knee_vel_sub = self.MRD_vh[:, [13]]
                data_right_ankle_x_vel_sub = self.MRD_vh[:, [14]]
                data_right_ankle_y_vel_sub = self.MRD_vh[:, [15]]
                data_right_ankle_z_vel_sub = self.MRD_vh[:, [16]]
                data_left_hip_x_vel_sub = self.MRD_vh[:, [17]]
                data_left_hip_y_vel_sub = self.MRD_vh[:, [18]]
                data_left_hip_z_vel_sub = self.MRD_vh[:, [19]]
                data_left_knee_vel_sub = self.MRD_vh[:, [20]]
                data_left_ankle_x_vel_sub = self.MRD_vh[:, [21]]
                data_left_ankle_y_vel_sub = self.MRD_vh[:, [22]]
                data_left_ankle_z_vel_sub = self.MRD_vh[:, [23]]
            
               
                if((-1 * np.power(-1, self.n)) == -1):  # right swing
                    self.data_abdomen_x = data_abdomen_x_sub[self.i_lis]
                    self.data_abdomen_y = data_abdomen_y_sub[self.i_lis]
                    self.data_abdomen_z = data_abdomen_z_sub[self.i_lis]
                    self.data_right_hip_x = data_left_hip_x_sub[self.i_lis]
                    self.data_right_hip_y = data_left_hip_y_sub[self.i_lis]
                    self.data_right_hip_z = data_left_hip_z_sub[self.i_lis]
                    self.data_right_knee = data_left_knee_sub[self.i_lis]
                    self.data_right_ankle_x = data_left_ankle_x_sub[self.i_lis]
                    self.data_right_ankle_y = data_left_ankle_y_sub[self.i_lis]
                    self.data_right_ankle_z = data_left_ankle_z_sub[self.i_lis]
                    self.data_left_hip_x = data_right_hip_x_sub[self.i_lis]
                    self.data_left_hip_y = data_right_hip_y_sub[self.i_lis]
                    self.data_left_hip_z = data_right_hip_z_sub[self.i_lis]
                    self.data_left_knee = data_right_knee_sub[self.i_lis]
                    self.data_left_ankle_x = data_right_ankle_x_sub[self.i_lis]
                    self.data_left_ankle_y = data_right_ankle_y_sub[self.i_lis]
                    self.data_left_ankle_z = data_right_ankle_z_sub[self.i_lis]  
                    self.data_abdomen_x_vel = data_abdomen_x_vel_sub[self.i_lis]
                    self.data_abdomen_y_vel = data_abdomen_y_vel_sub[self.i_lis]
                    self.data_abdomen_z_vel = data_abdomen_z_vel_sub[self.i_lis]
                    self.data_right_hip_x_vel = data_left_hip_x_vel_sub[self.i_lis]
                    self.data_right_hip_y_vel = data_left_hip_y_vel_sub[self.i_lis]
                    self.data_right_hip_z_vel = data_left_hip_z_vel_sub[self.i_lis]
                    self.data_right_knee_vel = data_left_knee_vel_sub[self.i_lis]
                    self.data_right_ankle_x_vel = data_left_ankle_x_vel_sub[self.i_lis]
                    self.data_right_ankle_y_vel = data_left_ankle_y_vel_sub[self.i_lis]
                    self.data_right_ankle_z_vel = data_left_ankle_z_vel_sub[self.i_lis]
                    self.data_left_hip_x_vel = data_right_hip_x_vel_sub[self.i_lis]
                    self.data_left_hip_y_vel = data_right_hip_y_vel_sub[self.i_lis]
                    self.data_left_hip_z_vel = data_right_hip_z_vel_sub[self.i_lis]
                    self.data_left_knee_vel = data_right_knee_vel_sub[self.i_lis]
                    self.data_left_ankle_x_vel = data_right_ankle_x_vel_sub[self.i_lis]
                    self.data_left_ankle_y_vel = data_right_ankle_y_vel_sub[self.i_lis]
                    self.data_left_ankle_z_vel = data_right_ankle_z_vel_sub[self.i_lis]  
                else:
                    self.data_abdomen_x = data_abdomen_x_sub[self.i_lis]
                    self.data_abdomen_y = data_abdomen_y_sub[self.i_lis]
                    self.data_abdomen_z = data_abdomen_z_sub[self.i_lis]
                    self.data_right_hip_x = data_right_hip_x_sub[self.i_lis]
                    self.data_right_hip_y = data_right_hip_y_sub[self.i_lis]
                    self.data_right_hip_z = data_right_hip_z_sub[self.i_lis]
                    self.data_right_knee = data_right_knee_sub[self.i_lis]
                    self.data_right_ankle_x = data_right_ankle_x_sub[self.i_lis]
                    self.data_right_ankle_y = data_right_ankle_y_sub[self.i_lis]
                    self.data_right_ankle_z = data_right_ankle_z_sub[self.i_lis]
                    self.data_left_hip_x = data_left_hip_x_sub[self.i_lis]
                    self.data_left_hip_y = data_left_hip_y_sub[self.i_lis]
                    self.data_left_hip_z = data_left_hip_z_sub[self.i_lis]
                    self.data_left_knee = data_left_knee_sub[self.i_lis]
                    self.data_left_ankle_x = data_left_ankle_x_sub[self.i_lis]
                    self.data_left_ankle_y = data_left_ankle_y_sub[self.i_lis]
                    self.data_left_ankle_z = data_left_ankle_z_sub[self.i_lis]  
                    self.data_abdomen_x_vel = data_abdomen_x_vel_sub[self.i_lis]
                    self.data_abdomen_y_vel = data_abdomen_y_vel_sub[self.i_lis]
                    self.data_abdomen_z_vel = data_abdomen_z_vel_sub[self.i_lis]
                    self.data_right_hip_x_vel = data_right_hip_x_vel_sub[self.i_lis]
                    self.data_right_hip_y_vel = data_right_hip_y_vel_sub[self.i_lis]
                    self.data_right_hip_z_vel = data_right_hip_z_vel_sub[self.i_lis]
                    self.data_right_knee_vel = data_right_knee_vel_sub[self.i_lis]
                    self.data_right_ankle_x_vel = data_right_ankle_x_vel_sub[self.i_lis]
                    self.data_right_ankle_y_vel = data_right_ankle_y_vel_sub[self.i_lis]
                    self.data_right_ankle_z_vel = data_right_ankle_z_vel_sub[self.i_lis]
                    self.data_left_hip_x_vel = data_left_hip_x_vel_sub[self.i_lis]
                    self.data_left_hip_y_vel = data_left_hip_y_vel_sub[self.i_lis]
                    self.data_left_hip_z_vel = data_left_hip_z_vel_sub[self.i_lis]
                    self.data_left_knee_vel = data_left_knee_vel_sub[self.i_lis]
                    self.data_left_ankle_x_vel = data_left_ankle_x_vel_sub[self.i_lis]
                    self.data_left_ankle_y_vel = data_left_ankle_y_vel_sub[self.i_lis]
                    self.data_left_ankle_z_vel = data_left_ankle_z_vel_sub[self.i_lis]  
                    
                    
                    
                p.resetJointState(self.MRchar, self.id_abdomen_x       , self.data_abdomen_x)
                p.resetJointState(self.MRchar, self.id_abdomen_y       , self.data_abdomen_y)
                p.resetJointState(self.MRchar, self.id_abdomen_z       , self.data_abdomen_z)
                p.resetJointState(self.MRchar, self.id_right_hip_x     , self.data_right_hip_x)
                p.resetJointState(self.MRchar, self.id_right_hip_y     , self.data_right_hip_y)
                p.resetJointState(self.MRchar, self.id_right_hip_z     , self.data_right_hip_z)
                p.resetJointState(self.MRchar, self.id_right_knee      , self.data_right_knee)
                p.resetJointState(self.MRchar, self.id_right_ankle_x   , self.data_right_ankle_x)
                p.resetJointState(self.MRchar, self.id_right_ankle_y   , self.data_right_ankle_y)
                p.resetJointState(self.MRchar, self.id_right_ankle_z   , self.data_right_ankle_z)
                p.resetJointState(self.MRchar, self.id_left_hip_x      , self.data_left_hip_x)
                p.resetJointState(self.MRchar, self.id_left_hip_y      , self.data_left_hip_y)
                p.resetJointState(self.MRchar, self.id_left_hip_z      , self.data_left_hip_z)
                p.resetJointState(self.MRchar, self.id_left_knee       , self.data_left_knee)
                p.resetJointState(self.MRchar, self.id_left_ankle_x    , self.data_left_ankle_x)
                p.resetJointState(self.MRchar, self.id_left_ankle_y    , self.data_left_ankle_y)
                p.resetJointState(self.MRchar, self.id_left_ankle_z    , self.data_left_ankle_z)  
        
            #if((-1 * np.power(-1, n)) == -1):  # right swing
            p.resetBasePositionAndOrientation(self.objs_b, [self.pos_FP_right[0], self.pos_FP_right[1], self.pos_FP_right[2]], [0.0, 0.0, 0.0, 1.0]) # left foot
            p.resetBasePositionAndOrientation(self.objs_r, [self.pos_FP_left[0], self.pos_FP_left[1], self.pos_FP_left[2]], [0.0, 0.0, 0.0, 1.0]) # right foot
            if(self.flag_MRD==True):
                qw, qx, qy, qz = _euler_to_quaternion_angle(0.0, 0.0, 0.0)#root_Z[i_lis])
                p.resetBasePositionAndOrientation(self.MRchar, [self.root_x_sum+self.root_x[self.i_lis], self.root_y_sum+self.root_y[self.i_lis], self.root_z_sum+self.root_z[self.i_lis]-0.9], [qx, qy, qz, qw]) # MR character
            
            if(self.flag_MRD==True):
                self.elapsed_time = self.elapsed_time + self.dt#_mpcap_samp
                if(self.elapsed_time >=self.dtime):
                    self.elapsed_time = 0.0
                    self.i_lis = self.i_lis + 1
                if(self.i_lis >= (len(self.data_num)-1)):
                    self.root_Y_sum += 0.0#self.root_Y[self.i_lis-1]
                    self.root_Z_sum += 0.0#self.root_Z[self.i_lis-1]
                    self.root_x_sum += self.root_x[self.i_lis-1]
                    self.root_y_sum += 0.0#self.root_y[self.i_lis-1]
                    self.root_z_sum += 0.0#root_z[i_lis-1]
                    self.i_lis = 0
                    self. n += 1

class Character:
    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01, rah=[], qh=[], mc=[], episode=0):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.rah = rah
        self.qh = qh
        self.motion_copy = mc
        self.reset(episode)
        self.data_abdomen_x = 0.0
        self.data_abdomen_y = 0.0
        self.data_abdomen_z = 0.0
        self.data_right_hip_x = 0.0
        self.data_right_hip_y = 0.0
        self.data_right_hip_z = 0.0
        self.data_right_knee = 0.0
        self.data_right_ankle_x = 0.0
        self.data_right_ankle_y = 0.0
        self.data_right_ankle_z = 0.0
        self.data_left_hip_x = 0.0
        self.data_left_hip_y = 0.0
        self.data_left_hip_z = 0.0
        self.data_left_knee = 0.0
        self.data_left_ankle_x = 0.0
        self.data_left_ankle_y = 0.0
        self.data_left_ankle_z = 0.0
    def reset(self, i_lis):
        self.initial_z = None
        objs = p.loadMJCF(self.urdfRootPath, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        self.human = objs[0]
        self.jdict = {}
        self.ordered_joints = []
        self.ordered_joint_indices = []    
        data_abdomen_x_sub = self.motion_copy[:, [7]]
        data_abdomen_y_sub = self.motion_copy[:, [8]]
        data_abdomen_z_sub = self.motion_copy[:, [9]]
        data_right_hip_x_sub = self.motion_copy[:, [10]]
        data_right_hip_y_sub = self.motion_copy[:, [11]]
        data_right_hip_z_sub = self.motion_copy[:, [12]]
        data_right_knee_sub = self.motion_copy[:, [13]]
        data_right_ankle_x_sub = self.motion_copy[:, [14]]
        data_right_ankle_y_sub = self.motion_copy[:, [15]]
        data_right_ankle_z_sub = self.motion_copy[:, [16]]
        data_left_hip_x_sub = self.motion_copy[:, [17]]
        data_left_hip_y_sub = self.motion_copy[:, [18]]
        data_left_hip_z_sub = self.motion_copy[:, [19]]
        data_left_knee_sub = self.motion_copy[:, [20]]
        data_left_ankle_x_sub = self.motion_copy[:, [21]]
        data_left_ankle_y_sub = self.motion_copy[:, [22]]
        data_left_ankle_z_sub = self.motion_copy[:, [23]]
        id_abdomen_x = 2
        id_abdomen_y = 3
        id_abdomen_z = 4
        id_right_hip_x = 6
        id_right_hip_y = 7
        id_right_hip_z = 8
        id_right_knee = 10
        id_right_ankle_x = 12
        id_right_ankle_y = 13
        id_right_ankle_z = 14
        id_left_hip_x = 16
        id_left_hip_y = 17
        id_left_hip_z = 18
        id_left_knee = 20
        id_left_ankle_x = 22
        id_left_ankle_y = 23
        id_left_ankle_z = 24
        for j in range(p.getNumJoints(self.human)):
            info = p.getJointInfo(self.human, j)
            # print(info)
            link_name = info[12].decode("ascii")
            if link_name == "left_foot":
                self.left_foot = j
            if link_name == "right_foot": 
                self.right_foot = j
            self.ordered_joint_indices.append(j)
            if info[2] != p.JOINT_REVOLUTE: continue
            jname = info[1].decode("ascii")
            self.jdict[jname] = j
            lower, upper = (info[8], info[9])
            self.ordered_joints.append((j, lower, upper))
            p.setJointMotorControl2(self.human, j, controlMode=p.VELOCITY_CONTROL, force=0)
        self.data_abdomen_x = 0.0#data_abdomen_x_sub[i_lis]
        self.data_abdomen_y = 0.0#data_abdomen_y_sub[i_lis]
        self.data_abdomen_z = 0.0#data_abdomen_z_sub[i_lis]
        self.data_right_hip_x = 0.0#data_right_hip_x_sub[i_lis]
        self.data_right_hip_y = 0.0#data_right_hip_y_sub[i_lis]
        self.data_right_hip_z = 0.0#data_right_hip_z_sub[i_lis]
        self.data_right_knee = 0.0#data_right_knee_sub[i_lis]
        self.data_right_ankle_x = 0.0#data_right_ankle_x_sub[i_lis]
        self.data_right_ankle_y = 0.0#data_right_ankle_y_sub[i_lis]
        self.data_right_ankle_z = 0.0#data_right_ankle_z_sub[i_lis]
        self.data_left_hip_x = 0.0#data_left_hip_x_sub[i_lis]
        self.data_left_hip_y = 0.0#data_left_hip_y_sub[i_lis]
        self.data_left_hip_z = 0.0#data_left_hip_z_sub[i_lis]
        self.data_left_knee = 0.0#data_left_knee_sub[i_lis]
        self.data_left_ankle_x = 0.0#data_left_ankle_x_sub[i_lis]
        self.data_left_ankle_y = 0.0#data_left_ankle_y_sub[i_lis]
        self.data_left_ankle_z = 0.0#data_left_ankle_z_sub[i_lis]
        p.resetJointState(self.human    , id_abdomen_x       , self.data_abdomen_x)
        p.resetJointState(self.human    , id_abdomen_y       , self.data_abdomen_y)
        p.resetJointState(self.human    , id_abdomen_z       , self.data_abdomen_z)
        p.resetJointState(self.human    , id_right_hip_x     , self.data_right_hip_x)
        p.resetJointState(self.human    , id_right_hip_y     , self.data_right_hip_y)
        p.resetJointState(self.human    , id_right_hip_z     , self.data_right_hip_z)
        p.resetJointState(self.human    , id_right_knee      , self.data_right_knee)
        p.resetJointState(self.human    , id_right_ankle_x   , self.data_right_ankle_x)
        p.resetJointState(self.human    , id_right_ankle_y   , self.data_right_ankle_y)
        p.resetJointState(self.human    , id_right_ankle_z   , self.data_right_ankle_z)
        p.resetJointState(self.human    , id_left_hip_x      , self.data_left_hip_x)
        p.resetJointState(self.human    , id_left_hip_y      , self.data_left_hip_y)
        p.resetJointState(self.human    , id_left_hip_z      , self.data_left_hip_z)
        p.resetJointState(self.human    , id_left_knee       , self.data_left_knee)
        p.resetJointState(self.human    , id_left_ankle_x    , self.data_left_ankle_x)
        p.resetJointState(self.human    , id_left_ankle_y    , self.data_left_ankle_y)
        p.resetJointState(self.human    , id_left_ankle_z    , self.data_left_ankle_z)
        p.setJointMotorControl2(self.human    , id_abdomen_x       , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=200 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_abdomen_y       , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=200 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_abdomen_z       , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=200 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_right_hip_x     , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=200 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_right_hip_y     , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=200 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_right_hip_z     , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=200 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_right_knee      , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=150 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_right_ankle_x   , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=90 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_right_ankle_y   , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=90 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_right_ankle_z   , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=90 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_left_hip_x      , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=200 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_left_hip_y      , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=200 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_left_hip_z      , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=200 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_left_knee       , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=150 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_left_ankle_x    , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=90 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_left_ankle_y    , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=90 * 0.082 * 0.0)
        p.setJointMotorControl2(self.human    , id_left_ankle_z    , controlMode=p.VELOCITY_CONTROL , targetVelocity=0.0  , force=90 * 0.082 * 0.0)    
        p.resetBaseVelocity(self.human, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        body_xyz, (qx, qy, qz, qw) = p.getBasePositionAndOrientation(self.human)
        self.getBaseState = [body_xyz[0], body_xyz[1], body_xyz[2], qx, qy, qz, qw]
        self.motor_names = ["abdomen_x", "abdomen_y", "abdomen_z"]
        self.motor_power = [200, 200, 200]
        self.motor_names += ["right_hip_x", "right_hip_y", "right_hip_z", "right_knee", "right_ankle_x", "right_ankle_y", "right_ankle_z"]
        self.motor_power += [200, 200, 200, 150, 90, 90, 90]
        self.motor_names += ["left_hip_x", "left_hip_y", "left_hip_z", "left_knee", "left_ankle_x", "left_ankle_y", "left_ankle_z"]
        self.motor_power += [200, 200, 200, 150, 90, 90, 90]
        self.motors = [self.jdict[n] for n in self.motor_names]
    
    def applyAction(self, actions):
        forces = [0.] * len(self.motors)
        for m in range(len(self.motors)):
            forces[m] = self.motor_power[m] * actions[m]  # *0.082
        p.setJointMotorControlArray(self.human, self.motors, controlMode=p.TORQUE_CONTROL, forces=forces)
        p.stepSimulation()

class Environment:
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second' : 50}
    def __init__(self, urdfRoot=pybullet_data.getDataPath(), actionRepeat=50, isEnableSelfCollision=True, renders=True, time_step=0.01, obs_dim=0):
        # print("init")
        self.Path = filename6
        self._timeStep = time_step
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._p = p
        if self._renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self._seed()
        self._obs_dim = obs_dim 
        filepath_prefix = filename8
        filename_list = open(filepath_prefix + "mocap_list.txt", "r")
        str_list = filename_list.readlines()
        filename_list.close()
        self.tm = []
        self.rah = []
        self.qh = []
        for i_lis in range(0, len(str_list), 1):
            if str_list[i_lis][-1] == "\n": 
                filepath = filepath_prefix + str_list[i_lis][:-1]
            else:
                filepath = filepath_prefix + str_list[i_lis][:]
            filename = open(filepath, "r")
            str_txt = filename.readlines()
            filename.close()
            time = []
            pos = []
            for i in range(0, len(str_txt), 1):
                pos_sub = []
                if len(str_txt[i]) > 10:
                    for j in str_txt[i][:].split(","):
                        pos_sub.append(np.float32(j))
                    if i == 0:
                        time = pos_sub[0].copy()
                        pos = pos_sub[1:].copy()
                    else:
                        time = np.vstack((time, pos_sub[0]))
                        pos = np.vstack((pos, pos_sub[1:]))
            time_hist = []  # np.zeros([len(pos), 24])
            qpos_hist = np.zeros([len(pos), 24])
            # qvel_hist = np.zeros([len(pos), 24])
            root_angle_hist = np.zeros([len(pos), 3])
            for i in range(0, len(pos), 1):
                qpos_sub, root_angle_sub = self._qpos_transform(pos[i])
                time_hist.append(time) 
                qpos_hist[i] = qpos_sub.transpose()
                root_angle_hist[i] = root_angle_sub.transpose()
            self.tm.append(time_hist)
            self.rah.append(root_angle_hist)
            self.qh.append(qpos_hist)
            self.curvature = []
        for i in range(0, len(self.qh), 1):
            x = []
            y = []
            z = []
            x_sub = self.qh[i][:, [0]].copy()
            y_sub = self.qh[i][:, [1]].copy()
            z_sub = self.qh[i][:, [2]].copy()
            for j in range(0, len(self.qh[i]), 1):
                x.append(x_sub[j][0].copy())
                y.append(y_sub[j][0].copy())
                z.append(z_sub[j][0].copy())
            ans = math.atan2(y[len(y) - 1] - y[0], x[len(x) - 1] - x[0])
            self.curvature.append(ans)    
        self.rvh = []
        self.vh = []
        for i in range(0, len(self.qh), 1):
            rah_temp = self.rah[i]
            qh_temp = self.qh[i]
            rvel_sub = np.zeros([len(self.rah[i]), len(self.rah[i][0])])
            vel_sub = np.zeros([len(self.qh[i]), len(self.qh[i][0])])
            for j in range(0, len(qh_temp), 1):
                root_X = rah_temp[:, [0]]
                root_Y = rah_temp[:, [1]]
                root_Z = rah_temp[:, [2]]
                root_x = qh_temp[:, [0]]
                root_y = qh_temp[:, [1]]
                root_z = qh_temp[:, [2]]
                root_qw = qh_temp[:, [3]]
                root_qx = qh_temp[:, [4]]
                root_qy = qh_temp[:, [5]]
                root_qz = qh_temp[:, [6]]
                data_abdomen_x = qh_temp[:, [7]]
                data_abdomen_y = qh_temp[:, [8]]
                data_abdomen_z = qh_temp[:, [9]]
                data_right_hip_x = qh_temp[:, [10]]
                data_right_hip_y = qh_temp[:, [11]]
                data_right_hip_z = qh_temp[:, [12]]
                data_right_knee = qh_temp[:, [13]]
                data_right_ankle_x = qh_temp[:, [14]]
                data_right_ankle_y = qh_temp[:, [15]]
                data_right_ankle_z = qh_temp[:, [16]]
                data_left_hip_x = qh_temp[:, [17]]
                data_left_hip_y = qh_temp[:, [18]]
                data_left_hip_z = qh_temp[:, [19]]
                data_left_knee = qh_temp[:, [20]]
                data_left_ankle_x = qh_temp[:, [21]]
                data_left_ankle_y = qh_temp[:, [22]]
                data_left_ankle_z = qh_temp[:, [23]]
                root_X_vel = self._pos_to_vel(root_X)
                root_Y_vel = self._pos_to_vel(root_Y)
                root_Z_vel = self._pos_to_vel(root_Z)
                root_x_vel = self._pos_to_vel(root_x)
                root_y_vel = self._pos_to_vel(root_y)
                root_z_vel = self._pos_to_vel(root_z)
                root_qw_vel = self._pos_to_vel(root_qw)
                root_qx_vel = self._pos_to_vel(root_qx)
                root_qy_vel = self._pos_to_vel(root_qy)
                root_qz_vel = self._pos_to_vel(root_qz)
                data_abdomen_x_vel = self._pos_to_vel(data_abdomen_x)
                data_abdomen_y_vel = self._pos_to_vel(data_abdomen_y)
                data_abdomen_z_vel = self._pos_to_vel(data_abdomen_z)
                data_right_hip_x_vel = self._pos_to_vel(data_right_hip_x)
                data_right_hip_y_vel = self._pos_to_vel(data_right_hip_y)
                data_right_hip_z_vel = self._pos_to_vel(data_right_hip_z)
                data_right_knee_vel = self._pos_to_vel(data_right_knee)
                data_right_ankle_x_vel = self._pos_to_vel(data_right_ankle_x)
                data_right_ankle_y_vel = self._pos_to_vel(data_right_ankle_y)
                data_right_ankle_z_vel = self._pos_to_vel(data_right_ankle_z)
                data_left_hip_x_vel = self._pos_to_vel(data_left_hip_x)
                data_left_hip_y_vel = self._pos_to_vel(data_left_hip_y)
                data_left_hip_z_vel = self._pos_to_vel(data_left_hip_z)
                data_left_knee_vel = self._pos_to_vel(data_left_knee)
                data_left_ankle_x_vel = self._pos_to_vel(data_left_ankle_x)
                data_left_ankle_y_vel = self._pos_to_vel(data_left_ankle_y)
                data_left_ankle_z_vel = self._pos_to_vel(data_left_ankle_z)
                rvel_sub[:, [0]] = root_X_vel
                rvel_sub[:, [1]] = root_Y_vel
                rvel_sub[:, [2]] = root_Z_vel
                vel_sub[:, [0]] = root_x_vel
                vel_sub[:, [1]] = root_y_vel
                vel_sub[:, [2]] = root_z_vel
                vel_sub[:, [3]] = root_qw_vel
                vel_sub[:, [4]] = root_qx_vel
                vel_sub[:, [5]] = root_qy_vel
                vel_sub[:, [6]] = root_qz_vel
                vel_sub[:, [7]] = data_abdomen_x_vel
                vel_sub[:, [8]] = data_abdomen_y_vel
                vel_sub[:, [9]] = data_abdomen_z_vel
                vel_sub[:, [10]] = data_right_hip_x_vel
                vel_sub[:, [11]] = data_right_hip_y_vel
                vel_sub[:, [12]] = data_right_hip_z_vel
                vel_sub[:, [13]] = data_right_knee_vel
                vel_sub[:, [14]] = data_right_ankle_x_vel
                vel_sub[:, [15]] = data_right_ankle_y_vel
                vel_sub[:, [16]] = data_right_ankle_z_vel
                vel_sub[:, [17]] = data_left_hip_x_vel
                vel_sub[:, [18]] = data_left_hip_y_vel
                vel_sub[:, [19]] = data_left_hip_z_vel
                vel_sub[:, [20]] = data_left_knee_vel
                vel_sub[:, [21]] = data_left_ankle_x_vel
                vel_sub[:, [22]] = data_left_ankle_y_vel
                vel_sub[:, [23]] = data_left_ankle_z_vel
            self.rvh.append(rvel_sub)
            self.vh.append(vel_sub)
        filename = open(filename9 , "r")
        str_txt = filename.readlines()
        filename.close()
        motion = []
        self.motion_copy = []
        for i in range(4, len(str_txt) - 1, 1):
            motion = []
            if len(str_txt[i]) > 10:
                for j in str_txt[i][3:-3].split(","):
                    motion.append(np.float32(j))
                if i == 4:
                    time = motion[0].copy()
                    self.motion_copy = motion[1:].copy()
                else:
                    time = np.vstack((time, motion[0]))
                    self.motion_copy = np.vstack((self.motion_copy, motion[1:]))
        qpos_hist = np.zeros([len(self.motion_copy), 24])
        for i in range(0, len(self.motion_copy) - 1, 1):
            qpos_sub, _ = self._qpos_transform(self.motion_copy[i])
            qpos_hist[i] = qpos_sub.transpose()
        self.motion_copy = qpos_hist.copy()
        self.pos_ini = [root_x[0], root_y[0], root_z[0], root_Z[0]]
        self.human_state = np.zeros([7])
        self._reset()
        #self._state, self._observation = self.getObservation()
        self._state = self.getObservation()
        self._observationDim = self._obs_dim  # len(state)
        #observation_high = np.array([np.finfo(np.float32).max] * self._observationDim)    
        self.viewer = None
    def _pos_to_vel(self, data):
        data_vel = np.diff(data.transpose())
        data_vel = data_vel.transpose()
        data_vel = np.vstack((data_vel[0], data_vel))
        return data_vel
    def _reset(self, episode=0):
        num = len(self.motion_copy)
        episode = episode % num
        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"))        
        self._humanoid = Character(urdfRootPath=self.Path, timeStep=self._timeStep, rah=self.rah, qh=self.qh, mc=self.motion_copy, episode=episode)
        body_xyz, body_XYZ= p.getBasePositionAndOrientation(self._humanoid.human)
        self.human_state[0] = body_xyz[0]
        self.human_state[1] = body_xyz[1]
        self.human_state[2] = body_xyz[2]
        self.human_state[3] = body_XYZ[0]
        self.human_state[4] = body_XYZ[1]
        self.human_state[5] = body_XYZ[2]
        self.human_state[6] = body_XYZ[3]
        self._MRD = MotionReferenceCharacter(self._timeStep, self.tm, self.rah, self.qh, self.curvature, self.rvh, self.vh, self.pos_ini, self.human_state)
        p.setGravity(0, 0, -10.0)
        self._envStepCounter = 0
        p.stepSimulation()
        #self._state, self._observation = self.getObservation()
        self._state = self.getObservation()
        return np.array(self._state)
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def getObservation(self):
        #self._observation = self._humanoid.getObservation()
        state = np.zeros([self._obs_dim])
        body_xyz, (qx, qy, qz, qw) = p.getBasePositionAndOrientation(self._humanoid.human)
        linearvelocity, angularvelocity = p.getBaseVelocity(self._humanoid.human)
        ex, ey, ez = _quaternion_to_euler_angle(qw, qx, qy, qz)
        state[0] = body_xyz[2]
        state[1] = body_xyz[0]
        state[2] = body_xyz[1]
        state[3] = body_xyz[2]
        state[4] = qw
        state[5] = qx
        state[6] = qy
        state[7] = qz
        state[8] = linearvelocity[0]
        state[9] = linearvelocity[1]
        state[10] = linearvelocity[2]
        state[11] = angularvelocity[0]
        state[12] = angularvelocity[1]
        state[13] = angularvelocity[2]
        Link = p.getLinkState(self._humanoid.human, 1)
        linkWorldPosition = Link[0]  # (0.0, 0.0, 0.9899999871850014)
        linkWorldOrientation = Link[1]  # (0.0, 0.0, 0.0, 1.0)
        dx = body_xyz[0] - linkWorldPosition[0]
        dy = body_xyz[1] - linkWorldPosition[1]
        dz = body_xyz[2] - linkWorldPosition[2]
        state[14] = dx  # relative_pos_x
        state[15] = dy  # relative_pos_y
        state[16] = dz  # relative_pos_z
        ex_link, ey_link, ez_link = _quaternion_to_euler_angle(linkWorldOrientation[3], linkWorldOrientation[0], linkWorldOrientation[1], 2)
        dex = ex - ex_link
        dey = ey - ey_link
        dez = ez - ez_link
        dqw, dqx, dqy, dqz = _euler_to_quaternion_angle(dex, dey, dez)
        state[17] = dqw  # relative_angle_w
        state[18] = dqx  # relative_angle_x
        state[19] = dqy  # relative_angle_y
        state[20] = dqz  # relative_angle_z
        dvel_x = dx / self._timeStep
        dvel_y = dy / self._timeStep
        dvel_z = dz / self._timeStep
        state[21] = dvel_x  # rel_vel_x
        state[22] = dvel_y  # rel_vel_y
        state[23] = dvel_z  # rel_vel_z
        dang_vel_x = dex / self._timeStep
        dang_vel_y = dey / self._timeStep
        dang_vel_z = dez / self._timeStep
        state[24] = dang_vel_x  # rel_angular_vel_x
        state[25] = dang_vel_y  # rel_angular_vel_y
        state[26] = dang_vel_z  # rel_angular_vel_z
        Link = p.getLinkState(self._humanoid.human, 5)
        linkWorldPosition = Link[0]  # (0.0, 0.0, 0.9899999871850014)
        linkWorldOrientation = Link[1]  # (0.0, 0.0, 0.0, 1.0)
        dx = body_xyz[0] - linkWorldPosition[0]
        dy = body_xyz[1] - linkWorldPosition[1]
        dz = body_xyz[2] - linkWorldPosition[2]
        state[27] = dx  # relative_pos_x
        state[28] = dy  # relative_pos_y
        state[29] = dz  # relative_pos_z
        ex_link, ey_link, ez_link = _quaternion_to_euler_angle(linkWorldOrientation[3], linkWorldOrientation[0], linkWorldOrientation[1], 2)
        dex = ex - ex_link
        dey = ey - ey_link
        dez = ez - ez_link
        dqw, dqx, dqy, dqz = _euler_to_quaternion_angle(dex, dey, dez)
        state[30] = dqw  # relative_angle_w
        state[31] = dqx  # relative_angle_x
        state[32] = dqy  # relative_angle_y
        state[33] = dqz  # relative_angle_z
        dvel_x = dx / self._timeStep
        dvel_y = dy / self._timeStep
        dvel_z = dz / self._timeStep
        state[34] = dvel_x  # rel_vel_x
        state[35] = dvel_y  # rel_vel_y
        state[36] = dvel_z  # rel_vel_z
        dang_vel_x = dex / self._timeStep
        dang_vel_y = dey / self._timeStep
        dang_vel_z = dez / self._timeStep
        state[37] = dang_vel_x  # rel_angular_vel_x
        state[38] = dang_vel_y  # rel_angular_vel_y
        state[39] = dang_vel_z  # rel_angular_vel_z
        Link = p.getLinkState(self._humanoid.human, 9)
        linkWorldPosition = Link[0]
        linkWorldOrientation = Link[1]    
        dx = body_xyz[0] - linkWorldPosition[0]
        dy = body_xyz[1] - linkWorldPosition[1]
        dz = body_xyz[2] - linkWorldPosition[2]
        state[40] = dx  # relative_pos_x
        state[41] = dy  # relative_pos_y
        state[42] = dz  # relative_pos_z
        ex_link, ey_link, ez_link = _quaternion_to_euler_angle(linkWorldOrientation[3], linkWorldOrientation[0], linkWorldOrientation[1], 2)
        dex = ex - ex_link
        dey = ey - ey_link
        dez = ez - ez_link
        dqw, dqx, dqy, dqz = _euler_to_quaternion_angle(dex, dey, dez)
        state[43] = dqw  # relative_angle_w
        state[44] = dqx  # relative_angle_x
        state[45] = dqy  # relative_angle_y
        state[46] = dqz  # relative_angle_z
        dvel_x = dx / self._timeStep
        dvel_y = dy / self._timeStep
        dvel_z = dz / self._timeStep
        state[47] = dvel_x  # rel_vel_x
        state[48] = dvel_y  # rel_vel_y
        state[49] = dvel_z  # rel_vel_z
        dang_vel_x = dex / self._timeStep
        dang_vel_y = dey / self._timeStep
        dang_vel_z = dez / self._timeStep
        state[50] = dang_vel_x  # rel_angular_vel_x
        state[51] = dang_vel_y  # rel_angular_vel_y
        state[52] = dang_vel_z  # rel_angular_vel_z
        Link = p.getLinkState(self._humanoid.human, 11)
        linkWorldPosition = Link[0]
        linkWorldOrientation = Link[1]    
        dx = body_xyz[0] - linkWorldPosition[0]
        dy = body_xyz[1] - linkWorldPosition[1]
        dz = body_xyz[2] - linkWorldPosition[2]
        state[53] = dx  # relative_pos_x
        state[54] = dy  # relative_pos_y
        state[55] = dz  # relative_pos_z
        ex_link, ey_link, ez_link = _quaternion_to_euler_angle(linkWorldOrientation[3], linkWorldOrientation[0], linkWorldOrientation[1], 2)
        dex = ex - ex_link
        dey = ey - ey_link
        dez = ez - ez_link
        dqw, dqx, dqy, dqz = _euler_to_quaternion_angle(dex, dey, dez)
        state[56] = dqw  # relative_angle_w
        state[57] = dqx  # relative_angle_x
        state[58] = dqy  # relative_angle_y
        state[59] = dqz  # relative_angle_z
        dvel_x = dx / self._timeStep
        dvel_y = dy / self._timeStep
        dvel_z = dz / self._timeStep
        state[60] = dvel_x  # rel_vel_x
        state[61] = dvel_y  # rel_vel_y
        state[62] = dvel_z  # rel_vel_z
        dang_vel_x = dex / self._timeStep
        dang_vel_y = dey / self._timeStep
        dang_vel_z = dez / self._timeStep
        state[63] = dang_vel_x  # rel_angular_vel_x
        state[64] = dang_vel_y  # rel_angular_vel_y
        state[65] = dang_vel_z  # rel_angular_vel_z
        Link = p.getLinkState(self._humanoid.human, 15)
        linkWorldPosition = Link[0]
        linkWorldOrientation = Link[1]
        dx = body_xyz[0] - linkWorldPosition[0]
        dy = body_xyz[1] - linkWorldPosition[1]
        dz = body_xyz[2] - linkWorldPosition[2]
        state[66] = dx  # relative_pos_x
        state[67] = dy  # relative_pos_y
        state[68] = dz  # relative_pos_z
        ex_link, ey_link, ez_link = _quaternion_to_euler_angle(linkWorldOrientation[3], linkWorldOrientation[0], linkWorldOrientation[1], 2)
        dex = ex - ex_link
        dey = ey - ey_link
        dez = ez - ez_link
        dqw, dqx, dqy, dqz = _euler_to_quaternion_angle(dex, dey, dez)
        state[69] = dqw  # relative_angle_w
        state[70] = dqx  # relative_angle_x
        state[71] = dqy  # relative_angle_y
        state[72] = dqz  # relative_angle_z
        dvel_x = dx / self._timeStep
        dvel_y = dy / self._timeStep
        dvel_z = dz / self._timeStep
        state[73] = dvel_x  # rel_vel_x
        state[74] = dvel_y  # rel_vel_y
        state[75] = dvel_z  # rel_vel_z
        dang_vel_x = dex / self._timeStep
        dang_vel_y = dey / self._timeStep
        dang_vel_z = dez / self._timeStep
        state[76] = dang_vel_x  # rel_angular_vel_x
        state[77] = dang_vel_y  # rel_angular_vel_y
        state[78] = dang_vel_z  # rel_angular_vel_z
        Link = p.getLinkState(self._humanoid.human, 19)
        linkWorldPosition = Link[0]
        linkWorldOrientation = Link[1]
        dx = body_xyz[0] - linkWorldPosition[0]
        dy = body_xyz[1] - linkWorldPosition[1]
        dz = body_xyz[2] - linkWorldPosition[2]
        state[79] = dx  # relative_pos_x
        state[80] = dy  # relative_pos_y
        state[81] = dz  # relative_pos_z
        ex_link, ey_link, ez_link = _quaternion_to_euler_angle(linkWorldOrientation[3], linkWorldOrientation[0], linkWorldOrientation[1], 2)
        dex = ex - ex_link
        dey = ey - ey_link
        dez = ez - ez_link
        dqw, dqx, dqy, dqz = _euler_to_quaternion_angle(dex, dey, dez)
        state[82] = dqw  # relative_angle_w
        state[83] = dqx  # relative_angle_x
        state[84] = dqy  # relative_angle_y
        state[85] = dqz  # relative_angle_z
        dvel_x = dx / self._timeStep
        dvel_y = dy / self._timeStep
        dvel_z = dz / self._timeStep
        state[86] = dvel_x  # rel_vel_x
        state[87] = dvel_y  # rel_vel_y
        state[88] = dvel_z  # rel_vel_z
        dang_vel_x = dex / self._timeStep
        dang_vel_y = dey / self._timeStep
        dang_vel_z = dez / self._timeStep
        state[89] = dang_vel_x  # rel_angular_vel_x
        state[90] = dang_vel_y  # rel_angular_vel_y
        state[91] = dang_vel_z  # rel_angular_vel_z
        Link = p.getLinkState(self._humanoid.human, 21)
        linkWorldPosition = Link[0]
        linkWorldOrientation = Link[1]
        dx = body_xyz[0] - linkWorldPosition[0]
        dy = body_xyz[1] - linkWorldPosition[1]
        dz = body_xyz[2] - linkWorldPosition[2]
        state[92] = dx  # relative_pos_x
        state[93] = dy  # relative_pos_y
        state[94] = dz  # relative_pos_z
        ex_link, ey_link, ez_link = _quaternion_to_euler_angle(linkWorldOrientation[3], linkWorldOrientation[0], linkWorldOrientation[1], 2)
        dex = ex - ex_link
        dey = ey - ey_link
        dez = ez - ez_link
        dqw, dqx, dqy, dqz = _euler_to_quaternion_angle(dex, dey, dez)
        state[95] = dqw  # relative_angle_w
        state[96] = dqx  # relative_angle_x
        state[97] = dqy  # relative_angle_y
        state[98] = dqz  # relative_angle_z
        dvel_x = dx / self._timeStep
        dvel_y = dy / self._timeStep
        dvel_z = dz / self._timeStep
        state[99] = dvel_x  # rel_vel_x
        state[100] = dvel_y  # rel_vel_y
        state[101] = dvel_z  # rel_vel_z
        dang_vel_x = dex / self._timeStep
        dang_vel_y = dey / self._timeStep
        dang_vel_z = dez / self._timeStep
        state[102] = dang_vel_x  # rel_angular_vel_x
        state[103] = dang_vel_y  # rel_angular_vel_y
        state[104] = dang_vel_z  # rel_angular_vel_z
        Link = p.getLinkState(self._humanoid.human, 25)
        linkWorldPosition = Link[0]
        linkWorldOrientation = Link[1]
        dx = body_xyz[0] - linkWorldPosition[0]
        dy = body_xyz[1] - linkWorldPosition[1]
        dz = body_xyz[2] - linkWorldPosition[2]
        state[105] = dx  # relative_pos_x
        state[106] = dy  # relative_pos_y
        state[107] = dz  # relative_pos_z
        ex_link, ey_link, ez_link = _quaternion_to_euler_angle(linkWorldOrientation[3], linkWorldOrientation[0], linkWorldOrientation[1], 2)
        dex = ex - ex_link
        dey = ey - ey_link
        dez = ez - ez_link
        dqw, dqx, dqy, dqz = _euler_to_quaternion_angle(dex, dey, dez)
        state[108] = dqw  # relative_angle_w
        state[109] = dqx  # relative_angle_x
        state[110] = dqy  # relative_angle_y
        state[111] = dqz  # relative_angle_z
        dvel_x = dx / self._timeStep
        dvel_y = dy / self._timeStep
        dvel_z = dz / self._timeStep
        state[112] = dvel_x  # rel_vel_x
        state[113] = dvel_y  # rel_vel_y
        state[114] = dvel_z  # rel_vel_z
        dang_vel_x = dex / self._timeStep
        dang_vel_y = dey / self._timeStep
        dang_vel_z = dez / self._timeStep
        state[115] = dang_vel_x  # rel_angular_vel_x
        state[116] = dang_vel_y  # rel_angular_vel_y
        state[117] = dang_vel_z  # rel_angular_vel_z
        state[118] = 0.0
        state[119] = 0.0
        cont_right_foot = p.getContactPoints(self._humanoid.human, -1, 15, -1)
        cont_left_foot = p.getContactPoints(self._humanoid.human, -1, 25, -1)
        if(len(cont_right_foot) > 0):
            # print("right foot contact")
            state[118] = 1.0
        if(len(cont_left_foot) > 0):
            # print("left foot contract")
            state[119] = 1.0
        foot_pos_right = p.getLinkState(self._humanoid.human, 15)  # right foot
        right_foot_dx = math.pow(math.fabs(self._MRD.foot_pos_right[0] - foot_pos_right[0][0]), 2)
        right_foot_dy = math.pow(math.fabs(self._MRD.foot_pos_right[1] - foot_pos_right[0][1]), 2)
        right_foot_dz = math.pow(math.fabs(self._MRD.foot_pos_right[2] - foot_pos_right[0][2]), 2)
        state[120] = right_foot_dx
        state[121] = right_foot_dy
        state[122] = right_foot_dz
        foot_pos_left = p.getLinkState(self._humanoid.human, 25)  # left foot
        left_foot_dx = math.pow(math.fabs(self._MRD.foot_pos_left[0] - foot_pos_left[0][0]), 2)
        left_foot_dy = math.pow(math.fabs(self._MRD.foot_pos_left[1] - foot_pos_left[0][1]), 2)
        left_foot_dz = math.pow(math.fabs(self._MRD.foot_pos_left[2] - foot_pos_left[0][2]), 2)
        state[123] = left_foot_dx
        state[124] = left_foot_dy
        state[125] = left_foot_dz    
        return state#, self._observation
    def _step(self, action):
        body_xyz, body_XYZ= p.getBasePositionAndOrientation(self._humanoid.human)
        self.human_state[0] = body_xyz[0]
        self.human_state[1] = body_xyz[1]
        self.human_state[2] = body_xyz[2]
        self.human_state[3] = body_XYZ[0]
        self.human_state[4] = body_XYZ[1]
        self.human_state[5] = body_XYZ[2]
        self.human_state[6] = body_XYZ[3]
        self._MRD._step(self.human_state)
        self._humanoid.applyAction(action)
        for i in range(self._actionRepeat):
            p.stepSimulation()
            #self._state, self._observation = self.getObservation()
            self._state = self.getObservation()
            if self._termination():
                break
            self._envStepCounter += 1
        reward = self._reward()
        done = self._termination()    
        return np.array(self._state), reward, done, {}
    def _termination(self):
        TorsoLink = p.getLinkState(self._humanoid.human, 5)
        done = bool(((self._state[0] < 0.5) or (self._state[0] > 1.1)) or (TorsoLink[0][2] <= self._state[0]))  # or self._envStepCounter>1000) 
        return done  # self._envStepCounter>1000
    def _pos_diff(self, A, B, w):
        return w * math.pow(A - B, 2)
    def _vel_diff(self, A, B, w):
        return w * math.pow(math.fabs(A - B), 2)
    def _reward_pos(self):
        ans = 0.0
        jointStates = p.getJointStates(self._humanoid.human, self._humanoid.motors)
        data_abdomen_x = jointStates[0][0]
        data_abdomen_y = jointStates[1][0]
        data_abdomen_z = jointStates[2][0]
        data_right_hip_x = jointStates[3][0]
        data_right_hip_y = jointStates[4][0]
        data_right_hip_z = jointStates[5][0]
        data_right_knee = jointStates[6][0]
        data_right_ankle_x = jointStates[7][0]
        data_right_ankle_y = jointStates[8][0]
        data_right_ankle_z = jointStates[9][0]
        data_left_hip_x = jointStates[10][0]
        data_left_hip_y = jointStates[11][0]
        data_left_hip_z = jointStates[12][0]
        data_left_knee = jointStates[13][0]
        data_left_ankle_x = jointStates[14][0]
        data_left_ankle_y = jointStates[15][0]
        data_left_ankle_z = jointStates[16][0]
        ans += self._pos_diff(self._MRD.data_abdomen_x     , data_abdomen_x      , 0.5)  # 0
        ans += self._pos_diff(self._MRD.data_abdomen_y     , data_abdomen_y      , 0.5)  # 1
        ans += self._pos_diff(self._MRD.data_abdomen_z     , data_abdomen_z      , 0.5)  # 2
        ans += self._pos_diff(self._MRD.data_right_hip_x    , data_right_hip_x     , 0.5)  # 3
        ans += self._pos_diff(self._MRD.data_right_hip_y    , data_right_hip_y     , 0.5)  # 4
        ans += self._pos_diff(self._MRD.data_right_hip_z    , data_right_hip_z     , 0.5)  # 5
        ans += self._pos_diff(self._MRD.data_right_knee     , data_right_knee      , 0.3)  # 6
        ans += self._pos_diff(self._MRD.data_right_ankle_x  , data_right_ankle_x   , 0.2)  # 7
        ans += self._pos_diff(self._MRD.data_right_ankle_y  , data_right_ankle_y   , 0.2)  # 8
        ans += self._pos_diff(self._MRD.data_right_ankle_z  , data_right_ankle_z   , 0.2)  # 9
        ans += self._pos_diff(self._MRD.data_left_hip_x   , data_left_hip_x    , 0.5)  # 10
        ans += self._pos_diff(self._MRD.data_left_hip_y   , data_left_hip_y    , 0.5)  # 11
        ans += self._pos_diff(self._MRD.data_left_hip_z   , data_left_hip_z    , 0.5)  # 12
        ans += self._pos_diff(self._MRD.data_left_knee    , data_left_knee     , 0.3)  # 13
        ans += self._pos_diff(self._MRD.data_left_ankle_x , data_left_ankle_x  , 0.2)  # 14
        ans += self._pos_diff(self._MRD.data_left_ankle_y , data_left_ankle_y  , 0.2)  # 15
        ans += self._pos_diff(self._MRD.data_left_ankle_z , data_left_ankle_z  , 0.2)  # 16
        #p.addUserDebugLine([self._MRD.data_abdomen_x, self._MRD.data_abdomen_y, self._MRD.data_abdomen_z], [data_abdomen_x, self._MRD.data_abdomen_y, self._MRD.data_abdomen_y], [0.0, 1.0, 0.0], 1.0, 0.1)
        return math.exp(-ans)
    def _reward_vel(self):
        ans = 0.0
        jointStates = p.getJointStates(self._humanoid.human, self._humanoid.motors)
        data_abdomen_x_vel = jointStates[0][1]
        data_abdomen_y_vel = jointStates[1][1]
        data_abdomen_z_vel = jointStates[2][1]
        data_right_hip_x_vel = jointStates[3][1]
        data_right_hip_y_vel = jointStates[4][1]
        data_right_hip_z_vel = jointStates[5][1]
        data_right_knee_vel = jointStates[6][1]
        data_right_ankle_x_vel = jointStates[7][1]
        data_right_ankle_y_vel = jointStates[8][1]
        data_right_ankle_z_vel = jointStates[9][1]
        data_left_hip_x_vel = jointStates[10][1]
        data_left_hip_y_vel = jointStates[11][1]
        data_left_hip_z_vel = jointStates[12][1]
        data_left_knee_vel = jointStates[13][1]
        data_left_ankle_x_vel = jointStates[14][1]
        data_left_ankle_y_vel = jointStates[15][1]
        data_left_ankle_z_vel = jointStates[16][1]
        ans += self._vel_diff(self._MRD.data_abdomen_x_vel     , data_abdomen_x_vel      , 0.5)
        ans += self._vel_diff(self._MRD.data_abdomen_y_vel     , data_abdomen_y_vel      , 0.5)
        ans += self._vel_diff(self._MRD.data_abdomen_z_vel     , data_abdomen_z_vel      , 0.5)
        ans += self._vel_diff(self._MRD.data_right_hip_x_vel    , data_right_hip_x_vel     , 0.5)
        ans += self._vel_diff(self._MRD.data_right_hip_y_vel    , data_right_hip_y_vel     , 0.5)
        ans += self._vel_diff(self._MRD.data_right_hip_z_vel    , data_right_hip_z_vel     , 0.5)
        ans += self._vel_diff(self._MRD.data_right_knee_vel     , data_right_knee_vel      , 0.3)
        ans += self._vel_diff(self._MRD.data_right_ankle_x_vel  , data_right_ankle_x_vel   , 0.2)
        ans += self._vel_diff(self._MRD.data_right_ankle_y_vel  , data_right_ankle_y_vel   , 0.2)
        ans += self._vel_diff(self._MRD.data_right_ankle_z_vel  , data_right_ankle_z_vel   , 0.2)
        ans += self._vel_diff(self._MRD.data_left_hip_x_vel   , data_left_hip_x_vel    , 0.5)
        ans += self._vel_diff(self._MRD.data_left_hip_y_vel   , data_left_hip_y_vel    , 0.5)
        ans += self._vel_diff(self._MRD.data_left_hip_z_vel   , data_left_hip_z_vel    , 0.5)
        ans += self._vel_diff(self._MRD.data_left_knee_vel    , data_left_knee_vel     , 0.3)
        ans += self._vel_diff(self._MRD.data_left_ankle_x_vel , data_left_ankle_x_vel  , 0.2)
        ans += self._vel_diff(self._MRD.data_left_ankle_y_vel , data_left_ankle_y_vel  , 0.2)
        ans += self._vel_diff(self._MRD.data_left_ankle_z_vel , data_left_ankle_z_vel  , 0.2)
        return math.exp(-ans)    
    def _reward_root(self):
        MRchar_body_xyz, _ = p.getBasePositionAndOrientation(self._MRD.MRchar)
        MR_root = MRchar_body_xyz[2]+0.9
        char_body_xyz, _ = p.getBasePositionAndOrientation(self._humanoid.human)
        Char_root = char_body_xyz[2]
        ans = math.pow(MR_root - Char_root , 2)
        return math.exp(-10.0 * ans)
    def _reward_com(self):
        linearvelocity, _ = p.getBaseVelocity(self._MRD.MRchar)
        MR_com_x = linearvelocity[0]
        MR_com_y = linearvelocity[1]
        MR_com_z = linearvelocity[2]
        linearvelocity, _ = p.getBaseVelocity(self._humanoid.human)
        Char_com_x = linearvelocity[0]
        Char_com_y = linearvelocity[1]
        Char_com_z = linearvelocity[2]
        ans_x = math.fabs(MR_com_x - Char_com_x)
        ans_x = math.pow(ans_x , 2)
        ans_y = math.fabs(MR_com_y - Char_com_y)
        ans_y = math.pow(ans_y , 2)
        ans_z = math.fabs(MR_com_z - Char_com_z)
        ans_z = math.pow(ans_z , 2)
        return math.exp(-(ans_x + ans_y + ans_z))
    def _reward_end(self, char_state):
        ans_right = ans_left = 0.0
        foot_pos_right = p.getLinkState(self._humanoid.human, 15)  # right foot
        ans_right = math.pow(math.fabs(self._MRD.pos_FP_right[0] - foot_pos_right[0][0]), 2)
        ans_right += math.pow(math.fabs(self._MRD.pos_FP_right[1] - foot_pos_right[0][1]), 2)
        ans_right += math.pow(math.fabs(self._MRD.pos_FP_right[2] - foot_pos_right[0][2]), 2)
        dist_right = np.sqrt(ans_right)
        foot_pos_left = p.getLinkState(self._humanoid.human, 25)  # left foot
        ans_left += math.pow(math.fabs(self._MRD.pos_FP_left[0] - foot_pos_left[0][0]), 2)
        ans_left += math.pow(math.fabs(self._MRD.pos_FP_left[1] - foot_pos_left[0][1]), 2)
        ans_left += math.pow(math.fabs(self._MRD.pos_FP_left[2] - foot_pos_left[0][2]), 2)
        dist_left = np.sqrt(ans_left)
        p.addUserDebugLine([self._MRD.pos_FP_right[0], self._MRD.pos_FP_right[1], self._MRD.pos_FP_right[2]], [foot_pos_right[0][0], foot_pos_right[0][1], foot_pos_right[0][2]], [1.0, 0.5, 0.5], 1.0, 0.1)
        p.addUserDebugLine([self._MRD.pos_FP_left[0], self._MRD.pos_FP_left[1], self._MRD.pos_FP_left[2]], [foot_pos_left[0][0], foot_pos_left[0][1], foot_pos_left[0][2]], [1.0, 0.5, 0.5], 1.0, 0.1)
        if(bool(char_state[118]) & bool(char_state[119]) & (dist_right <= 0.15) & (dist_left <= 0.15)):
           flag_next_ok = True
        else:
           flag_next_ok = False
        return math.exp(-(ans_right+ans_left)), flag_next_ok
    def _reward_heading(self):
        _, (qx, qy, qz, qw) = p.getBasePositionAndOrientation(self._MRD.MRchar)
        MR_root_x, MR_root_y, MR_root_z = _quaternion_to_euler_angle(qw, qx, qy, qz)
        _, (qx, qy, qz, qw) = p.getBasePositionAndOrientation(self._humanoid.human)
        Char_root_x, Char_root_y, Char_root_z = _quaternion_to_euler_angle(qw, qx, qy, qz)
        return 0.5 * math.cos(MR_root_z - Char_root_z) + 0.5
    def _reward(self,char_state):
        w = [0.5, 0.05, 0.1, 0.1, 0.2, 0.1]
        
        reward_pos = w[0] * self._reward_pos()
        
        reward_vel = w[1] * self._reward_vel()
        
        reward_root = w[2] * self._reward_root()
        
        reward_com = w[3] * self._reward_com()
        
        reward_end, flag_next_ok = self._reward_end(char_state)
        reward_end = w[4] * reward_end
        
        reward_heading = w[5] * self._reward_heading()
        #return reward_pos + reward_vel + reward_root + reward_com + reward_end# + reward_heading
        return reward_pos + reward_root + reward_end, flag_next_ok# + reward_heading
    def _qpos_transform(self, pos):
        qpos = np.zeros([24, 1])
        root_angle = np.zeros([3, 1])
        qpos[0] = pos[0].copy()
        qpos[1] = pos[2].copy()
        qpos[2] = pos[1].copy()
        qpos[3] = pos[3].copy()
        qpos[4] = pos[4].copy()
        qpos[5] = pos[5].copy()
        qpos[6] = pos[6].copy()
        X_root, Y_root, Z_root = _quaternion_to_euler_angle(pos[3], pos[4], pos[5], pos[6])
        root_angle[0] = X_root / 180.0 * 3.1415 * -1.0
        root_angle[1] = Z_root / 180.0 * 3.1415 * -1.0
        root_angle[2] = Y_root / 180.0 * 3.1415 * -1.0    
        X_torso, Y_torso, Z_torso = _quaternion_to_euler_angle(pos[7], pos[8], pos[9], pos[10])
        qpos[7] = X_torso / 180.0 * 3.1415 * -1.0
        qpos[8] = Z_torso / 180.0 * 3.1415 * -1.0
        qpos[9] = Y_torso / 180.0 * 3.1415 * -1.0
        X_right_hip, Y_right_hip, Z_right_hip = _quaternion_to_euler_angle(pos[11], pos[12], pos[13], pos[14])
        qpos[10] = X_right_hip / 180.0 * 3.1415 * -1.0
        qpos[11] = Z_right_hip / 180.0 * 3.1415 * -1.0
        qpos[12] = Y_right_hip / 180.0 * 3.1415 * -1.0
        qpos[13] = pos[15].copy()
        X_right_ankle, Y_right_ankle, Z_right_ankle = _quaternion_to_euler_angle(pos[16], pos[17], pos[18], pos[19])
        qpos[14] = X_right_ankle / 180.0 * 3.1415 * -1.0
        qpos[15] = Z_right_ankle / 180.0 * 3.1415 * -1.0
        qpos[16] = Y_right_ankle / 180.0 * 3.1415 * -1.0        
        X_left_hip, Y_left_hip, Z_left_hip = _quaternion_to_euler_angle(pos[20], pos[21], pos[22], pos[23])
        qpos[17] = X_left_hip / 180.0 * 3.1415 * -1.0
        qpos[18] = Z_left_hip / 180.0 * 3.1415 * -1.0
        qpos[19] = Y_left_hip / 180.0 * 3.1415 * -1.0
        qpos[20] = pos[24].copy()
        X_left_ankle, Y_left_ankle, Z_left_ankle = _quaternion_to_euler_angle(pos[25], pos[26], pos[27], pos[28])
        qpos[21] = X_left_ankle / 180.0 * 3.1415 * -1.0
        qpos[22] = Z_left_ankle / 180.0 * 3.1415 * -1.0
        qpos[23] = Y_left_ankle / 180.0 * 3.1415 * -1.0
        qpos.setflags(write=0)
        return qpos, root_angle

def _euler_to_quaternion_angle(X, Y, Z):
    pitch = Y
    roll = X
    yaw = Z
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    w = cy * cr * cp + sy * sr * sp;
    x = cy * sr * cp - sy * cr * sp;
    y = cy * cr * sp + sy * sr * cp;
    z = sy * cr * cp - cy * sr * sp;
    return w, x, y, z
def _quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    return X, Y, Z

class Central_agent:
    def __init__(self):
        with tf.name_scope("central_agent"):
            self.logger = Logger()
            self.value = Value(obs_dim, hid1_mult)
            self.policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar)
            self.num_tuple = 0
         
    def update_parameter_server(self, episode, batches, name):
        self.num_tuple += len(batches)
        if len(batches) < batch_size:
            return        
        observes, actions, advantages, disc_sum_rew = self.build_train_set(batches)    
        self.policy._update(observes, actions, advantages, self.logger)  # update policy
        self.value._update(observes, disc_sum_rew, self.logger)  # update value function
        self.log_batch_stats(observes, actions, advantages, disc_sum_rew, self.logger, episode)
        self.logger.write(display=False)  # write logger results to file and stdout
        print(['thread_name: ' + name + ', episode: ' + str(episode) + ', tuples: ' + str(self.num_tuple)]) 
        if(episode % batch_size == 0):
            #print(['stop'])
            self.policy._save(episode)
            self.value._save(episode)
    def build_train_set(self, batches):
        observes = np.concatenate([t['observes'] for t in batches])
        actions = np.concatenate([t['actions'] for t in batches])
        disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in batches])
        advantages = np.concatenate([t['advantages'] for t in batches])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        return observes, actions, advantages, disc_sum_rew
    def log_batch_stats(self, observes, actions, advantages, disc_sum_rew, logger, episode):
        """ Log various batch statistics """
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
        self.env = Environment(renders=flag_render , time_step=Ts, obs_dim=obs_dim-1)
    def run(self):
        while self.episode < num_episodes:
            batches = self.run_policy(self.env, self.central_agent.policy, 0, batch_size, self.episode)
            self.episode += len(batches)
            self.add_value(batches, self.central_agent.value)  # add estimated values to episodes
            self.add_disc_sum_rew(batches, gamma)  # calculated discounted sum of Rs
            self.add_gae(batches, gamma, lam)  # calculate advantage
            self.central_agent.update_parameter_server(self.episode, batches, self.name)
    def run_episode(self, env, policy, animate, episode):
        obs = env._reset(episode=episode)
        observes, actions, rewards = [], [], []
        done = False
        step = 0.0
        while not done:
            if animate:
                env._render()
            obs = obs.astype(np.float32).reshape((1, -1))
            obs = np.append(obs, [[step]], axis=1)  # add time step feature
            observes.append(obs)
            action = self.central_agent.policy._act(obs).reshape((1, -1)).astype(np.float32)
            actions.append(action)
            obs, reward, done, _ = self.env._step(np.squeeze(action, axis=0))
            # print(env.model.data.qpos)
            
            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            rewards.append(reward)
            step += 1e-3  # increment time step feature
            # print(len(observes))
        return (np.concatenate(observes), np.concatenate(actions),
                np.array(rewards, dtype=np.float64))
    def run_policy(self, env, policy, logger, episodes, episode):
        total_steps = 0
        batches = []
        for e in range(episodes):
            observes, actions, rewards = self.run_episode(env, policy, False, episode + e)
            total_steps += observes.shape[0]
            batch = {'observes': observes,
                          'actions': actions,
                          'rewards': rewards}
            batches.append(batch)
        return batches
    def discount(self, x, gamma):
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]
    def add_disc_sum_rew(self, batches, gamma):
        for batch in batches:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = batch['rewards'] * (1 - gamma)
            else:
                rewards = batch['rewards']
            disc_sum_rew = self.discount(rewards, gamma)
            batch['disc_sum_rew'] = disc_sum_rew
    def add_value(self, batches, val_func):
        for batch in batches:
            observes = batch['observes']
            values = val_func._predict(observes)
            batch['values'] = values
    def add_gae(self, batches, gamma, lam):
        for batch in batches:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = batch['rewards'] * (1 - gamma)
            else:
                rewards = batch['rewards']
            values = batch['values']
            tds = rewards - values + np.append(values[1:] * gamma, 0)
            advantages = self.discount(tds, gamma * lam)
            batch['advantages'] = advantages
    
SESS = tf.Session()

'''
def main():
    central_agent = Central_agent()
    thread_name = "local_thread"+str(1)
    agent = Agent(thread_name, "learning", central_agent)
    agent.run()
    
if __name__ == '__main__':
    main()
'''
