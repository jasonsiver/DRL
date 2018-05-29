import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from cupy import float32
import math

def quaternion_to_euler_angle(w, x, y, z):
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

def mass_center(model):
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

def motion_transform(motion, qpos):
    d_motion = np.zeros(len(qpos))
    d_motion[0]=motion[1]-qpos[0]
    d_motion[1]=motion[3]-qpos[1]
    d_motion[2]=motion[2]-qpos[2]
    d_motion[3]=motion[4]-qpos[3]
    d_motion[4]=motion[5]-qpos[4]
    d_motion[5]=motion[6]-qpos[5]
    d_motion[6]=motion[7]-qpos[6]
    
    X_torso,Y_torso,Z_torso=quaternion_to_euler_angle(motion[8], motion[9],motion[10],motion[11])
    d_motion[7]=(X_torso/180.0*3.1415*-1.0)-qpos[7]
    d_motion[8]=(Z_torso/180.0*3.1415*-1.0)-qpos[8]
    d_motion[9]=(Y_torso/180.0*3.1415*-1.0)-qpos[9]
        
    #right_hip_w,right_hip_x,right_hip_y,right_hip_z
    X_right_hip,Y_right_hip,Z_right_hip=quaternion_to_euler_angle(motion[12],motion[13],motion[14],motion[15])
    d_motion[10]=(X_right_hip/180.0*3.1415*-1.0)-qpos[10]
    d_motion[11]=(Z_right_hip/180.0*3.1415*-1.0)-qpos[11]
    d_motion[12]=(Y_right_hip/180.0*3.1415*-1.0)-qpos[12]
    
    #right_knee
    d_motion[13]=motion[16]-qpos[13]
        
    #right_ankle_w,right_ankle_x,right_ankle_y,right_ankle_z
    X_right_ankle,Y_right_ankle,Z_right_ankle=quaternion_to_euler_angle(motion[17],motion[18],motion[19],motion[20])
    d_motion[14]=(X_right_ankle/180.0*3.1415*-1.0)-qpos[14]
    d_motion[15]=(Z_right_ankle/180.0*3.1415*-1.0)-qpos[15]
    d_motion[16]=(Y_right_ankle/180.0*3.1415*-1.0)-qpos[16]
    
    #left_hip_w,left_hip_x,left_hip_y,left_hip_z
    X_left_hip,Y_left_hip,Z_left_hip=quaternion_to_euler_angle(motion[21],motion[22],motion[23],motion[24])
    d_motion[17]=(X_left_hip/180.0*3.1415*-1.0)-qpos[17]
    d_motion[18]=(Z_left_hip/180.0*3.1415*-1.0)-qpos[18]
    d_motion[19]=(Y_left_hip/180.0*3.1415*-1.0)-qpos[19]
        
    #left_knee
    d_motion[20]=motion[25]-qpos[20]

    #left_ankle_w,left_ankle_x,left_ankle_y,left_ankle_z
    X_left_ankle,Y_left_ankle,Z_left_ankle=quaternion_to_euler_angle(motion[26],motion[27],motion[28],motion[29])
    d_motion[21]=(X_left_ankle/180.0*3.1415*-1.0)-qpos[21]
    d_motion[22]=(Z_left_ankle/180.0*3.1415*-1.0)-qpos[22]
    d_motion[23]=(Y_left_ankle/180.0*3.1415*-1.0)-qpos[23]
    '''
    #right_shoulder_w,right_shoulder_x,right_shoulder_y,right_shoulder_z
    X_right_shoulder,Y_right_shoulder,Z_right_shoulder=quaternion_to_euler_angle(motion[30],motion[31],motion[32],motion[33])
    d_motion[24]=(X_right_shoulder/180.0*3.1415*-1.0)-qpos[24]
    d_motion[25]=(Z_right_shoulder/180.0*3.1415*-1.0)-qpos[25]
    d_motion[26]=(Y_right_shoulder/180.0*3.1415*-1.0)-qpos[26]
        
    #right_elbow
    d_motion[27]=motion[34]-qpos[27]
        
    #right_wrist_w,right_wrist_x,right_wrist_y,right_wrist_z
    X_right_wrist,Y_right_wrist,Z_right_wrist=quaternion_to_euler_angle(motion[35],motion[36],motion[37],motion[38])
    d_motion[28]=(X_right_wrist/180.0*3.1415*-1.0)-qpos[28]
    d_motion[29]=(Z_right_wrist/180.0*3.1415*-1.0)-qpos[29]
    d_motion[30]=(Y_right_wrist/180.0*3.1415*-1.0)-qpos[30]
        
    #left_shoulder_w,left_shoulder_x,left_shoulder_y,left_shoulder_z
    X_left_shoulder,Y_left_shoulder,Z_left_shoulder=quaternion_to_euler_angle(motion[39],motion[40],motion[41],motion[42])
    d_motion[31]=(X_left_shoulder/180.0*3.1415*-1.0)-qpos[31]
    d_motion[32]=(Z_left_shoulder/180.0*3.1415*-1.0)-qpos[32]
    d_motion[33]=(Y_left_shoulder/180.0*3.1415*-1.0)-qpos[33]
    
    #left_elbow        for i in str[1][8:-3].split(','):
    d_motion[34]=motion[43]-qpos[34]
        
    #left_wrist_w,left_wrist_x,left_wrist_y,left_wrist_z
    X_left_wrist,Y_left_wrist,Z_left_wrist=quaternion_to_euler_angle(motion[44],motion[45],motion[46],motion[47])
    d_motion[35]=(X_left_wrist/180.0*3.1415*-1.0)-qpos[35]
    d_motion[36]=(Z_left_wrist/180.0*3.1415*-1.0)-qpos[36]
    d_motion[37]=(Y_left_wrist/180.0*3.1415*-1.0)-qpos[37]
    '''
    
    #d_motion = np.power(d_motion,2)

    return d_motion

class HumanoidasimoMRD3Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoidasimoMRD3.xml', 5)
        utils.EzPickle.__init__(self)
        
        self.pos=[]
        self.vel=[]
        fileHandle = open ( '/home/initial/my_project_folder/my_project/src/python_code3/trpo-master/data/states/biped3d_sim_walk_state-asimo.txt', 'r' )
        str = fileHandle.readlines()
        fileHandle.close()
        for i in str[1][8:-3].split(','):
            self.pos.append(float32(i))
        for i in str[2][7:-2].split(','):
            self.vel.append(float32(i))

        fileHandle = open ( '/home/initial/my_project_folder/my_project/src/python_code3/trpo-master/data/motions/mocap/asimo/0007_Walking001_motion_00000_retargeted_asimo.txt', 'r' )
        str = fileHandle.readlines()
        fileHandle.close()
        self.motion=[]
        for i in range(4,31,1):
            motion_sub=[]
            for j in str[i][1:-3].split(','):
                motion_sub.append(float32(j))
            self.motion.append(motion_sub)
            
        self.time_step = 0.0#e-3
        self.i = 0

    def _get_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def _step(self, a, flag=True):
        pos_before = mass_center(self.model)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model)
        alive_bonus = 5.0
        data = self.model.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.model.data.qpos
        head_pos=self.model.data.geom_xpos[2,2];
        #done = bool((qpos[2] < 0.40) or (qpos[2] > 2.0)) 
        done = bool(((qpos[2] < 0.5) or (qpos[2] > 1.1)) or (head_pos <= qpos[2])) 
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
  
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
