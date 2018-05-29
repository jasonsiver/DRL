import pybullet as p
import time
import math
import numpy as np
from awscli.customizations.emr.constants import TRUE

filename4 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/character/sphere_blue.urdf'
filename5 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/character/sphere_red.urdf'
filename7 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/character/MRchar.urdf'
filename8 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/mocap/biped3d_walk/'
filename9 = '/home/initial/eclipse-workspace4/test/trpo-master/src/DRLmaster/mocap/biped3d_walk.txt'
filename0 = '/home/initial/my_project_folder/my_project/src/bullet3/data/plane.urdf'#/home/initial/my_project_folder/my_project/src/python_code4/test/trpo-master/src/DRLmaster/character/plane100.urdf'

def eular_to_quaternion_angle(X, Y, Z):
	cy = math.cos(Z * 0.5)
	sy = math.sin(Z * 0.5)
	cr = math.cos(Y * 0.5)
	sr = math.sin(Y * 0.5)
	cp = math.cos(X * 0.5)
	sp = math.sin(X * 0.5)
	w = cy * cr * cp + sy * sr * sp
	x = cy * sr * cp - sy * cr * sp
	y = cy * cr * sp + sy * sr * cp
	z = sy * cr * cp - cy * sr * sp
	return w, x, y, z

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

def qpos_transform(pos):
    qpos = np.zeros([24, 1])
    root_angle = np.zeros([3, 1])
    qpos[0] = pos[0].copy()
    qpos[1] = pos[2].copy()
    qpos[2] = pos[1].copy()
    qpos[3] = pos[3].copy()
    qpos[4] = pos[4].copy()
    qpos[5] = pos[5].copy()
    qpos[6] = pos[6].copy()
    
    X_root, Y_root, Z_root = quaternion_to_euler_angle(pos[3], pos[4], pos[5], pos[6])
    root_angle[0] = X_root / 180.0 * 3.1415 * -1.0
    root_angle[1] = Z_root / 180.0 * 3.1415 * -1.0
    root_angle[2] = Y_root / 180.0 * 3.1415 * -1.0
    
    # for i in range(7,46,4):
    # torso_w,torso_x,torso_y,torso_z
    X_torso, Y_torso, Z_torso = quaternion_to_euler_angle(pos[7], pos[8], pos[9], pos[10])
    qpos[7] = X_torso / 180.0 * 3.1415 * -1.0
    qpos[8] = Z_torso / 180.0 * 3.1415 * -1.0
    qpos[9] = Y_torso / 180.0 * 3.1415 * -1.0
        
    # right_hip_w,right_hip_x,right_hip_y,right_hip_z
    X_right_hip, Y_right_hip, Z_right_hip = quaternion_to_euler_angle(pos[11], pos[12], pos[13], pos[14])
    qpos[10] = X_right_hip / 180.0 * 3.1415 * -1.0
    qpos[11] = Z_right_hip / 180.0 * 3.1415 * -1.0
    qpos[12] = Y_right_hip / 180.0 * 3.1415 * -1.0

    # right_knee
    qpos[13] = pos[15].copy()
        
    # right_ankle_w,right_ankle_x,right_ankle_y,right_ankle_z
    X_right_ankle, Y_right_ankle, Z_right_ankle = quaternion_to_euler_angle(pos[16], pos[17], pos[18], pos[19])
    qpos[14] = X_right_ankle / 180.0 * 3.1415 * -1.0
    qpos[15] = Z_right_ankle / 180.0 * 3.1415 * -1.0
    qpos[16] = Y_right_ankle / 180.0 * 3.1415 * -1.0        
        
    # left_hip_w,left_hip_x,left_hip_y,left_hip_z
    X_left_hip, Y_left_hip, Z_left_hip = quaternion_to_euler_angle(pos[20], pos[21], pos[22], pos[23])
    qpos[17] = X_left_hip / 180.0 * 3.1415 * -1.0
    qpos[18] = Z_left_hip / 180.0 * 3.1415 * -1.0
    qpos[19] = Y_left_hip / 180.0 * 3.1415 * -1.0
        
    # left_knee
    qpos[20] = pos[24].copy()

    # left_ankle_w,left_ankle_x,left_ankle_y,left_ankle_z
    X_left_ankle, Y_left_ankle, Z_left_ankle = quaternion_to_euler_angle(pos[25], pos[26], pos[27], pos[28])
    qpos[21] = X_left_ankle / 180.0 * 3.1415 * -1.0
    qpos[22] = Z_left_ankle / 180.0 * 3.1415 * -1.0
    qpos[23] = Y_left_ankle / 180.0 * 3.1415 * -1.0

    qpos.setflags(write=0)
       
    return qpos, root_angle
   
cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
	p.connect(p.GUI)
p.setPhysicsEngineParameter(numSolverIterations=10)
dt = 0.001
p.setTimeStep(dt)
logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "visualShapeBench.json")
p.loadURDF(filename0, useMaximalCoordinates=True)
MRchar=p.loadURDF(filename7)
for j in range( p.getNumJoints(MRchar) ):
	info = p.getJointInfo(MRchar, j)
	print(info)
	link_name = info[12].decode("ascii")
	if info[2] != p.JOINT_REVOLUTE: continue	
	p.setJointMotorControl2(MRchar, j, controlMode=p.VELOCITY_CONTROL, force=0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
shift = [0, -0.02, 0]
meshScale = [0.1, 0.1, 0.1]
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.stopStateLogging(logId)
p.setGravity(0, 0, -10)
colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
currentColor = 0
foot_step = 0.518  # m/sec
foot_width = 0.17
x_ini = - foot_step / 2.0
objs_b = p.loadURDF(filename4, [x_ini, -foot_width / 2.0, 0.0])
objs_r = p.loadURDF(filename5, [x_ini, foot_width / 2.0, 0.0])

pos_MRD = [x_ini, 0.0, 0.01, 0.0]  # x,y,z,the
pos_FP_left = [x_ini, -foot_width / 2.0, 0.0, 0.0]
pos_FP_right = [x_ini, foot_width / 2.0, 0.0, 0.0]
dthe = 0.0
total_step = 10
import math
	
count = 0
time = 0.0
swing = 1.0
n = 0

p.resetBasePositionAndOrientation(objs_b, [pos_FP_right[0], pos_FP_right[1], pos_FP_right[2]], [0.0, 0.0, 0.0, 1.0])
p.resetBasePositionAndOrientation(objs_r, [pos_FP_left[0], pos_FP_left[1], pos_FP_left[2]], [0.0, 0.0, 0.0, 1.0])


'''
dt_inv = int(1 / dt)

n += 1

if((-1 * np.power(-1, n)) == -1):
	p.resetBasePositionAndOrientation(objs_b, [old_pos_FP[0], old_pos_FP[1], old_pos_FP[2]], [0.0, 0.0, 0.0, 1.0])
	p.resetBasePositionAndOrientation(objs_r, [pos_FP[0], pos_FP[1], pos_FP[2]], [0.0, 0.0, 0.0, 1.0])
else:
	p.resetBasePositionAndOrientation(objs_b, [pos_FP[0], pos_FP[1], pos_FP[2]], [0.0, 0.0, 0.0, 1.0])
	p.resetBasePositionAndOrientation(objs_r, [old_pos_FP[0], old_pos_FP[1], old_pos_FP[2]], [0.0, 0.0, 0.0, 1.0])
'''
filepath_prefix = filename8
filename_list = open(filepath_prefix + "mocap_list.txt", "r")
str_list = filename_list.readlines()
filename_list.close()

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')

tm = []
rah = []
qh = []
for i_lis in range(0, len(str_list), 1):
	if str_list[i_lis][-1] == "\n": 
		filepath = filepath_prefix + str_list[i_lis][:-1]
	else:
		filepath = filepath_prefix + str_list[i_lis][:]
	filename = open(filepath, "r")
	str = filename.readlines()
	filename.close()
	time = []
	pos = []
	for i in range(0, len(str), 1):
		pos_sub = []
		if len(str[i]) > 10:
			for j in str[i][:].split(","):
				pos_sub.append(np.float32(j))
			if i == 0:
				time = pos_sub[0].copy()
				pos = pos_sub[1:].copy()
			else:
				time = np.vstack((time, pos_sub[0]))
				pos = np.vstack((pos, pos_sub[1:]))
	time_hist = []
	qpos_hist = np.zeros([len(pos), 24])
	root_angle_hist = np.zeros([len(pos), 3])
	for i in range(0, len(pos), 1):
		qpos_sub, root_angle_sub = qpos_transform(pos[i])
		time_hist.append(time) 
		qpos_hist[i] = qpos_sub.transpose()
		root_angle_hist[i] = root_angle_sub.transpose()
	tm.append(time_hist)
	rah.append(root_angle_hist)
	qh.append(qpos_hist)
curvature = []

filename = open(filename9,"r")
str = filename.readlines()
filename.close()
motion = []
motion_copy = []
for i in range(4,len(str)-1,1):
	motion = []
	if len(str[i]) > 10:
		for j in str[i][3:-3].split(","):
			motion.append(np.float32(j))
		if i == 4:
			time = motion[0].copy()
			motion_copy = motion[1:].copy()
		else:
			time = np.vstack((time, motion[0]))
			motion_copy = np.vstack((motion_copy, motion[1:]))
qpos_hist = np.zeros([len(motion_copy),24])
for i in range(0,len(motion_copy),1):
	qpos_sub, _ = qpos_transform(motion_copy[i])
	qpos_hist[i] = qpos_sub.transpose()
motion_copy=qpos_hist.copy()
    
for i in range(0, len(qh), 1):
	x = []
	y = []
	z = []
	x_sub = qh[i][:, [0]].copy()
	y_sub = qh[i][:, [1]].copy()
	z_sub = qh[i][:, [2]].copy()
	for j in range(0, len(qh[i]), 1):
		x.append(x_sub[j][0].copy())
		y.append(y_sub[j][0].copy())
		z.append(z_sub[j][0].copy())
	ans = math.atan2(y[len(y) - 1] - y[0], x[len(x) - 1] - x[0])
	curvature.append(ans)	

t=[]
a=[]
i=10
MRD_qh = qh[i]
test_time=tm[i][0]
test_right_knee = MRD_qh[:,[13]]

fig = plt.figure()
test_right_knee_vel = np.diff(test_right_knee.transpose())
test_right_knee_vel = test_right_knee_vel.transpose()
test_right_knee_vel = np.vstack((test_right_knee_vel[0], test_right_knee_vel))

def find_nearest(array, value):
    idx = (np.fabs(array - value)).argmin()
    return idx, array[idx]


i_lis=0
id_abdomen_x        = 2
id_abdomen_y        = 3
id_abdomen_z        = 4
id_right_hip_x      = 5
id_right_hip_y      = 6
id_right_hip_z      = 7
id_right_knee       = 8
id_right_ankle_x    = 9
id_right_ankle_y    = 10
id_right_ankle_z    = 11
id_left_hip_x       = 12
id_left_hip_y       = 13
id_left_hip_z       = 14
id_left_knee        = 15
id_left_ankle_x     = 16
id_left_ankle_y     = 17
id_left_ankle_z     = 18

data_abdomen_x = []
data_abdomen_y = []
data_abdomen_z = []
data_right_hip_x = []
data_right_hip_y = []
data_right_hip_z = []
data_right_knee = []
data_right_ankle_x = []
data_right_ankle_y = []
data_right_ankle_z = []
data_left_hip_x = []
data_left_hip_y = []
data_left_hip_z = []
data_left_knee = []
data_left_ankle_x = []
data_left_ankle_y = []
data_left_ankle_z = []
flag_MRD = False
elapsed_time = 0.0
#flag_continue = True
root_X_sum = 0.0
root_Y_sum = 0.0
root_Z_sum = 0.0
root_x_sum = 0.0#0.25
root_y_sum = 0.0
root_z_sum = 0.0
dt_mpcap_samp = 0.033332
dt_mocap = 0.633
pos_FP = np.array([0,0,0.0,0.0,0.0])
dt_inv = int(dt_mocap/dt)#0.63330802693963051
flag_next_ok = False
total_count = 0
pos_MRD_hist = []
for i in range(0, dt_inv, 1):
	count += 1
	rd = np.random.randn() * 0.05
	dthe = 0.0 / 180.0 * 3.1415# + rd
	rd2 = np.random.randn() * 0.15
	foot_step2 = foot_step/dt_mocap# + rd2
	old_pos_MRD = pos_MRD.copy()
	pos_MRD[3] = pos_MRD[3] + dthe
	pos_MRD[0] = pos_MRD[0] + foot_step2 * dt * math.cos(pos_MRD[3])
	pos_MRD[1] = pos_MRD[1] + foot_step2 * dt * math.sin(pos_MRD[3])
	p.addUserDebugLine([old_pos_MRD[0], old_pos_MRD[1], old_pos_MRD[2]], [pos_MRD[0], pos_MRD[1], pos_MRD[2]], [1.0, 0.5, 0.5], 1.0, 100.0)
	if(count == dt_inv):
		pos_MRD_hist.append(pos_MRD.copy())


###
id = 26#, Y = find_nearest(np.asarray(curvature), d_pos_FP)
MRD_tm = tm[id]
MRD_rah = rah[id]
MRD_qh = qh[id]
root_X=MRD_rah[:,[0]]
root_Y=MRD_rah[:,[1]]
root_Z=MRD_rah[:,[2]]
root_x=MRD_qh[:,[0]]
root_y=MRD_qh[:,[1]]
root_z=MRD_qh[:,[2]]
root_qw=MRD_qh[:,[3]]
root_qx=MRD_qh[:,[4]]
root_qy=MRD_qh[:,[5]]
root_qz=MRD_qh[:,[6]]
data_abdomen_x        = MRD_qh[:,[7]]
data_abdomen_y        = MRD_qh[:,[8]]
data_abdomen_z        = MRD_qh[:,[9]]
data_right_hip_x      = MRD_qh[:,[10]]
data_right_hip_y      = MRD_qh[:,[11]]
data_right_hip_z      = MRD_qh[:,[12]]
data_right_knee       = MRD_qh[:,[13]]
data_right_ankle_x    = MRD_qh[:,[14]]
data_right_ankle_y    = MRD_qh[:,[15]]
data_right_ankle_z    = MRD_qh[:,[16]]
data_left_hip_x       = MRD_qh[:,[17]]
data_left_hip_y       = MRD_qh[:,[18]]
data_left_hip_z       = MRD_qh[:,[19]]
data_left_knee        = MRD_qh[:,[20]]
data_left_ankle_x     = MRD_qh[:,[21]]
data_left_ankle_y     = MRD_qh[:,[22]]
data_left_ankle_z     = MRD_qh[:,[23]]

if((-1 * np.power(-1, n)) == -1):  # right swing            
	p.resetJointState(MRchar,id_abdomen_x       ,data_abdomen_x[i_lis])
	p.resetJointState(MRchar,id_abdomen_y       ,data_abdomen_y[i_lis])
	p.resetJointState(MRchar,id_abdomen_z       ,data_abdomen_z[i_lis])
	p.resetJointState(MRchar,id_right_hip_x     ,data_right_hip_x[i_lis])
	p.resetJointState(MRchar,id_right_hip_y     ,data_right_hip_y[i_lis])
	p.resetJointState(MRchar,id_right_hip_z     ,data_right_hip_z[i_lis])
	p.resetJointState(MRchar,id_right_knee      ,data_right_knee[i_lis])
	p.resetJointState(MRchar,id_right_ankle_x   ,data_right_ankle_x[i_lis])
	p.resetJointState(MRchar,id_right_ankle_y   ,data_right_ankle_y[i_lis])
	p.resetJointState(MRchar,id_right_ankle_z   ,data_right_ankle_z[i_lis])
	p.resetJointState(MRchar,id_left_hip_x      ,data_left_hip_x[i_lis])
	p.resetJointState(MRchar,id_left_hip_y      ,data_left_hip_y[i_lis])
	p.resetJointState(MRchar,id_left_hip_z      ,data_left_hip_z[i_lis])
	p.resetJointState(MRchar,id_left_knee       ,data_left_knee[i_lis])
	p.resetJointState(MRchar,id_left_ankle_x    ,data_left_ankle_x[i_lis])
	p.resetJointState(MRchar,id_left_ankle_y    ,data_left_ankle_y[i_lis])
	p.resetJointState(MRchar,id_left_ankle_z    ,data_left_ankle_z[i_lis])  
else:
	p.resetJointState(MRchar,id_abdomen_x       ,data_abdomen_x[i_lis])
	p.resetJointState(MRchar,id_abdomen_y       ,data_abdomen_y[i_lis])
	p.resetJointState(MRchar,id_abdomen_z       ,data_abdomen_z[i_lis])
	p.resetJointState(MRchar,id_right_hip_x     ,data_left_hip_x[i_lis])
	p.resetJointState(MRchar,id_right_hip_y     ,data_left_hip_y[i_lis])
	p.resetJointState(MRchar,id_right_hip_z     ,data_left_hip_z[i_lis])
	p.resetJointState(MRchar,id_right_knee      ,data_left_knee[i_lis])
	p.resetJointState(MRchar,id_right_ankle_x   ,data_left_ankle_x[i_lis])
	p.resetJointState(MRchar,id_right_ankle_y   ,data_left_ankle_y[i_lis])
	p.resetJointState(MRchar,id_right_ankle_z   ,data_left_ankle_z[i_lis])
			
	p.resetJointState(MRchar,id_left_hip_x      ,data_right_hip_x[i_lis])
	p.resetJointState(MRchar,id_left_hip_y      ,data_right_hip_y[i_lis])
	p.resetJointState(MRchar,id_left_hip_z      ,data_right_hip_z[i_lis])
	p.resetJointState(MRchar,id_left_knee       ,data_right_knee[i_lis])
	p.resetJointState(MRchar,id_left_ankle_x    ,data_right_ankle_x[i_lis])
	p.resetJointState(MRchar,id_left_ankle_y    ,data_right_ankle_y[i_lis])
	p.resetJointState(MRchar,id_left_ankle_z    ,data_right_ankle_z[i_lis])  
				

pos_FP[0] = pos_MRD_hist[-1][0] - math.sin(pos_MRD_hist[-1][3]) * foot_width/2.0 * (-1.0 * np.power(-1.0, n))
pos_FP[1] = pos_MRD_hist[-1][1] + math.cos(pos_MRD_hist[-1][3]) * foot_width/2.0 * (-1.0 * np.power(-1.0, n))
pos_FP[3] = pos_MRD_hist[-1][3]
if((-1 * np.power(-1, n)) == -1):
	pos_FP_left = pos_FP.copy()
else:
	
	pos_FP_right = pos_FP.copy()	
#if((-1 * np.power(-1, n)) == -1):  # right swing
p.resetBasePositionAndOrientation(objs_b, [pos_FP_right[0], pos_FP_right[1], pos_FP_right[2]], [0.0, 0.0, 0.0, 1.0]) # left foot
p.resetBasePositionAndOrientation(objs_r, [pos_FP_left[0], pos_FP_left[1], pos_FP_left[2]], [0.0, 0.0, 0.0, 1.0]) # right foot
qw, qx, qy, qz = eular_to_quaternion_angle(0.0, 0.0, 0.0)#root_Z[i_lis])
p.resetBasePositionAndOrientation(MRchar, [0.0, 0, root_z_sum+root_z[i_lis]-0.9], [qx, qy, qz, qw]) # MR character
	
###
n=1
count = 0 
while (1):
	total_count += 1
	
	if(total_count % dt_inv == 0.0):
		flag_next_ok = True
	else:
		flag_next_ok = False
	
	if(count<dt_inv):
		count += 1
		rd = np.random.randn() * 0.05
		dthe = 0.0 / 180.0 * 3.1415# + rd
		rd2 = np.random.randn() * 0.15
		foot_step2 = foot_step/dt_mocap# + rd2
		old_pos_MRD = pos_MRD.copy()
		pos_MRD[3] = pos_MRD[3] + dthe
		pos_MRD[0] = pos_MRD[0] + foot_step2 * dt * math.cos(pos_MRD[3])
		pos_MRD[1] = pos_MRD[1] + foot_step2 * dt * math.sin(pos_MRD[3])
		p.addUserDebugLine([old_pos_MRD[0], old_pos_MRD[1], old_pos_MRD[2]], [pos_MRD[0], pos_MRD[1], pos_MRD[2]], [1.0, 0.5, 0.5], 1.0, 100.0)
		if(count == dt_inv):
			pos_MRD_hist.append(pos_MRD.copy())
		
		if((count == dt_inv) & flag_next_ok):
			count = 0
			pos_FP[0] = pos_MRD_hist[-1][0] - math.sin(pos_MRD_hist[-1][3]) * foot_width/2.0 * (-1.0 * np.power(-1.0, n))
			pos_FP[1] = pos_MRD_hist[-1][1] + math.cos(pos_MRD_hist[-1][3]) * foot_width/2.0 * (-1.0 * np.power(-1.0, n))
			pos_FP[3] = pos_MRD_hist[-1][3]
			if((-1 * np.power(-1, n)) == -1):
				pos_FP_left = pos_FP.copy()
			else:
				pos_FP_right = pos_FP.copy()		
			# nearest motion in the MRD filenaes
			id = 26#, Y = find_nearest(np.asarray(curvature), d_pos_FP)
			MRD_tm = tm[id]
			MRD_rah = rah[id]
			MRD_qh = qh[id]
		
			flag_MRD=True
			dtime = MRD_tm[0][0][0]#1.0/len(MRD_tm)

		if(flag_MRD==True):
			root_X=MRD_rah[:,[0]]
			root_Y=MRD_rah[:,[1]]
			root_Z=MRD_rah[:,[2]]
			root_x=MRD_qh[:,[0]]
			root_y=MRD_qh[:,[1]]
			root_z=MRD_qh[:,[2]]
			root_qw=MRD_qh[:,[3]]
			root_qx=MRD_qh[:,[4]]
			root_qy=MRD_qh[:,[5]]
			root_qz=MRD_qh[:,[6]]
			data_abdomen_x        = MRD_qh[:,[7]]
			data_abdomen_y        = MRD_qh[:,[8]]
			data_abdomen_z        = MRD_qh[:,[9]]
			data_right_hip_x      = MRD_qh[:,[10]]
			data_right_hip_y      = MRD_qh[:,[11]]
			data_right_hip_z      = MRD_qh[:,[12]]
			data_right_knee       = MRD_qh[:,[13]]
			data_right_ankle_x    = MRD_qh[:,[14]]
			data_right_ankle_y    = MRD_qh[:,[15]]
			data_right_ankle_z    = MRD_qh[:,[16]]
			data_left_hip_x       = MRD_qh[:,[17]]
			data_left_hip_y       = MRD_qh[:,[18]]
			data_left_hip_z       = MRD_qh[:,[19]]
			data_left_knee        = MRD_qh[:,[20]]
			data_left_ankle_x     = MRD_qh[:,[21]]
			data_left_ankle_y     = MRD_qh[:,[22]]
			data_left_ankle_z     = MRD_qh[:,[23]]
			
				
			if((-1 * np.power(-1, n)) == -1):  # right swing            
				p.resetJointState(MRchar,id_abdomen_x       ,data_abdomen_x[i_lis])
				p.resetJointState(MRchar,id_abdomen_y       ,data_abdomen_y[i_lis])
				p.resetJointState(MRchar,id_abdomen_z       ,data_abdomen_z[i_lis])
				p.resetJointState(MRchar,id_right_hip_x     ,data_left_hip_x[i_lis])
				p.resetJointState(MRchar,id_right_hip_y     ,data_left_hip_y[i_lis])
				p.resetJointState(MRchar,id_right_hip_z     ,data_left_hip_z[i_lis])
				p.resetJointState(MRchar,id_right_knee      ,data_left_knee[i_lis])
				p.resetJointState(MRchar,id_right_ankle_x   ,data_left_ankle_x[i_lis])
				p.resetJointState(MRchar,id_right_ankle_y   ,data_left_ankle_y[i_lis])
				p.resetJointState(MRchar,id_right_ankle_z   ,data_left_ankle_z[i_lis])
				p.resetJointState(MRchar,id_left_hip_x      ,data_right_hip_x[i_lis])
				p.resetJointState(MRchar,id_left_hip_y      ,data_right_hip_y[i_lis])
				p.resetJointState(MRchar,id_left_hip_z      ,data_right_hip_z[i_lis])
				p.resetJointState(MRchar,id_left_knee       ,data_right_knee[i_lis])
				p.resetJointState(MRchar,id_left_ankle_x    ,data_right_ankle_x[i_lis])
				p.resetJointState(MRchar,id_left_ankle_y    ,data_right_ankle_y[i_lis])
				p.resetJointState(MRchar,id_left_ankle_z    ,data_right_ankle_z[i_lis])  
			else:
				p.resetJointState(MRchar,id_abdomen_x       ,data_abdomen_x[i_lis])
				p.resetJointState(MRchar,id_abdomen_y       ,data_abdomen_y[i_lis])
				p.resetJointState(MRchar,id_abdomen_z       ,data_abdomen_z[i_lis])
				p.resetJointState(MRchar,id_right_hip_x     ,data_right_hip_x[i_lis])
				p.resetJointState(MRchar,id_right_hip_y     ,data_right_hip_y[i_lis])
				p.resetJointState(MRchar,id_right_hip_z     ,data_right_hip_z[i_lis])
				p.resetJointState(MRchar,id_right_knee      ,data_right_knee[i_lis])
				p.resetJointState(MRchar,id_right_ankle_x   ,data_right_ankle_x[i_lis])
				p.resetJointState(MRchar,id_right_ankle_y   ,data_right_ankle_y[i_lis])
				p.resetJointState(MRchar,id_right_ankle_z   ,data_right_ankle_z[i_lis])
				p.resetJointState(MRchar,id_left_hip_x      ,data_left_hip_x[i_lis])
				p.resetJointState(MRchar,id_left_hip_y      ,data_left_hip_y[i_lis])
				p.resetJointState(MRchar,id_left_hip_z      ,data_left_hip_z[i_lis])
				p.resetJointState(MRchar,id_left_knee       ,data_left_knee[i_lis])
				p.resetJointState(MRchar,id_left_ankle_x    ,data_left_ankle_x[i_lis])
				p.resetJointState(MRchar,id_left_ankle_y    ,data_left_ankle_y[i_lis])
				p.resetJointState(MRchar,id_left_ankle_z    ,data_left_ankle_z[i_lis])  
				
		p.resetBasePositionAndOrientation(objs_b, [pos_FP_right[0], pos_FP_right[1], pos_FP_right[2]], [0.0, 0.0, 0.0, 1.0]) # left foot
		p.resetBasePositionAndOrientation(objs_r, [pos_FP_left[0], pos_FP_left[1], pos_FP_left[2]], [0.0, 0.0, 0.0, 1.0]) # right foot
		if(flag_MRD==True):
			qw, qx, qy, qz = eular_to_quaternion_angle(0.0, 0.0, 0.0)#root_Z[i_lis])
			p.resetBasePositionAndOrientation(MRchar, [root_x_sum+root_x[i_lis], root_y_sum+root_y[i_lis], root_z_sum+root_z[i_lis]-0.9], [qx, qy, qz, qw]) # MR character

			
		if(flag_MRD==True):
			elapsed_time = elapsed_time + dt#_mpcap_samp

			if(elapsed_time >=dtime):
				elapsed_time = 0.0
				i_lis = i_lis + 1

			
			if(i_lis >= (len(data_abdomen_x)-1)):
				root_X_sum += 0.0#root_X[i_lis-1]
				root_Y_sum += 0.0#root_Y[i_lis-1]
				root_Z_sum += 0.0#root_Z[i_lis-1]
				root_x_sum += root_x[i_lis-1]
				root_y_sum += 0.0#root_y[i_lis-1]
				root_z_sum += 0.0#root_z[i_lis-1]
				i_lis = 0
				n += 1
				#flag_continue = False

		p.stepSimulation()