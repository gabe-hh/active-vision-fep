import sys, getopt
import matplotlib.pyplot as plt
import numpy as np
import rospy
import cv2
import imutils
import math
import time
from time import sleep
from sensor_msgs.msg import JointState
from pathlib import Path
import random
try:
    from rakuda2.replay import Replay
except ImportError:
    pass

OVERWRITE = False

def p_control(current_states, delta, gain, clip):
    head_angle = current_states + np.clip((gain * delta), -clip, clip)
    return head_angle

class Robot:
    def __init__(self) -> None:
        self.joint_state = None

    def joint_states_callback(self, data):
        self.joint_state = list(data.position)

    def sample_data(self, observations, views, sample_size):
        total_size = len(observations)
        print(total_size)
        if sample_size > total_size:
            if input('SCENE BELOW DESIRED LENGTH! KEEP? y/n').lower() == 'y':
                return observations, views
            else:
                return None, None
        elif sample_size == total_size:
            return observations, views
        selected_indices = random.sample(range(total_size), sample_size)
        selected_indices.sort()
        sampled_observations = [observations[i] for i in selected_indices]
        sampled_views = [views[i] for i in selected_indices]
        return sampled_observations, sampled_views

    def collect_scene_sweep(self, x_range=[-math.pi, math.pi], y_range=[-math.pi, math.pi], num_y=4, error_tolerance=0.00001, scene_len=50, num_runs=1):
        replay = Replay()
        rospy.Subscriber('actual_joint_states', JointState, self.joint_states_callback, queue_size=1)
        initial_joint_state = replay.previous_state

        dataset_path = Path('data/actvis/nobg/scenes/')
        dataset_path.mkdir(parents=True, exist_ok=True)
        files = list(dataset_path.glob('scene_*.npz'))
        start_index = max([int(file.stem.split('_')[1]) for file in files] + [0]) + 1 if files else 0

        Y = np.linspace(y_range[0], y_range[1], num_y)
        avg_t = 0
        for n in range(num_runs):
            t0 = time.time()
            scene_observations = []
            scene_views = []
            print('--------! R E S E T  T H E  S C E N E !--------')
            sleep(5)
            print(f'Starting run {n+1}')
            current_state = initial_joint_state[1:3]
            step = 0
            for i in range(num_y):
                # Move to start
                if i % 2 == 0:
                    x_start = x_range[0]
                    x_end = x_range[1]
                else:
                    x_start = x_range[1]
                    x_end = x_range[0]
                y_tgt = Y[i]
                x_tgt = x_start
                tgt = np.array([x_tgt, y_tgt])
                print(f'Tgt: {tgt}')
                reached_tgt = False
                while not reached_tgt:
                    if self.joint_state is None:
                            continue
                    else:
                        angles = self.joint_state[1:3]
                        p_tgt = p_control(angles, tgt-angles, 0.8, 0.2)
                        joint_call = np.array([initial_joint_state[0], *p_tgt, *initial_joint_state[3:]])
                        replay.send(joint_call, 5)
                        err = tgt - np.array(self.joint_state[1:3])
                        if np.linalg.norm(err) < error_tolerance:
                            reached_tgt = True

                #Sweep across x axis
                #y_tgt = current_state[1]
                x_tgt = x_end
                tgt = np.array([x_tgt, y_tgt])
                print(f'Tgt: {tgt}')
                reached_tgt = False
                while not reached_tgt:
                    if self.joint_state is None:
                        continue
                    else:
                        image_state = replay.get_image(96*2, 128*2)
                        angles = self.joint_state[1:3]
                        p_tgt = p_control(angles, tgt-angles, 1, random.uniform(0.07, 0.081))
                        joint_call = np.array([initial_joint_state[0], *p_tgt, *initial_joint_state[3:]])
                        replay.send(joint_call, 5)
                        err = tgt - np.array(self.joint_state[1:3])
                        scene_views.append(angles)
                        scene_observations.append(image_state[0])
                        #plt.imsave(f"{dataset_path}/recorded_obs/obs{step}.jpg", image_state[0])
                        step += 1
                        if np.linalg.norm(err) < error_tolerance:
                            reached_tgt = True

            #scene_observations, scene_views = self.sample_data(scene_observations, scene_views, scene_len)
            if scene_observations is not None:
                print("saving")
                scene_observations = np.stack(scene_observations)
                scene_views = np.stack(scene_views)
                np.savez(f'{dataset_path}/scene_{start_index + n}.npz', observations=scene_observations, views=scene_views)
            else:
                print("not saving anything")
            t1 = time.time()
            dt = t1-t0
            if n == 0:
                avg_t = dt
            else:
                avg_t += (1/(n+2)) * (dt - avg_t)
            tleft = (num_runs - (n+1)) * avg_t
            mins, secs = divmod(tleft, 60)
            print(f"SWEEP TIME: {dt} MEAN TIME: {avg_t} TIME LEFT: {mins}:{secs}")

if __name__ == '__main__':
    n = 1
    scene_len = 50
    opts,_ = getopt.getopt(sys.argv[1:], "on:l:")
    for opt, arg in opts:
        if opt == "-o":
            OVERWRITE = True
        elif opt == "-n":
            n = int(arg)
        elif opt == "-l":
            scene_len = int(arg)
    robot = Robot()
    robot.collect_scene_sweep([-0.62,0.62], [0.88, 1.11], 6, 0.05, scene_len=scene_len, num_runs=n)
