# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import cv2
import time
import numpy as np
from collections import deque
from tqdm import tqdm
from gripper_controller import GripperController
from multiprocessing.managers import SharedMemoryManager
from umi.common.precise_sleep import precise_sleep
from umi.common.latency_util import get_latency
from matplotlib import pyplot as plt

# %%
@click.command()
@click.option('-f', '--frequency', type=float, default=30)
def main(frequency):
    duration = 10.0
    sample_dt = 1 / 100
    k = int(duration / sample_dt)
    sample_t = np.linspace(0, duration, k)
    value = np.sin(sample_t * duration / 1.5) * 0.015 + 0.015
    width = value

    with SharedMemoryManager() as shm_manager:
        with GripperController(
            shm_manager=shm_manager,
            frequency=frequency,
            move_max_speed=1.5,
            get_max_k=int(k*1.2),
            command_queue_size=int(k*1.2),
            verbose=True) as gripper:

            gripper.schedule_waypoint(width[0], time.time() + 0.3)
            precise_sleep(1.0)

            timestamps = time.time() + sample_t + 1.0
            for i in range(k):
                gripper.schedule_waypoint(width[i], timestamps[i])
                time.sleep(0.0)
            precise_sleep(duration + 1.0)

            states = gripper.get_all_state()

    latency, info = get_latency(
        x_target=width,
        t_target=timestamps,
        x_actual=states['gripper_position'],
        t_actual=states['gripper_receive_timestamp']
    )
    print(f"End-to-end latency: {latency}sec")

    # plot everything
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(15, 5, forward=True)

    ax = axes[0]
    ax.plot(info['lags'], info['correlation'])
    ax.set_xlabel('lag')
    ax.set_ylabel('cross-correlation')
    ax.set_title("Cross Correlation")

    ax = axes[1]
    ax.plot(timestamps, width, label='target')
    ax.plot(states['gripper_receive_timestamp'], states['gripper_position'], label='actual')
    ax.set_xlabel('time')
    ax.set_ylabel('gripper-width')
    ax.legend()
    ax.set_title("Raw observation")

    ax=axes[2]
    t_samples = info['t_samples'] - info['t_samples'][0]
    ax.plot(t_samples, info['x_target'], label='target')
    ax.plot(t_samples-latency, info['x_actual'], label='actual-latency')
    ax.set_xlabel('time')
    ax.set_ylabel('gripper-width')
    ax.legend()
    ax.set_title(f"Aligned with latency={latency}")
    plt.show()

# %%
if __name__ == '__main__':
    main()
