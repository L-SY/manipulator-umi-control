import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

import rospy
import sys
import os
workspace_path = "/home/lsy/manipulator_control"
sys.path.insert(0, os.path.join(workspace_path, "devel/lib/python3/dist-packages"))
from manipulator_msgs.msg import GripperState, GripperCmd
from threading import Lock

class GripperRosInterface:
    def __init__(self):
        rospy.init_node('gripper_command_publisher', anonymous=False)
        self.cmd_pub = rospy.Publisher('/controllers/gripper_controller/command', GripperCmd, queue_size=10)
        self.state_sub = rospy.Subscriber('/controllers/gripper_controller/state', GripperState, self.state_callback)
        self.gripper_cmd = GripperCmd()
        self.gripper_cmd.mode = 0
        self.gripper_cmd.des_pos = 0.033
        self.gripper_cmd.des_vel = 0.0
        self.gripper_cmd.des_eff = 0.0
        self.gripper_cmd.des_kd = 0.0
        self.gripper_cmd.des_kp = 0.0
        self.gripper_pos = 0
        self.gripper_vel = 0
        self.gripper_eff = 0

    def test(self):
        print('ros_test',self.gripper_pos)
    def state_callback(self, msg):
        self.gripper_pos = msg.gripper_pos
        self.gripper_vel = msg.gripper_vel
        self.gripper_eff = msg.gripper_eff
        print('ros',self.gripper_pos)
        self.test()

    def publish_command(self):
        # Read data from ros_queue and update gripper_cmd
        # if not self.ros_queue.empty():
        #     command_data = self.ros_queue.get()
        #     self.gripper_cmd.mode = command_data['mode']
        #     self.gripper_cmd.des_pos = command_data['des_pos']
        #     self.gripper_cmd.des_vel = command_data['des_vel']
        #     self.gripper_cmd.des_eff = command_data['des_eff']
        #     self.gripper_cmd.des_kd = command_data['des_kd']
        #     self.gripper_cmd.des_kp = command_data['des_kp']
        #     # Publish command using ROS interface
        self.cmd_pub.publish(self.gripper_cmd)

class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class GripperController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 frequency=30,
                 move_max_speed=1.5,
                 get_max_k=None,
                 command_queue_size=1024,
                 receive_latency=0.0,
                 verbose=True
                 ):
        super().__init__(name="GripperController")
        self.frequency = frequency
        self.move_max_speed = move_max_speed
        self.receive_latency = receive_latency
        self.scale = 1.0
        self.verbose = verbose
        self.lock = Lock()
        if get_max_k is None:
            get_max_k = int(frequency * 10)

        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )

        # Define an example dictionary for the ros_queue
        # example_ros_queue = {
        #     'mode': 0,           # Corresponds to GripperCmd.mode
        #     'des_pos': 0.0,    # Corresponds to GripperCmd.des_pos
        #     'des_vel': 0.0,      # Corresponds to GripperCmd.des_vel
        #     'des_eff': 0.0,      # Corresponds to GripperCmd.des_eff
        #     'des_kd': 0.0,       # Corresponds to GripperCmd.des_kd
        #     'des_kp': 0.0,       # Corresponds to GripperCmd.des_kp
        #     'gripper_pos': 0,    # Example gripper position
        #     'gripper_vel': 0,    # Example gripper velocity
        #     'gripper_eff': 0     # Example gripper efficiency
        # }

        # Create the ros_queue with the example data
        # self.ros_queue = SharedMemoryQueue.create_from_examples(
        #     shm_manager=shm_manager,
        #     examples=example_ros_queue,
        #     buffer_size=command_queue_size
        # )

        # build ring buffer
        example = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        self.ros_interface = GripperRosInterface()
        self.ready_event = mp.Event()

    def start(self, wait=False):
        super().start()
        if self.verbose:
            print(f"[GripperController] Controller process spawned at {self.pid}")

    def stop(self, wait=False):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run(self):
        # start connection
        try:
            # get initial
            curr_pos = 0.14
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos, 0, 0, 0, 0, 0]]
            )

            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            while keep_running:
                # while not rospy.is_shutdown():
                t_now = time.monotonic()
                dt = 1 / self.frequency
                t_target = t_now
                target_pos = pose_interp(t_target)[0]
                target_vel = (target_pos - pose_interp(t_target - dt)[0]) / dt

                self.ros_interface.gripper_cmd.des_pos = target_pos
                self.ros_interface.gripper_cmd.des_vel = target_vel
                # Publish command using ROS interface
                self.ros_interface.publish_command()

                # get state from robot
                state = {
                    'gripper_state': 0,
                    'gripper_position': self.ros_interface.gripper_pos,
                    'gripper_velocity': self.ros_interface.gripper_vel,
                    'gripper_force': self.ros_interface.gripper_eff,
                    'gripper_measure_timestamp': time.time(),
                    'gripper_receive_timestamp': time.time(),
                    'gripper_timestamp': time.time()
                }
                self.ros_interface.test()
                print('run',self.ros_interface.gripper_pos)
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos']
                        target_time = command['target_time']
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[GripperController] Disconnected from robot")
