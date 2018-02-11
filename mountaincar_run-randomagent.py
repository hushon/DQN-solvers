## argument parser
## pass an OpenAI Gym environment name to specify the environment
import argparse
parser = argparse.ArgumentParser(description="Run a random agent in an OpenAI Gym environment")
parser.add_argument('-e', '--env-id', nargs='?', default='MountainCar-v0', help='Select an OpenAI Gym environment to run')
args = parser.parse_args()

import gym
from time import sleep

## define agent
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

if __name__ == '__main__':
    ## make environment
    env = gym.make(args.env_id)
    env.reset()

    ## execute
    episode_count = 10
    max_step = 200
    agent = RandomAgent(env.action_space)

    for ep in range(episode_count):
        print("Episode #", ep)
        env.reset()
        done = False
        for st in range(max_step):
            obs, reward, done, _ = env.step(agent.act())
            env.render()
            sleep(0.01)
            if done:
                print("Terminated at step #", st)
                break

    env.close()