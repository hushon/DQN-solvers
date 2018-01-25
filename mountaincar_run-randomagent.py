import argparse
import gym

## define agent
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

if __name__ == '__main__':

    ## argument parser
    ## pass an OpenAI Gym environment name to specify the environment
    default_env = 'MountainCar-v0'
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default=default_env, help='Select the environment to run')
    args = parser.parse_args()

    ## define environment
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
            if done:
                print("Terminated at step #", st)
                break

    env.close()