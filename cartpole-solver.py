## Original implementation from https://github.com/ageron/tiny-dqn
## Edited to solve OpenAI Gym classic environments by github.com/hushon

from __future__ import division, print_function, unicode_literals

# Handle arguments (before slow imports so --help can be fast)
import argparse
parser = argparse.ArgumentParser(
    description="Train a DQN net to play OpenAI Gym classic environments.")
parser.add_argument("-e", "--environment", action="store", default="CartPole-v0",
    help="name of the Gym environment")
parser.add_argument("-n", "--number-steps", type=int, default=10000,
    help="total number of training steps")
parser.add_argument("-l", "--learn-iterations", type=int, default=4,
    help="number of game iterations between each training step")
parser.add_argument("-s", "--save-steps", type=int, default=400,
    help="number of training steps between saving checkpoints")
parser.add_argument("-c", "--copy-steps", type=int, default=100,
    help="number of training steps between copies of online DQN to target DQN")
parser.add_argument("-r", "--render", action="store_true", default=False,
    help="render the game during training or testing")
parser.add_argument("-p", "--path", default="./CartPole-v0/my_dqn.ckpt",
    help="path of the checkpoint file")
parser.add_argument("-t", "--test", action="store_true", default=False,
    help="test (no learning and minimal epsilon)")
parser.add_argument("-v", "--verbosity", action="count", default=0,
    help="increase output verbosity")
args = parser.parse_args()

from time import sleep
from collections import deque
import gym
from gym import wrappers
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make(args.environment)
# if args.test: env = wrappers.Monitor(env, args.path+'/cartpole-experiment-1')
done = True  # env needs to be reset

# First let's build the two DQNs (online & target)
n_outputs = env.action_space.n  # 3 discrete actions are available
num_outputs_list = [200, 500] # number of units in input layer and hidden layer
activation_list = [tf.nn.relu, tf.nn.relu] # activation function in input layer and hidden layer

def q_network(X_state, name):
    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        for num_outputs, activation in zip(num_outputs_list, activation_list):
            prev_layer = tf.contrib.layers.fully_connected(
                prev_layer,
                num_outputs,
                activation_fn=activation,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None)
        outputs = tf.contrib.layers.fully_connected(
            prev_layer,
            n_outputs,
            activation_fn=tf.nn.relu,
            normalizer_fn=None,
            normalizer_params=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=None,
            biases_initializer=tf.zeros_initializer(),
            biases_regularizer=None,
            reuse=None,
            variables_collections=None,
            outputs_collections=None,
            trainable=True,
            scope=None)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

state_shape = (np.prod(env.observation_space.shape), )
X_state = tf.placeholder(tf.float32, shape=[None]+list(state_shape))
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

# We need an operation to copy the online DQN to the target DQN
copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

# Now for the training operations
learning_rate = 0.001
momentum = 0.95

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1, keep_dims=True)
    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Let's implement a simple replay memory
replay_memory_size = 20000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
           cols[4].reshape(-1, 1))

# And on to the epsilon-greedy policy with decaying epsilon
eps_min, eps_max = (0.1, 1.0) if not args.test else (0.0, 0.0)
eps_decay_steps = args.number_steps // 2

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action

## We need to preprocess the images to speed up training
## preprocessor stacks two observations and returnes a flattened array
# def preprocess_observation(obs1, obs2):
#     obs_stacked = np.vstack((obs1, obs2))
#     return obs_stacked.reshape(-1)
def preprocess_observation(obs1):
    return obs1

# TensorFlow - Execution phase
training_start = 0  # start training after 10,000 game iterations
discount_rate = 0.9
skip_start = 0  # Skip the start of every game (it's just waiting time).
batch_size = 50
iteration = 0  # game iterations
done = True # env needs to be reset

# We will keep track of the max Q-Value over time and compute the mean per game
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0

# for plotting loss
plt.figure(1)
loss_list = np.zeros(args.number_steps)

with tf.Session() as sess:
    if os.path.isfile(args.path + ".index"):
        saver.restore(sess, args.path)
    else:
        init.run()
        copy_online_to_target.run()
    while True:
        step = global_step.eval()
        if step >= args.number_steps:
            break
        iteration += 1
        if args.verbosity > 0:
            print("\rIteration {}   Training step {}/{} ({:.1f})%   "
                  "Loss {:5f}    Mean Max-Q {:5f}   ".format(
            iteration, step, args.number_steps, step * 100 / args.number_steps,
            loss_val, mean_max_q), end="")
        if done: # game over, start again
            print("Game over")
            obs = env.reset()
            for skip in range(skip_start): # skip the start of each game
                obs, reward, done, info = env.step(0)
            state = preprocess_observation(obs)
            obs_old = obs

        if args.render:
            env.render()
            sleep(0.05)

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)
        print(("left", "stay", "right")[action])
        # Online DQN plays
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)
        obs_old = obs

        # Let's memorize what happened
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        # Skip below when executed in test mode
        if args.test:
            continue

        # Compute statistics for tracking progress (not shown in the book)
        total_max_q += q_values.max()
        game_length += 1
        if done:
            mean_max_q = total_max_q / game_length
            total_max_q = 0.0
            game_length = 0

        if iteration < training_start or iteration % args.learn_iterations != 0:
            continue # only train after warmup period and at regular intervals
        
        # Sample memories and use the target DQN to produce the target Q-Value
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            sample_memories(batch_size))
        next_q_values = target_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        # Train the online DQN
        _, loss_val = sess.run([training_op, loss], feed_dict={
            X_state: X_state_val, X_action: X_action_val, y: y_val})
        loss_list[step] = loss_val

        # Regularly copy the online DQN to the target DQN
        if step % args.copy_steps == 0:
            copy_online_to_target.run()

        # And save regularly
        if step % args.save_steps == 0:
            saver.save(sess, args.path)

plt.plot(np.arange(args.number_steps), loss_list)
plt.savefig('./CartPole-v0/loss_chart.png')
