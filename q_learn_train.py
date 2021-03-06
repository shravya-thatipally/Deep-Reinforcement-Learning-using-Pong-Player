import gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
import os


GAMMA = 0.99
OBSERVE = 500.
EXPLORE = 500.
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 1.0
REPLAY_MEMORY = 590000
BATCH = 32
K = 1

filename = './qlearn.ckpt'

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# model initialization
image_size = 80
D = image_size * image_size # input dimensionality: 80x80 grid

def createNetwork():
    # model definition in tensorflow
    tf.reset_default_graph()
    observations = tf.placeholder(tf.float32, [None, None, D] , name="input_x")

    x_image = tf.reshape(observations, [-1,image_size,image_size,1])

    # define the first layer: convolution + ReLU
    W_conv1 = tf.get_variable("W1", shape=[12,12,1,32], initializer=tf.contrib.layers.xavier_initializer())
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1))
    h_pool1 = max_pool_2x2(h_conv1)

    # define the second layer: convolution + ReLU
    W_conv2 = tf.get_variable("W2", shape=[8,8,32,48], initializer=tf.contrib.layers.xavier_initializer())
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2))
    h_pool2 = max_pool_2x2(h_conv2)

    # define the third layer: densely connected layer
    W_fc1 = tf.get_variable("W3", shape=[20 * 20 * 48, 256], initializer=tf.contrib.layers.xavier_initializer())
    h_pool2_flat = tf.reshape(h_pool2, [-1, 20*20*48])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1))

    # softmax laye
    W_fc2 = tf.get_variable("W4", shape=[256, 1], initializer=tf.contrib.layers.xavier_initializer())
    y_conv = tf.matmul(h_fc1, W_fc2)

    # now we get the probability of moving up
    probability = tf.nn.sigmoid(y_conv)

    #From here we define the parts of the network needed for learning a good policy.
    actions = tf.placeholder(tf.float32,[None,1], name="actions")
    reward = tf.placeholder(tf.float32, [None], name="reward")
    readout_action = tf.reduce_sum(tf.mul(probability, actions), reduction_indices = 1)

    cost = tf.reduce_mean((tf.square(reward - readout_action)) )

    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    return observations, probability, actions, reward, train_step

def choose_action(action_distribution):
    r = np.random.random()
    if r < action_distribution:
        return 2
    else:
        return 3

def preprocess_image(image):
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return image.astype(np.float).ravel()



if __name__ == "__main__":

    observations, probability, actions, reward, train_step = createNetwork()

    session = tf.Session()

    saver = tf.train.Saver()
    if os.path.exists(filename + '.index'):
        saver.restore(session, filename)
    else :
        init_op = tf.global_variables_initializer()
        session.run(init_op)

    epsilon = INITIAL_EPSILON
    t = 0

    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = np.zeros(D)
    env.reset()

    observation = preprocess_image(observation)
    movement = observation - prev_x
    prev_x = observation

    s_t = movement.reshape([1, 80 * 80])

    D = deque()

    while True:

        # choose an action epsilon greedily
        #env.render()
        action_distribution = session.run(probability, feed_dict={observations: [s_t]})

        if random.random() <= epsilon or t <= OBSERVE:
            action = choose_action(random.random())
        else:
            action = choose_action(action_distribution)

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        for i in range(0, K):
            # run the selected action and observe next state and reward
            observation, r_t, done, info = env.step(action)
            observation = preprocess_image(observation)
            movement = observation - prev_x
            prev_x = observation

            s_t1 = movement.reshape([1, 80 * 80])

            # store the transition in D
            D.append((s_t, action, r_t, s_t1, done))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            a_batch = np.vstack(a_batch)

            prob_j1_batch = session.run(probability, feed_dict={observations: s_j1_batch})
            for i in range(0, len(minibatch)):
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(prob_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict={
                reward: y_batch,
                actions: a_batch,
                observations: s_j_batch}, session=session)

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(session, filename)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
            print "TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action, "/ REWARD", r_t
