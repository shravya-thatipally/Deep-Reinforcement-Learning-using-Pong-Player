import numpy as np
import sys
import gym
import pickle
import os

class PongPlayer(object):

    def __init__(self):
        self.env = gym.make('Pong-v0')
        self.filename = "policy_gradients_weights.p"
        self.input_layer_size = 6400
        self.hidden_layer_size = 200
        self.output_layer_size = 1
        self.discount_factor = 0.98
        self.gradient_step_size = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.learning_rate = 1e-4
        self.batch_size = 10
        self.rewards = []
        self.reward_count = 0
        self.accuracy_list = []
        self.won_count = 0
        self.load_network();

    # loads the saved model after each episode
    def load_network(self):
        if os.path.exists(self.filename):
            model = pickle.load(open(self.filename, 'rb'))
            self.W1 = model['W1']
            self.W2 = model['W2']
        else :
            self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size) / np.sqrt(self.input_layer_size)
            self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size) / np.sqrt(self.hidden_layer_size)
            # update buffers that add up gradients over a batch
        self.gradiant_buffer = {k: np.zeros_like(v)
                        for k, v in {"W1" :  self.W1, "W2" : self.W2}.iteritems()}
        self.m = {k: np.zeros_like(v) for k, v in self.gradiant_buffer.iteritems()}
        self.v = {k: np.zeros_like(v) for k, v in self.gradiant_buffer.iteritems()}

        if os.path.exists('accuracy'):
            with open('accuracy', 'rb') as fp:
                self.accuracy_list = pickle.load(fp)

    def save_network(self):
        model = {}
        model['W1'] = self.W1
        model['W2'] = self.W2
        pickle.dump(model, open(self.filename, 'wb'))



    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # same computation as karpathy blog
    def forward(self, movement):
        hidden_state = movement.dot(self.W1)
        hidden_state[hidden_state<0] = 0
        log_prob = hidden_state.dot(self.W2)
        prob = self.sigmoid(log_prob)
        return prob, hidden_state

    def back_prop(self, hidden_states, movements, gradients):
        new_W2 = hidden_states.T.dot(gradients)
        hs = np.outer(gradients, self.W2)
        hs[hidden_states <= 0] = 0
        new_W1 = movements.T.dot(hs)
        return {"W1" : new_W1, "W2" :new_W2}

    # not needed
    def finite_difference(self, f, model):
        numgrad = np.zeros(model.shape)
        perturb = np.zeros(model.shape)
        e = 1e-4
        for i in range(perturb.size):
            perturb.flat[i] = e
            loss1 = f(model - perturb)
            loss2 = f(model + perturb)
            numgrad.flat[i] = (loss2 - loss1) / (2 * e)
            perturb.flat[i] = 0
        return numgrad

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            if rewards[t] != 0:
                discounted_rewards[t] = rewards[t]
            else:
                discounted_rewards[t] = discounted_rewards[t+1] * self.discount_factor
        return discounted_rewards

    #
    def choose_action(self, action_distribution):
        r = np.random.random()
        if r < action_distribution:
            return 2, 1 #up
        else :
            return 3, 0 #down

    def preprocess_image(self, image):
        image = image[35:195]  # crop
        image = image[::2, ::2, 0]  # downsample by factor of 2
        image[image == 144] = 0  # erase background (background type 1)
        image[image == 109] = 0  # erase background (background type 2)
        image[image != 0] = 1  # everything else (paddles, ball) just set to 1
        return image.astype(np.float).ravel()


    def train(self):
        #env.monitor.start(outdir)
        #experiment_filename = './pong-experiment-1'
        #self.env.monitor.start(experiment_filename, force=True)
        observation = self.env.reset()
        for episode in range(300000):
            observation = self.env.reset()

            observed_states = []
            executed_actions = []
            observed_rewards = []
            hidden_states = []
            gradients = []
            prev_imag = np.zeros(self.input_layer_size)

            step = 0
            while True:
                step += 1

                observation = self.preprocess_image(observation)
                movement = observation - prev_imag
                prev_imag = observation
                observed_states.append(movement)
                action_distribution, hidden_state = self.forward(movement) # prob and state

                action, y = self.choose_action(action_distribution)

                observation, reward, done, info = self.env.step(action)

                hidden_states.append(hidden_state)
                executed_actions.append(action)
                gradients.append(y - action_distribution)
                observed_rewards.append(reward)

                if reward != 0:
                    self.reward_count += 1
                    print "reward %d, step %d, episode %d" % (reward, step, episode)

                if reward == 1:
                    self.won_count +=1
                    print "----------------------won---------------"

                if done or step > self.env.spec.timestep_limit:
                    print 'finished episode', episode, 'steps', step
                    # should write fews lines to use in learning curves
                    break

            steps = step

            observed_states = np.vstack(observed_states)
            hidden_states = np.vstack(hidden_states)
            observed_rewards = np.vstack(observed_rewards)
            gradients = np.vstack(gradients)

            discounted_rewards = self.discount_rewards(observed_rewards)
            gradients *= discounted_rewards
            gradient_buffer = self.back_prop(hidden_states, observed_states, gradients)
            for k in {"W1" :  self.W1, "W2" : self.W2}:
                self.gradiant_buffer[k] += gradient_buffer[k]

            # perform rmsprop parameter update every batch_size episodes
            if episode % self.batch_size == 0:
                print("*************************************")
                print("Batch number:", (episode / self.batch_size))
                print("Total rewards: ", self.reward_count)
                print("won:", self.won_count)
                print("accuracy:", (float(self.won_count) / float(self.reward_count)) * 100)
                self.accuracy_list.append((float(self.won_count) / float(self.reward_count)) * 100)
                with open('accuracy', 'wb') as fp:
                    pickle.dump(self.accuracy_list, fp)
                self.reward_count = 0
                self.won_count = 0
                print("\nUpdating parameters\n")
                for k, val in {'W1' :  self.W1, 'W2' : self.W2}.iteritems():
                    #print(k)
                    grad = self.gradiant_buffer[k]
                    self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad
                    self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grad ** 2)
                    if k in 'W1':
                        self.W1 += - self.learning_rate * self.m[k] / (np.sqrt(self.v[k]) + 1e-8)
                    else:
                        self.W2 += - self.learning_rate * self.m[k] / (np.sqrt(self.v[k]) + 1e-8)
                    #print(self.W2)
                    self.gradiant_buffer[k] = np.zeros_like(val)  # reset batch gradient buffer
                    self.save_network()
        #self.env.monitor.close()

if __name__ == "__main__":
    np.random.seed(0)
    pong = PongPlayer()
    pong.train()