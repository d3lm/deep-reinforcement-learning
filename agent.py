import sys
import numpy as np
import warnings
import utils
from enum import Enum
from time import time, sleep
import matplotlib.pyplot as plt
from policy import EpsGreedyPolicy
from memory import ExperienceReplay
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model

TEST = 0
SIMPLE = 1
DOUBLE = 2

class Agent:
  def __init__(self, game, mode=SIMPLE, nb_epoch=10000, memory_size=1000, batch_size=50, nb_frames=4, epsilon=1., discount=.9, learning_rate=.1, model=None):

    self.game = game
    self.mode = mode
    self.target_model = None
    self.rows, self.columns = game.field_shape()
    self.nb_epoch = nb_epoch
    self.nb_frames = nb_frames
    self.nb_actions = game.nb_actions()

    if mode == TEST:
      print('Training Mode: Loading model...')
      self.model = load_model(model)
    elif mode == SIMPLE:
      print('Using Plain DQN: Building model...')
      self.model = self.build_model()
    elif mode == DOUBLE:
      print('Using Double DQN: Building primary and target model...')
      self.model = self.build_model()
      self.target_model = self.build_model()
      self.update_target_model()

    # Trades off the importance of sooner versus later rewards.
    # A factor of 0 means it rather prefers immediate rewards
    # and it will mostly consider current rewards. A factor of 1
    # will make it strive for a long-term high reward.
    self.discount = discount

    # The learning rate or step size determines to what extent the newly
    # acquired information will override the old information. A factor
    # of 0 will make the agent not learn anything, while a factor of 1
    # would make the agent consider only the most recent information
    self.learning_rate = learning_rate

    # Use epsilon-greedy exploration as our policy.
    # Epsilon determines the probability for choosing random actions.
    # This factor will decrease linear by the number of epoches. So we choose
    # a random action by the probability 'eps'. Without this policy the network
    # is greedy and it will it settles with the first effective strategy it finds.
    # Hence, we introduce certain randomness.
    # Epislon reaches its minimum at 1/2 of the games
    epsilon_end = self.nb_epoch - (self.nb_epoch / 2)
    self.policy = EpsGreedyPolicy(self.model, epsilon_end, self.nb_actions, epsilon, .1)

    # Create new experience replay memory. Without this optimization
    # the training takes extremely long even on a GPU and most
    # importantly the approximation of Q-values using non-linear
    # functions, that is used for our NN, is not very stable.
    self.memory = ExperienceReplay(self.model, self.target_model, self.nb_actions, memory_size, batch_size, self.discount, self.learning_rate)

    self.frames = None

  def build_model(self):
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(self.nb_frames, self.rows, self.columns), data_format="channels_first"))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(self.nb_actions))
    model.compile(Adam(), 'MSE')

    return model

  def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())

  def get_frames(self):
    frame = self.game.get_state()
    if self.frames is None:
      self.frames = [frame] * self.nb_frames
    else:
      self.frames.append(frame)
      self.frames.pop(0)

    # Expand frames to match the input shape for the CNN (4D)
    # 1D      = # batches
    # 2D      = # frames per batch
    # 3D / 4D = game board
    return np.expand_dims(self.frames, 0)

  def clear_frames(self):
    self.frames = None

  def print_stats(self, data, y_label, x_label='Epoch', marker='-'):
    data = np.array(data)
    x, y = data.T
    p = np.polyfit(x, y, 3)

    fig = plt.figure()

    plt.plot(x, y, marker)
    plt.plot(x, np.polyval(p, x), 'r:')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    words = y_label.split()
    file_name = '_'.join(map(lambda x: x.lower(), words))
    path = './plots/{name}_{size}x{size}_{timestamp}'
    fig.savefig(path.format(size=self.game.grid_size, name=file_name, timestamp=int(time())))

  def train(self, update_freq=10):
    total_steps = 0
    max_steps = self.game.grid_size**2 * 3
    loops = 0
    nb_wins = 0
    cumulative_reward = 0
    duration_buffer = []
    reward_buffer = []
    steps_buffer = []
    wins_buffer = []

    for epoch in range(self.nb_epoch):
      loss = 0.
      duration = 0
      steps = 0

      self.game.reset()
      self.clear_frames()
      done = False

      # Observe the initial state
      state_t = self.get_frames()

      start_time = time()

      while(not done):
        # Explore or Exploit
        action = self.policy.select_action(state_t, epoch)

        # Act on the environment
        _, reward, done, is_victory = self.game.act(action)
        state_tn = self.get_frames()

        cumulative_reward += reward
        steps += 1
        total_steps += 1

        if steps == max_steps and not done:
          loops += 1
          done = True

        # Build transition and remember it (Experience Replay)
        transition = [state_t, action, reward, state_tn, done]
        self.memory.remember(*transition)
        state_t = state_tn

        # Get batch of batch_size samples
        # A batch generally approximates the distribution of the input data
        # better than a single input. The larger the batch, the better the
        # approximation. However, larger batches take longer to process.
        batch = self.memory.get_batch()

        if batch:
          inputs, targets = batch
          loss += float(self.model.train_on_batch(inputs, targets))

        if self.game.is_victory():
          nb_wins += 1

        if done:
          duration = utils.get_time_difference(start_time, time())

        if self.mode == DOUBLE and self.target_model is not None and total_steps % (update_freq) == 0:
          self.update_target_model()

      current_epoch = epoch + 1
      reward_buffer.append([current_epoch, cumulative_reward])
      duration_buffer.append([current_epoch, duration])
      steps_buffer.append([current_epoch, steps])
      wins_buffer.append([current_epoch, nb_wins])

      summary = 'Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Time(ms) {:3.3f} | Steps {:.2f} | Wins {} | Loops {}'
      print(summary.format(current_epoch, self.nb_epoch, loss, self.policy.get_eps(), duration, steps, nb_wins, loops))

    # Generate plots
    self.print_stats(reward_buffer, 'Cumulative Reward')
    self.print_stats(duration_buffer, 'Duration per Game')
    self.print_stats(steps_buffer, 'Steps per Game')
    self.print_stats(wins_buffer, 'Wins')

    path = './models/model_{mode}_{size}x{size}_{epochs}_{timestamp}.h5'
    mode = 'dqn' if self.mode == SIMPLE else 'ddqn'
    self.model.save(path.format(mode=mode, size=self.game.grid_size, epochs=self.nb_epoch, timestamp=int(time())))

  def play(self, nb_games=5, interval=.7):
    nb_wins = 0
    accuracy = 0
    summary = '{}\n\nAccuracy {:.2f}% | Game {}/{} | Wins {}'

    for epoch in range(nb_games):
      self.game.reset()
      self.clear_frames()
      done = False

      state_t = self.get_frames()

      self.print_state(summary, state_t[:,-1], accuracy, epoch, nb_games, nb_wins, 0)

      while(not done):
        q = self.model.predict(state_t)
        action = np.argmax(q[0])

        _, _, done, is_victory = self.game.act(action)
        state_tn = self.get_frames()

        state_t = state_tn

        if is_victory:
          nb_wins += 1

        accuracy = 100. * nb_wins / nb_games

        self.print_state(summary, state_t[:,-1], accuracy, epoch, nb_games, nb_wins, interval)

  def print_state(self, summary, state, accuracy, epoch, nb_games, nb_wins, interval):
    utils.clear_screen()
    print(summary.format(state, accuracy, epoch + 1, nb_games, nb_wins))
    sleep(interval)
