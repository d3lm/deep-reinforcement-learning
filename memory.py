import numpy as np
from random import sample

class ExperienceReplay():
  def __init__(self, model, target_model, nb_actions, memory_size=100, batch_size=50, discount=.9, learning_rate=.1):
      self.memory = []
      self.model = model
      self.target_model = target_model
      self.nb_actions = nb_actions
      self.memory_size = memory_size
      self.batch_size = batch_size
      self.discount = discount
      self.learning_rate = learning_rate

      # input_shape = (frames, rows, columns)
      self.input_shape = model.input_shape[1:]

      # input_dim = frames * rows * columns
      self.input_dim = np.prod(self.input_shape)

  def remember(self, state_t, action, reward, state_tn, done):
    # Store flatten array for better memory efficiency
    # transition = [state_t, action_t, reward_t, state_tn, done]
    # memory[i] = transition
    transition = np.concatenate([
      state_t.flatten(),
      np.array(action).flatten(),
      np.array(reward).flatten(),
      state_tn.flatten(),
      1 * np.array(done).flatten()
    ])

    self.memory.append(transition)

    if len(self.memory) > self.memory_size:
      self.memory.pop(0)

  def reset_memory(self):
    self.memory = []

  def get_batch(self):
    batch_size = self.batch_size

    if len(self.memory) < batch_size:
      batch_size = len(self.memory)

    experience = np.array(sample(self.memory, batch_size))

    bare_transitions = self.extract_transition(experience, batch_size)
    states_t, actions, rewards, states_tn, done = self.reshape(batch_size, *bare_transitions)
    batch = np.concatenate([states_t, states_tn], axis=0)

    # First n (= batch_size) rows are the states_t
    # and the next m rows are the states_tn
    # [[ 0 0 0 0 0 ... #nb_actions ]
    #  [ 0 0 0 0 0 ... #nb_actions ]
    #   ... #states_t + #states_tn ]
    q_t = self.model.predict(batch)

    # q-values for the next states (states_tn)
    q_tn = self.get_q_next(q_t, states_tn, batch_size)

    # Delta (learning rate). Determines how aggressively
    # the q-values should be updated. 1 means very a
    # aggresive (replacing the q-value completely) and
    # 0 means not updating the values at all
    delta = np.zeros((batch_size, self.nb_actions))
    delta[np.arange(batch_size), actions] = self.learning_rate

    inputs = states_t

    # Update q-values based on the next states (states_tn)
    # q_t[:batch_size] = q-values for the current states (states_t)
    targets = (1 - delta) * q_t[:batch_size] + delta * (rewards + self.discount * (1 - done) * q_tn)

    return inputs, targets

  def get_q_next(self, q_t, states_tn, batch_size):
    if not self.target_model:
      # Plain DQN
      # A single network for action selection and generation of target q-values
      # Take max q-value from each next state (state_tn) and reshape into
      # [[ .5 .5 .5 .5 .5 ] | max q for state_tn[0]
      #  [ .2 .2 .2 .2 .2 ] | max q for state_tn[1]
      #   ... #state_tn ]
      q_next = np.max(q_t[batch_size:], axis=1)
    else:
      # Double DQN
      # The problem with plain DQN is that it tends to overestimate the q-values due to the
      # 'max' used in the formula to update the targets. The 'max' leads to a positive bias
      # because the highest q-value is propagated to previous states.
      # The solution is to have two separate networks, one primary network for determining the
      # action and a second (target) network to genrate the target q-values for that action.
      # By decoupling the action choice from the target Q-value generation, we are able to
      # substantially reduce the overestimation, and train faster and more reliably.

      # Select max action from primary network (from states_tn)
      next_actions = np.argmax(q_t[batch_size:], axis=1)

      # Generate target q-values with secondary (target) network
      target_q_values = self.target_model.predict(states_tn)

      # Take the highest q-values
      q_next = target_q_values[range(batch_size), next_actions]

    return q_next.repeat(self.nb_actions).reshape((batch_size, self.nb_actions))

  def extract_transition(self, experience, batch_size):
    input_dim = self.input_dim

    states_t = experience[:, 0:input_dim]
    actions = experience[:, input_dim]
    rewards = experience[:, input_dim + 1]
    states_tn = experience[:, input_dim + 2 : 2 * input_dim + 2]
    done = experience[:, 2 * input_dim + 2]

    return states_t, actions, rewards, states_tn, done

  def reshape(self, batch_size, states_t, actions, rewards, states_tn, done):
    states_t = states_t.reshape((batch_size, ) + self.input_shape)
    actions = np.cast['int'](actions)
    rewards = rewards.repeat(self.nb_actions).reshape((batch_size, self.nb_actions))
    states_tn = states_tn.reshape((batch_size, ) + self.input_shape)
    done = done.repeat(self.nb_actions).reshape((batch_size, self.nb_actions))

    return states_t, actions, rewards, states_tn, done
