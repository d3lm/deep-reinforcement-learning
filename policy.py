import numpy as np

# Policies are used by the agent to choose actions.

class LinearControlSignal():
  def __init__(self, start, end, nb_epoch):
    self.eps = start
    self.start = start
    self.end = end
    self.nb_epoch = nb_epoch

    # Linear coefficient
    self.coefficient = (end - start) / nb_epoch

  def get_value(self, epoch):
    if epoch < self.nb_epoch:
      value = epoch * self.coefficient + self.start
    else:
      value = self.end

    return value

class EpsGreedyPolicy():
  def __init__(self, model, nb_epoch, nb_actions, start=1., end=.1):
    self.eps = start
    self.end = end
    self.model = model
    self.nb_epoch = nb_epoch
    self.nb_actions = nb_actions
    self.eps_linear = LinearControlSignal(start, end, nb_epoch)

  def get_eps(self):
    return self.eps

  def select_action(self, state_t, epoch):
    self.eps = self.eps_linear.get_value(epoch)

    if np.random.uniform() < self.eps:
      action = np.random.random_integers(0, self.nb_actions-1)
    else:
      q_values = self.model.predict(state_t)
      action = int(np.argmax(q_values[0]))

    return action
