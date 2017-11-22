import numpy as np
import random
import warnings
from scipy.spatial import distance

WALL_SYMBOL = 1
APPLE_SYMBOL = 3
HEAD_SYMBOL = 8
TAIL_SYMBOL = 2

# 0 = Left, 1 = Right, 2 = Up, 3 = Down
ACTIONS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

class Snake():
  def __init__(self, grid_size=10, snake_length=3, walls=False):
    self.walls = walls
    self.snake_length = snake_length
    self.grid_size = grid_size

    if self.grid_size < 2:
      warnings.warn("Grid too small: Falling back to 2x2")
      self.grid_size = 2

    if walls and self.grid_size < 4:
      warnings.warn("Grid too small for having walls: Falling back to 4x4")
      self.grid_size = 4

    self.grid_cells = self.grid_size * self.grid_size
    self.spaces = self.grid_cells

    if walls:
      self.spaces = self.grid_cells - ((4 * self.grid_size) - 4)

    self.cells_per_direction = self.grid_size if not walls else self.grid_size - 2

    if snake_length > self.cells_per_direction:
      self.snake_length = self.cells_per_direction
      message = "Snake too long for grid: Falling back to {} body parts".format(self.snake_length)
      warnings.warn(message)

    self.reset()

  def nb_actions(self):
    return len(ACTIONS)

  def field_shape(self):
    return self.field.shape

  def act(self, action):
    self.play(action)
    reward = self.get_reward()
    is_done = self.is_done()
    is_victory = self.is_victory()
    return self.get_state(), reward, is_done, is_victory

  def play(self, action):
    action = int(action)

    if action not in ACTIONS:
        return

    self.scored = False
    self.move_snake(self.snake, self.apple, action)

    if self.scored and not self.is_victory():
        self.apple = self.generate_apple(self.field)

    self.update_state()

  def move_snake(self, snake, apple, action):
    action = ACTIONS[action]

    if self.is_opposite(self.prev_action, action):
        action = self.prev_action
    else:
        self.prev_action = action

    (dir_x, dir_y) = action
    (row, column) = snake[0]

    row += 1 * dir_x
    column += 1 * dir_y

    new_head = self.wrap_boundaries((row, column))

    if self.is_collision(new_head, apple):
        self.scored = True
        self.score += 1
    else:
        snake.pop()

    snake.insert(0, new_head)

  def reset(self):
    self.game_over = False
    self.scored = False
    self.score = 0
    self.field, self.border = self.init_field()
    self.snake = self.init_snake(self.field)
    self.apple = self.generate_apple(self.field)
    self.prev_action = (0, 1)
    self.update_state()

  def is_opposite(self, prev_action, curr_action):
    opposite = tuple([x * -1 for x in prev_action])
    return curr_action == opposite

  def is_done(self):
    return self.is_game_over() or self.is_victory()

  def is_game_over(self):
    head, tail = self.snake[0], self.snake[1:]
    return any(self.is_collision(segment, head) for segment in tail) or head in self.border

  def is_victory(self):
    if self.walls:
      return self.get_snake_len() == self.spaces
    else:
      return self.get_snake_len() == self.grid_cells

  def is_collision(self, a, b):
    return a == b

  def get_score(self):
    return self.score

  def get_snake_len(self):
    return len(self.snake)

  def get_reward(self):
    score = 0

    if self.is_victory():
      score = 1
    elif self.is_game_over():
      score = -1
    elif self.scored:
      score = .5
    else:
      score = -.1

    return score

  def get_state(self):
    return self.field

  def update_state(self):
    self.field = self.clear_field()
    self.render_apple(self.field, self.apple)
    self.render_snake(self.field, self.snake)

  def init_field(self):
    border = []
    field = np.zeros((self.grid_size, self.grid_size), dtype=np.int)

    if self.walls:
      field = np.ones((self.grid_size, self.grid_size), dtype=np.int)
      field[1:-1, 1:-1] = 0

      for z in range(self.grid_size):
        border += [(z, 0), (z, self.grid_size - 1), (0, z), (self.grid_size - 1, z)]

    return field, border

  def render_apple(self, field, apple):
    field.itemset(apple, APPLE_SYMBOL)

  def render_snake(self, field, snake):
    for index, (row, column) in enumerate(snake):
      symbol = HEAD_SYMBOL if index == 0 else TAIL_SYMBOL
      field.itemset((row, column), symbol)

  def clear_field(self):
    field, _ = self.init_field()

    if self.walls:
      field[1:-1, 1:-1] = 0

    return field

  def init_snake(self, field):
    row, column = self.get_free_cell(field)

    snake = [(row, column)]

    for index in range(1, self.snake_length):
      if not self.walls:
        body_segment = self.wrap_boundaries((row, column - index))
      else:
        curr_segment = snake[index - 1]
        body_segment = self.get_next_segment(field, curr_segment)

      snake.append(body_segment)
      self.render_snake(field, snake)

    return snake

  def get_next_segment(self, field, prev_segment):
    row, column = prev_segment

    if self.is_free_cell(field, row, column - 1):
      return (row, column - 1)
    elif self.is_free_cell(field, row - 1, column):
      return (row - 1, column)
    elif self.is_free_cell(field, row + 1, column):
      return (row + 1, column)
    else:
      return (row, column + 1)

  def generate_apple(self, field):
    return self.get_free_cell(field)

  def wrap_boundaries(self, (row, column)):
    return (row % self.grid_size, column % self.grid_size)

  def random_cell(self, max, min=0):
    return random.randint(min, max)

  def is_free_cell(self, field, row, column):
    return field[row, column] == 0

  def get_free_cell(self, field):
    start = 0
    end = self.grid_size - 1

    if self.walls:
      start += 1
      end = self.cells_per_direction

    row = self.random_cell(end, start)
    column = self.random_cell(end, start)

    while(not self.is_free_cell(field, row, column)):
        row = self.random_cell(end, start)
        column = self.random_cell(end, start)

    return (row, column)
