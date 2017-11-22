import os
from games import Snake
import utils

game = Snake(grid_size=5, walls=True)

def start_game():
  game.reset()
  reward = 0
  cummulative_reward = 0
  done = False

  while(not done):
    print('{}\n\nReward {}\nCummulative Reward {:.1f}'.format(game.get_state(), reward, cummulative_reward))
    print('\n0 = Left\n1 = Right\n2 = Up\n3 = Down\n---------')
    action = raw_input("Action: ")
    game.play(action)
    reward = game.get_reward()
    done = game.is_done()
    cummulative_reward += reward
    utils.clear_screen()

  print('GAME OVER', reward)

restart = ''

while(restart == '' or restart == 'y'):
  utils.clear_screen()
  start_game()
  restart = raw_input("Restart (Y/n): ")