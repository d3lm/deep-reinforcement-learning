from games import Snake
from agent import Agent, TEST
import argparse

boolean = lambda x: (str(x).lower() == 'true')

# Command line argumentss
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='?', type=boolean, const=True, default=True)
parser.add_argument("--model", nargs='?', const=True)
parser.add_argument("--mode", nargs='?', type=int, const=True, default=1, choices=[0,1,2])
parser.add_argument("--update-freq", nargs='?', type=int, const=True, default=10)
parser.add_argument("--grid-size", nargs='?', type=int, const=True, default=10)
parser.add_argument("--frames", nargs='?', type=int, const=True, default=4)
parser.add_argument("--epochs", nargs='?', type=int, const=True, default=10000)
parser.add_argument("--memory-size", nargs='?', type=int, const=True, default=1000)
parser.add_argument("--batch-size", nargs='?', type=int, const=True, default=32)
parser.add_argument("--epsilon", nargs='?', type=float, const=True, default=1.)
parser.add_argument("--discount", nargs='?', type=float, const=True, default=.9)
parser.add_argument("--learning_rate", nargs='?', type=float, const=True, default=.1)
parser.add_argument("--walls", nargs='?', type=boolean, const=True, default=True)
parser.add_argument("--games", nargs='?', type=int, const=True, default=5)
parser.add_argument("--interval", nargs='?', type=float, const=True, default=.1)

args = parser.parse_args()

if not args.train and args.model is None:
  parser.error("Non-training mode requires a model")

print(args)

game = Snake(grid_size=args.grid_size, walls=args.walls)

# Hyper parameter for the neural net and the agent
nb_frames = args.frames
nb_epoch = args.epochs
memory_size = args.memory_size
batch_size = args.batch_size
epsilon = args.epsilon
discount = args.discount
learning_rate = args.learning_rate
nb_actions = game.nb_actions()
mode = args.mode if args.train else TEST
update_freq = args.update_freq

agent = Agent(game, mode, nb_epoch, memory_size, batch_size, nb_frames, epsilon, discount, learning_rate, model=args.model)

if args.train:
  agent.train(update_freq=update_freq)
else:
  agent.play(nb_games=args.games, interval=args.interval)
