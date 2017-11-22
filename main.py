from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from games import Snake
from keras.models import load_model
from agent import Agent
import argparse

boolean = lambda x: (str(x).lower() == 'true')

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='?', type=boolean, const=True, default=True)
parser.add_argument("--model", nargs='?', const=True)
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
rows, columns = game.field_shape()
nb_frames = args.frames
nb_epoch = args.epochs
memory_size = args.memory_size
batch_size = args.batch_size
epsilon = args.epsilon
discount = args.discount
learning_rate = args.learning_rate
nb_actions = game.nb_actions()

model = None

if args.train:
  model = Sequential()
  model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(nb_frames, rows, columns), data_format="channels_first"))
  model.add(Conv2D(64, (2, 2), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Flatten())
  model.add(Dropout(0.1))
  model.add(Dense(512, activation='relu'))
  model.add(Dense(nb_actions))
  model.compile(Adam(), 'MSE')
else:
  model = load_model(args.model)

agent = Agent(game, model, nb_epoch, memory_size, batch_size, nb_frames, epsilon, discount, learning_rate)

if args.train:
  agent.train()
else:
  agent.play(nb_games=args.games, interval=args.interval)
