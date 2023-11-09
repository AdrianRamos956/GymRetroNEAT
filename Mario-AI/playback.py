import retro
import numpy as np
import cv2 
import neat
import pickle

env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2.state')

imgarray = []
xpos_end = 0

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

with open('winnerlvl2.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)

ob = env.reset()
ac = env.action_space.sample()

inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness= 0
fitness_current = 0
frame = 0
counter = 0
xpos = 0
xpos_max = 0

done = False

while True:
    
    env.render()
    frame += 1

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx,iny))
    ob = np.interp(np.ndarray.flatten(ob), (0, 254), (-1, +1))

    imgarray = np.ndarray.flatten(ob)
    nnOutput = net.activate(imgarray)

    
    ob, rew, done, info = env.step(nnOutput)
    
    xpos = info['x']
    xpos_end = info['endOfLevel']
    fitness = 0
    counter = 0

    while True:
        ob, rew, done, info = env.step(nnOutput)

        xpos = info['x']
        xpos_end = info['endOfLevel']

        if xpos > xpos_max:
            fitness += 1
            xpos_max = xpos
            counter = 0
        else:
            counter += 1

        if xpos_end == 1:
            fitness += 100000
            print("Ive Been Activated we have a winner")
            break

        if done or counter == 300:
            break