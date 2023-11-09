import retro        # pip install gym-retro
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        
    def work(self):
        
        self.env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
        
        self.env.reset()
        
        ob, _, _, _ = self.env.step(self.env.action_space.sample())
        
        inx = int(ob.shape[0]/8)
        iny = int(ob.shape[1]/8)
        done = False
        
        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)

        
        fitness = 0
        xpos = 0
        xpos_max = 0
        coutner = 0
        fitness_current = 0
        current_max_fitness = 0
        imgarray = []
        
        while not done:
            # self.env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            imgarray = np.ndarray.flatten(ob)
            imgarray = np.interp(imgarray, (0, 254), (-1, +1))
            actions = net.activate(imgarray)
            
            ob, rew, done, info = self.env.step(actions)
            
            xpos = info['x']
            xpos_end = info['screen_x_end']

            if xpos > xpos_max:
                fitness_current += 1
                xpos_max = xpos
                coutner = 0
            else:
                coutner += 1

            if xpos == xpos_end and xpos > 500:
                fitness_current += 100000
                done = True

            fitness_current += rew

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
            
            if done or coutner == 300:
                done = True
    
        print(fitness_current, xpos, xpos_end)  
        return fitness_current

def eval_genomes(genome, config):
    
    worky = Worker(genome, config)
    return worky.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)
#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-79')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

if __name__ == "__main__":

    pe = neat.ParallelEvaluator(10, eval_genomes)

    winner = p.run(pe.evaluate)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

