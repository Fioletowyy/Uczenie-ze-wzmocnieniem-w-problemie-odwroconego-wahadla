import numpy as np
import gym
import random
import matplotlib
import matplotlib.pyplot as plt

goal_steps = 20000000000000 #ilosc wywolan pojedynczej akcji/gry
neurons=16   #ilosc neuronow

env = gym.make("CartPole-v0")
env.reset()
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:                                 #inicjalizacja odswiezania wyswietlania
    from IPython import display

plt.ion()

def sigmoid(x):                                 #inicjalizacja f-cji aktywacji sigma
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):                      #inicjalizacja pochodnej sigmy
    return x * (1.0 - x)

class NeuralNetwork:            #inicjalizacja sieci neuronowej
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],neurons) 
        self.weights2   = np.random.rand(neurons,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.predict    = 0

    def feedforward(self):      
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # wsteczna propagacja / nauka sieci
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # aktualizacja wag
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    def prediction(self,action):            # predykcja
        self.layer1 = sigmoid(np.dot(action, self.weights1))
        self.predict = sigmoid(np.dot(self.layer1, self.weights2))

if __name__ == "__main__":         #trenowanie sieci
    X=[]
    y=[]
    
    for i in range(neurons):
        x1=random.uniform(-0.2,0.2)
        x2=random.uniform(-1.4,1.4)
        X.append([x1,x2]) 
        if x1>0:
            if x2<-1:
                y.append([0])
            else:
                y.append([1])
        else:
            if x2>1:
                y.append([1])
            else:
                y.append([0])
    X=np.array(X)
    y=np.array(y)
    nn = NeuralNetwork(X,y)

    for i in range(2000):
        nn.feedforward()
        nn.backprop()

    print(nn.output)


def plot_durations(scores):    #wyswietlanie wykresu
    plt.figure(2)
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(scores)

  

    plt.pause(0.001)  
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

scores = []
choices = []
for each_game in range(30):         #inicjalizacja gier po nauczeniu sieci
    score = 0
    game_memory = []
    prev_obs = []
    actions = []
    env.reset()
    for t in range(goal_steps):
        env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
       
            observ=[]
            observ.append(prev_obs[2])
            observ.append(prev_obs[3])
            actions = nn.prediction(observ)
            action=round(nn.predict[0])
            action=int(action)
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done:
          
            plot_durations(scores)
            break

    scores.append(score)
plt.ioff()
plt.show()