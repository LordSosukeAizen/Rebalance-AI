from collections import deque
import numpy as np
from tqdm import tqdm
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from make_env import portfolio_env
import matplotlib.pyplot as plt
import time
import random


opt = Adam(learning_rate=0.01)


class Agent():
    def __init__(self, env, num_stocks, num_actions=3, batch_size=32, num_features=4):
        self.env = env
        self.epsilon = 1
        self.min_epsilon = 0.3
        self.decay = 0.8
        self.gamma = 0.6
        self.memory = deque(maxlen=2000)
        self.treward = 0
        self.num_features = num_features
        self.num_stocks = num_stocks
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.max_reward = 0
        self.state_dim =  self.env.observation_space.shape[0]
        self.model = self._build_model()
        
        
        
    
    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.state_dim,)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.num_stocks * self.num_actions, activation='linear'))
        model.compile(optimizer=opt, loss='mse')
        return model
    
    
    def act(self, state):
        
        # Exploration
        if np.random.rand() < self.epsilon:
            random_actions = np.random.randint(0, 3, size=self.num_stocks)
            return random_actions
        
        # Exploitation
        actions = []
    
        # Add batch dimension if missing
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        
        predicted_policies = self.model.predict(state, verbose=0)[0]  # shape: [num_stocks * num_actions]
        
        for i in range(0, len(predicted_policies), self.num_actions):
            policies_per_stock = predicted_policies[i:i + self.num_actions]
            best_action = np.argmax(policies_per_stock)
            actions.append(best_action)
        
        return actions
        
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return  # not enough samples

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0] for sample in minibatch])
        next_states = np.array([sample[1] for sample in minibatch])
        
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        for idx, (_, _, action, reward, done) in enumerate(minibatch):
            for stock_idx in range(self.num_stocks):
                act = action[stock_idx]
                target = reward
                if not done:
                    max_next_q = np.max(next_q_values[idx][stock_idx * self.num_actions:(stock_idx + 1) * self.num_actions])
                    target = reward + self.gamma * max_next_q
                
                # Update Q-value
                q_values[idx][stock_idx * self.num_actions + act] = target

        self.model.fit(states, q_values, epochs=1, verbose=0)
        if self.epsilon > self.min_epsilon :
            self.epsilon *= self.decay
            self.epsilon = max(self.epsilon, self.min_epsilon)

    
    def learn(self, num_episodes=10, iterations_per_episode=100):
        
        for _ in tqdm(range(num_episodes)):
            state = self.env.reset()
            self.treward = 0
            for _ in (range(iterations_per_episode)):
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.treward += reward
                self.memory.append((state, next_state, action, reward, done))
                if done:
                    self.max_reward = max(self.max_reward, self.treward)
                    log = f'Total reward: {self.treward} | Max Reward: {self.max_reward}'
                    print(log)
                    break
                
                state = next_state
                
            if len(self.memory) > 1000:
                self.replay()



    def test(self, num_steps=200):
        state = self.env.reset()
        self.env.epsilon = 0  # Disable exploration
        self.portfolio_history = [self.env.portfolio_value]

        for _ in range(num_steps):
            action = self.act(state)
            state, _, done, _, _ = self.env.step(action)
            self.portfolio_history.append(self.env.portfolio_value)
            if done:
                break

        # Plot the balance/portfolio value
        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio_history, label='Portfolio Value')
        plt.xlabel("Step")
        plt.ylabel("Portfolio Value")
        plt.title("Agent Portfolio Performance Over Time")
        plt.legend()
        plt.grid()
        plt.show()

    
stock_tickers = [
    'AAPL',
    'MSFT',
    'GOOGL',
    'AMZN',
    'NVDA',
    'META',
    'TSLA',
    'JPM',
    'JNJ',
    'BRK-B'
]



agent = Agent(portfolio_env, num_stocks=len(stock_tickers))
agent.learn(num_episodes=100, iterations_per_episode=200)

print("\n\nTesting is started wait for 10sec.....\n\n\n")
time.sleep(10)
agent.test()