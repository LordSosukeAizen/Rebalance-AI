
import gym
import numpy as np
import pandas as pd
import yfinance as yf
import time


WINDOW_VOLUME = 10
INDICATORS = ['Close', 'SMA', 'EMA', f'{WINDOW_VOLUME}_DAY_VOLUME']


def add_indicators(data, window_sma=10, window_ema=10, window_volume=WINDOW_VOLUME):
  
    
    data['SMA'] = data.groupby(level=1)['Close'].rolling(window=window_sma).mean().values
    data['EMA'] = data.groupby(level=1)['Close'].ewm(span=window_ema, adjust=False).mean().values
    data[f'{window_volume}_DAY_VOLUME'] = data.groupby(level=1)['Volume'].rolling(window=window_volume).mean().values
  
    return data


class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=1e6, max_steps=300, num_stocks=10, num_features=4):
        super(PortfolioEnv, self).__init__()
        self.df = df  
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.num_stocks = num_stocks
        self.num_features = num_features 

        self.action_space = gym.spaces.MultiDiscrete([3] * self.num_stocks)  # 0: hold, 1: buy, 2: sell

        # [balance] + [stock prices] + [technical features] + [stock holdings]
        self.obs_len = 1 + num_stocks * (num_features) + num_stocks
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_len,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.shares_held = [0] * self.num_stocks
        self.current_step = WINDOW_VOLUME
        return self.curr_obs()

    def curr_obs(self):
        
        row = self.df.iloc[self.current_step: self.current_step+self.num_stocks, :]
        
        obs = [self.balance]
        obs.extend(self.shares_held)
        for indicator in INDICATORS:
            obs.extend(row[indicator].values)
    
        return np.array(obs, dtype=np.float32)
    
    

    def step(self, actions):
        
        
        prev_value = self.portfolio_value
        row = self.df.iloc[self.current_step: self.current_step+self.num_stocks, :]
        prices = row['Close'].values

        # Execute actions
        for stock_index, action in enumerate(actions):
            price = prices[stock_index]
            if action == 1 and self.balance >= price:
                self.balance -= price
                self.shares_held[stock_index] += 1
            elif action == 2 and self.shares_held[stock_index] > 0:
                self.balance += price
                self.shares_held[stock_index] -= 1

        self.current_step += self.num_stocks
        done = (self.current_step >= self.max_steps)

        # Update portfolio value
        next_row = self.df.iloc[self.current_step: self.current_step+self.num_stocks, :]
        new_prices = next_row['Close'].values

        self.portfolio_value = self.balance + sum([sh * p for sh, p in zip(self.shares_held, new_prices)])

        reward = self.portfolio_value - prev_value
        obs = self.curr_obs()
        return obs, reward, done, False, {}


    def render(self, mode='human'):
        print(f"Step: {self.current_step/10 - 1}, Balance: {self.balance:.2f}, Shares: {self.shares_held}, Portfolio Value: {self.portfolio_value:.2f}")

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

data = yf.download(stock_tickers, start='2020-01-01', end='2024-01-01', group_by='ticker', auto_adjust=True)

data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index()
data.set_index(['Date', 'Ticker'], inplace=True)

data = add_indicators(data)

portfolio_env = PortfolioEnv(data)


print("\n\n")
NUMBER_OF_SIMULATIONS = 10
for _ in range(NUMBER_OF_SIMULATIONS):
    random_actions = np.random.randint(0, 3, size=10)
    _, reward, _, _, _ = portfolio_env.step(random_actions)
    print("Reward: ", reward)
    portfolio_env.render()
    print("\n\n\n\n")
    time.sleep(3)
