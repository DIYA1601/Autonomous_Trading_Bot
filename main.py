import pandas as pd
from environment.trading_env import TradingEnv
from agent.dqn_agent import DQN
import torch

# Load historical stock data (CSV with columns: Open, High, Low, Close)
data = pd.read_csv("historical_stock_data.csv")

env = TradingEnv(data)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQN(state_size, action_size)

state = env.reset()
done = False
while not done:
    action = agent(torch.tensor(state, dtype=torch.float32)).argmax().item()
    next_state, reward, done, _ = env.step(action)
    state = next_state

print("Final Balance:", env.balance)
