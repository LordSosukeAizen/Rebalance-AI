from agent import agent
import matplotlib.pyplot as plt


class BackTestAgent():
    def __init__(self, agent):
        self.agent = agent
        self.initial_step = self.agent.env.max_steps
        
    def test(self, num_steps=200):
        state = self.agent.env.reset(initial_step=self.agent.env.max_steps)
        self.agent.env.epsilon = 0  # Disable exploration
        self.portfolio_history = [self.agent.env.portfolio_value]

        for _ in range(num_steps):
            action = self.agent.act(state)
            state, _, done, _, _ = self.agent.env.step(action)
            self.portfolio_history.append(self.agent.env.portfolio_value)
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

        
backtestagent = BackTestAgent(agent)
backtestagent.test()