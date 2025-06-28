# Rebalance-AI

Rebalance AI is a reinforcement learning (RL) driven portfolio management system designed to simulate and learn optimal rebalancing strategies in a stock portfolio using key technical indicators such as:

- **SMA** (Simple Moving Average)
- **EMA** (Exponential Moving Average)
- **RSI** (Relative Strength Index)
- **10-Day Average Volume**

---

## 🧠 Objective

The core goal is to **train an intelligent agent** that learns to manage and rebalance a stock portfolio over time. The agent interacts with a custom `PortfolioEnv` environment built with `OpenAI Gym` and learns optimal buy/sell/hold strategies through experience.

---

## 📊 Technical Indicators Used

| Indicator         | Description |
|------------------|-------------|
| **SMA**          | Measures the average price over a fixed number of past days to smoothen price trends. |
| **EMA**          | A more responsive average compared to SMA, giving more weight to recent prices. |
| **RSI**          | Indicates overbought or oversold conditions by measuring the magnitude of recent price changes. |
| **10-Day Volume**| Average volume over 10 days, helpful for assessing liquidity and market activity. |

---
