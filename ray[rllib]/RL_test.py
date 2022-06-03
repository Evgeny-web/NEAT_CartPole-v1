import gym
import numpy as np

env = gym.make("CartPole-v1")


for ep in range(5):
    print(f"Episode is {ep+1}")
    obs = env.reset()
    s = 0
    while True:
        obs, reward, done, _ = env.step(0)
        s += 1
        print(f'step {s} Pole angle {np.degrees(obs[2])}', end=" ")
        print(f'Reward is {reward}, done: {done}')
        env.render()
        if done:
            break

input('Enter: ')
env.close()
