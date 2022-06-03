import gym
import pygame

env = gym.make('CartPole-v1')
observation = env.reset()

print(observation)
print(env.action_space)

done = False
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    print(env.action_space.sample())

    env.render()

input('Enter: ')
env.close()