from ray.rllib.agents.ppo.ppo import PPOTrainer
from gym.wrappers import RecordVideo
import gym


agent = PPOTrainer(config={"env": "CartPole-v1",
                           "evaluation_interval": 2,
                           "evaluation_num_episodes": 20,
                           }
                   )

agent.restore(r"C:\Users\ykudj\ray_results\PPO\PPO_CartPole-v1_26cff_00000_0_2022-05-28_21-40-48\checkpoint_000032\checkpoint-32")


env = RecordVideo(gym.make("CartPole-v1"), "ppo_agent_video")
obs = env.reset()

while True:
    action = agent.compute_action(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break

env.close()