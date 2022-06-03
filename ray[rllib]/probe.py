import ray
import redis

from ray import tune


ray.init()

tune.run("DQN",
         config={"env": "CartPole-v1",
                 "evaluation_interval": 2,
                 "evaluation_num_episodes": 20,
                 },
         # local_dir="<namefile>"
         checkpoint_freq=2,
         )
