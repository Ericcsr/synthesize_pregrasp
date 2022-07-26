import wandb
import time
import numpy as np
wandb.init(project="my-test-project")

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}

for i in range(1000):
    wandb.log({"loss": np.random.random(1)})
    time.sleep(0.05)

# Optional

