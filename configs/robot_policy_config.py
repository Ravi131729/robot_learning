"""Default architecture configuration for the robot policy prototype."""

# Vision
NUM_VISION_TOKENS = 197
DINO_DIM = 768
NUM_POOL_QUERIES = 12
NUM_POOL_HEADS = 8
POOL_MLP_DIM = 4 * DINO_DIM
NUM_CAMERAS = 3

# Diffusion Transformer
DIT_DIM = 1536
DIT_HEADS = 24
DIT_DEPTH = 32
DIT_MLP_DIM = 4 * DIT_DIM

# Inputs and outputs
STATE_DIM = 14
TASK_DIM = 512
ACTION_DIM = 14
ACTION_HORIZON = 30
TIME_EMBED_DIM = 256
