import os
from dotenv import load_dotenv, set_key

# Config
load_dotenv()
batch_size=int(os.getenv('BATCH_SIZE'))
lr=float(os.getenv('LR'))
hidden_feats=int(os.getenv('HIDDEN_FEATS'))
epoch_count=int(os.getenv('EPOCH_COUNT'))
save_per_epoch=int(os.getenv('SAVE_PER_EPOCH'))
num_layers=int(os.getenv('NUM_LAYERS'))