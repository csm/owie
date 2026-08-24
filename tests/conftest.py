import torch
from hypothesis import settings

# Deterministic by default. Phase 0-2 run batch one with no sampling; tests
# follow the same discipline so a failure is always reproducible.
torch.manual_seed(0)

settings.register_profile("default", deadline=None, max_examples=50)
settings.load_profile("default")
