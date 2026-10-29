"""
train_split.py   (gsplat container, run from gsplat/examples)
Runs gsplat's simple_trainer unchanged, except that the train/val split comes from
<data_dir>/split.json ({"train": [...], "test": [...]}) instead of every-8th-image.
Usage: python /workspace/statue3dgs/train_split.py default --data_dir ... (same args as simple_trainer.py)
"""
import json, os, runpy, sys
import numpy as np

sys.path.insert(0, os.getcwd())
import datasets.colmap as dc

_orig_init = dc.Dataset.__init__


def _init(self, parser, split="train", *a, **k):
    _orig_init(self, parser, split, *a, **k)
    names = json.load(open(os.path.join(parser.data_dir, "split.json")))["train" if split == "train" else "test"]
    self.indices = np.array([i for i, n in enumerate(parser.image_names) if n in set(names)])
    assert len(self.indices) == len(names), (split, len(self.indices), len(names))
    print(f"[train_split] {split}: {len(self.indices)} images")


dc.Dataset.__init__ = _init
sys.argv[0] = "simple_trainer.py"
runpy.run_path("simple_trainer.py", run_name="__main__")
