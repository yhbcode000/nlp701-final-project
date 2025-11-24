Minecraft Dataset Layout
========================

This directory stores the MineDojo-style Minecraft trajectories that we use
for frame prediction and action recognition experiments. The dataset is
organized as a simple two-level hierarchy (dataset → episode) to make it easy
to drop in new creative scenes:

```
datasets/minecraft/
├── data/
│   ├── creative:0/
│   │   ├── 000000.npy
│   │   ├── 000001.npy
│   │   └── ...
│   ├── creative:1/
│   └── ...
├── preprocess_data.py
├── filter_data.py
├── show_data.py
├── read_data.py
└── README.md
```

- `data/` — Root directory that holds all creative-mode episodes.
- `creative:{episode_id}/` — A single creative-mode episode (index comes from
  the original MineDojo dump). Each episode directory contains one `.npy` file
  per time step.
- `000123.npy` — Zero-padded frame index within the episode. Files are ordered
  lexicographically so iterating over them yields the temporal sequence of the
  episode.

### Frame file contents

Every `.npy` stores a Python dictionary with two fields:

- `action`: `np.ndarray` with shape `(3,)`. The three integers encode the agent
  control that was applied to reach the next frame:
  - `action[0]` (“straight”): `0` noop, `1` forward, `2` backward.
  - `action[1]` (“pan”): `0` noop, `1` left, `2` right.
  - `action[2]` (“jump”): `0` noop, `1` jump.
- `voxel`: `np.ndarray` with shape `(5, 5, 5)` containing the block name at
  each `(x, y, z)` coordinate around the agent. Values are plain strings like
  `"air"`, `"stone"`, etc. The array uses MineDojo’s convention (origin at the
  lower south-west corner, iterating x→y→z as in `read_data.voxel2word`).

When consumed through `minecraft_dataset.MinecraftDataset`, consecutive frames
`t` and `t+1` are paired so that `(voxel_t, action_t)` predicts `voxel_{t+1}`,
or (for action recognition) `(voxel_t, voxel_{t+1})` predicts `action_t`.

### Adding new trajectories

1. Create a new `creative:{episode_id}` folder inside `data/`.
2. Dump each step as a `dict(action=<np.ndarray>, voxel=<np.ndarray>)` and save
   with zero-padded filenames that preserve chronological order.

Keeping these conventions ensures the existing loaders continue to work
without modification.

### Utility scripts

- `preprocess_data.py` — Relabel actions by inferring per-axis motion between
  consecutive frames. The script independently scores shifts along x
  (forward/back), z (left/right), and y (jump), then writes the corresponding
  `[straight, pan, jump]` vector back to each frame. Run:
  `python3 preprocess_data.py --data-dir data`.
- `filter_data.py` — Collapse consecutive `[0, 0, 0]` actions within each
  episode, keeping only the first occurrence and reindexing the remaining
  frames (filenames reset to `000000.npy`, …). Run:
  `python3 filter_data.py --data-dir data`.
- `show_data.py` — Visualize frames in 3D (squares on the X/Z ground plane,
  Y vertical) and include the action for each frame. With `--output` you can
  save static PNGs or animated GIFs (for a creative directory) without an
  interactive backend:
  - Single frame: `python3 show_data.py --path data/creative:0/000000.npy --dim 5 --output frame.png`
  - Animation: `python3 show_data.py --path data/creative:0 --dim 5 --output episode.gif`

### Loader expectations

- The training code walks `data/` recursively and loads each `creative:*`
  directory separately, preserving the continuity of every episode. Frame names
  therefore must stay lexicographically sorted (`000000.npy`, `000001.npy`, …)
  with no gaps if you want the entire rollout to be used.
- Episode boundaries are respected: no frame pair crosses from one
  `creative:*` directory into the next. If you intentionally split an episode,
  place each chunk in its own directory.
- If you receive data in a flat structure, reorganize it so that every episode
  sits inside a `creative:{episode_id}` directory under `data/`. A simple
  script can copy files into this layout as long as it preserves the
  chronological order of frames.
- `MinecraftDataset` lets you cap how many creative scenes load via
  `max_creative_scenes` (default 10) and will spawn `max(1, cpu_count // 2)`
  workers per `DataLoader`. When CUDA is available, batches are automatically
  moved onto the GPU-ready device in the collate step.
