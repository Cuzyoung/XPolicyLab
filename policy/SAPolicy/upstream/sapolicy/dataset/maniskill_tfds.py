#!/usr/bin/env python3
"""
Efficient Open X-Embodiment Dataset using TFDS with tf.data pipeline optimizations
Following the pattern from tfds_example.py for optimal performance
"""

import os
import sys
import numpy as np
import pickle
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import math
import torch
from torch.utils.data import IterableDataset
import torch.distributed as dist

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sapolicy.logger import Log

def collate_fn(batch):
    # batch is a list of length 1
    obs = {k: torch.as_tensor(batch[0]["observation"][k]) for k in batch[0]["observation"]}
    actions = torch.as_tensor(batch[0]["action"])
    prompt = batch[0]["prompt"]
    episode_id = batch[0]["episode_id"]
    step_id = batch[0]["step_id"]
    return {"observation": obs, "action": actions, "prompt": prompt, "episode_id": episode_id, "step_id": step_id}

class ManiSkillTFDSDataset(IterableDataset):
    """
    Efficient dataset using TFDS with tf.data pipeline optimizations.
    Uses IterableDataset to stream data efficiently without loading all into memory.
    """

    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        split: str = 'train',
        dataset_names: Optional[List[str]] = None,
        max_episodes_per_dataset: Optional[int] = None,
        use_train_val_split: bool = False,
        train_val_split_ratio: float = 0.9,
        split_seed: int = 42,
        action_dim: int = 7,
        observation_keys: Optional[List[str]] = None,
        action_sequence_length: int = 4,
        normalize_actions: bool = True,
        use_task_description: bool = True,
        tcp_preprocessed_dir: Optional[str] = None,
        transforms: Optional[List] = None,
        min_depth: float = 0.1,
        max_depth: float = 5,
        cache_data: bool = False,  # Whether to cache the dataset in memory
        shuffle_buffer_size=5000,
        batch_size: int = 1,
        action_space: str = 'joint_position', # Not used for now
        filter_dict_path: Optional[str] = None,
        resize_params: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize the efficient TFDS dataset with tf.data pipeline"""
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.split = split
        self.dataset_names = dataset_names or ['maniskill_dataset_converted_externally_to_rlds']
        self.max_episodes_per_dataset = max_episodes_per_dataset
        self.use_train_val_split = use_train_val_split
        self.train_val_split_ratio = train_val_split_ratio
        self.split_seed = split_seed
        self.action_dim = int(action_dim)
        self.action_sequence_length = int(action_sequence_length)
        self.observation_keys = observation_keys or ['image', 'tcp_pixel_coords']
        self.normalize_actions = normalize_actions
        self.use_task_description = use_task_description
        self.tcp_preprocessed_dir = Path(tcp_preprocessed_dir) if tcp_preprocessed_dir else None
        self.transforms = transforms or []
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.cache_data = cache_data
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = int(batch_size)
        self.filter_dict_path = filter_dict_path
        self.resize_params = resize_params
        self.cur_img_size = {}

        if dist.is_initialized():
            self.batch_size = self.batch_size // dist.get_world_size()

        print(f"batch_size: {self.batch_size}")

        # Load TCP preprocessed data mappings
        self._load_tcp_mappings()
        
        # Build the tf.data pipeline
        self._build_tf_data_pipeline()

        Log.info(f"Initialized {self.dataset_name} with tf.data pipeline")

    def _build_tf_data_pipeline(self):
        """Build efficient tf.data pipeline following tfds_example.py pattern"""
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import tensorflow as tf
        import tensorflow_datasets as tfds
        # Suppress TensorFlow warnings and use CPU only to avoid conflicts with PyTorch
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # Configure TensorFlow to use CPU only
        tf.config.set_visible_devices([], 'GPU')

        AUTOTUNE = tf.data.AUTOTUNE

        # Collect all datasets
        all_datasets = []

        # 检查是否在 DDP 模式
        world_size, rank = 1, 0
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            Log.info(f"Building dataset with shard: world_size={world_size}, rank={rank}")

        for dataset_name in self.dataset_names:
            try:
                # Load the dataset
                ds_builder = tfds.builder(dataset_name, data_dir=self.data_root)

                # Get appropriate split
                split_name = 'train'
                if split_name not in ds_builder.info.splits:
                    split_name = list(ds_builder.info.splits.keys())[0]

                # Load dataset
                builder = tfds.builder(dataset_name, data_dir=self.data_root)

                # Use proper split configuration
                split_name = 'train'
                if self.max_episodes_per_dataset:
                    split_name = f'train[:{self.max_episodes_per_dataset}]'
                elif self.use_train_val_split:
                    train_pct = int(self.train_val_split_ratio * 100)
                    if self.split == 'train':
                        split_name = f'train[:{train_pct}%]'
                    else:
                        split_name = f'train[{train_pct}%:]'
                
                Log.info(f"Using split: {split_name}")
                    
                ds = builder.as_dataset(
                    split=split_name,
                    shuffle_files=False,
                    as_supervised=False,
                    decoders={"steps": tfds.decode.SkipDecoding()},
                    read_config=tfds.ReadConfig(
                        skip_prefetch=True,
                        num_parallel_calls_for_interleave_files=1,
                        interleave_cycle_length=1,
                    ),
                )

                tcp_dataset = None
                if self.tcp_preprocessed_dir:
                    tfrecord_path = self.tcp_preprocessed_dir / f"{dataset_name}" / "train_100pct.tfrecord"
                    if not tfrecord_path.exists():
                        Log.warn(f"TCP TFRecord not found: {tfrecord_path}")
                        continue

                    # Parse TFRecords and build lookup dictionary
                    tcp_dataset = tf.data.TFRecordDataset(str(tfrecord_path), num_parallel_reads=1)
                    feature_description = {
                        'episode_id': tf.io.FixedLenFeature([], tf.string),
                        'dataset_name': tf.io.FixedLenFeature([], tf.string),
                        'num_steps': tf.io.FixedLenFeature([], tf.int64),

                        # 每个 step 都是长度为 3 的 float 向量
                        'tcp_pixel_coords': tf.io.FixedLenSequenceFeature([3], tf.float32, allow_missing=True),
                        'tcp_pos': tf.io.FixedLenSequenceFeature([3], tf.float32, allow_missing=True),
                        'tcp_dir_x': tf.io.FixedLenSequenceFeature([2], tf.float32, allow_missing=True),
                        'tcp_dir_y': tf.io.FixedLenSequenceFeature([2], tf.float32, allow_missing=True),
                        'tcp_dir_z': tf.io.FixedLenSequenceFeature([2], tf.float32, allow_missing=True),
                        'tcp_orn': tf.io.FixedLenSequenceFeature([9], tf.float32, allow_missing=True),
                        'camera_intrinsics': tf.io.FixedLenSequenceFeature([9], tf.float32, allow_missing=True),
                        'camera_extrinsics': tf.io.FixedLenSequenceFeature([16], tf.float32, allow_missing=True),

                        # step_indices 还是一维 int 序列
                        'step_indices': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    }

                    def _parse_fn(traj):
                        parsed = tf.io.parse_single_example(traj, feature_description)

                        # num_steps = tf.cast(parsed['num_steps'], tf.int32)

                        # 注意：这里 parse 出来的已经是 [num_steps, 3] 的 dense tensor
                        # 不需要 sparse.to_dense() 也不需要 reshape 了
                        return {
                            'episode_id': parsed['episode_id'],
                            'tcp_pixel_coords': parsed['tcp_pixel_coords'],  # shape: [num_steps, 3]
                            'tcp_pos': parsed['tcp_pos'],                    # shape: [num_steps, 3]
                            'tcp_dir_x': parsed['tcp_dir_x'],                # shape: [num_steps, 2]
                            'tcp_dir_y': parsed['tcp_dir_y'],                # shape: [num_steps, 2]
                            'tcp_dir_z': parsed['tcp_dir_z'],                # shape: [num_steps, 2]
                            'tcp_orn': parsed['tcp_orn'],                    # shape: [num_steps, 9]
                            'camera_intrinsics': parsed['camera_intrinsics'],  # shape: [num_steps, 9]
                            'camera_extrinsics': parsed['camera_extrinsics'],  # shape: [num_steps, 16]
                            # 'step_indices': parsed['step_indices'],          # shape: [num_steps]
                        }

                    tcp_dataset = tcp_dataset.map(_parse_fn)
                    if self.use_train_val_split:
                        selected_episode_ids = [traj["episode_metadata"]["episode_id"].numpy().decode("utf-8") 
                            if isinstance(traj["episode_metadata"]["episode_id"], bytes) else traj["episode_metadata"]["episode_id"].numpy()
                                for traj in ds
                        ]
                        selected_ids_set = set(selected_episode_ids)
                        def filter_tcp(episode_id):
                            return episode_id.numpy() in selected_ids_set
                        tcp_dataset = tcp_dataset.filter(lambda traj: tf.py_function(filter_tcp, [traj["episode_id"]], Tout=tf.bool))
                    # filter out tcp_pixel_coords[0] < 0 or tcp_pixel_coords[1] < 0:

                ## Filter out any unsuccessful trajectories -- we use the file name to check this
                # ds = ds.filter(
                #     lambda traj: tf.strings.regex_full_match(
                #         traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
                #     )
                # )
                # Load the filter dictionary if provided.
                # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
                # (e.g.,
                # {
                #     "<episode key>": [[0, 100], [200, 300]]
                # }
                # means keep frames 0-99 and 200-299).
                # if self.filter_dict_path is not None:
                #     cached_filter_dict_path = download.maybe_download(self.filter_dict_path)
                #     with Path(cached_filter_dict_path).open("r") as f:
                #         filter_dict = json.load(f)

                #     logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

                #     keys_tensor = []
                #     values_tensor = []

                #     for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                #         for start, end in ranges:
                #             for t in range(start, end):
                #                 frame_key = f"{episode_key}--{t}"
                #                 keys_tensor.append(frame_key)
                #                 values_tensor.append(True)
                #     self.filter_table = tf.lookup.StaticHashTable(
                #         tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
                #     )
                #     logging.info("Filter hash table initialized")
                # else:
                #     self.filter_table = tf.lookup.StaticHashTable(
                #         tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
                #     )

                # Repeat dataset so we never run out of data.
                # ds = ds.repeat()

                first_elem = next(iter(ds))
                self.cur_img_size[dataset_name] = tf.shape(tf.io.decode_image(first_elem["steps"]["observation"]["image"][0], expand_animations=False, dtype=tf.uint8))[:2]

                def restructure(traj):
                    """Reformat observation and action keys, sample language instruction."""
                    actions = traj["steps"]["action"]
                    # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
                    # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
                    exterior_img = traj["steps"]["observation"]["image"]
                    depth = traj["steps"]["observation"]["depth"]
                    # Randomly sample one of the three language instructions
                    instruction = traj["steps"]["language_instruction"]

                    # traj_len = tf.shape(actions)[0]
                    # indices = tf.as_string(tf.range(traj_len))

                    # Data filtering:
                    # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
                    # and each step's time step index. This will index into the filter hash table, and if it returns true,
                    # then the frame passes the filter.
                    # step_id = (
                    #     traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                    #     + "--"
                    #     + traj["traj_metadata"]["episode_metadata"]["file_path"]
                    #     + "--"
                    #     + indices
                    # )
                    # passes_filter = self.filter_table.lookup(step_id)

                    return {
                        "action": actions,
                        "observation": {
                            "image": exterior_img,
                            "depth": depth,
                        },
                        "prompt": instruction,
                        # "step_id": step_id,
                        # "passes_filter": passes_filter,
                        "episode_id": traj["episode_metadata"]["episode_id"],
                    }

                # ds = ds.traj_map(restructure, AUTOTUNE)
                ds = ds.map(restructure, AUTOTUNE) # Traj-level map

                if tcp_dataset is not None:
                    ds = tf.data.Dataset.zip((ds, tcp_dataset))

                # def make_kv_dataset(ds, source_name):
                #     return ds.map(lambda x: (x['episode_id'], (source_name, x)))

                # ds_a_kv = make_kv_dataset(ds, "a")
                # ds_b_kv = make_kv_dataset(tcp_dataset, "b")

                # # 合并
                # merged = ds_a_kv.concatenate(ds_b_kv)

                # # 按 key 分组，每个 episode_id 对应一个小batch
                # def key_func(key, val):
                #     return key

                # def reduce_func(key, dataset):
                #     return dataset.batch(2)  # 因为每个id在两个数据集中各出现一次

                # grouped = merged.group_by_window(
                #     key_func=key_func,
                #     reduce_func=reduce_func,
                #     window_size=2
                # )

                # # 最终拼接
                # def merge_sources(batch):
                #     # batch 是 [(“a”, example_a), (“b”, example_b)]
                #     ex_a = batch[0][1] if batch[0][0] == b"a" else batch[1][1]
                #     ex_b = batch[1][1] if batch[1][0] == b"b" else batch[0][1]
                #     return ex_a, ex_b

                # ds = grouped.map(merge_sources)

                def merge_tcp_data(traj, tcp_data):
                    """Merge TCP data into the trajectory."""
                    # Use tf.debugging.assert_equal for tensor comparison
                    tf.debugging.assert_equal(
                        traj["episode_id"],
                        tcp_data["episode_id"],
                        message="Episode ID mismatch"
                    )

                    traj["observation"]["tcp_pixel_coords"] = tcp_data["tcp_pixel_coords"]
                    traj["observation"]["tcp_pos"] = tcp_data["tcp_pos"]
                    traj["observation"]["tcp_dir_x"] = tcp_data["tcp_dir_x"]
                    traj["observation"]["tcp_dir_y"] = tcp_data["tcp_dir_y"]
                    traj["observation"]["tcp_dir_z"] = tcp_data["tcp_dir_z"]
                    traj["observation"]["tcp_orn"] = tcp_data["tcp_orn"]
                    traj["observation"]["camera_intrinsics"] = tcp_data["camera_intrinsics"]
                    traj["observation"]["camera_extrinsics"] = tcp_data["camera_extrinsics"]

                    return {
                        "observation": traj["observation"],
                        "prompt": traj["prompt"],
                        "action": traj["action"],
                        "episode_id": tf.repeat(traj["episode_id"], repeats=tf.shape(traj["action"])[0]),
                    }

                if tcp_dataset is not None:
                    ds = ds.map(merge_tcp_data, AUTOTUNE)
                    
                def chunk_actions(traj):
                    """Splits episode into action chunks."""
                    traj_len = tf.shape(traj["action"])[0]

                    # For each step in the trajectory, construct indices for the next n actions
                    action_chunk_indices = tf.broadcast_to(
                        tf.range(self.action_sequence_length)[None],
                        [traj_len, self.action_sequence_length],
                    ) + tf.broadcast_to(
                        tf.range(traj_len)[:, None],
                        [traj_len, self.action_sequence_length],
                    )

                    # Cap to length of the sequence --> final chunks will repeat the last action
                    # This makes sense, since we are using absolute joint + gripper position actions
                    action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

                    # Gather the actions for each chunk
                    traj["action"] = tf.gather(traj["action"], action_chunk_indices)
                    traj["step_id"] = tf.range(traj_len)

                    return traj

                # ds = ds.traj_map(chunk_actions, num_parallel_calls=AUTOTUNE)
                ds = ds.map(chunk_actions, num_parallel_calls=AUTOTUNE) # Traj-level map

                ds = ds.interleave(
                    lambda traj: tf.data.Dataset.from_tensor_slices(traj),
                    cycle_length=AUTOTUNE,
                    num_parallel_calls=AUTOTUNE,
                )

                def _preprocess_step(step):
                    """Preprocess a single step (to be executed in tf.data pipeline)"""
                    observation = step['observation']

                    def constrain_to_multiple_of(x, multiple_of, min_val=None, max_val=None):
                        """Constrain a value to be a multiple of `multiple_of` in TF graph."""
                        # 四舍五入到 nearest multiple
                        y = tf.round(x / multiple_of) * multiple_of

                        # clip to max_val if specified
                        if max_val is not None:
                            y = tf.where(y > max_val, tf.math.floor(x / multiple_of) * multiple_of, y)
                        # clip to min_val if specified
                        if min_val is not None:
                            y = tf.where(y < min_val, tf.math.ceil(x / multiple_of) * multiple_of, y)

                        return tf.cast(y, tf.int32)

                    def get_size(width, height, resize_params):
                        """
                        width, height: tf.int32 or tf.float32
                        resize_params: dict with keys ['height','width','keep_aspect_ratio','resize_method']
                        """
                        __height = tf.cast(resize_params['height'], tf.float32)
                        __width = tf.cast(resize_params['width'], tf.float32)
                        __keep_aspect_ratio = resize_params['keep_aspect_ratio']
                        __resize_method = resize_params['resize_method']
                        __multiple_of = resize_params['ensure_multiple_of']

                        width = tf.cast(width, tf.float32)
                        height = tf.cast(height, tf.float32)

                        scale_height = __height / height
                        scale_width = __width / width

                        if __keep_aspect_ratio:
                            scale = tf.constant(1.0, dtype=tf.float32)
                            if __resize_method == "lower_bound":
                                scale = tf.maximum(scale_width, scale_height)
                            elif __resize_method == "upper_bound":
                                scale = tf.minimum(scale_width, scale_height)
                            elif __resize_method == "minimal":
                                scale = tf.cond(
                                    tf.abs(1.0 - scale_width) < tf.abs(1.0 - scale_height),
                                    lambda: scale_width,
                                    lambda: scale_height
                                )
                            else:
                                raise ValueError(f"resize_method {__resize_method} not implemented")
                            scale_width = scale
                            scale_height = scale

                        # Apply constraint to multiple_of
                        if __resize_method == "lower_bound":
                            new_height = constrain_to_multiple_of(scale_height * height, __multiple_of, min_val=__height)
                            new_width = constrain_to_multiple_of(scale_width * width, __multiple_of, min_val=__width)
                        elif __resize_method == "upper_bound":
                            new_height = constrain_to_multiple_of(scale_height * height, __multiple_of, max_val=__height)
                            new_width = constrain_to_multiple_of(scale_width * width, __multiple_of, max_val=__width)
                        elif __resize_method == "minimal":
                            new_height = constrain_to_multiple_of(scale_height * height, __multiple_of)
                            new_width = constrain_to_multiple_of(scale_width * width, __multiple_of)
                        else:
                            raise ValueError(f"resize_method {__resize_method} not implemented")

                        return new_width, new_height

                    def resize_image(image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
                        """Resizes an image using Lanczos3 interpolation. Expects & returns uint8."""
                        assert image.dtype == tf.uint8
                        image = tf.image.resize(image, size, method="lanczos3", antialias=True)
                        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
                        return image

                    def resize_depth_image(depth_image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
                        """Resizes a depth image using bilinear interpolation. Expects & returns float32 in arbitrary range."""
                        assert depth_image.dtype == tf.float32
                        if len(depth_image.shape) < 3:
                            depth_image = tf.image.resize(
                                depth_image[..., None], size, method="nearest", antialias=True
                            )[..., 0]
                        else:
                            depth_image = tf.image.resize(
                                depth_image, size, method="nearest", antialias=True
                            )
                        return depth_image

                    resize_size = get_size(self.cur_img_size[dataset_name][1], self.cur_img_size[dataset_name][0], self.resize_params)
                    # Process image
                    if 'image' in observation:
                        image = observation['image']
                        if image.dtype == tf.string:
                            if tf.strings.length(image) == 0:
                                # this is a padding image
                                image = tf.zeros((*resize_size, 3), dtype=tf.uint8)
                            else:
                                image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
                        elif image.dtype != tf.uint8:
                            raise ValueError(f"Unsupported image dtype: found image with dtype {image.dtype}")

                        # Normalize tcp_pixel_coords
                        if 'tcp_pixel_coords' in observation:
                            # Normalize coordinates to [0, 1] range
                            u_norm = observation['tcp_pixel_coords'][0] / tf.cast(self.cur_img_size[dataset_name][1]-1, tf.float32)
                            v_norm = observation['tcp_pixel_coords'][1] / tf.cast(self.cur_img_size[dataset_name][0]-1, tf.float32)
                            depth_norm = tf.clip_by_value(observation['tcp_pixel_coords'][2], self.min_depth, self.max_depth)
                            depth_norm = (depth_norm - self.min_depth) / (self.max_depth - self.min_depth)

                            # Clamp to [0, 1] to handle any edge cases
                            depth_norm = tf.clip_by_value(depth_norm, 0.0, 1.0)

                            normalized_coords = tf.stack([
                                u_norm,
                                v_norm,
                                depth_norm  # Keep depth normed
                            ])
                            observation['tcp_pixel_coords'] = normalized_coords

                        image = resize_image(image, size=resize_size)
                        image = tf.cast(image, tf.float32) / 255.0
                        # Channel first
                        observation['image'] = tf.transpose(image, (2, 0, 1))


                    # Process depth
                    if 'depth' in observation:
                        depth = observation['depth']

                        if depth.dtype == tf.string:
                            if tf.strings.length(depth) == 0:
                                depth = tf.zeros((*resize_size, 1), dtype=tf.float32)
                            else:
                                depth = tf.io.decode_image(depth, expand_animations=False, dtype=tf.float32)
                        elif depth.dtype != tf.float32:
                            raise ValueError(f"Unsupported depth dtype: found depth with dtype {depth.dtype}")
                        depth = resize_depth_image(depth, size=resize_size)
                        # Channel first
                        depth = tf.transpose(depth, (2, 0, 1))

                        depth = tf.cast(depth, tf.float32) / 2**10 # translate to meters
                        # Clip depth
                        depth = tf.clip_by_value(depth, self.min_depth, self.max_depth)
                        # depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
                        observation['depth'] = depth

                    return step
                
                # Apply preprocessing
                ds = ds.map(
                    _preprocess_step,
                    num_parallel_calls=AUTOTUNE
                )
                if tcp_dataset is not None:
                    # Filter out trajectories with invalid TCP coordinates (< 0 means out of bounds)
                    # Filter out trajectories with invalid TCP coordinates
                    def is_valid_tcp(traj):
                        tcp_u = traj["observation"]["tcp_pixel_coords"][0]
                        tcp_v = traj["observation"]["tcp_pixel_coords"][1]
                        
                        # Check bounds [0, 1]
                        in_bounds = tf.reduce_all(tf.greater_equal(tcp_u, 0.0)) and \
                                   tf.reduce_all(tf.greater_equal(tcp_v, 0.0)) and \
                                   tf.reduce_all(tf.less_equal(tcp_u, 1.0)) and \
                                   tf.reduce_all(tf.less_equal(tcp_v, 1.0))
                        
                        # Check for NaN or Inf
                        no_nan = tf.reduce_all(tf.logical_not(tf.math.is_nan(tcp_u))) and \
                                tf.reduce_all(tf.logical_not(tf.math.is_nan(tcp_v)))
                        no_inf = tf.reduce_all(tf.logical_not(tf.math.is_inf(tcp_u))) and \
                                tf.reduce_all(tf.logical_not(tf.math.is_inf(tcp_v)))
                        
                        return in_bounds and no_nan and no_inf
                    
                    ds = ds.filter(is_valid_tcp)

                all_datasets.append(ds)

                Log.info(f"Added dataset: {dataset_name}")

            except Exception as e:
                Log.error(f"Error loading dataset {dataset_name}: {e}")
                import traceback; traceback.print_exc()

        # Combine all datasets
        if len(all_datasets) == 0:
            raise ValueError("No datasets could be loaded")
        elif len(all_datasets) == 1:
            combined_ds = all_datasets[0]
        else:
            # Interleave multiple datasets for better mixing
            combined_ds = tf.data.Dataset.sample_from_datasets(
                all_datasets,
                weights=[1.0] * len(all_datasets)
            )

        # Apply caching if requested (before shuffle for efficiency)
        if self.cache_data:
            combined_ds = combined_ds.cache()

        # Only shuffle for training, not for validation (before batching)
        if self.split == 'train':
            combined_ds = combined_ds.shuffle(self.shuffle_buffer_size)

        # Batch with drop_remainder to ensure consistent batch sizes across ranks
        combined_ds = combined_ds.batch(self.batch_size, drop_remainder=True)

        # Repeat after batching to ensure infinite iteration
        combined_ds = combined_ds.repeat()

        # Shard for DDP - MUST be after repeat to avoid different data sizes per rank
        if world_size > 1:
            combined_ds = combined_ds.shard(num_shards=world_size, index=rank)
            Log.info(f"Applied shard for DDP: rank={rank}/{world_size}")

        # Prefetch for performance
        combined_ds = combined_ds.prefetch(AUTOTUNE)

        # Reduce memory usage
        # combined_ds = combined_ds.with_ram_budget(1)

        # Store the pipeline
        self.tf_dataset = combined_ds

    def _load_tcp_mappings(self):
        """Load TCP data batch mappings"""
        self.tcp_data_cache = {}

        if not self.tcp_preprocessed_dir:
            raise ValueError("TCP preprocessed directory is not set")
            return

        for dataset_name in self.dataset_names:
            dataset_dir = self.tcp_preprocessed_dir / dataset_name
            if not dataset_dir.exists():
                continue

            # Load all TCP data batches for this dataset
            for batch_file in dataset_dir.glob('tcp_data_batch_*.pkl'):
                try:
                    with open(batch_file, 'rb') as f:
                        batch_data = pickle.load(f)

                    # Store in cache
                    for _, episode_data in batch_data.items():
                        episode_idx = episode_data['episode_idx']
                        key = f'{dataset_name}_{episode_idx}'
                        self.tcp_data_cache[key] = episode_data

                except Exception as e:
                    Log.warn(f"Error loading TCP batch {batch_file}: {e}")

            Log.info(f"Loaded TCP data for {dataset_name}")

    def _get_tcp_data(self, dataset_name: str, episode_id: str, step_idx: Optional[int] = None) -> Optional[Dict]:
        """Get preprocessed TCP data for a specific step"""
        # Debug logging
        import tensorflow as tf
        tf.print(f"DEBUG: _get_tcp_data called with dataset_name={dataset_name}, episode_id={episode_id} (type: {type(episode_id)})")
        tf.print(f"DEBUG: Cache keys available: {list(self.tcp_data_cache.keys())[:5]}...")  # Show first 5 keys
        
        # Convert episode_id to integer if it's a string
        try:
            if isinstance(episode_id, str):
                episode_idx = int(episode_id)
            else:
                episode_idx = episode_id
            tf.print(f"DEBUG: Converted episode_id to episode_idx: {episode_idx}")
        except (ValueError, TypeError) as e:
            # If conversion fails, return None
            tf.print(f"DEBUG: Failed to convert episode_id '{episode_id}' to int: {e}")
            return None
            
        key = f'{dataset_name}_{episode_idx}'
        tf.print(f"DEBUG: Looking for key: {key}")
        
        if key not in self.tcp_data_cache:
            tf.print(f"DEBUG: Key {key} not found in cache. Cache size: {len(self.tcp_data_cache)}")
            return None

        episode_tcp = self.tcp_data_cache[key]
        if step_idx is None:
            tf.print(f"DEBUG: Returning episode_tcp for key {key}")
            return episode_tcp

        if 'steps' in episode_tcp and step_idx < len(episode_tcp['steps']):
            tf.print(f"DEBUG: Returning step {step_idx} from episode_tcp")
            return episode_tcp['steps'][step_idx]
        tf.print(f"DEBUG: No valid step found for step_idx {step_idx}")
        return None

    def __iter__(self):
        yield from self.tf_dataset.as_numpy_iterator()
