"""Select and validate the action datasets used by a YAM inference recipe."""

from omegaconf import OmegaConf


def resolve_action_dataset(config):
    """Return a representative action leaf without rewriting training branches.

    GeneralDataModule uses the last CombinedLoader branch for action learning.
    Without CombinedLoader, dataset_opts is one dataset or a concatenated list.
    Every action leaf must describe the same inference contract.
    """
    cfg = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
    train = cfg['data']['train_dataset']
    branch = train['dataset_opts']
    if 'combined_loader_opts' in train:
        if not isinstance(branch, (list, tuple)) or not branch:
            raise ValueError('CombinedLoader requires nonempty dataset_opts branches')
        branch = branch[-1]

    def leaves(node):
        if isinstance(node, dict):
            return [node]
        if isinstance(node, (list, tuple)) and node:
            return [leaf for child in node for leaf in leaves(child)]
        raise ValueError('Action dataset branch must contain nonempty dataset mappings')

    def enabled(value):
        return value is True or str(value).lower() == 'true'

    datasets = leaves(branch)
    reference = None
    for index, dataset in enumerate(datasets):
        if not enabled(dataset.get('body_frame_actions')) or not enabled(
            dataset.get('use_relative_actions', cfg.get('use_relative_actions'))
        ):
            raise ValueError(f'Action dataset {index} requires body-frame relative actions')
        camera_names = dataset.get('camera_names')
        if camera_names is None:
            camera_names = dataset.get('camera_pair_choices', {}).get('canonical_names')
        contract = {
            'transforms': dataset.get('transforms'),
            'camera_names': camera_names,
            'action_orn_mode': dataset.get('action_orn_mode', cfg.get('action_orn_mode')),
            'action_sequence_length': int(dataset.get('action_sequence_length', cfg['action_sequence_length'])),
            'obs_hist_length': int(dataset.get('obs_hist_length', cfg.get('obs_hist_length', 1))),
            'num_arms': int(dataset.get('num_arms', 2)),
            'arm_obs_prefixes': dataset.get('arm_obs_prefixes', ['left', 'right']),
            'use_state': enabled(dataset.get('use_state', cfg.get('use_state'))),
            'use_depth': enabled(dataset.get('use_depth', cfg.get('use_depth'))),
            'normalize_actions': enabled(dataset.get('normalize_actions', True)),
            'norm_type': dataset.get('norm_type', cfg.get('norm_type')),
        }
        if reference is not None:
            differences = [key for key in contract if contract[key] != reference[key]]
            if differences:
                raise ValueError(f'Action dataset {index} has inconsistent inference fields: {differences}')
        reference = contract
    # Preserve the existing flat-list recipe's final action dataset selection.
    return OmegaConf.create(datasets[-1])
