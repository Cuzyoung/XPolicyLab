import re
from omegaconf import DictConfig
from sapolicy.logger import Log


class LRTable:
    def __init__(
        self,
        default_lr: float = 1e-4,
        default_weight_decay: float = 0.01,
        prefix_table: DictConfig = None,
        postfix_table: DictConfig = None,
        else_table: DictConfig = None,
        decay_table: DictConfig = None,
        backbone_layer_decay: float = 1.0,
        backbone_num_layers: int = 12,
    ):
        #
        self.default_lr = default_lr
        self.default_weight_decay = default_weight_decay
        self.prefix_table = prefix_table
        self.postfix_table = postfix_table
        self.else_table = else_table
        self.decay_table = decay_table
        self.tables = [self.prefix_table, self.postfix_table, self.else_table]
        self.tags = ["prefix_", "postfix_", "else_"]
        self.backbone_layer_decay = backbone_layer_decay
        self.backbone_num_layers = backbone_num_layers

    def match_table(self, key) -> bool:
        if self.prefix_table is not None:
            for prefix in self.prefix_table:
                if key.startswith(prefix):
                    return 0, prefix

        if self.postfix_table is not None:
            for postfix in self.postfix_table:
                if key.endswith(postfix):
                    return 1, postfix

        if self.else_table is not None:
            for else_key in self.else_table:
                if else_key in key and self.tables[2][else_key] is not None:
                    return 2, else_key

        return 3, None

    def _get_backbone_layer_index(self, key):
        """Extract block index from backbone parameter name (e.g. 'pretrained.blocks.7.attn' -> 7).
        Returns None if not a block parameter. patch_embed gets index -1 (earliest)."""
        m = re.search(r'\.blocks\.(\d+)\.', key)
        if m:
            return int(m.group(1))
        if 'patch_embed' in key:
            return -1
        return None

    def get_lr(self, key) -> float:
        table_idx, match_key = self.match_table(key)
        if table_idx == 3:
            Log.debug(
                f"{key} is not matched to any table, use default lr: {self.default_lr}"
            )
            return "default", self.default_lr
        else:
            base_lr = self.tables[table_idx][match_key]
            group_name = self.tags[table_idx] + match_key

            # Apply layer-wise decay for backbone parameters
            if self.backbone_layer_decay < 1.0 and match_key in ('pretrained', 'depth_pretrained'):
                layer_idx = self._get_backbone_layer_index(key)
                if layer_idx is not None:
                    # Layer N gets decay^(num_layers - 1 - N), so top layer = 1x, bottom = decay^(N-1)
                    num_layers = self.backbone_num_layers
                    exponent = num_layers - 1 - layer_idx  # block 11 -> 0, block 0 -> 11, patch_embed(-1) -> 12
                    scale = self.backbone_layer_decay ** exponent
                    lr = base_lr * scale
                    group_name = f"{group_name}_layer{layer_idx}"
                    Log.debug(f"{key} -> {group_name}: base_lr={base_lr:.2e} * decay^{exponent}={scale:.4f} -> lr={lr:.2e}")
                    return group_name, lr

            Log.debug(f"{key} is matched to table {self.tags[table_idx]}: {match_key}")
            return group_name, base_lr

    def get_weight_decay(self, key) -> float:
        if self.decay_table is not None:
            for name in self.decay_table.keys():
                if key.startswith(name):
                    return self.decay_table[name]
        
        return self.default_weight_decay