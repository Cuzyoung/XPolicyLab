import torch
from torch import nn
import torch.nn.functional as F
import logging
import os

# NaN/Inf guards call .any()/.item() on CUDA tensors: each one is a host sync that drains the GPU queue.
# Off by default; set SA_DEBUG_NONFINITE=1 to re-enable them when debugging a divergence.
_DEBUG_NONFINITE = os.environ.get("SA_DEBUG_NONFINITE", "0") == "1"

log = logging.getLogger(__name__)

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt((diff_log ** 2).mean() -
                          self.lambd * (diff_log.mean() ** 2))

        return loss


class AffineInvarientLoss(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        # pred = pred[valid_mask]
        # target = target[valid_mask]
        
        N = pred.shape[0]
        pred_m, _ = torch.median(pred.reshape(N, -1), 1)
        norm_pred = pred - pred_m.reshape(N, 1, 1)
        norm_pred = norm_pred / (torch.mean(torch.abs(norm_pred).reshape(N, -1), axis=1).reshape(N, 1, 1) + 1e-5)
        print('pred:', norm_pred.detach().cpu().numpy().min(), norm_pred.detach().cpu().numpy().max())
        
        target_m, _ = torch.median(target.reshape(N, -1), 1)
        norm_gt = target - target_m.reshape(N, 1, 1)
        norm_gt = norm_gt / (torch.mean(torch.abs(norm_gt).reshape(N, -1), axis=1).reshape(N, 1, 1) + 1e-5)
        print('gt: ', norm_gt.detach().cpu().numpy().min(), norm_gt.detach().cpu().numpy().max())

        loss = self.lambd * torch.mean(torch.abs(norm_pred - norm_gt))
        return loss, norm_pred, norm_gt


class MultiScaleGradLoss(nn.Module):
    def __init__(self, lambd=0.5, scales=4):
        super().__init__()
        self.lambd = lambd
        self.scales = scales
        self.scale_ratio = [1.0, 0.85, 0.75, 0.50]
        self.eps = 1e-5

    def compute_gradient(self, img):
        # Compute gradients along x and y directions
        grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return grad_x, grad_y

    def forward(self, pred, target, valid_mask): # N, H, W
        valid_mask = valid_mask.detach()
        pred = torch.unsqueeze(pred, 1)
        target = torch.unsqueeze(target, 1)
        total_loss = torch.zeros((), device=pred.device, dtype=torch.float32)
        for i in range(self.scales):
            scale_factor = 1.0 / (2.0**i)
            pred_scaled = nn.functional.interpolate(pred, scale_factor=scale_factor, mode='bicubic', align_corners=True)
            target_scaled = nn.functional.interpolate(target, scale_factor=scale_factor, mode='bicubic', align_corners=True)
            # pred_scaled = pred_scaled.clamp(self.eps, 1e6)
            # target_scaled = target_scaled.clamp(self.eps, 1e6)
            # diff_log = torch.log(target_scaled) - torch.log(pred_scaled)
            # grad_x, grad_y = self.compute_gradient(diff_log)
            
            diff = target_scaled - pred_scaled
            grad_x, grad_y = self.compute_gradient(diff)
            loss = torch.abs(grad_x).mean() + torch.abs(grad_y).mean()

            total_loss += loss * self.scale_ratio[i]

        return total_loss * self.lambd


class TotalLoss(nn.Module):
    def __init__(self, lambd=[1.0, 3.0]):
        super().__init__()
        self.affine_loss = AffineInvarientLoss(lambd=lambd[0])
        self.grad_loss = MultiScaleGradLoss(lambd=lambd[1])

    def forward(self, pred, target, valid_mask):
        loss0, norm_pred, norm_gt = self.affine_loss(pred, target, valid_mask) 
        loss1 = self.grad_loss(norm_pred, norm_gt, valid_mask)

        loss = loss0 + loss1
        return loss


class MultiScaleGradLossV2(nn.Module):
    def __init__(self, lambd=0.5, scales=4):
        super().__init__()
        self.lambd = lambd
        self.scales = scales
        self.scale_ratio = [1.0, 1.0, 1.0, 1.0]

    def compute_gradient(self, img):
        # Compute gradients along x and y directions
        grad_x = img[:, :, :-1] - img[:, :, 1:]
        grad_y = img[:, :-1, :] - img[:, 1:, :]
        return grad_x, grad_y

    def forward(self, pred, target, valid_mask): # N, H, W
        valid_mask = valid_mask.detach()

        total_loss = torch.zeros((), device=pred.device, dtype=torch.float32)
        for i in range(self.scales):
            step = int(2**i)
            pred_scaled = pred[:, ::step, ::step]
            target_scaled = target[:, ::step, ::step]
            
            diff = target_scaled - pred_scaled
            grad_x, grad_y = self.compute_gradient(diff)
            loss = torch.abs(grad_x).mean() + torch.abs(grad_y).mean()

            total_loss += loss * self.scale_ratio[i]

        return total_loss * self.lambd


class TotalLossv2(nn.Module):
    def __init__(self, lambd=[1.0, 3.0], affine=True):
        super().__init__()
        self.lambd = lambd
        self.grad_loss = MultiScaleGradLossV2(lambd=lambd[1])
        self.affine = affine
        
    # least-squares solver
    def compute_scale_and_shift(self, prediction, target, mask):
        if mask is None:
            mask = torch.ones(*prediction.shape, device=prediction.device)
            
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        valid = det.nonzero()

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1
    
    def forward(self, pred, target, valid_mask):
        """
        pred: N x H x W
        target: N x H x W
        """
        if self.affine:
            scale, shift = self.compute_scale_and_shift(pred, target, None)
            affine_pred = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)
        else:
            affine_pred = pred
        # print('pred:', affine_pred.detach().cpu().numpy().min(), affine_pred.detach().cpu().numpy().max())
        # print('gt:', target.detach().cpu().numpy().min(), target.detach().cpu().numpy().max())
        
        diff = affine_pred - target
        loss0 = torch.abs(diff).mean() * self.lambd[0]
        
        loss1 = self.grad_loss(affine_pred, target, valid_mask)
        loss = loss0 + loss1
        
        return loss
    
    
class CELoss(nn.Module):
    def __init__(self, classes=3, invalid=None):
        super().__init__()
        self.ce_func = nn.CrossEntropyLoss()
        self.classes = classes
        self.invalid = invalid
        
    def forward(self, pred, target):
        '''
        pred: B x C x U x V x D
        target: B x U x V x D
        '''
        pred = pred.permute(0, 2, 3, 4, 1)
        if self.invalid is None:
            return self.ce_func(pred.reshape(-1, self.classes), target.reshape(-1))
        else:
            pred = pred.reshape(-1, self.classes)
            target = target.reshape(-1)
            indices = torch.where(target == invalid)
            sel_pred = pred[indices, :]
            sel_target = target[indices]
            return self.ce_func(sel_pred, sel_target)
        
        
class GeoLoss(nn.Module):
    def __init__(self, weight=100):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred_barrier, target):
        '''
        pred_barrier: B x U x V x D
        target: B x U x V x D
        '''
        pred_barrier = pred_barrier.float()
        empty_probs = 1 - pred_barrier
        target_barrier = (target == 0).float()
        target_free = (target > 0).float()
        
        eps = 1e-6
        intersection = (target_barrier * pred_barrier).sum()
        precision = intersection / (pred_barrier.sum() + eps)
        recall = intersection / (target_barrier.sum() + eps)
        spec = (target_free * empty_probs).sum() / (target_free.sum()+eps)
        
        weight_map = target_barrier + torch.ones_like(target_barrier) / self.weight
        loss0 = F.binary_cross_entropy(pred_barrier, target_barrier, weight=weight_map)
        loss1 = F.binary_cross_entropy(precision, torch.ones_like(precision))
        loss2 = F.binary_cross_entropy(recall, torch.ones_like(recall))
        loss3 = F.binary_cross_entropy(spec, torch.ones_like(spec))

        loss = loss0 + loss1 + loss2 + loss3
        return loss
    
    
class DepthProbLoss(nn.Module):
    def __init__(self, downsample=7, d_range=[0, 2.0], ds=0.02):
        super().__init__()
        self.downsample = downsample
        self.D = int((d_range[1] - d_range[0]) / ds)
        self.d_range = d_range
        self.ds = ds
        self.celoss = nn.CrossEntropyLoss()

    def depthprob(self, inv_depth):
        depth = 1 / inv_depth

        # spatial downsample
        B, H, W = depth.shape
        sH = int(H//self.downsample)
        sW = int(W//self.downsample)
        depth = depth.reshape(B, sH, self.downsample, sW, self.downsample)
        depth = depth.permute(0, 1, 3, 2, 4).reshape(B, sH, sW, -1).min(axis=-1).values

        # channel expand
        depth = (depth - self.d_range[0]) / self.ds
        depth_idx = torch.where((depth < self.D) & (depth >= 0.0), depth, torch.ones_like(depth)*self.D)
        depth_Onehot = F.one_hot(depth_idx.long(), num_classes=self.D + 1)[..., :-1].permute(0, 3, 1, 2)

        return depth_idx.long(), depth_Onehot

    def forward(self, depth_prob, inv_depth):
        '''
        depth_prob: B x D x H' x W'
        inv_depth: B x H x W
        '''
        assert depth_prob.shape[1] == self.D

        gt_depth_idx, gt_depth_Onehot = self.depthprob(inv_depth.float())
        loss0 = ((depth_prob.float() - gt_depth_Onehot)**2).mean()

        mask = gt_depth_idx < self.D
        tmp = depth_prob.permute(0, 2, 3, 1).float()
        loss1 = self.celoss(tmp[mask, :], gt_depth_idx[mask])

        loss = loss0 + loss1
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in heatmap supervision"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred: [B, C, H, W] prediction heatmap
        target: [B, C, H, W] target heatmap
        """
        with torch.cuda.amp.autocast(enabled=False):
            pred = pred.float()
            target = target.float()

            # Clamp to valid range for BCE (CRITICAL: prevents CUDA assert)
            pred = pred.clamp(0.0, 1.0)
            target = target.clamp(0.0, 1.0)

            # Validate inputs
            if _DEBUG_NONFINITE and (torch.isnan(pred).any() or torch.isinf(pred).any()):
                print(f"WARNING: NaN/Inf in pred heatmap before BCE")
                pred = torch.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0)

            if _DEBUG_NONFINITE and (torch.isnan(target).any() or torch.isinf(target).any()):
                print(f"WARNING: NaN/Inf in target heatmap before BCE")
                target = torch.nan_to_num(target, nan=0.5, posinf=1.0, neginf=0.0)

            bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss for robust coordinate regression"""
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta,
                          0.5 * (diff ** 2) / self.beta,
                          diff - 0.5 * self.beta)
        return loss.mean()


class CoordinateConsistencyLoss(nn.Module):
    """Ensures consistency between heatmap and coordinate predictions"""
    def __init__(self, sigma=0.1):
        super(CoordinateConsistencyLoss, self).__init__()
        self.sigma = sigma

    def forward(self, pred_coords, pred_heatmap):
        """
        pred_coords: [B, 3] predicted coordinates [u, v, depth]
        pred_heatmap: [B, 1, H, W] predicted heatmap
        """
        B, _, H, W = pred_heatmap.shape
        device = pred_coords.device

        # Clamp predicted coordinates to valid range [0, 1] without inplace operation
        pred_u = torch.clamp(pred_coords[:, 0], 0.0, 1.0)
        pred_v = torch.clamp(pred_coords[:, 1], 0.0, 1.0)
        pred_coords_clamped = torch.stack([pred_u, pred_v, pred_coords[:, 2]], dim=1)

        # Check for NaN values
        if _DEBUG_NONFINITE and (torch.isnan(pred_coords_clamped).any()):
            return torch.zeros((), device=device, dtype=torch.float32)

        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Normalize coordinates (handle H=1 or W=1 edge case)
        if W > 1:
            x_coords_norm = x_coords / (W - 1)
        else:
            raise ValueError(f"Invalid width: {W}")
            x_coords_norm = x_coords * 0.0 + 0.5  # Center at 0.5

        if H > 1:
            y_coords_norm = y_coords / (H - 1)
        else:
            raise ValueError(f"Invalid height: {H}")
            y_coords_norm = y_coords * 0.0 + 0.5  # Center at 0.5

        consistency_loss = torch.zeros((), device=device, dtype=torch.float32)
        for i in range(B):
            pred_u, pred_v = pred_coords_clamped[i, 0], pred_coords_clamped[i, 1]

            # Generate expected heatmap from predicted coordinates (using normalized coords)
            expected_heatmap = torch.exp(-((x_coords_norm - pred_u)**2 + (y_coords_norm - pred_v)**2) / (2.0 * self.sigma**2))
            expected_heatmap = expected_heatmap.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            # Compute KL divergence between predicted and expected heatmaps
            pred_hm = pred_heatmap[i:i+1]
            eps = 1e-8
            pred_hm = pred_hm / (pred_hm.sum() + eps)  # Normalize
            expected_heatmap = expected_heatmap / (expected_heatmap.sum() + eps)  # Normalize

            kl_loss = F.kl_div(torch.log(pred_hm + eps), expected_heatmap, reduction='sum')
            consistency_loss += kl_loss

        return consistency_loss / B


class TCPDirection3DLoss(nn.Module):
    """
    Loss for TCP 2D direction prediction with optional 3D geometric constraints.
    """
    def __init__(self, lambda_orth=1.0, lambda_norm=0.1):
        """
        Args:
            lambda_orth: weight for 3D orthogonality loss
            lambda_norm: weight for 3D unit norm loss
        """
        super().__init__()
        self.l2_loss = nn.MSELoss()
        self.lambda_orth = lambda_orth
        self.lambda_norm = lambda_norm

    def forward(self, pred_dir_x, pred_dir_y, pred_dir_z,
                gt_dir_x, gt_dir_y, gt_dir_z,
                depth_map=None,
                K=None):
        """
        Args:
            pred_dir_x/y/z: predicted 2D directions, shape [B, 2] if pooled, or [B, 2, H, W]
            gt_dir_x/y/z: ground-truth 2D directions, same shape as pred
            depth_map: [B], TCP depth for back-projection
            K: [B, 9] or [B, 3, 3] camera intrinsics matrix
        """
        # ---------- 1. 2D MSE loss ----------
        loss_2d = self.l2_loss(pred_dir_x, gt_dir_x) + \
                  self.l2_loss(pred_dir_y, gt_dir_y) + \
                  self.l2_loss(pred_dir_z, gt_dir_z)

        # ---------- 2. Optional 3D constraints ----------
        if depth_map is not None and K is not None:
            B = pred_dir_x.shape[0]
            device = pred_dir_x.device

            # reshape K if necessary
            if K.ndim == 2 and K.shape[1] == 9:
                K = K.view(B, 3, 3)  # [B,3,3]

            K_inv = torch.linalg.inv(K)  # [B,3,3]

            def backproject_2d_to_3d(dir2d, depth):
                """
                dir2d: [B,2]
                depth: [B]
                return: [B,3] unit vector in camera frame
                """
                ones = torch.ones(B, 1, device=device)
                u_v = torch.cat([dir2d, ones], dim=1).unsqueeze(2)  # [B,3,1]
                vec3d = torch.bmm(K_inv, u_v).squeeze(2)  # [B,3]
                vec3d = vec3d * depth.view(B,1)
                vec3d = F.normalize(vec3d, p=2, dim=1)
                return vec3d

            dir_x_3d = backproject_2d_to_3d(pred_dir_x, depth_map)
            dir_y_3d = backproject_2d_to_3d(pred_dir_y, depth_map)
            dir_z_3d = backproject_2d_to_3d(pred_dir_z, depth_map)

            # ---------- 2a. Orthogonality loss ----------
            dot_xy = torch.sum(dir_x_3d * dir_y_3d, dim=1)
            dot_yz = torch.sum(dir_y_3d * dir_z_3d, dim=1)
            dot_zx = torch.sum(dir_z_3d * dir_x_3d, dim=1)
            loss_orth = torch.mean(dot_xy**2 + dot_yz**2 + dot_zx**2)

            # ---------- 2b. Unit norm loss ----------
            loss_norm = torch.mean((dir_x_3d.norm(dim=1)-1)**2 +
                                   (dir_y_3d.norm(dim=1)-1)**2 +
                                   (dir_z_3d.norm(dim=1)-1)**2)
        else:
            loss_orth = torch.zeros((), device=pred_dir_x.device)
            loss_norm = torch.zeros((), device=pred_dir_x.device)

        # ---------- 3. Total loss ----------
        total_loss = loss_2d + self.lambda_orth*loss_orth + self.lambda_norm*loss_norm

        return total_loss, loss_2d, loss_orth, loss_norm


class AdvancedTCPLoss(nn.Module):
    """
    Advanced Multi-objective Loss for TCP Prediction

    Combines:
    - L1 + L2 coordinate losses for robustness
    - Focal loss for heatmap supervision
    - Direction prediction losses
    - Coordinate-heatmap consistency loss
    - Multi-scale gradient losses for spatial awareness
    """
    def __init__(self,
                 coord_l1_weight=1.0,
                 coord_l2_weight=1.0,
                 coord_smooth_weight=0.5,
                 heatmap_weight=0.1,
                 heatmap_focal_weight=0.05,
                 direction_weight=1.0,
                 consistency_weight=0.1,
                 gradient_weight=0.01,
                 lambda_orth=1.0,
                 lambda_norm=0.1,
                 sigma=2.0,
                 focal_alpha=1.0,
                 focal_gamma=2.0):
        super(AdvancedTCPLoss, self).__init__()

        # Loss weights
        self.coord_l1_weight = coord_l1_weight
        self.coord_l2_weight = coord_l2_weight
        self.coord_smooth_weight = coord_smooth_weight
        self.heatmap_weight = heatmap_weight
        self.heatmap_focal_weight = heatmap_focal_weight
        self.direction_weight = direction_weight
        self.consistency_weight = consistency_weight
        self.gradient_weight = gradient_weight

        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.smooth_l1_loss = SmoothL1Loss(beta=1.0)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.consistency_loss = CoordinateConsistencyLoss(sigma=sigma)
        self.direction_loss = TCPDirection3DLoss(lambda_orth=lambda_orth, lambda_norm=lambda_norm)

        self.sigma = sigma

    def create_gaussian_heatmap(self, coords, H, W, device):
        """Create Gaussian heatmap targets from coordinates"""
        B = coords.shape[0]
        heatmaps = []

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Normalize coordinates (handle H=1 or W=1 edge case)
        if W > 1:
            x_coords_norm = x_coords / (W - 1)
        else:
            raise ValueError(f"Invalid width: {W}")
            x_coords_norm = x_coords * 0.0 + 0.5  # Center at 0.5

        if H > 1:
            y_coords_norm = y_coords / (H - 1)
        else:
            raise ValueError(f"Invalid height: {H}")
            y_coords_norm = y_coords * 0.0 + 0.5  # Center at 0.5

        for i in range(B):
            gt_u, gt_v = coords[i, 0], coords[i, 1]

            # Create Gaussian heatmap
            if 0 <= gt_u <= 1 and 0 <= gt_v <= 1 and torch.isfinite(gt_u) and torch.isfinite(gt_v):
                # 计算高斯heatmap (using normalized coords)
                heatmap = torch.exp(-((x_coords_norm - gt_u)**2 + (y_coords_norm - gt_v)**2) / (2.0 * self.sigma**2))
            else:
                log.warning(f"Invalid ground truth coordinates: {gt_u}, {gt_v} for batch {i}")
                heatmap = torch.zeros((H, W), device=device, dtype=torch.float32)

            heatmaps.append(heatmap)

        return torch.stack(heatmaps, dim=0).unsqueeze(1)  # [B, 1, H, W]

    def compute_gradient_loss(self, pred_heatmap, gt_heatmap):
        """Compute gradient loss for spatial consistency"""
        def compute_gradients(x):
            grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
            grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
            return grad_x, grad_y

        pred_grad_x, pred_grad_y = compute_gradients(pred_heatmap)
        gt_grad_x, gt_grad_y = compute_gradients(gt_heatmap)

        grad_loss_x = F.l1_loss(pred_grad_x, gt_grad_x)
        grad_loss_y = F.l1_loss(pred_grad_y, gt_grad_y)

        return grad_loss_x + grad_loss_y

    def forward(self, predictions, targets):
        """
        predictions: dict containing:
            - tcp_pixel_coords: [B, 3] predicted coordinates
            - tcp_dir_x: [B, 2] predicted x direction
            - tcp_dir_y: [B, 2] predicted y direction
            - tcp_dir_z: [B, 2] predicted z direction
            - tcp_heatmap: [B, 1, H, W] predicted heatmap

        targets: dict containing:
            - tcp_pixel_coords: [B, 3] ground truth coordinates
            - tcp_dir_x: [B, 2] ground truth x direction
            - tcp_dir_y: [B, 2] ground truth y direction
            - tcp_dir_z: [B, 2] ground truth z direction
        """
        # Ensure all inputs are float32 (without inplace modification)
        pred_coords = predictions['tcp_pixel_coords'].float()
        pred_dir_x = predictions['tcp_dir_x'].float() if 'tcp_dir_x' in predictions else None
        pred_dir_y = predictions['tcp_dir_y'].float() if 'tcp_dir_y' in predictions else None
        pred_dir_z = predictions['tcp_dir_z'].float() if 'tcp_dir_z' in predictions else None
        pred_heatmap = predictions['tcp_heatmap'].float()

        gt_coords = targets['tcp_pixel_coords'].float()
        gt_dir_x = targets['tcp_dir_x'].float() if 'tcp_dir_x' in targets else None
        gt_dir_y = targets['tcp_dir_y'].float() if 'tcp_dir_y' in targets else None
        gt_dir_z = targets['tcp_dir_z'].float() if 'tcp_dir_z' in targets else None
        K = targets['intrinsics'].float()

        B, _, H, W = pred_heatmap.shape
        device = pred_coords.device

        # 1. Multi-objective coordinate losses
        coord_l1_loss = self.l1_loss(pred_coords, gt_coords)
        coord_l2_loss = self.l2_loss(pred_coords, gt_coords)
        coord_smooth_loss = self.smooth_l1_loss(pred_coords, gt_coords)

        # 2. Direction losses
        if pred_dir_x is not None and pred_dir_y is not None and pred_dir_z is not None and gt_dir_x is not None and gt_dir_y is not None and gt_dir_z is not None:
            dir_total_loss, loss_dir_2d, loss_dir_orth, loss_dir_norm = self.direction_loss(pred_dir_x, pred_dir_y, pred_dir_z,
                                                        gt_dir_x, gt_dir_y, gt_dir_z,
                                                        depth_map=pred_coords[:, 2],
                                                        K=K)
        else:
            dir_total_loss = torch.zeros((), device=pred_coords.device)
            loss_dir_2d = torch.zeros((), device=pred_coords.device)
            loss_dir_orth = torch.zeros((), device=pred_coords.device)
            loss_dir_norm = torch.zeros((), device=pred_coords.device)

        # 3. Heatmap supervision with Gaussian targets (skip for 1x1 feature maps)
        # For ResNet with global pooling (H=1, W=1), heatmaps don't make sense
        # Only use heatmap losses for spatial feature maps (e.g., DINOv2 patches)
        if H > 1 and W > 1:
            gt_heatmaps = self.create_gaussian_heatmap(gt_coords, H, W, device)
            heatmap_mse_loss = self.l2_loss(pred_heatmap, gt_heatmaps)
            heatmap_focal_loss = self.focal_loss(pred_heatmap, gt_heatmaps)

            # 4. Coordinate-heatmap consistency loss
            consistency_loss = self.consistency_loss(pred_coords, pred_heatmap)

            # 5. Gradient loss for spatial awareness
            gradient_loss = self.compute_gradient_loss(pred_heatmap, gt_heatmaps)
        else:
            # Skip heatmap-based losses for non-spatial features (H=1 or W=1)
            heatmap_mse_loss = torch.zeros((), device=device, dtype=torch.float32)
            heatmap_focal_loss = torch.zeros((), device=device, dtype=torch.float32)
            consistency_loss = torch.zeros((), device=device, dtype=torch.float32)
            gradient_loss = torch.zeros((), device=device, dtype=torch.float32)

        # Combine all losses
        total_loss = (
            self.coord_l1_weight * coord_l1_loss +
            self.coord_l2_weight * coord_l2_loss +
            self.coord_smooth_weight * coord_smooth_loss +
            self.direction_weight * dir_total_loss +
            self.heatmap_weight * heatmap_mse_loss +
            self.heatmap_focal_weight * heatmap_focal_loss +
            # self.consistency_weight * consistency_loss +
            self.gradient_weight * gradient_loss
        ).float()

        # Return detailed loss breakdown for monitoring
        loss_dict = {
            'total_loss': total_loss,
            'coord_l1_loss': coord_l1_loss,
            'coord_l2_loss': coord_l2_loss,
            'coord_smooth_loss': coord_smooth_loss,
            'dir_total_loss': dir_total_loss,
            'loss_dir_2d': loss_dir_2d,
            'loss_dir_orth': loss_dir_orth,
            'loss_dir_norm': loss_dir_norm,
            'heatmap_mse_loss': heatmap_mse_loss,
            'heatmap_focal_loss': heatmap_focal_loss,
            'consistency_loss': consistency_loss,
            'gradient_loss': gradient_loss
        }

        return total_loss, loss_dict


class TCP3DPoseLoss(nn.Module):
    """
    Multi-task loss for 3D TCP pose prediction.

    Primary objectives:
    - 3D orientation loss (geodesic + Frobenius)

    Auxiliary objectives (optional):
    - 2D pixel coordinate loss (L1 + L2)
    - Heatmap loss (MSE + Focal)
    - Consistency loss (3D→2D projection consistency)
    """

    def __init__(
        self,
        # 3D position loss weights
        coord_l1_weight: float = 0.5,
        coord_l2_weight: float = 0.5,
        coord_smooth_weight=0.5,
        # 3D pose loss weights
        orn_geodesic_weight: float = 1.0,
        orn_frobenius_weight: float = 0.1,
        heatmap_weight=0.1,
        heatmap_focal_weight: float = 0.05,
        consistency_weight: float = 0.1,
        gradient_weight: float = 0.01,
        # Heatmap generation params
        sigma: float = 0.05,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
    ):
        super(TCP3DPoseLoss, self).__init__()

        # 3D pose loss weights
        self.orn_geodesic_weight = orn_geodesic_weight
        self.orn_frobenius_weight = orn_frobenius_weight

        # Auxiliary 2D loss weights
        self.coord_l1_weight = coord_l1_weight
        self.coord_l2_weight = coord_l2_weight
        self.coord_smooth_weight = coord_smooth_weight
        self.heatmap_weight = heatmap_weight
        self.heatmap_focal_weight = heatmap_focal_weight
        self.consistency_weight = consistency_weight
        self.gradient_weight = gradient_weight

        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.smooth_l1_loss = SmoothL1Loss(beta=1.0)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.consistency_loss = CoordinateConsistencyLoss(sigma=sigma)

        self.sigma = sigma

    def create_gaussian_heatmap(self, coords, H, W, device):
        """Create Gaussian heatmap targets from coordinates"""
        B = coords.shape[0]
        heatmaps = []

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Normalize coordinates (handle H=1 or W=1 edge case)
        if W > 1:
            x_coords_norm = x_coords / (W - 1)
        else:
            x_coords_norm = x_coords * 0.0 + 0.5  # Center at 0.5

        if H > 1:
            y_coords_norm = y_coords / (H - 1)
        else:
            y_coords_norm = y_coords * 0.0 + 0.5  # Center at 0.5

        for i in range(B):
            gt_u, gt_v = coords[i, 0], coords[i, 1]

            # Create Gaussian heatmap
            if 0 <= gt_u <= 1 and 0 <= gt_v <= 1 and torch.isfinite(gt_u) and torch.isfinite(gt_v):
                # 计算高斯heatmap (using normalized coords)
                heatmap = torch.exp(-((x_coords_norm - gt_u)**2 + (y_coords_norm - gt_v)**2) / (2.0 * self.sigma**2))
            else:
                log.warning(f"Invalid ground truth coordinates: {gt_u}, {gt_v} for batch {i}")
                heatmap = torch.zeros((H, W), device=device, dtype=torch.float32)

            heatmaps.append(heatmap)

        return torch.stack(heatmaps, dim=0).unsqueeze(1)  # [B, 1, H, W]

    def forward(self, predictions, targets, camera_names):
        """
        Args:
            predictions: dict with keys
                - 'tcp_orn': [B, 3, 3] predicted rotation matrix
                - 'tcp_pixel_coords': [B, 3] predicted pixel coords
                - 'tcp_heatmap': [B, 1, H, W] predicted heatmap
            targets: dict with keys
                - 'tcp_orn': [B, 3, 3] or [B, 9] ground truth rotation
                - 'tcp_pixel_coords': [B, 3] ground truth pixel coords
                - 'intrinsics': [B, 3, 3] or [B, 9] camera intrinsics

        Returns:
            total_loss: scalar tensor
            loss_dict: dictionary with individual loss components
        """
        total_loss = 0.0
        loss_dict = {}
        for camera_name in camera_names:
            pred_orn = predictions['tcp_orn'][camera_name].float()
            pred_coords = predictions['tcp_pixel_coords'][camera_name].float()
            pred_heatmap = predictions['tcp_heatmap'][camera_name].float()

            gt_orn = targets['tcp_orn'][camera_name].float()
            gt_coords = targets['tcp_pixel_coords'][camera_name].float()
            K = targets['intrinsics'][camera_name].float()

            B, _, H, W = pred_heatmap.shape
            device = pred_coords.device

            if _DEBUG_NONFINITE and (torch.isnan(pred_orn).any() or torch.isinf(pred_orn).any()):
                print(f"ERROR: NaN/Inf in pred_orn - min: {pred_orn.min().item()}, max: {pred_orn.max().item()}")
                raise ValueError("NaN or Inf detected in predicted TCP orientation")

            if _DEBUG_NONFINITE and (torch.isnan(gt_orn).any() or torch.isinf(gt_orn).any()):
                print(f"ERROR: NaN/Inf in gt_orn - min: {gt_orn.min().item()}, max: {gt_orn.max().item()}")
                raise ValueError("NaN or Inf detected in ground truth TCP orientation")

            # Reshape gt_orn if needed
            if len(gt_orn.shape) == 2 and gt_orn.shape[1] == 9:
                gt_orn = gt_orn.reshape(-1, 3, 3)

            # 1. Orientation loss
            orn_geodesic_loss = self._geodesic_loss(pred_orn, gt_orn)
            orn_frobenius_loss = self._frobenius_loss(pred_orn, gt_orn)

            orn_loss = self.orn_geodesic_weight * orn_geodesic_loss + self.orn_frobenius_weight * orn_frobenius_loss

            # 2. Multi-objective normalized coordinate losses (L1 + L2 + Smooth L1)
            coord_l1_loss = self.l1_loss(pred_coords, gt_coords)
            coord_l2_loss = self.l2_loss(pred_coords, gt_coords)
            coord_smooth_loss = self.smooth_l1_loss(pred_coords, gt_coords)
            coord_loss = (
                self.coord_l1_weight * coord_l1_loss +
                self.coord_l2_weight * coord_l2_loss + 
                self.coord_smooth_weight * coord_smooth_loss
            )

            # 3. Heatmap supervision with Gaussian targets (skip for 1x1 feature maps)
            # For ResNet with global pooling (H=1, W=1), heatmaps don't make sense
            # Only use heatmap losses for spatial feature maps (e.g., DINOv2 patches)
            if H > 1 and W > 1:
                gt_heatmaps = self.create_gaussian_heatmap(gt_coords, H, W, device)
                heatmap_mse_loss = self.l2_loss(pred_heatmap, gt_heatmaps)
                heatmap_focal_loss = self.focal_loss(pred_heatmap, gt_heatmaps)
                heatmap_loss = (
                    self.heatmap_weight * heatmap_mse_loss +
                    self.heatmap_focal_weight * heatmap_focal_loss
                )

                # 4. Coordinate-heatmap consistency loss
                consistency_loss = self.consistency_loss(pred_coords, pred_heatmap) * self.consistency_weight

                # 5. Gradient loss for spatial awareness
                gradient_loss = self.compute_gradient_loss(pred_heatmap, gt_heatmaps) * self.gradient_weight
            else:
                # Skip heatmap-based losses for non-spatial features (H=1 or W=1)
                heatmap_loss = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
                heatmap_mse_loss = torch.zeros((), device=device, dtype=torch.float32)
                heatmap_focal_loss = torch.zeros((), device=device, dtype=torch.float32)
                consistency_loss = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
                gradient_loss = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

            # Validate each loss component before summation
            def validate_loss(loss, name):
                if _DEBUG_NONFINITE and (torch.isnan(loss) or torch.isinf(loss)):
                    print(f"ERROR: NaN/Inf in {name} = {loss.item()}")
                    return torch.zeros((), device=loss.device, dtype=loss.dtype, requires_grad=True)
                return loss

            orn_loss = validate_loss(orn_loss, "orn_loss")
            coord_loss = validate_loss(coord_loss, "coord_loss")
            heatmap_loss = validate_loss(heatmap_loss, "heatmap_loss")
            gradient_loss = validate_loss(gradient_loss, "gradient_loss")
            consistency_loss = validate_loss(consistency_loss, "consistency_loss")

            # ===== Total Loss =====
            total_loss_tmp = (orn_loss + coord_loss + heatmap_loss + gradient_loss).float()
            total_loss += total_loss_tmp

            # Final validation: Check for NaN/Inf in total loss
            if _DEBUG_NONFINITE and (torch.isnan(total_loss) or torch.isinf(total_loss)):
                print("=" * 80)
                print("ERROR: NaN/Inf detected in total loss!")
                print(f"  {camera_name} total_loss: {total_loss_tmp.item()}")
                print(f"  {camera_name} orn_loss: {orn_loss.item()}")
                print(f"  {camera_name} coord_loss: {coord_loss.item()}")
                print(f"  {camera_name} heatmap_loss: {heatmap_loss.item()}")
                print(f"  {camera_name} gradient_loss: {gradient_loss.item()}")
                print(f"  consistency_loss: {consistency_loss.item()}")
                print(f"  {camera_name} orn_geodesic_loss: {orn_geodesic_loss.item()}")
                print(f"  {camera_name} orn_frobenius_loss: {orn_frobenius_loss.item()}")
                print(f"  {camera_name} coord_l1_loss: {coord_l1_loss.item()}")
                print(f"  {camera_name} coord_l2_loss: {coord_l2_loss.item()}")
                print(f"  coord_smooth_loss: {coord_smooth_loss.item()}")
                print(f"  {camera_name} heatmap_mse_loss: {heatmap_mse_loss.item()}")
                print(f"  {camera_name} heatmap_focal_loss: {heatmap_focal_loss.item()}")
                print("=" * 80)
                raise ValueError("NaN or Inf in total loss")

            loss_dict.update({
                f'{camera_name}/total_loss': total_loss_tmp,
                f'{camera_name}/coord_loss': coord_loss,
                f'{camera_name}/coord_l1_loss': coord_l1_loss,
                f'{camera_name}/coord_l2_loss': coord_l2_loss,
                f'{camera_name}/coord_smooth_loss': coord_smooth_loss,
                f'{camera_name}/orn_loss': orn_loss,
                f'{camera_name}/orn_geodesic_loss': orn_geodesic_loss,
                f'{camera_name}/orn_frobenius_loss': orn_frobenius_loss,
                f'{camera_name}/heatmap_loss': heatmap_loss,
                f'{camera_name}/heatmap_mse_loss': heatmap_mse_loss,
                f'{camera_name}/heatmap_focal_loss': heatmap_focal_loss,
                f'{camera_name}/consistency_loss': consistency_loss,
                f'{camera_name}/gradient_loss': gradient_loss
            })

        return total_loss, loss_dict

    def compute_gradient_loss(self, pred_heatmap, gt_heatmap):
        """Compute gradient loss for spatial consistency"""
        def compute_gradients(x):
            grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
            grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
            return grad_x, grad_y

        # Validate inputs
        if _DEBUG_NONFINITE and (torch.isnan(pred_heatmap).any() or torch.isinf(pred_heatmap).any()):
            print(f"WARNING: NaN/Inf in pred_heatmap for gradient loss")
            return torch.zeros((), device=pred_heatmap.device, requires_grad=True)

        if _DEBUG_NONFINITE and (torch.isnan(gt_heatmap).any() or torch.isinf(gt_heatmap).any()):
            print(f"WARNING: NaN/Inf in gt_heatmap for gradient loss")
            return torch.zeros((), device=gt_heatmap.device, requires_grad=True)

        pred_grad_x, pred_grad_y = compute_gradients(pred_heatmap)
        gt_grad_x, gt_grad_y = compute_gradients(gt_heatmap)

        grad_loss_x = F.l1_loss(pred_grad_x, gt_grad_x)
        grad_loss_y = F.l1_loss(pred_grad_y, gt_grad_y)

        grad_loss = grad_loss_x + grad_loss_y

        # Validate output
        if _DEBUG_NONFINITE and (torch.isnan(grad_loss) or torch.isinf(grad_loss)):
            print(f"WARNING: NaN/Inf in gradient loss output")
            return torch.zeros((), device=pred_heatmap.device, requires_grad=True)

        return grad_loss

    def _geodesic_loss(self, pred_orn, gt_orn):
        """Geodesic distance on SO(3) manifold with numerical stability"""
        # Check for NaN/Inf in inputs
        if _DEBUG_NONFINITE and (torch.isnan(pred_orn).any() or torch.isinf(pred_orn).any()):
            print("Warning: NaN/Inf in pred_orn")
            return torch.zeros((), device=pred_orn.device, requires_grad=True)
        if _DEBUG_NONFINITE and (torch.isnan(gt_orn).any() or torch.isinf(gt_orn).any()):
            print("Warning: NaN/Inf in gt_orn")
            return torch.zeros((), device=gt_orn.device, requires_grad=True)

        R_rel = torch.bmm(pred_orn.transpose(1, 2), gt_orn)
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        # Increase margin to prevent acos domain errors
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -0.99999, 0.99999)
        geodesic_dist = torch.acos(cos_angle)

        # Check for NaN in output
        if _DEBUG_NONFINITE and (torch.isnan(geodesic_dist).any()):
            print(f"Warning: NaN in geodesic_dist, trace={trace.mean().item()}, cos_angle={cos_angle.mean().item()}")
            return torch.zeros((), device=pred_orn.device, requires_grad=True)

        return torch.mean(geodesic_dist)

    def _frobenius_loss(self, pred_orn, gt_orn):
        """Frobenius norm loss"""
        diff = pred_orn - gt_orn
        frobenius_norm = torch.norm(diff.reshape(-1, 9), p='fro', dim=1)
        return torch.mean(frobenius_norm)


# ---------------------------------------------------------------------------
# New loss classes for TCP FM head + consistency
# ---------------------------------------------------------------------------

# ---- Reusable SO(3) helpers ----

def _geodesic_distance(R1, R2):
    """Geodesic distance on SO(3) between two batches of rotation matrices.

    Args:
        R1: (B, 3, 3)
        R2: (B, 3, 3)
    Returns:
        (B,) geodesic angles in radians
    """
    R_rel = torch.bmm(R1.transpose(1, 2), R2)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -0.99999, 0.99999)
    return torch.acos(cos_angle)


def _frobenius_distance(R1, R2):
    """Frobenius norm between two batches of rotation matrices.

    Args:
        R1: (B, 3, 3)
        R2: (B, 3, 3)
    Returns:
        (B,) Frobenius norms
    """
    return torch.norm((R1 - R2).reshape(-1, 9), p='fro', dim=1)



def pixel_depth_to_camera_3d(pixel_coords_norm, intrinsics, image_h, image_w,
                              min_depth=0.1, max_depth=5.0):
    """Convert normalized pixel coords (u, v, depth) to 3D camera frame.

    Fully differentiable.  No extrinsics needed.

    Args:
        pixel_coords_norm: [B, 3]  (u_norm, v_norm, d_norm) all in [0, 1].
        intrinsics:        [B, 3, 3] or [B, 9] camera K matrix.
        image_h, image_w:  int – original image resolution before any resize.
        min_depth, max_depth: float – depth denormalization range.

    Returns:
        [B, 3]  (x, y, z) in camera coordinate frame (meters).
    """
    if intrinsics.ndim == 2 and intrinsics.shape[-1] == 9:
        intrinsics = intrinsics.reshape(-1, 3, 3)

    u_norm = pixel_coords_norm[:, 0]   # [B]
    v_norm = pixel_coords_norm[:, 1]   # [B]
    d_norm = pixel_coords_norm[:, 2]   # [B]

    # Denormalize
    u_pixel = u_norm * (image_w - 1)
    v_pixel = v_norm * (image_h - 1)
    depth   = d_norm * (max_depth - min_depth) + min_depth

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    x = (u_pixel - cx) * depth / (fx + 1e-8)
    y = (v_pixel - cy) * depth / (fy + 1e-8)
    z = depth

    return torch.stack([x, y, z], dim=-1)


# Dead-end loss classes removed in cleanup (Mar 25, 2026):
# - CrossViewGeometricContrastiveLoss (R118) — feature matching failed
# - CrossViewTokenContrastiveLoss (R117) — worse than global SimCLR
# - MoCoContrastiveLoss (R75/R81) — worse than SimCLR
# - RelativeGeometryLoss (R119) — CLS-level aux prediction failed

def project_points(K, points_3d, eps=1e-6):
    """
    K: (..., 3, 3) camera intrinsics
    points_3d: (..., 3) in camera coordinates
    return: (..., 2) pixel UV, valid_mask (Z > eps)
    """
    X = points_3d[..., 0]
    Y = points_3d[..., 1]
    Z = points_3d[..., 2]

    valid = Z > eps

    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    Z_safe = torch.clamp(Z, min=eps)
    u = fx * (X / Z_safe) + cx
    v = fy * (Y / Z_safe) + cy
    uv = torch.stack([u, v], dim=-1)

    return uv, valid


class TCPAuxiliaryLoss(nn.Module):
    """TCP auxiliary losses (Ericonaldo, Apr 1 2026).

    Two roles:
    - As a weight holder: sa_policy.py reads self.uv_w / self.pos_w / self.rot_w
      and computes uv/3d/6d losses inline (camera-frame GT + predictions).
    - As a functional module: .forward() also computes `proj_gt` (pred_3d → UV
      must match gt_uv) and `proj_self` (pred_3d → UV must match pred_uv; with
      detach on pred_uv so only the 3d head catches the 2d head). These add
      internal geometric self-consistency between the uv_head and pos3d_head.

    Note: sa_policy.py currently only reads the weight attributes; .forward is not
    invoked (projection bridge unnecessary after GT was moved to camera frame).
    The forward path is kept for future re-activation.
    """

    def __init__(
        self,
        uv_weight=1.0,
        pos3d_weight=1.0,
        rot6d_weight=1.0,
        proj_gt_weight=1.0,
        proj_self_weight=0.1,
    ):
        super().__init__()
        self.uv_w = uv_weight
        self.pos_w = pos3d_weight
        self.rot_w = rot6d_weight
        self.proj_gt_w = proj_gt_weight
        self.proj_self_w = proj_self_weight

    def masked_l1(self, pred, target, mask=None):
        loss = torch.abs(pred - target)
        if mask is None:
            return loss.mean()
        
        mask = mask.to(device=loss.device, dtype=loss.dtype)
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        loss = loss * mask.float()
        denom = mask.float().sum() * loss.shape[-1]
        return loss.sum() / torch.clamp(denom, min=1.0)

    def forward(self, pred_uv, pred_3d, pred_6d, gt_uv, gt_3d, gt_6d, K, valid_mask=None):
        with torch.autocast("cuda", enabled=False):
            loss_uv = self.masked_l1(pred_uv, gt_uv, valid_mask)
            loss_3d = self.masked_l1(pred_3d, gt_3d, valid_mask)
            loss_6d = self.masked_l1(pred_6d, gt_6d, valid_mask)

            proj_uv, proj_valid = project_points(K, pred_3d)
            if valid_mask is not None:
                proj_valid = proj_valid & valid_mask.bool()

            loss_proj_gt = self.masked_l1(proj_uv, gt_uv, proj_valid)
            loss_proj_self = self.masked_l1(proj_uv, pred_uv.detach(), proj_valid)

            total = (
                self.uv_w * loss_uv
                + self.pos_w * loss_3d
                + self.rot_w * loss_6d
                + self.proj_gt_w * loss_proj_gt
                + self.proj_self_w * loss_proj_self
            )

        return {
            "loss": total,
            "loss_uv": loss_uv,
            "loss_3d": loss_3d,
            "loss_6d": loss_6d,
            "loss_proj_gt": loss_proj_gt,
            "loss_proj_self": loss_proj_self,
        }


def rot6d_to_matrix(x):
    """Convert 6D rotation representation to rotation matrix. x: (...,6)"""
    a1 = F.normalize(x[..., 0:3], dim=-1)
    a2_raw = x[..., 3:6]
    a2 = F.normalize(
        a2_raw - (a2_raw * a1).sum(dim=-1, keepdim=True) * a1,
        dim=-1,
    )
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack((a1, a2, a3), dim=-1)  # shape (...,3,3)


def matrix_to_rot6d(mat):
    # Copy from official Rotation matrix to 6D: take first two columns and flatten
    return mat[..., :3, 0:2].reshape(*mat.shape[:-2], 6)


class ActionConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_tcp,
        pred_action,
        gt_tcp,
        gt_action,
        camera_intrinsics=None,
        image_hw=None,
        uv_weight=0.0,
    ):
        """
        pred_tcp: 3d pos + 6d representation pose [B, 9]
        pred_action: 3d pos + 6d representation relative pose [B, T, 9]
        gt_tcp: 3d pos + 6d representation pose [B, 9]
        gt_action: 3d pos + 6d representation relative pose [B, T, 9]

        Return: pred/gt future pose sequence MSE
        """
        with torch.autocast("cuda", enabled=False):
            pred_tcp = pred_tcp.float()
            pred_action = pred_action.float()
            gt_tcp = gt_tcp.float()
            gt_action = gt_action.float()
            if camera_intrinsics is not None:
                camera_intrinsics = camera_intrinsics.float()
            # 计算future pose: 累加相对动作到初始pose
            # 假设3d位置和6d表现在前3和后6
            pred_tcp_pos, pred_tcp_rot6d = pred_tcp[:, :3], pred_tcp[:, 3:]   # [B, 3], [B, 6]
            gt_tcp_pos, gt_tcp_rot6d = gt_tcp[:, :3], gt_tcp[:, 3:]          # [B, 3], [B, 6]

            pred_action_pos, pred_action_rot6d = pred_action[..., :3], pred_action[..., 3:]  # [B, T, 3], [B, T, 6]
            gt_action_pos, gt_action_rot6d = gt_action[..., :3], gt_action[..., 3:]         # [B, T, 3], [B, T, 6]

            # 每步 delta 都以 cur tcp 为起点展开（非链式递推）：p_future[t] = p_cur + delta[t]
            #pred_future_pos = pred_tcp_pos.unsqueeze(1) + pred_action_pos         # [B, T, 3]
            #gt_future_pos = gt_tcp_pos.unsqueeze(1) + gt_action_pos               # [B, T, 3]

            # 这里不能直接将6D旋转向量进行位姿变换。具体地，初始6D旋转先转为旋转矩阵，再与相对旋转（也需转为旋转矩阵）做前向链式乘法，最后再转回6D。此处为SE(3)运动合成。
            # Forward compose in SE(3) for delta-6d (relative rotation)
            B, T = pred_action_rot6d.shape[:2]
            pred_rot = rot6d_to_matrix(pred_tcp_rot6d)   # [B,3,3]
            gt_rot = rot6d_to_matrix(gt_tcp_rot6d)        # [B,3,3]
            pred_action_rotmat = rot6d_to_matrix(pred_action_rot6d)    # [B,T,3,3]
            gt_action_rotmat = rot6d_to_matrix(gt_action_rot6d)        # [B,T,3,3]
            pred_future_pos = pred_tcp_pos.unsqueeze(1) + torch.einsum('bij,btj->bti', pred_rot, pred_action_pos)
            gt_future_pos = gt_tcp_pos.unsqueeze(1) + torch.einsum('bij,btj->bti', gt_rot, gt_action_pos)
            # 前向合成(T steps): 每个delta都是相对于当前初始观测（cur tcp），即所有动作都是基于起始姿态进行累加，而非基于上一步姿态
            # Body/tcp-frame delta composition: R_future[t] = R_delta[t] @ R_cur (left-multiply current by delta)
            # 不做链式递推，即每一步都与初始旋转合成，不是逐步递推
            pred_future_rotmat = torch.matmul(
                pred_rot.unsqueeze(1).expand(-1, T, -1, -1), pred_action_rotmat
            )  # [B, T, 3, 3]
            gt_future_rotmat = torch.matmul(
                gt_rot.unsqueeze(1).expand(-1, T, -1, -1), gt_action_rotmat
            )    # [B, T, 3, 3]

            pred_future_rot6d = matrix_to_rot6d(pred_future_rotmat)  # [B, T, 6]
            gt_future_rot6d = matrix_to_rot6d(gt_future_rotmat)      # [B, T, 6]

            # 拼接future pose [B, T, 9]
            pred_future_pose = torch.cat([pred_future_pos, pred_future_rot6d], dim=-1)
            gt_future_pose = torch.cat([gt_future_pos, gt_future_rot6d], dim=-1)

            pose_loss = F.mse_loss(pred_future_pose, gt_future_pose)
            uv_loss = pose_loss.new_tensor(0.0)
            if camera_intrinsics is not None and image_hw is not None:
                K, (H, W), eps = camera_intrinsics, image_hw, 1e-6
                def proj(p):
                    Z = p[..., 2].clamp(min=eps)
                    u = K[:, 0:1] * p[..., 0] / Z + K[:, 2:3]
                    v = K[:, 1:2] * p[..., 1] / Z + K[:, 3:4]
                    return torch.stack([u / max(W - 1, 1), v / max(H - 1, 1)], dim=-1)
                pred_uv = proj(pred_future_pos)
                gt_uv = proj(gt_future_pos)
                uv_loss = F.mse_loss(pred_uv, gt_uv)

        return {
            "loss": pose_loss + uv_weight * uv_loss,
            "loss_pose": pose_loss,
            "loss_uv": uv_loss,
        }
