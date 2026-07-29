import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class GradientLoss(nn.Module):
    """Calcola la differenza sui gradienti (operatori di Sobel) per incrementare la nitidezza dei bordi."""
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        
        # Replica il filtro per i 3 canali RGB (Depthwise Conv)
        self.register_buffer("kernel_x", kernel_x.repeat(3, 1, 1, 1))
        self.register_buffer("kernel_y", kernel_y.repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1, groups=3)
        
        target_grad_x = F.conv2d(target, self.kernel_x, padding=1, groups=3)
        target_grad_y = F.conv2d(target, self.kernel_y, padding=1, groups=3)
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        return loss_x + loss_y


class MinimalLLLoss(nn.Module):
    def __init__(self, w_charb=1.0, w_ssim=1.0, w_color=0.2, w_grad=0.5):
        super().__init__()
        self.w_charb = w_charb
        self.w_ssim = w_ssim
        self.w_color = w_color
        self.w_grad = w_grad
        
        self.charb = CharbonnierLoss(eps=1e-3)
        self.ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        self.grad = GradientLoss()

    def color_loss(self, pred, target):
        pred_mean = pred.mean(dim=[2, 3])
        target_mean = target.mean(dim=[2, 3])
        return F.l1_loss(pred_mean, target_mean)

    def forward(self, pred, target):
        l_charb = self.charb(pred, target)
        l_ssim = 1.0 - self.ssim(pred, target)
        l_color = self.color_loss(pred, target)
        l_grad = self.grad(pred, target)

        total = (
            self.w_charb * l_charb 
            + self.w_ssim * l_ssim 
            + self.w_color * l_color 
            + self.w_grad * l_grad
        )

        return total, {
            "total": total.detach(),
            "charb": l_charb.detach(),
            "ssim": l_ssim.detach(),
            "color": l_color.detach(),
            "grad": l_grad.detach(),
        }