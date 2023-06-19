import sys
import shutil
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.registry import register_model
from compressai.optimizers import net_aux_optimizer
from compressai.losses import RateDistortionLoss
# from compressai.ans import BufferedRansEncoder, RansDecoder

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.registry import register_model

from compressai.models.base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from compressai.models.utils import conv, deconv
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):  # d == images
        d = d.to(device)

        optimizer.zero_grad()  # reset gradient 清空过往梯度
        aux_optimizer.zero_grad()

        out_net = model(d)  # outputs

        out_criterion = criterion(out_net, d)  # loss 计算损失函数

        # backward
        out_criterion["loss"].backward()  # backward 反向传播，计算当前梯度
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()  # optimizer 根据梯度更新网络参数

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

def configure_optimizers(net):
    learning_rate = 1e-4
    aux_learning_rate = 1e-3
    conf = {
        "net": {"type": "Adam", "lr": learning_rate},
        "aux": {"type": "Adam", "lr": aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]

@register_model("cheng2020-attn")
class Cheng2020Attention(CompressionModel):

    def __init__(self, N=192, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),  # 注意力机制
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),  # 注意力机制
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 8 // 3, N * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            N, 2 * N, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)  # 高斯建模
        self.N = int(N)

    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)  # cat(dim=1)按照第二维(列)拼接
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)  # chunk(a,b) : a表示分成的块数，b=0按横向分割，b=1按纵向分割
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

cfgs = {
    "cheng2020-attn": {
            1: (128,),  # 如果元组中只有一个元素的话，需要在元素的后面加上一个逗号,否则视为int
            2: (128,),
            3: (128,),
            4: (192,),
            5: (192,),
            6: (192,),
        },
}

def _load_model(
    quality, pretrained=False, progress=True, **kwargs
):
    model = Cheng2020Attention(*cfgs["cheng2020-attn"][quality], **kwargs)
    return model

def main(argv):
    # 超参数
    patch_size = (256, 256)
    dataset = '/root/autodl-fs/datasets/'
    batch_size = 16
    num_workers = 8
    test_batch_size = 64
    lmbda = 1e-2
    epochs = 12
    clip_max_norm = 1.0
    checkpoint = 'checkpoint_best_loss.pth.tar'

    # 数据预处理：数据集中的图片裁剪
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(patch_size), transforms.ToTensor()]
    )

    # from compressai.datasets import ImageFolder
    train_dataset = ImageFolder(dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(dataset, split="test", transform=test_transforms)

    # 使用GPU进行运算
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载训练集和测试集
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # 1、加载模型
    net = _load_model(quality=3)
    net = net.to(device)

    # 并行运算
    if torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    # 2、加载优化算法Adam，用到了learning_rate,aux_learning_rate,调用了from compressai.optimizers import net_aux_optimizer
    optimizer, aux_optimizer = configure_optimizers(net)
    # 调整学习率，当监控的数量停止减少时，lr会减少，用的是torch中的函数
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # 3、加载学习准则，调用的from compressai.losses import RateDistortionLoss
    criterion = RateDistortionLoss(lmbda=lmbda)

    # 加载之前没训练完的结点，如果有，接着训练
    last_epoch = 0
    if checkpoint:  # load from previous checkpoint
        print("Loading", checkpoint)
        checkpoint = torch.load(checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    print("last_epoch = ", last_epoch)
    # 开始训练
    best_loss = float("inf")
    for epoch in range(last_epoch, epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,  # 损失标准
            train_dataloader,
            optimizer,
            aux_optimizer,  # 辅助优化器
            epoch,
            clip_max_norm,  # 梯度裁剪
        )
        # 测试集上计算loss
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        # 7、损失收敛后训练完成，保存模型以及相应的节点
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": net.state_dict(),  # 训练的权重
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            is_best,
        )

if __name__ == "__main__":
    main(sys.argv[1:])