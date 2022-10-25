import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# load environmental variables in .env file
load_dotenv()

from pathlib import Path

import torch
import torch_optimizer as optim
from loguru import logger
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, MinMetric

import wandb
from groovis import Architecture, Vision
from groovis.data import SIMCLR_AUG_RELAXED, Imagenette
from groovis.data.dataset import Splits
from groovis.loss import SimCLRLoss
from groovis.schema import load_config

# initialize wandb server
# 일반적으로 config 값은 따로 가져와서 wandb에 업데이트 하는 방식으로 진행함..
RUN_NAME = "optimizer-test-2"
wandb.init(
    project="groovis",
    group="first try",
    name=RUN_NAME,
    mode="online",
)
config = load_config("config.yaml")  # 실험 설정 값은 yaml 파일에 저장하여 관리
wandb.config.update(
    OmegaConf.to_container(config)
)  # wandb config를 update할때는, 딕셔너리 타입으로 변환해줄 것!!

Path(f"build/{RUN_NAME}").mkdir(
    parents=True, exist_ok=True
)  # best practice for control file system in python


# 지정 파일에 로그를 저장(add)
logger.add("logs/train_{time}.log")
logger.info(f"Configuration: {config}")

# initialize weight parameter
architecture = Architecture(
    patch_size=config.patch_size, channels=config.channels, embed_dim=config.embed_dim
)

vision = Vision(architecture=architecture)

# logging parameters in Vision instance to wandb server
# Hooks into the torch model to collect gradients and the topology.
wandb.watch(models=vision, log="all", log_freq=config.log_interval)

# load data
splits: list[Splits] = ["train", "validation"]

dataloader: dict[Splits, DataLoader] = {
    split: DataLoader(
        dataset=Imagenette(transforms=SIMCLR_AUG_RELAXED, split=split),
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
    )
    for split in splits
}

# define variables related to train and validation process
epoch_steps = len(dataloader["train"])
total_steps = config.epochs * epoch_steps
warmup_steps = config.warmup_epochs * epoch_steps
global_step = 0
patience = 0

# define loss function and metric
loss_fn = SimCLRLoss(temperature=config.temperature)

metric: dict[Splits, MeanMetric] = {split: MeanMetric() for split in splits}
best_validation_loss = MinMetric()
best_validation_loss.update(1e9)

# set optimizer
optimizer = optim.LARS(params=vision.parameters(), lr=config.base_lr)

# set scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=config.base_lr,
    total_steps=total_steps,
    pct_start=config.warmup_epochs / config.epochs,
    anneal_strategy="linear",
    div_factor=config.base_lr / config.warmup_lr,
    final_div_factor=1e6,
)

for epoch in range(config.epochs):
    # step for training
    for step, images in enumerate(dataloader["train"]):
        images_1, images_2 = images

        # compute representations
        representations_1 = vision(images_1)
        representations_2 = vision(images_2)

        # measure quality
        loss = loss_fn(representations_1, representations_2)

        # calculate gradient
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(
            parameters=vision.parameters(),
            max_norm=config.clip_grad,
            error_if_nonfinite=True,
        )

        # update parameter and learning rate
        optimizer.step()  # equal to parameter -= lr * parameter.grad
        optimizer.zero_grad(set_to_none=True)  # equal to parameter.grad.zero_()
        scheduler.step()  # equal to code for settting learning rate scheduler

        metric["train"].update(loss)

        # optim param_groups: https://ko.n4zc.com/article/huauu78z.html
        lr = optimizer.param_groups[0]["lr"]

        # synchronize log for metric per log interval step(wandb)
        if not (global_step + 1) % config.log_interval:
            wandb.log(
                data={
                    "train": {"lr": lr, "loss": metric["train"].compute()},
                },
                step=global_step,
                commit=False if step == len(dataloader["train"]) - 1 else True,
            )

            metric["train"].reset()

        # logging with loguru
        logger.info(
            f"Train: "
            f"[{epoch}/{config.epochs}][{step}/{len(dataloader['train'])}]\t"
            f"lr {lr:.4f}\t"
            f"loss {loss:.8f}\t"
            # f"grad_norm {grad_norm:.4f}"
        )

        global_step += 1

    # step for validation per epoch
    with torch.no_grad():
        for step, images in enumerate(dataloader["validation"]):
            images_1, images_2 = images

            # compute representations
            representations_1 = vision(images_1)
            representations_2 = vision(images_2)

            # measure quality
            loss = loss_fn(representations_1, representations_2)

            metric["validation"].update(loss)

            logger.info(
                f"Validation: "
                f"[{epoch}/{config.epochs}][{step}/{len(dataloader['validation'])}]\t"
                f"loss {loss:.8f}\t"
            )

    validation_loss = metric["validation"].compute()
    wandb.log(
        data={"validation": {"loss": validation_loss}},
        step=global_step - 1,
        commit=True,
    )

    metric["validation"].reset()

    torch.save(
        {
            "vision": vision.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        f"build/{RUN_NAME}/{epoch:02d}-{validation_loss:.2f}.pth",
    )

    if validation_loss < best_validation_loss.compute():
        patience = 0
        best_validation_loss.update(validation_loss)
    else:
        patience += 1

    if patience == config.patience:
        break
