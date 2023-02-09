import os
import time

import torch
import hydra
import pytorch_lightning as pl
from h_tsp import HTSP_PPO
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


@hydra.main(config_path=".", config_name="config_ppo")
def run(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    cfg.run_name = cfg.run_name or cfg.default_run_name
    
    if not os.path.isabs(cfg.low_level_load_path):
        cfg.low_level_load_path = os.path.join(hydra.utils.get_original_cwd(), cfg.low_level_load_path)

    if not os.path.isabs(cfg.val_data_path):
        cfg.val_data_path = os.path.join(hydra.utils.get_original_cwd(), cfg.val_data_path)

    if cfg.save_dir is None:
        root_dir = (os.getcwd(),)
    elif os.path.isabs(cfg.save_dir):
        root_dir = cfg.save_dir
    else:
        root_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.save_dir)
    root_dir = os.path.join(root_dir, f"{cfg.run_name}")
    log_dir = root_dir

    # build  High Level Agent
    high_level_agent = HTSP_PPO(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/mean_tour_length",
        dirpath=root_dir,
        filename=cfg.encoder_type + "{epoch:02d}-{val/mean_tour_length:.2f}",
        save_top_k=3,
        save_last=True,
        # mode="min",
        every_n_epochs=1,
    )
    tensorboard_logger = TensorBoardLogger(name=cfg.run_name, save_dir=log_dir,)
    loggers = [tensorboard_logger]
    # wandb logger
    if cfg.wandb:
        os.makedirs(os.path.join(os.path.abspath(log_dir), "wandb"), exist_ok=True)
        wandb_logger = WandbLogger(
            name=cfg.run_name,
            save_dir=log_dir,
            project=cfg.wandb_project,
            log_model=False,
            save_code=True,
            group=time.strftime("%Y%m%d", time.localtime()),
            tags=cfg.default_run_name.split("-")[:-1],
        )
        wandb_logger.log_hyperparams(cfg)
        wandb_logger.watch(high_level_agent, log="all", log_freq=10)
        loggers.append(wandb_logger)

    # build trainer
    if cfg.load_path:
        print("--------------------------------------------")
        print(f"Loading model from {cfg.load_path}")
        print("--------------------------------------------")
        high_level_agent.load_state_dict(
            torch.load(os.path.join(hydra.utils.get_original_cwd(), cfg.load_path))[
                "state_dict"
            ]
        )

    trainer = pl.Trainer(
        default_root_dir=root_dir,
        gpus=cfg.gpus,
        strategy="ddp",
        #  sync_batchnorm=True,
        precision=cfg.precision,
        max_epochs=cfg.total_epoch,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        logger=loggers,
        log_every_n_steps=1,
        check_val_every_n_epoch=cfg.val_freq,
        reload_dataloaders_every_n_epochs=1,
    )

    # training and save ckpt
    trainer.fit(high_level_agent)

    trainer.save_checkpoint(
        os.path.join(root_dir, "pretrained_models", "high_level_agent_checkpoint.ckpt")
    )


if __name__ == "__main__":
    run()
