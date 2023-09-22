import argparse
import contextlib
import logging
import os
import sys
import pdb

class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


def main(args, extras) -> None:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    if args.typecheck:
        from jaxtyping import install_import_hook

        install_import_hook("threestudio", "typeguard.typechecked")

    import threestudio
    from threestudio.systems.base import BaseSystem
    from threestudio.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.misc import get_rank
    from threestudio.utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            if not args.gradio:
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                handler.addFilter(ColoredFilter())
            else:
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    # pdb.set_trace()
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    dm = threestudio.find(cfg.data_type)(cfg.data)
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    if args.gradio:
        fh = logging.FileHandler(os.path.join(cfg.trial_dir, "logs"))
        fh.setLevel(logging.INFO)
        if args.verbose:
            fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            CodeSnapshotCallback(
                os.path.join(cfg.trial_dir, "code"), use_version=False
            ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]
        if args.gradio:
            callbacks += [
                ProgressCallback(save_path=os.path.join(cfg.trial_dir, "progress"))
            ]
        else:
            callbacks += [CustomProgressBar(refresh_rate=1)]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # make tensorboard logging dir to suppress warning
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
            CSVLogger(cfg.trial_dir, name="csv_logs"),
        ] + system.get_loggers()
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()

    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )

    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

    # pdb.set_trace()
    # args = Namespace(config='configs/magic123-coarse-sd.yaml', gpu='0', train=True, validate=False, test=False, export=False, gradio=False, verbose=False, typecheck=False)
    # extras = ['data.image_path=load/images/luffy_medal_rgba.png', 'system.prompt_processor.prompt=a high-resolution DSLR image of Luffy medal']
    # cfg = {'name': 'magic123-coarse-sd', 'description': '', 
    # 'tag': 'luffy_medal_rgba.png-a_high-resolution_DSLR_image_of_Luffy_medal', 'seed': 0, 'use_timestamp': True, 
    # 'timestamp': '@20230916-065506', 'exp_root_dir': 'outputs', 'exp_dir': 'outputs/magic123-coarse-sd', 
    # 'trial_name': 'luffy_medal_rgba.png-a_high-resolution_DSLR_image_of_Luffy_medal@20230916-065506', 
    # 'trial_dir': 'outputs/magic123-coarse-sd/luffy_medal_rgba.png-a_high-resolution_DSLR_image_of_Luffy_medal@20230916-065506', 
    # 'n_gpus': 1, 'resume': None, 'data_type': 'single-image-datamodule', 
    # 'data': {'image_path': 'load/images/luffy_medal_rgba.png', 'height': 128, 'width': 128, 
    # 'default_elevation_deg': 0.0, 'default_azimuth_deg': 0.0, 'default_camera_distance': 2.5, 
    # 'default_fovy_deg': 40.0, 'requires_depth': True, 'requires_normal': False, 
    # 'random_camera': {'batch_size': 1, 'height': 128, 'width': 128, 'elevation_range': [-45, 45], 
    # 'azimuth_range': [-180, 180], 'camera_distance_range': [2.5, 2.5], 'fovy_range': [40.0, 40.0], 
    # 'camera_perturb': 0.0, 'center_perturb': 0.0, 'up_perturb': 0.0, 'eval_height': 512, 'eval_width': 512, 
    # 'eval_elevation_deg': 0.0, 'eval_camera_distance': 2.5, 'eval_fovy_deg': 40.0, 'n_val_views': 4, 
    # 'n_test_views': 120}}, 'system_type': 'magic123-system', 'system': {'geometry_type': 'implicit-volume', 
    # 'geometry': {'radius': 1.0, 'normal_type': 'analytic', 'density_bias': 'blob_magic3d', 
    # 'density_activation': 'softplus', 'density_blob_scale': 10.0, 'density_blob_std': 0.5, 
    # 'pos_encoding_config': {'otype': 'ProgressiveBandHashGrid', 'n_levels': 16, 'n_features_per_level': 2, 
    # 'log2_hashmap_size': 19, 'base_resolution': 16, 'per_level_scale': 1.447269237440378, 'start_level': 8, 
    # 'start_step': 2000, 'update_steps': 300}, 
    # 'mlp_network_config': {'otype': 'VanillaMLP', 'activation': 'ReLU', 'output_activation': 'none', 'n_neurons': 64, 
    # 'n_hidden_layers': 3}}, 'material_type': 'no-material', 'material': {'requires_normal': True}, 
    # 'background_type': 'solid-color-background', 'renderer_type': 'nerf-volume-renderer', 
    # 'renderer': {'radius': 1.0, 'estimator': 'occgrid', 'num_samples_per_ray': 256, 'return_comp_normal': True}, 
    # 'guidance_3d_type': 'zero123-unified-guidance', 'guidance_3d': {'guidance_type': 'sds', 'guidance_scale': 5.0, 
    # 'min_step_percent': 0.2, 'max_step_percent': 0.6, 'cond_image_path': 'load/images/luffy_medal_rgba.png', 
    # 'cond_elevation_deg': 0.0, 'cond_azimuth_deg': 0.0, 'cond_camera_distance': 2.5}, 
    # 'prompt_processor_type': 'stable-diffusion-prompt-processor', 
    # 'prompt_processor': {'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5', 
    # 'prompt': 'a high-resolution DSLR image of Luffy medal'}, 
    # 'guidance_type': 'stable-diffusion-unified-guidance', 
    # 'guidance': {'guidance_type': 'sds', 'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5', 
    # 'guidance_scale': 100.0, 'min_step_percent': 0.2, 'max_step_percent': 0.6}, 
    # 'loggers': {'wandb': {'enable': False, 'project': 'threestudio', 'name': 'None'}}, 
    # 'loss': {'lambda_rgb': 1000.0, 'lambda_mask': 100.0, 'lambda_sd': 0.025, 'lambda_3d_sd': 1.0, 
    # 'lambda_orient': 2, 'lambda_normal_smoothness_2d': 1000.0, 'lambda_sparsity': 0.0, 'lambda_opaque': 0.0, 
    # 'lambda_depth': 0.0, 'lambda_depth_rel': 0.2, 'lambda_normal': 0.0}, 
    # 'optimizer': {'name': 'Adam', 'args': {'lr': 0.01, 'betas': [0.9, 0.99], 'eps': 1e-08}, 
    # 'params': {'geometry.encoding': {'lr': 0.01}, 'geometry.density_network': {'lr': 0.001}, 
    # 'geometry.feature_network': {'lr': 0.001}}}}, 
    # 'trainer': {'max_steps': 5000, 'log_every_n_steps': 1, 'num_sanity_val_steps': 0, 'val_check_interval': 100, 
    # 'enable_progress_bar': True, 'precision': 32}, 'checkpoint': {'save_last': True, 'save_top_k': -1, 
    # 'every_n_train_steps': 5000}}
    if args.train:
        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        trainer.test(system, datamodule=dm)
        if args.gradio:
            # also export assets if in gradio mode
            trainer.predict(system, datamodule=dm)
    elif args.validate:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.test:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.export:
        set_system_status(system, cfg.resume)
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    parser.add_argument(
        "--gradio", action="store_true", help="if true, run in gradio mode"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )

    args, extras = parser.parse_known_args()
    # pdb.set_trace()
    if args.gradio:
        # FIXME: no effect, stdout is not captured
        with contextlib.redirect_stdout(sys.stderr):
            main(args, extras)
    else:
        main(args, extras)
