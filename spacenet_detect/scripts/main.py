from pathlib import Path
from args import *
from UNet_monai import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import SpaceNet6DataModule


if __name__ == "__main__":
    args = get_main_args()
    assert args.base_dir, "Data directory not specified"

    callbacks = []
    model = Unet(args)

    if args.exec_mode == 'train':
        Path(args.ckpt_path).mkdir(parents=True, exist_ok=True)
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    model_ckpt = ModelCheckpoint(dirpath="./checkpoints/save", filename="best_model",
                                monitor="dice_mean", mode="max", save_last=True)
    callbacks.append(model_ckpt)
    dm = SpaceNet6DataModule(args)
    trainer = Trainer(callbacks=callbacks, enable_checkpointing=True, max_epochs=args.num_epochs, 
                    enable_progress_bar=True, gpus=1, accelerator="cpu", amp_backend='apex', profiler='simple')

    # train the model
    if args.exec_mode == 'train':
        trainer.fit(model, dm)
    elif args.exec_mode == 'predict':
        trainer.predict(model, datamodule=dm, ckpt_path=args.ckpt_path)
    else:
	    AssertionError("Exec mode must be either train or predict") 
