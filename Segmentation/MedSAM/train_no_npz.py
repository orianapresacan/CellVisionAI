import argparse
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import monai
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from typing import Any, Iterable
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.dataset import MedSamDataset
import matplotlib.pyplot as plt

def dice_score(preds, targets):
    smooth = 1.0
    assert preds.size() == targets.size()

    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    dice = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice


class TrainMedSam:
    BEST_VAL_LOSS = float("inf")
    BEST_EPOCH = 0

    def __init__(
        self,
        lr: float = 3e-4,
        batch_size: int = 4,
        epochs: int = 100,
        device: str = "cuda:0",
        model_type: str = "vit_h",
        image_dir="data/train/images",
        mask_dir="data/train/masks",
        checkpoint: str = "sam_vit_h_4b8939.pth",
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.sam_checkpoint_dir = checkpoint
        self.model_type = model_type

    def __call__(self, train_df, test_df, val_df, image_col, mask_col):
        """Entry method
        prepare `dataset` and `dataloader` objects

        """
        train_ds = MedSamDataset(
            train_df,
            image_col,
            mask_col,
            self.image_dir,
            self.mask_dir,
        )
        val_ds = MedSamDataset(
            val_df,
            image_col,
            mask_col,
            self.image_dir,
            self.mask_dir,
        )
        test_ds = MedSamDataset(
            test_df,
            image_col,
            mask_col,
            self.image_dir,
            self.mask_dir,
        )
        # Define dataloaders
        train_loader = DataLoader(
            dataset=train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            dataset=val_ds, batch_size=self.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            dataset=test_ds, batch_size=self.batch_size, shuffle=False
        )
        # get the model
        model = self.get_model()
        model.to(self.device)

        # Train and evaluate model
        self.train(model, train_loader, val_loader)
        # Evaluate model on test data
        loss, dice_score = self.test(model, test_loader, desc="Testing")

        del model
        torch.cuda.empty_cache()

        self.BEST_EPOCH = 0
        self.BEST_VAL_LOSS = float("inf")

        return dice_score

    def get_model(self):
        sam_model = sam_model_registry[self.model_type](
            checkpoint=self.sam_checkpoint_dir
        ).to(self.device)

        return sam_model

    @torch.inference_mode()
    def evaluate(self, model, val_loader, desc="Validating") -> float:
        """Perform evaluation on test or validation data
        Args:
            model (_type_): pytorch model
            val_loader (_type_): dataloader
            desc (str, optional): _description_. Defaults to "Validating".

        Returns:
            np.array: (mean validation loss, mean validation dice)
        """
        seg_loss = monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )

        progress_bar = tqdm(val_loader, total=len(val_loader))
        val_loss = []
        val_dice = []

        for image, mask, bbox in progress_bar:
            image = image.to(self.device)
            mask = mask.to(self.device)
            # resize image to 1024 by 1024
            image = TF.resize(image, (1024, 1024), antialias=True)
            
            H, W = mask.shape[-2], mask.shape[-1]

            sam_trans = ResizeLongestSide(model.image_encoder.img_size)

            box = sam_trans.apply_boxes(bbox, (H, W))
            box_tensor = torch.as_tensor(box, dtype=torch.float, device=self.device)

            # Get predictioin mask

            image_embeddings = model.image_encoder(image)  # (B,256,64,64)

            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=box_tensor,
                masks=None,
            )

            mask_predictions, _ = model.mask_decoder(
                image_embeddings=image_embeddings.to(self.device),  # (B, 256, 64, 64)
                image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            mask_predictions = (mask_predictions > 0.5).float()

            low_res_pred = F.interpolate(
                mask_predictions,
                size=(2048, 2048),
                mode="bilinear",
                align_corners=False,
            )  # (1, 1, gt.shape)
            # low_res_pred = low_res_pred.squeeze().detach().cpu().numpy()  # (256, 256)
            # low_res_pred = (low_res_pred > 0.5).astype(np.uint8)

            # get the dice loss
            loss = seg_loss(low_res_pred, mask)
            
            low_res_pred = (low_res_pred > 0.5).float()
            dice = dice_score(low_res_pred, mask)

            val_loss.append(loss.detach().item())
            val_dice.append(dice.detach().item())

            # Update the progress bar
            progress_bar.set_description(desc)
            progress_bar.set_postfix(
                eval_loss=np.mean(val_loss), eval_dice=np.mean(val_dice)
            )
            progress_bar.update()
        return np.mean(val_loss), np.mean(val_dice)

    @torch.inference_mode()
    def test(self, model, val_loader, desc="Testing") -> float:
        """Perform evaluation on test or validation data
        Args:
            model (_type_): pytorch model
            val_loader (_type_): dataloader
            desc (str, optional): _description_.

        Returns:
            float: mean validation loss
        """
        seg_loss = monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )
        progress_bar = tqdm(val_loader, total=len(val_loader))
        val_loss = []
        dice_scores = []

        for image, mask, bbox in progress_bar:
            image = image.to(self.device)
            mask = mask.to(self.device)
            # resize image to 1024 by 1024
            image = TF.resize(image, (1024, 1024), antialias=True)
            H, W = mask.shape[-2], mask.shape[-1]
            sam_trans = ResizeLongestSide(model.image_encoder.img_size)

            box = sam_trans.apply_boxes(bbox, (H, W))
            box_tensor = torch.as_tensor(box, dtype=torch.float, device=self.device)
            # Get predictioin mask
            image_embeddings = model.image_encoder(image)  # (B,256,64,64)

            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=box_tensor,
                masks=None,
            )

            mask_predictions, _ = model.mask_decoder(
                image_embeddings=image_embeddings.to(self.device),  # (B, 256, 64, 64)
                image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            low_res_pred = F.interpolate(
                mask_predictions,
                size=(2048, 2048),
                mode="bilinear",
                align_corners=False,
            )  # (1, 1, gt.shape)
            # low_res_pred = low_res_pred.squeeze().detach().cpu().numpy()  # (256, 256)
            # low_res_pred = (low_res_pred > 0.5).astype(np.uint8)

            # get the dice loss
            loss = seg_loss(low_res_pred, mask)

            low_res_pred = (low_res_pred > 0.5).float()
            dice = dice_score(low_res_pred, mask)

            val_loss.append(loss.item())
            dice_scores.append(dice.detach().item())

            # Update the progress bar
            progress_bar.set_description(desc)
            progress_bar.set_postfix(
                eval_loss=np.mean(val_loss), eval_dice=np.mean(dice_scores)
            )
            progress_bar.update()
        return np.mean(val_loss), np.mean(dice_scores)

    def train(self, model, train_loader: Iterable, val_loader: Iterable, logg=True):
        """Train the model"""

        sam_trans = ResizeLongestSide(model.image_encoder.img_size)
        writer = SummaryWriter()

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.01, verbose=True
        )
        seg_loss = monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )
        model.train()
        for epoch in range(self.epochs):
            epoch_loss = []
            epoch_dice = []
            progress_bar = tqdm(train_loader, total=len(train_loader))
            for image, mask, bbox in progress_bar:
                image = image.to(self.device)
                mask = mask.to(self.device)
                # resize image to 1024 by 1024
                image = TF.resize(image, (1024, 1024), antialias=True)
                H, W = mask.shape[-2], mask.shape[-1]

                box = sam_trans.apply_boxes(bbox, (H, W))
                box_tensor = torch.as_tensor(box, dtype=torch.float, device=self.device)

                # Get predictioin mask
                with torch.inference_mode():
                    image_embeddings = model.image_encoder(image)  # (B,256,64,64)

                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=None,
                        boxes=box_tensor,
                        masks=None,
                    )

                mask_predictions, _ = model.mask_decoder(
                    image_embeddings=image_embeddings.to(
                        self.device
                    ),  # (B, 256, 64, 64)
                    image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
                )
                low_res_pred = F.interpolate(
                    mask_predictions,
                    size=(2048, 2048),
                    mode="bilinear",
                    align_corners=False,
                )  # (1, 1, gt.shape)
                # low_res_pred = low_res_pred.squeeze().detach().cpu().numpy()  # (256, 256)
                # low_res_pred = (low_res_pred > 0.5).astype(np.uint8)

            
                # Calculate loss
                loss = seg_loss(low_res_pred, mask)
                low_res_pred = (low_res_pred > 0.5).float()
                
                dice = dice_score(low_res_pred, mask)
              
                # first_image1 = low_res_pred[0, 0, :, :]
                # first_image2 = mask[0, 0, :, :]

                # # Convert the torch tensors to numpy arrays
                # first_image_np1 = first_image1.cpu().detach().numpy()
                # first_image_np2 = first_image2.cpu().detach().numpy()

                # # Create a figure with two subplots
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                # # Plot the first image on the first subplot
                # axes[0].imshow(first_image_np1, cmap='gray')  # Assuming the image is grayscale
                # axes[0].axis('off')  # Remove axis

                # # Plot the second image on the second subplot
                # axes[1].imshow(first_image_np2, cmap='gray')  # Assuming the image is grayscale
                # axes[1].axis('off')  # Remove axis

                # plt.show()
                # exit()

                epoch_loss.append(loss.detach().item())
                epoch_dice.append(dice.detach().item())

                # empty gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.set_description(f"Epoch {epoch+1}/{self.epochs}")
                progress_bar.set_postfix(
                    loss=np.mean(epoch_loss), dice=np.mean(epoch_dice)
                )
                progress_bar.update()
            # Evaluate every two epochs
            if epoch % 1 == 0:
                self.save_model(model, epoch=epoch)
                validation_loss, validation_dice = self.evaluate(
                    model, val_loader, desc=f"Validating"
                )
                scheduler.step(torch.tensor(validation_loss))

                if self.early_stopping(model, validation_loss, epoch):
                    print(f"[INFO:] Early Stopping!!")
                    break

            if logg:
                writer.add_scalars(
                    "loss",
                    {
                        "train": round(np.mean(epoch_loss), 4),
                        "val": round(validation_loss, 4),
                    },
                    epoch,
                )

                writer.add_scalars(
                    "dice",
                    {
                        "train": round(np.mean(epoch_dice), 4),
                        "val": round(validation_dice, 4),
                    },
                    epoch,
                )

    def save_model(self, model, epoch):
        date_postfix = datetime.now().strftime("%Y-%m-%d-%H-%S")
        model_name = f"medsam_finetune_{epoch}.pth"
        save_path = "finetune_weights"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        print(f"[INFO:] Saving model to {os.path.join(save_path,model_name)}")
        torch.save(model.state_dict(), os.path.join(save_path, model_name))

    def early_stopping(
        self,
        model,
        val_loss: float,
        epoch: int,
        patience: int = 10,
        min_delta: int = 0.001,
    ):
        """Helper function for model training early stopping
        Args:
            val_loss (float): _description_
            epoch (int): _description_
            patience (int, optional): _description_. Defaults to 10.
            min_delta (int, optional): _description_. Defaults to 0.01.
        """

        if self.BEST_VAL_LOSS - val_loss >= min_delta:
            print(
                f"[INFO:] Validation loss improved from {self.BEST_VAL_LOSS} to {val_loss}"
            )
            self.BEST_VAL_LOSS = val_loss
            self.BEST_EPOCH = epoch
            # self.save_model(model)
            return False

        if (
            self.BEST_VAL_LOSS - val_loss < min_delta
            and epoch - self.BEST_EPOCH >= patience
        ):
            return True
        return False


class CrossValidate(TrainMedSam):
    def __init__(
        self,
        lr: float = 3e-4,
        batch_size: int = 4,
        epochs: int = 100,
        device: str = "cuda:0",
        model_type: str = "vit_h",
        image_dir="data/train/images",
        mask_dir="data/train/masks",
        checkpoint: str = "sam_vit_h_4b8939.pth",
    ):
        super().__init__(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            model_type=model_type,
            image_dir=image_dir,
            mask_dir=mask_dir,
            checkpoint=checkpoint,
        )

    def __call__(self, train_df, test_df, image_col, mask_col, k: int = 5) -> Any:
        """Performs kfold cross validation
        Args:
            k (int, optional): Fold size. Defaults to 5.
        """
        # Define the cross-validation splitter
        kf = KFold(n_splits=k, shuffle=True)
        # loop over each fold
        fold_scores = {}
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            print(f"Cross validating for fold {fold + 1}")
            # Define training and validation sets for this fold
            f_train_df = train_df.iloc[train_idx]
            f_val_df = train_df.iloc[val_idx]

            dice_score = super().__call__(
                f_train_df, test_df, f_val_df, image_col, mask_col
            )

            fold_scores[f"fold_{fold + 1}_mean_dice"] = dice_score

        return fold_scores


def main():
    # set up parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=False, help="Path to the CSV file", default="data/dataset.csv")

    parser.add_argument(
        "--image_col",
        type=str,
        default='image_filename',
        help="Name of the column on the dataframe that holds the image file names",
    )

    parser.add_argument(
        "--mask_col",
        type=str,
        default='mask_filename',
        help="the name of the column on the dataframe that holds the mask file names",
    )
    parser.add_argument(
        "--image", type=str, required=False, help="Path to the input image directory"
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=False,
        help="Path to the ground truth mask directory",
    )
    parser.add_argument(
        "--num_epochs", type=int, required=False, default=100, help="number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, required=False, default=3e-4, help="learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=8, help="batch size"
    )
    parser.add_argument(
        "-k",
        type=int,
        default=None,
        required=False,
        help="Number of folds for cross validation",
    )
    parser.add_argument("--model_type", type=str, required=False, default="vit_h")
    parser.add_argument(
        "--checkpoint", type=str, required=False, help="Path to SAM checkpoint", default='sam_vit_h_4b8939.pth'
    )

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"{args.csv} does not exist")

    # split the dataset into train and test set
    train_df, test_df = train_test_split(df, train_size=0.8, random_state=2023)
    # if `k` argument is specified, run the cross validation
    if args.k:
        print(f"[INFO] Starting {args.k} fold cross validation ....")
        if args.k < 5:
            raise ValueError("K should be a value greater than or equal to 5")

        cross_validate = CrossValidate(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.num_epochs,
            image_dir=args.image,
            mask_dir=args.mask,
            checkpoint=args.checkpoint,
        )
        scores = cross_validate(
            train_df, test_df, args.image_col, args.mask_col, args.k
        )

        # write cross-validation scores to file
        with open("medsam.json", "w") as f:
            json.dump(scores, f)
    # if `k` is not specified, normal training mode
    if not args.k:
        print(f"[INFO] Starting training for {args.num_epochs} epochs ....")
        # create a validation set
        train_df, val_df = train_test_split(train_df, train_size=0.8, random_state=2023)

        train = TrainMedSam(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.num_epochs,
            image_dir=args.image,
            mask_dir=args.mask,
            checkpoint=args.checkpoint,
        )

        train(train_df, test_df, val_df, args.image_col, args.mask_col)


if __name__ == "__main__":
    main()
