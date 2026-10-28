import argparse
import os
import torch
import pandas as pd
import numpy as np
import gc

from torch.utils.data import ConcatDataset
from utils.model_utils import generate_concept_logits
from datasets.concept_dataset import ConceptDataset
from models.full_models import joint_model, standard_model, concepts_model, main_model
from trainers.trainer_cbm import CBMTrainer
from evaluators.evaluator_cbm import CBMEvaluator

def read_concept_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    base = os.path.dirname(os.path.abspath(csv_path))

    def resolve(p):
        if pd.isna(p) or p == "":
            return p  # good images have no mask
        return os.path.normpath(os.path.join(base, p))

    for col in ("image_path", "mask_path"):
        if col in df.columns:
            df[col] = df[col].apply(resolve)
    return df

def train_model(category: str,
                dataset_path: str, #main training dataframe path
                use_gen_anomalies: bool,
                contaminate: bool,  #add some original images to the training set of generated anomalies
                n_per_type: int,  #number of original images per defect to add
                model_type: str,
                save_dir: str,
                device: torch.device,
                lambda_: float,
                batch_size: int = 16,
                optimizer: str = "adam",
                lr: float = 1e-3,
                epochs: int = 100,
                model_path: str = None, #path to pretrained backbone model
                save_path_pred_concepts: str = None, #path to df of predicted concepts
                seed: int = 42):

    def init_optimizer(params):
        if optimizer == "adam":
            opt = torch.optim.Adam(params, lr=lr)
        elif optimizer == "sgd":
            opt = torch.optim.SGD(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5)
        return opt, scheduler

    # construct save path
    if use_gen_anomalies:
        file_name = "unsupervised_model.pth"
        if contaminate:
            file_name = f"semisupervised_{n_per_type}_model.pth"
    else:
        file_name = "supervised_model.pth"

    if save_dir:
        if model_type =="joint":
            save_path = os.path.join(save_dir, file_name)
        elif model_type in ["sequential", "independent"]:
            save_path = os.path.join(save_dir, file_name)
            save_path_concepts = os.path.join(save_dir, f"concepts_{file_name}.pth")
    else:
        save_path = None

    print(f"\nTraining {model_type} model for {category} category...")

    if use_gen_anomalies:
        dataframe_path = os.path.join(dataset_path, category, "concepts", f"{category}_dataset_gen.csv")
        if contaminate:
            dataframe_path_original = os.path.join(dataset_path, category, "concepts", f"{category}_dataset_gen_to_real.csv")
    else:
        dataframe_path = os.path.join(dataset_path, category, "concepts", f"{category}_dataset.csv")

    dataframe = read_concept_csv(dataframe_path)
    dataframe_original = read_concept_csv(dataframe_path_original) if contaminate else None

    state_dict = torch.load(model_path) if model_path else None

    train_dataset = ConceptDataset(dataframe, "train", contaminate=contaminate,
                                   n_per_type=n_per_type, original_df=dataframe_original, random_state=seed)
    val_dataset = ConceptDataset(dataframe, "val", random_state=seed)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

    num_attr = len(train_dataset.attr_cols)

    if model_type == "independent":
        #build dataset without images for main task prediction
        train_dataset_no_img = ConceptDataset(dataframe, "train", load_image=False, contaminate=contaminate,
                                            n_per_type=n_per_type, original_df=dataframe_original, random_state=seed)
        val_dataset_no_img = ConceptDataset(dataframe, "val", load_image=False, random_state=seed)

        train_dataloader_no_img = torch.utils.data.DataLoader(train_dataset_no_img, batch_size)
        val_dataloader_no_img = torch.utils.data.DataLoader(val_dataset_no_img, batch_size)

    print(f"\nNumber of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    imbalance_ratio, contamination_ratio = train_dataset.find_class_imbalance("main")
    print("Imbalance Ratio (negatives per positive) in training set:", imbalance_ratio)
    print("Contamination ratio in training set:", contamination_ratio)

    imbalance_ratio_attr = train_dataset.find_class_imbalance("attributes")

    if model_type == "joint":
        model = joint_model(num_attr=num_attr, use_relu=True, use_sigmoid=False, model_state_dict=state_dict)
        # save train and val dataframe in the state dict
        model.train_df = pd.concat([train_dataset.df, val_dataset.df], ignore_index=True)
        model.to(device)
        optimizer, scheduler = init_optimizer(model.parameters())
        trainer = CBMTrainer(model, num_attr, train_dataloader, val_dataloader, optimizer, scheduler,
                             device, lambda_=lambda_, num_epochs=epochs, weight_main=imbalance_ratio,
                             weight_attr=imbalance_ratio_attr, save_path=save_path)

        val_main_f1, val_attr_f1 = trainer.train()

    elif model_type == "independent" or model_type == "sequential":
        # common first step: attribute prediction
        concept_model = concepts_model(num_attr=num_attr, model_state_dict=state_dict)
        concept_model.to(device)
        concept_optimizer, concept_scheduler = init_optimizer(concept_model.parameters())

        trainer_concepts = CBMTrainer(concept_model, num_attr, train_dataloader, val_dataloader, concept_optimizer,
                                      concept_scheduler,
                                      device, lambda_=lambda_, num_epochs=epochs, concepts=True,
                                      bottleneck=True, weight_attr=imbalance_ratio_attr, save_path=save_path_concepts)

        val_attr_f1 = trainer_concepts.train()

        # main model: common
        main_task_model = main_model(num_attr=num_attr, expand_dim=8)
        main_task_model.to(device)
        main_optimizer, main_scheduler = init_optimizer(main_task_model.parameters())

        if model_type == "sequential":
            # generate (and save) concept logits
            dataframe_new = generate_concept_logits(concept_model, dataframe, save_path=save_path_pred_concepts, device=device)

            train_dataset_no_img = ConceptDataset(dataframe_new, "train", load_image=False, contaminate=contaminate,
                                                n_per_type=n_per_type, original_df=dataframe_original, random_state=seed)
            val_dataset_no_img = ConceptDataset(dataframe_new, "val", load_image=False, random_state=seed)

            train_dataloader_no_img = torch.utils.data.DataLoader(train_dataset_no_img, batch_size)
            val_dataloader_no_img = torch.utils.data.DataLoader(val_dataset_no_img, batch_size)

        trainer_main = CBMTrainer(main_task_model, num_attr, train_dataloader_no_img, val_dataloader_no_img,
                                  main_optimizer, main_scheduler, device, lambda_=lambda_, num_epochs=epochs,
                                  concepts=False, main_only=True, weight_main=imbalance_ratio, save_path=save_path)

        val_main_f1 = trainer_main.train()

    torch.cuda.empty_cache()
    gc.collect()


def test_model(category: str,
               dataset_path: str,
               use_gen_anomalies: bool,
               contaminate: bool,
               n_per_type: int,
               model_type: str,
               save_dir: str,
               device: torch.device,
               batch_size: int = 16,
               mode: str = "eval",
               image_path: str = None,):

    # construct save path
    if use_gen_anomalies:
        file_name = "unsupervised_model.pth"
        if contaminate:
            file_name = f"semisupervised_{n_per_type}_model.pth"
    else:
        file_name = "supervised_model.pth"

    if model_type == "joint":
        save_path = os.path.join(save_dir, file_name)
        save_path_concepts = None
    elif model_type in ["sequential", "independent"]:
        save_path = os.path.join(save_dir, file_name)
        save_path_concepts = os.path.join(save_dir, f"concepts_{file_name}.pth")


    print(f"\nTesting {model_type} model for {category} category...")

    if use_gen_anomalies:
        dataframe_path = os.path.join(dataset_path, category, "concepts", f"{category}_dataset_gen_to_real.csv")
    else:
        dataframe_path = os.path.join(dataset_path, category, "concepts", f"{category}_dataset.csv")

    dataframe = read_concept_csv(dataframe_path)

    print(f"Loading state dict from {save_path}")

    state_dict = torch.load(save_path, weights_only=False) if save_path else None
    state_dict_concepts = torch.load(save_path_concepts) if save_path_concepts else None

    test_dataset = ConceptDataset(dataframe, "test")
    num_attr = len(test_dataset.attr_cols)
    attr_cols = test_dataset.attr_cols

    if use_gen_anomalies:
        val_dataset = ConceptDataset(dataframe, "val")
        train_dataset = ConceptDataset(dataframe, "train")
        test_df = pd.concat([train_dataset.df, val_dataset.df, test_dataset.df], ignore_index=True)

        train_df = state_dict.get("train_df", None) if isinstance(state_dict, dict) else None
        if train_df is not None:
            # remove from train_df the test split
            train_df = train_df[train_df["split"].isin(["train", "val"])]

            # to check occurrences, we can use the image paths, but only the filename and dirname of the parent folder and not the full path
            def get_rel_path(path):
                return os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))

            train_df["rel_path"] = train_df["image_path"].apply(get_rel_path)
            # find original anomalies which are the ones that don't have "gen" in the filepath
            train_df = train_df[~train_df["image_path"].str.contains("gen")]

            test_df["rel_path"] = test_df["image_path"].apply(get_rel_path)

            # remove rows that occur both in train_df and dataframe (test set)
            test_df = test_df[~test_df["rel_path"].isin(train_df["rel_path"])]
            test_df = test_df.drop(columns=["rel_path"])

        # repeated code
        test_dataset = ConceptDataset(test_df, "test")
        num_attr = len(test_dataset.attr_cols)
        attr_cols = test_dataset.attr_cols
        val_dataset = ConceptDataset(test_df, "val")
        train_dataset = ConceptDataset(test_df, "train")
        test_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    if mode == "eval":
        print(f"\nNumber of test images: {len(test_dataset)}")
    elif mode == "inference":
        print(f"\nPerforming inference on image stored in {image_path}...")

    if model_type == "joint":
        model = joint_model(num_attr=num_attr, use_relu=True, use_sigmoid=False, model_state_dict=state_dict, mode="test")

        model.to(device)

        evaluator = CBMEvaluator(model, num_attr, attr_cols, test_dataloader, device)

        if mode == "eval":
            auc_main, f1_main, auc_attr, f1_attr, image_preds = evaluator.evaluate()
        elif mode == "inference":
            evaluator.inference(image_path)

    elif model_type in ["independent", "sequential"]:
        # first step: attribute prediction
        concept_model = concepts_model(num_attr=num_attr, model_state_dict=state_dict_concepts, mode="test")
        concept_model.to(device)

        concept_evaluator = CBMEvaluator(concept_model, num_attr, attr_cols, test_dataloader, device, concepts=True, bottleneck=True)

        if mode == "eval":
            auc_attr, f1_attr = concept_evaluator.evaluate()
        elif mode == "inference":
            concept_evaluator.inference(image_path)

        # second step: main prediction
        main_task_model = main_model(num_attr=num_attr, model_state_dict=state_dict)
        main_task_model.to(device)

        # generate concept logits
        dataframe_new = generate_concept_logits(concept_model, dataframe, splits=["test"], save_path=None, device=device)
        test_dataset_no_img = torch.utils.data.DataLoader(dataframe_new, "test", load_image=False)
        test_dataloader_no_img = torch.utils.data.DataLoader(test_dataset_no_img, batch_size)

        main_evaluator = CBMEvaluator(main_task_model, num_attr, attr_cols, test_dataloader_no_img, device, main_only=True)

        if mode == "eval":
            auc_main, f1_main = main_evaluator.evaluate()

        elif mode == "inference":
            main_evaluator.inference(image_path)

    if mode == "eval":
        return auc_main, auc_attr, f1_main, f1_attr
    else:
        return None, None, None, None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=["train", "eval", "inference"], help="Whether to train, test or perform inference on one image")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--use_gen_anomalies", action="store_true", help="Unsupervised or semi-supervised scenario: use generated anomalies to train")
    parser.add_argument("--contaminate", action="store_true",
                        help="Semi-supervised scenario: contaminate the generated dataset with some original anomalous samples")
    parser.add_argument("--n_per_type", type=int, default=0,
                        help="How many original images to add to the generated dataset")
    parser.add_argument("--model_type", type=str, default="joint",
                        help="Which model to train, e.g. 'independent', 'joint', ...")
    parser.add_argument("--save_dir", type=str, default=None, help = "Directory to save the models in")
    parser.add_argument("--category", type=str, help="Which category to train/test")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size to use")
    parser.add_argument("--lambda_", type=float, default=0.55, help="How much weight to give to the attributes")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate to use")
    parser.add_argument("--epochs", type=int, default=100, help="How many epochs to run the training for")
    parser.add_argument("--model_path", type=str, default=None,
                        help="If specified, loads the state dict of a pretrained backbone")
    parser.add_argument("--save_path_pred_concepts", action="store_true", help="Whether to save the predicted concepts dataframe here")
    parser.add_argument("--seed", type=int, default=42, help="Execution seed")
    parser.add_argument("--image_path", default=None, help="Path of the image to perform inference on")
    parser.add_argument("--gemini_logo_mask_path", default=None,
                        help="Path to the Gemini logo mask to be applied to all images")

    args = parser.parse_args()

    #set the gemini logo mask as an environment variable
    os.environ["GEMINI_LOGO_MASK_PATH"] = args.gemini_logo_mask_path or ""

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model_save_dir = os.path.join(args.save_dir, args.category, "cbm", args.model_type) if args.save_dir else None

    if args.mode == "train":
        os.makedirs(model_save_dir, exist_ok=True) if args.save_dir else None
        train_model(category=args.category,
                    dataset_path=args.dataset_path,
                    use_gen_anomalies=args.use_gen_anomalies,
                    contaminate=args.contaminate,
                    n_per_type=args.n_per_type,
                    model_type=args.model_type,
                    save_dir=model_save_dir,
                    device=device,
                    lambda_=args.lambda_,
                    batch_size=args.batch_size,
                    optimizer=args.optimizer,
                    lr=args.lr,
                    epochs=args.epochs,
                    model_path=args.model_path,
                    save_path_pred_concepts=args.save_path_pred_concepts,
                    seed=args.seed)
    else:
        auc_main, auc_attr, f1_main, f1_attr = test_model(category=args.category,
                                                          dataset_path=args.dataset_path,
                                                          use_gen_anomalies=args.use_gen_anomalies,
                                                          contaminate=args.contaminate,
                                                          n_per_type=args.n_per_type,
                                                          model_type=args.model_type,
                                                          save_dir=model_save_dir,
                                                          device=device,
                                                          batch_size=args.batch_size,
                                                          mode=args.mode,
                                                          image_path=args.image_path)


if __name__ == "__main__":
    main()