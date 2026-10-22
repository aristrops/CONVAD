import argparse
import torch
import os
import pandas as pd

from datasets.concept_dataset import ConceptDataset
from trainers.trainer_stfpm import STFPMTrainer
from evaluators.evaluator_stfpm import STFPMEvaluator
from models.model_backbones import BackboneModelFeatures


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

def build_teacher_path(save_dir, use_gen_anomalies, contaminate, n_per_type, model_type = "joint"):
    save_dir = os.path.join(save_dir, "cbm", model_type)

    if use_gen_anomalies:
        file_name = "unsupervised_model.pth"
        if contaminate:
            file_name = f"semisupervised_{n_per_type}_model.pth"
    else:
        file_name = "supervised_model.pth"

    teacher_path = os.path.join(save_dir, file_name)

    return teacher_path

def build_student_path(save_dir, use_gen_anomalies, contaminate, n_per_type, model_type = "joint"):
    save_dir = os.path.join(save_dir, "stfpm", model_type)

    if use_gen_anomalies:
        file_name = "unsupervised_model.pth"
        if contaminate:
            file_name = f"semisupervised_{n_per_type}_model.pth"
    else:
        file_name = "supervised_model.pth"

    student_path = os.path.join(save_dir, file_name)

    return student_path

def train_model(category: str,
                dataset_path: str,
                use_gen_anomalies: bool,
                contaminate: bool,
                n_per_type: int,
                save_dir: str,
                device: torch.device,
                batch_size: int = 8,
                optimizer: str = "adam",
                lr: float = 1e-3,
                epochs: int = 100):
    
    def init_optimizer(params):
        if optimizer == "adam":
            opt = torch.optim.Adam(params, lr = lr)
        elif optimizer == "sgd":
            opt = torch.optim.SGD(params, lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode = "min", factor = 0.1, patience = 5)
        return opt, scheduler

    print(f"Training student model for {category} category...")

    if use_gen_anomalies:
        dataframe_path = os.path.join(dataset_path, category, "concepts", f"{category}_dataset_gen.csv")
        dataframe_path_original = os.path.join(dataset_path, category, "concepts", f"{category}_dataset_gen_to_real.csv")
    else:
        dataframe_path = os.path.join(dataset_path, category, "concepts", f"{category}_dataset.csv")

    dataframe = read_concept_csv(dataframe_path)
    dataframe_original = read_concept_csv(dataframe_path_original) if use_gen_anomalies else None

    save_dir = os.path.join(save_dir, category)

    #load trained CBM model (supervised, semi-supervised or unsupervised depending on the setting)
    teacher_path = build_teacher_path(save_dir, use_gen_anomalies, contaminate, n_per_type)
    state_dict = torch.load(teacher_path, weights_only=False)
    state_dict = {
        k.replace("first_model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("first_model.")
    }
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc_layers.")}

    #build student path for saving the model
    student_path = build_student_path(save_dir, use_gen_anomalies, contaminate, n_per_type)

    normal_images = dataframe[(dataframe["split"] == "train") & (dataframe["label_index"] == 0)]
    train_dataset = ConceptDataset(normal_images, split="train", use_attr=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True,
                                                   num_workers=8, pin_memory=True, persistent_workers=True)

    if use_gen_anomalies:
        val_dataset = ConceptDataset(dataframe_original, split = "val", use_attr=False, load_mask=True)
    else:
        val_dataset = ConceptDataset(dataframe, split = "val", use_attr=False, load_mask=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False,
                                                 num_workers=8, pin_memory=True, persistent_workers=True)

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    teacher_model = BackboneModelFeatures(pretrained = True)
    teacher_model.load_state_dict(state_dict)

    student_model = BackboneModelFeatures(pretrained=False)

    optimizer, scheduler = init_optimizer(student_model.parameters())

    trainer = STFPMTrainer(teacher_model, student_model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs, save_path = student_path)
    trainer.train()


def test_model(category: str,
               dataset_path: str,
               use_gen_anomalies: bool,
               contaminate: bool,
               n_per_type: int,
               save_dir: str,
               device: torch.device,
               batch_size: int = 8,
               evaluate_one: bool = False,
               save_path_image: str = None):

    print(f"\nTesting STFPM model for {category} category...")

    if use_gen_anomalies:
        dataframe_path = os.path.join(dataset_path, category, "concepts", f"{category}_dataset_gen_to_real.csv")
    else:
        dataframe_path = os.path.join(dataset_path, category, "concepts", f"{category}_dataset.csv")

    dataframe = read_concept_csv(dataframe_path)

    # load trained CBM model (teacher)
    save_dir = os.path.join(save_dir, category)

    teacher_path = build_teacher_path(save_dir, use_gen_anomalies, contaminate, n_per_type)
    state_dict = torch.load(teacher_path)
    state_dict = {
        k.replace("first_model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("first_model.")
    }

    # load trained student model
    student_path = build_student_path(save_dir, use_gen_anomalies, contaminate, n_per_type)
    state_dict_student = torch.load(student_path)
    
    teacher_model = BackboneModelFeatures(pretrained=True)
    teacher_model.load_state_dict(state_dict)

    student_model = BackboneModelFeatures(pretrained=True)
    student_model.load_state_dict(state_dict_student, strict = False)

    test_dataset = ConceptDataset(dataframe, split = "test", use_attr=False, load_mask=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False,
                                                  num_workers=8, pin_memory=True)

    evaluator = STFPMEvaluator(teacher_model, student_model, test_dataloader, device)

    if evaluate_one:
        anomalous_indices = test_dataset.df.index[test_dataset.df["label_index"] == 1].tolist()
        for idx in anomalous_indices:
            print(f"Creating heatmap for image {idx}...")
            image, label, mask = test_dataset[idx]
            evaluator.visualize(image, mask, save_path=save_path_image)
    
    else:
        print(f"Number of test images: {len(test_dataset)}")
        print(f"Evaluating STFPM model for {category} category...")
        evaluator.evaluate()
    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=["train", "test", "visualize"], help="Whether to train, test or visualize")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--use_gen_anomalies", action="store_true", help="Unsupervised or semi-supervised scenario: use generated anomalies to train")
    parser.add_argument("--contaminate", action="store_true",
                        help="Semi-supervised scenario: contaminate the generated dataset with some original anomalous samples")
    parser.add_argument("--n_per_type", type=int, default=0,
                        help="How many original images to add to the generated dataset")
    parser.add_argument("--category", type=str, help="Which category to train/test")
    parser.add_argument("--save_dir", type=str, default=None, help = "Directory to save the models in")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--batch_size", type=int, default = 8, help="Batch size to use")
    parser.add_argument("--optimizer", type=str, default = "adam", help="Which optimizer to use")
    parser.add_argument("--lr", type = float, default = 1e-3, help="Learning rate to use")
    parser.add_argument("--epochs", type=int, default=100, help="How many epochs to run the training for")
    parser.add_argument("--save_path_image", type=str, help="Path where to save the generated heatmap")
    parser.add_argument("--seed", type=int, default=42, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.mode == "train":
        train_model(args.category, args.dataset_path, args.use_gen_anomalies, args.contaminate, args.n_per_type, args.save_dir, device, args.batch_size, args.optimizer, args.lr, args.epochs)
    elif args.mode == "test":
        test_model(args.category, args.dataset_path, args.use_gen_anomalies, args.contaminate, args.n_per_type, args.save_dir, device, args.batch_size)
    elif args.mode == "visualize":
        test_model(args.category, args.dataset_path, args.use_gen_anomalies, args.contaminate, args.n_per_type, args.save_dir, device, args.batch_size, evaluate_one=True, save_path_image = args.save_path_image)

if __name__ == "__main__":
    main()