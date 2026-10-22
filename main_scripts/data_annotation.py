import json
import os
import argparse

from utils import dataset_utils

def create_concept_dataset(dataset: str,
                           dataset_path: str,
                           category: str,
                           model_name: str,
                           use_gen_anomalies: bool,
                           annotate_true_imgs: bool):

    print(f"\nCreating concept dataset for {category} category...")

    #for annotation of real images from artifical concepts, simply load the predefined concepts list (run only after creating the concepts vocab)
    if annotate_true_imgs:
        final_vocab_path = os.path.join(dataset_path, category, "concepts_vocabulary", "filtered", "concepts_vocab_gen.json")
        if not os.path.exists(final_vocab_path):
            raise FileNotFoundError(
                f"{final_vocab_path} not found. Run the generated-anomalies annotation first (use_gen_anomalies=True, annotate_true_imgs=False)."
            )
        with open(final_vocab_path) as f:
            final_concepts = json.load(f)
        print(f"Loaded {len(final_concepts)} concepts from {final_vocab_path}")

    #for real anomalies and generated ones, run the full pipeline
    else:
        #step 1: create concept list
        print("\nQuerying the VLM to extract the concept list...")
        concept_list = dataset_utils.create_concept_list(dataset, dataset_path, category, model_name, use_gen_anomalies=use_gen_anomalies)

        #step 2: aggregate and filter concept list
        print("\nReducing the dimensionality of the concept set...")
        filtered_concept_list = dataset_utils.aggregate_concepts(concept_list, category, model_name)
        print("Removing too similar concepts...")
        filtered_concept_list = dataset_utils.compute_concept_similarity(filtered_concept_list)
        final_concepts = dataset_utils.compute_class_similarity(filtered_concept_list, category)
        print(f"Final number of concepts kept for {category} category: {len(final_concepts)}")
    
    #step 3: create final dataset
    print("\nAutomatically annotating the dataset...")
    if annotate_true_imgs:
        final_df = dataset_utils.create_final_dataset(dataset_path, dataset, category, final_concepts, model_name, use_gen_anomalies = False)
    else:
        final_df = dataset_utils.create_final_dataset(dataset_path, dataset, category, final_concepts, model_name, use_gen_anomalies = use_gen_anomalies)

    #step 4: split the dataset into train, test and validation
    print("\nSplitting the dataset into train, test and validation...")
    final_df = dataset_utils.modify_columns(final_df)
    final_df = dataset_utils.split_dataframe(final_df)

    if not annotate_true_imgs:
        #step 5: remove uninformative and highly correlated concepts
        print("\nRemoving uninformative and highly correlated concepts...")
        final_df, remaining_concepts = dataset_utils.drop_concepts(final_df, final_concepts)
        final_df, remaining_concepts = dataset_utils.compute_correlation(final_df, remaining_concepts)

        #save final concepts
        remaining_concepts_path = os.path.join(
            dataset_path, category, "concepts_vocabulary", "filtered",
            f"concepts_vocab{'_gen' if use_gen_anomalies else ''}.json",
        )
        with open(remaining_concepts_path, "w") as f:
            json.dump(remaining_concepts, f)

        print(f"Final number of concepts kept for {category} category after dataset annotation: {len(remaining_concepts)}")

        # step 6: statistical analysis of concepts
        print("\nComputing statistical significance of concepts...")
        dataset_utils.chi_square_test(final_df, remaining_concepts)

    #save full dataset
    if annotate_true_imgs:
        file_name = f"{category}_dataset_gen_to_real.csv"
    elif use_gen_anomalies and not annotate_true_imgs:
        file_name = f"{category}_dataset_gen.csv"
    else:
        file_name = f"{category}_dataset.csv"

    save_path = os.path.join(dataset_path, category, "concepts", file_name)

    final_df.to_csv(save_path, index = False)
    print(f"Final dataframe saved in {save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="Type of dataset to use")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset")
    parser.add_argument("--category", type = str, nargs="+", help="Which categories to annotate")
    parser.add_argument("--model_name", type=str, help="Which VLM to use")
    parser.add_argument("--use_gen_anomalies", action = "store_true", help = "Annotate dataset with the generated anomalies")

    args = parser.parse_args()

    create_concept_dataset(args.dataset_path, args.category, args.model_name, args.save_path, args.use_gen_anomalies)

if __name__ == "__main__":
    main()
    