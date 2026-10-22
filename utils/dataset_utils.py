import json
import re
import os
import torch
import clip
import numpy as np
import pandas as pd

from tqdm import tqdm
from ollama import Client
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency

from datasets.mvtec_dataset import MVTecDataset
from datasets.realiad_dataset import RealIadDataset
from datasets.visa_dataset import VisaDataset

client = Client(host=os.environ.get("CLIENT_HOST", "http://localhost:6000"))

#parse JSON object
def extract_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Raw response:", text)
        return {}

#convert concepts to binary vectors
def concepts_to_vectors(concept_json, concept_list):
        return [int(concept_json.get(concept, False)) for concept in concept_list]

#first VLM query
def first_vlm_query(category, model_name, sample):
    image_path = sample["image_path"]
    label = sample["label"]
    
    if label == "good":
        message = f"You are an expert evaluating an industrial image to detect anomalies. I provide an image of a {category}. The image has been classified as normal from an industrial point of view, so there is no visible defect, anomaly or issue."\
        f"Knowing this fact, please provide a general description of the image, providing a characterization of what is visible, for example information about the texure, the color, any relevant features, ..."\
        f"Then, from the description, extract the most meaningful concepts. The concepts should be defined in relationship to the image, in such a way that, observing the image, it is possible to clearly answer with yes or no about the presence of such a concept"\
        "Concepts should be referred to visible features only, avoiding speculations or assumptions."\
        "Avoid concepts that are diffused and cannot be grounded on an area of the image, e.g. AVOID concepts such as 'surface discontinuity', 'irregular shape'."\
        "You can output five concepts or less, concepts can have more than one word, if this adds information."\
        "Your ONLY output should be the concepts, nothing else, written as a valid JSON array of strings."

    else:
        message = f"You are an expert evaluating an industrial image to detect anomalies. I provide an image of a {category}. The image has been classified as anomalous, so there is a part of it that contains a defect with respect to the standard. The defect is {label}."\
        f"Knowing these facts, please focus on its area and first provide a general description of it, and then from the description extract the most meaningful concepts."\
        "The concepts should be defined in relationship to the image, in such a way that, observing the image, it is possible to clearly answer with yes or no about the presence of such a concept"\
        "Concepts should be referred to visible features only, avoiding speculations or assumptions."\
        "Avoid concepts that are diffused and cannot be grounded on an area of the image, e.g. AVOID concepts such as 'surface discontinuity', 'irregular shape'."\
        "You can output five concepts or less, concepts can have more than one word, if this adds information."\
        "Your ONLY output should be the concepts, nothing else, written as a valid JSON array of strings."

    
    response = client.chat(model=model_name, messages=[{"role": "user", "content": message, "images": [image_path]}])

    try:
        concept_json = extract_json(response["message"]["content"])
    except Exception as e:
        print(f"Error parsing response for {image_path}: {e}")
        concept_json = []
    
    return concept_json

#second VLM query
def second_vlm_query(category, model_name, sample, concept_list):
    image_path = sample["image_path"]
    label = sample["label"]

    if label == "good":
        message = f"You are an expert evaluating an industrial image to detect anomalies. I provide an image of a {category}"\
                "The image has been classified as normal, which implies that there is no visible defect, anomaly or issue."\
                f"Knowing this, choose which concepts you see in the image among the following list of attributes: {concept_list}."\
                "Output the result as a JSON object of this form: {concept_1: true, concept_2: false, ...}. Output ONLY the JSON object, nothing else."
        
        response = client.chat(model=model_name, messages=[{"role": "user", "content": message, "images": [image_path]}])

    else:
        message = f"You are an expert evaluating an industrial image to detect anomalies. The first image I provide is an image of a {category}"\
                f"The image has been classified as anomalous, which implies that it shows a visible defect, alteration or damage. The defect is {label}."\
                f"Knowing this, choose which concepts you see in the image among the following list of attributes: {concept_list}."\
                "Output the result as a JSON object of this form: {concept_1: true, concept_2: false, ...}. Output ONLY the JSON object, nothing else."

        response = client.chat(model=model_name, messages=[{"role": "user", "content": message, "images": [image_path]}])

    try:
        concept_json = extract_json(response["message"]["content"])
    except Exception as e:
            print(f"Error parsing response for {image_path}: {e}")
            concept_json = {}
    
    vector = concepts_to_vectors(concept_json, concept_list)

    return vector


#create concept list
def create_concept_list(dataset: str,
                        dataset_path: str,
                        category: str,
                        model_name: str,
                        use_gen_anomalies: bool = False):

    concepts = set()

    if dataset == "mvtec":
        train_dataset = MVTecDataset(root = dataset_path, category = category, split = "train")
        train_dataset.load_dataset()    

        test_dataset = MVTecDataset(root = dataset_path, category = category, split = "test")
        test_dataset.load_dataset(use_gen_anomalies=use_gen_anomalies)

        full_dataset = pd.concat([train_dataset.samples, test_dataset.samples])

        #extract 5% of the images
        subset = full_dataset.groupby("label", group_keys=False).sample(frac = 0.05, random_state = 42)
    
    elif dataset == "realiad":
        full_dataset = RealIadDataset(root = dataset_path, category=category)
        full_dataset.load_dataset()
        subset = full_dataset.samples.groupby("label", group_keys=False).sample(frac = 0.05, random_state = 42)

    elif dataset == "visa":
        full_dataset = VisaDataset(root_path=dataset_path, category=category)
        subset = full_dataset.samples.groupby("label", group_keys=False).sample(frac=0.05, random_state=42)

    print(f"Number of images to analyze: {len(subset)}")

    for i in tqdm(range(len(subset))):
        sample = subset.iloc[i]
        concept_json = first_vlm_query(category, model_name, sample)
        concepts.update(c.lower() for c in concept_json)

    concepts = list(concepts)
    print(f"Original number of concepts for {category} category:", len(concepts))

    concepts_path = os.path.join(
        dataset_path, category, "concepts_vocabulary", "raw",
        f"concepts_vocab{'_gen' if use_gen_anomalies else ''}.json",
    )
    with open(concepts_path, "w") as f:
        json.dump(concepts, f)

    return concepts


#aggregate similar concepts
def aggregate_concepts(concepts: list,
                       category: str,
                       model_name: str):

    message = "You are an industrial expert that is performing the task of visual anomaly detection."\
            f"Given the following list of visual concepts: {concepts}, related to images of a {category}, please group together those that refer to the same LITERAL meaning, i.e. if they share key words, spelling..."\
            "Ignore semantic relationships, focus only on literal wording and string similarity."\
            "Moreover, choose a representative attribute that best summarizes the group."\
            "I provide two examples: 1. 'Elliptical shape': ['ellipsoidal shape', 'ellipsoid shape', 'elliptical shape']"\
            "2. 'Smooth texture': ['natural texture, 'smooth texture', 'smooth surface texture', 'smooth appearance', 'organic texture', 'uniform texture', 'consistent texture']"\
            "DO NOT create groups that are too general, e.g. grouping together all colours or all textures. Representative concepts should still be able to clearly discriminate between normal and anomalous images."\
            "Return the output as a JSON dictionary where each key is the representative concept, and its value is the list of similar concepts grouped with it."\
            "In case of concepts that cannot be grouped with any other concept, the dictionary should have both as key and value the concept itself."\
            "Output ONLY the JSON dictionary."

    response = client.chat(model=model_name, messages=[{"role": "user", "content": message}])

    try:
        concept_json = extract_json(response["message"]["content"])
    except Exception as e:
        print(f"Error parsing response: {e}")
        concept_json = []
    
    return list(concept_json.keys())
    

#remove concepts with high cosine similarity between each other
def compute_concept_similarity(concepts, threshold = 0.9):
    
    model, preprocess = clip.load("ViT-B/32", jit = False)
    model.eval()
    model.to("cpu")

    with torch.no_grad():
        text_tokens = clip.tokenize(concepts).to("cpu")
        text_features = model.encode_text(text_tokens)
    
        #normalize features
        text_features /= text_features.norm(dim = -1, keepdim = True)

    #compute cosine similarity between pairs
    similarities = (text_features @ text_features.T).cpu().numpy()
    np.fill_diagonal(similarities, 0.0) #ignore self-similarity

    high_sim_counts = (similarities > threshold).sum(axis = 1)

    #remove concepts with more than two high-similarity matches
    concepts_to_remove = set(np.array(concepts)[high_sim_counts >= 2])

    #for remaining pairs, remove one of each
    remaining_idx = [i for i, c in enumerate(concepts) if c not in concepts_to_remove]
    remaining_concepts = [concepts[i] for i in remaining_idx]
    remaining_similarities = similarities[np.ix_(remaining_idx, remaining_idx)]

    #track which indices to drop
    already_removed = set()
    for i in range(len(remaining_idx)):
        for j in range(i + 1, len(remaining_idx)):
            if remaining_similarities[i, j] > threshold:
                if remaining_concepts[i] not in already_removed and remaining_concepts[j] not in already_removed:
                    already_removed.add(remaining_concepts[i])

    final_concepts = [c for c in remaining_concepts if c not in already_removed]

    return final_concepts


#remove concepts that have a high cosine similarity with the class ("anomalous") or the category
def compute_class_similarity(concepts, category, classes = ["anomalous", "normal"], threshold = 0.9):
    
    model, preprocess = clip.load("ViT-B/32", jit = False)
    model.eval()
    model.to("cpu")

    references = [category] + classes

    with torch.no_grad():
        ref_tokens = clip.tokenize(references).to("cpu")
        ref_features = model.encode_text(ref_tokens)
        ref_features = ref_features / ref_features.norm(dim = -1, keepdim = True)

        text_tokens = clip.tokenize(concepts).to("cpu")
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim = -1, keepdim = True)
    
    similarities = (text_features @ ref_features.T).cpu().numpy()

    concepts_to_remove = set()
    for i, concept in enumerate(concepts):
        for j, ref in enumerate(references):
            sim = similarities[i, j]
            if sim > threshold:
                concepts_to_remove.add(concept)
                break  
    
    final_concepts = [c for c in concepts if c not in concepts_to_remove]

    return final_concepts


#create annotated dataset
def create_final_dataset(dataset_path: str,
                         dataset: str,
                         category: str, 
                         final_concepts: list,
                         model_name: str,
                         use_gen_anomalies: bool = False):
    
    image_concepts = []
    
    if dataset == "mvtec":
        train_dataset = MVTecDataset(root = dataset_path, category = category, split = "train")
        train_dataset.load_dataset()

        test_dataset = MVTecDataset(root = dataset_path, category = category, split = "test")
        test_dataset.load_dataset(use_gen_anomalies=use_gen_anomalies)

        samples = pd.concat([train_dataset.samples, test_dataset.samples])
    
    elif dataset == "realiad":
        dataset = RealIadDataset(root = dataset_path, category=category)
        dataset.load_dataset()
        samples = dataset.samples

    elif dataset == "visa":
        dataset = VisaDataset(root_path=dataset_path, category=category)
        samples = dataset.samples
    
    for i in tqdm(range(len(samples))):
        sample = samples.iloc[i]
        concept_vector = second_vlm_query(category, model_name, sample, final_concepts)
        image_concepts.append(concept_vector)

    concepts_df = pd.DataFrame(image_concepts, columns = [f"{c}" for c in final_concepts])
    final_df = pd.concat([samples.reset_index(drop=True), concepts_df], axis=1)

    return final_df


#modify column names
def modify_columns(df):
    #define the anomaly type
    if "label" in df.columns:
        df = df.rename(columns={"label": "anomaly_type"})

    #drop unnecessary columns
    if "split" in df.columns and "path" in df.columns:
        df = df.drop(columns = ["path", "split"])

    return df

#split df into train, test and validation
def split_dataframe(df, train_size = 0.8, test_size = 0.5):   
    train_df, val_test_df = train_test_split(df, train_size = train_size, stratify=df["anomaly_type"], shuffle = True, random_state=42)

    val_df, test_df = train_test_split(val_test_df, test_size=test_size, shuffle = True, random_state=42)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    return df


#remove uninformative concepts
def drop_concepts(df, concepts):
    df_concepts = [col for col in concepts if col in df.columns]
    #drop concepts that appear in less than 10 images
    invalid_concepts = df[df_concepts].sum()[lambda x: x < 10].index
    print("Concepts that appear in less than 10 images:", invalid_concepts)

    df = df.drop(columns=invalid_concepts)

    remaining_concepts = [col for col in df_concepts if col not in invalid_concepts]
    #drop concepts that appear in more than 95% of the images
    always_present_concepts = df[remaining_concepts].sum()[lambda x: x > len(df) * 0.95].index
    print("Concepts that appear in more than 95% of images:", always_present_concepts)

    df = df.drop(columns=always_present_concepts)

    remaining_concepts = [col for col in concepts if col in df.columns]

    return df, remaining_concepts

#compute correlation between concepts
def compute_correlation(df, concepts, threshold=0.95):
    concepts_to_use = [col for col in concepts if col in df.columns]
    df_concepts = df[concepts_to_use].copy()
    
    #iteratively remove correlated features
    while True:
        corr_matrix = df_concepts.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        max_corr = upper_triangle.max().max()
        
        if max_corr < threshold:
            break
        
        #drop one of the two most correlated features
        to_drop = upper_triangle.stack().idxmax()[1]
        df_concepts = df_concepts.drop(columns=[to_drop])
    
    #get remaining concept names
    remaining_concepts = list(df_concepts.columns)
    
    #drop the removed concept columns from the full df, but keep all others
    df_result = df.drop(columns=[col for col in concepts if col not in remaining_concepts])
    
    return df_result, remaining_concepts


#discriminative analysis of concepts
def chi_square_test(df, concepts):
    for col in concepts:
        contingency = pd.crosstab(df[col], df["label_index"])
        chi2, p, dof, ex = chi2_contingency(contingency)
        print(f"\n{col}: p-value = {p:.2e}")
        prob_anomalous = df[df["label_index"] == 1][col].mean()
        prob_normal = df[df["label_index"] == 0][col].mean()
        print(f"{col}: P(concept = 1 | anomaly) = {prob_anomalous:.2f}, P(concept = 1 | normal) = {prob_normal:.2f}")