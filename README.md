<div align="center">

# Explainable Visual Anomaly Detection<br>via Concept Bottleneck Models

</div>
This repository contains the code for our work "Explainable Visual Anomaly Detection via Concept Bottleneck Models", through which we try to enhance interpretability in Industrial Visual Anomaly Detection (VAD) by employing Concept Bottleneck Models (CBMs), i.e. neural models that learn meaningful concepts and thus can provide human-understandable descriptions of anomalies and defects. 

## Paper Abstract
In recent years, Visual Anomaly Detection (VAD) has attracted significant attention due to its high interpretability, as it provides visual explanations by highlighting anomalous regions within an image. However, although these visual explanations can be helpful, they lack a direct and semantically meaningful interpretation for users. To address this limitation, we propose extending Concept Bottleneck Models (CBMs) to the VAD setting. By learning meaningful concepts, the network can provide human-interpretable descriptions of anomalies, offering a novel and more insightful way to explain them. Our main contributions are: (i) we introduce a concept-based framework for anomaly explanation by extending CBMs to the VAD setting for the first time; (ii) we evaluate multiple supervision regimes, ranging from supervised to unsupervised anomaly settings; (iii) we propose a dual-branch architecture that combines a CBM-based Concept branch for concept-level explanations with a Visual branch for pixel-level anomaly localization, bridging semantic and spatial interpretability. When evaluated across three well-established VAD benchmarks, our approach, Concept-Aware Visual Anomaly Detection (CONVAD), achieves performance comparable to classic VAD methods (up to 0.97 I-AUROC on MVTec AD) while providing richer, concept-driven explanations. A user study (582 responses) shows that CONVAD explanations are rated effective in 73.5% of cases (vs. 60.7% for the SOTA VLM-based explainer Anomaly-OV) and misleading in only 3.1% (vs. 11.7%). Beyond explanation quality, CBMs enable human-in-the-loop correction: concept-level interventions further boost detection in the challenging unsupervised setting, closing part of the gap to fully supervised performance.

## How to use
The repository is organized as follows:
- [/datasets](/datasets) defines the classes for the original VAD datasets and the final concept dataset that was used for training the Concept branch;
- [/models](/models) contains the scripts necessary to build the models;
- [/trainers](/trainers) and [/evaluators](/evaluators) contain the scripts for respectively training and evaluating both the Concept branch and the Visual branch;
- [/main_scripts](/main_scripts) contains the scripts for:
  * Performing data annotation with concepts (`data_annotation.py`)
  * Concept branch training/testing (`cbm.py`)
  * Concept branch intervention (`intervention.py`)
  * Visual branch training/testing (`stfpm.py`)
  * Evaluating the full pipeline with the production of combined metrics (`combined_branches.py`)

See `main.sh` for an example of how to run the main scripts.


## Dataset
We tested our dataset creation pipeline starting from the [MVTec-AD dataset](/https://www.mvtec.com/company/research/datasets/mvtec-ad), which was modified to include artificially generated anomalous images, together with the original ones, and concept annotation. Our dataset is available at [huggingface.co/datasets/aristrops/mvtec-concepts](https://huggingface.co/datasets/aristrops/mvtec-concepts).

## Citation
If you use our method in your scientific work, please cite:

```
@article{stropeni2025explainable,
  title={Explainable Visual Anomaly Detection via Concept Bottleneck Models},
  author={Stropeni, Arianna and Zaccaria, Valentina and Borsatti, Francesco and Pezze, Davide Dalle and Barusco, Manuel and Susto, Gian Antonio},
  journal={arXiv preprint arXiv:2511.20088},
  year={2025}
}
```
