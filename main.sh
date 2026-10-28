python -m main_scripts.cbm \
    --mode train \
    --dataset_path CONVAD-MVTec \
    --model_type joint \
    --save_dir cbm_models \
    --category bottle \
    --device cpu \
    --model_path cbm_models/backbone/fine-tuned-mobilenet.pth \
    --gemini_logo_mask_path datasets/gemini_logo_mask.png