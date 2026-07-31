# Denoiser

This repository contains a U-Net model and the necessary tools to train it on two distinct multimedia signal denoising tasks: speech dereverberation and low-light image enhancement.
Below are the instructions for training both models.
The first thing to do for both tasks is to create a venv and install the necessary dependencies:

```bash
python -m venv venv

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

The training can be monitored with the following command
```bash
tensorboard --logdir=artifacts/logs/
```


## Audio (Speech Dereverberation)

The ground-truth clean audio dataset is LibriSpeech (which is downloaded automatically by the data preparation script). To simulate environmental reverberation, Room Impulse Responses (RIRs) from the Aachen Impulse Response (AIR) database are utilized.

### Download and Data Preparation (Audio)

1. **Download the RIR dataset:**
   - [Aachen Impulse Response (AIR) Database](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)
   
   Extract the downloaded files into a temporary directory (e.g., `data/AIR/`).
   Then run the following script to split the AIR dataset into three parts for training, validation and testing without room overlap.
   ```bash
   python src/create_air_splits.py --source_dir data/AIR --output_dir data/rir/real
   ```

2. **Generate the synthetic RIRs:**
   Run the generation script, which will create the required RIRs via ShoeBox simulation.
   ```bash
   # train
   python src/generate_rirs.py --output_dir data/rir/synthetic/train --num_rirs 30000 --sample_rate 16000 --format pt --normalize none --seed 42

   # validation
   python src/generate_rirs.py --output_dir data/rir/synthetic/validation --num_rirs 5000 --sample_rate 16000 --format pt --normalize none --seed 999

   # test
   python src/generate_rirs.py --output_dir data/rir/synthetic/test --num_rirs 5000 --sample_rate 16000 --format pt --normalize none --seed 1234
   ```

3. **Generate static datasets (Validation / Test):**
   Run the setup script. This script will automatically download the required LibriSpeech subsets, generate fixed pairs of clean and reverberated audio (using both synthetic and real RIRs) and save them to disk for the validation and test sets.
   ```bash
   # validation
   python src/generate_static_datasets.py --split_name validation --ls_subset dev-clean --synth_rir_dir data/rir/synthetic/validation/ --real_rir_dir data/rir/real/validation/ --target_duration 3.0 --output_dir data/audio/validation --source_dir data/libriSpeech/

   # test
   python src/generate_static_datasets.py --split_name test --ls_subset test-clean --synth_rir_dir data/rir/synthetic/test/ --real_rir_dir data/rir/real/test/ --output_dir data/audio/test --source_dir data/libriSpeech/
   ```

### Training and evaluation (Audio)

A dedicated script is provided to calculate baseline and oracle metrics (PESQ and STOI) on the static test dataset generated in the previous step. The resulting baseline metrics are passed to the testing code to calculate the delta directly (the values for the datasets created as above are already hardcoded inside the training/testing script, but can be changed there for different datasets)
```bash
python src/oracle_test.py --test_data_dir data/audio/test
```

To start training the U-Net on the speech dereverberation task, run the following command, which will also download the necessary Librispeech training subset if it isn't already in the provided source_dir. The script will also launch a testing run at the end. The training parameters were optimized to run on a RTX 5070 TI with 16 GB of VRAM and on Linux.
```bash
python src/audio_training.py --experiment_name rir_mr-stft-loss_train-clean-100_mixed_32_masking_v47 --output_folder artifacts --training_subset train-clean-100 --batch_size 64 --base_features 32 --train_data_dir data/libriSpeech --val_data_dir data/audio/validation --test_data_dir data/audio/test
```



## Images (Low-Light Image Enhancement)

For the photographic task of recovering images in low-light conditions, the model relies on the LOL datasets.

### Download and Data Preparation (Images)

Download the following datasets and place them in the designated images data directory (e.g., `data/images/`):
- [LOL v1 Dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset)
- [LOL v2 Dataset](https://www.kaggle.com/datasets/tanhyml/lol-v2-dataset?select=LOL-v2)


Download the LOL dataset family and organize them under a single root folder (e.g., `./data`). Extract the contents of the archives while preserving the original directory structure. This ensures the dataloaders can correctly map the input "low-light" images to their respective well-lit ground-truth targets.

### Training, Evaluation and Testing (Images)

The low-light enhancement pipeline is handled by `image_training.py`.  
It expects the LOL datasets to be arranged as described in [Download and Data Preparation](#dataset-setup) and will automatically merge LOL v1, v2 Real and v2 Synthetic for training.

```bash
# training
python src/image_training.py --experiment_name LL_combinedLoss_FINAL_checkingsomething --output_folder artifacts --data_dir data  

# evaluation
python src/LL_evaluation_test.py --checkpoint artifacts/checkpoints/LL_combinedLoss_FINAL/lowlight-best-epoch=35-val_loss/total=0.1292.ckpt --data_dir data --output_dir artifacts/eva_results_combinedLoss_FINAL --base_features 48

# testing
python src/image_inference.py --checkpoint artifacts/checkpoints/LL_combinedLoss_64f_v1/lowlight-best-epoch=78-val_loss/total=0.1256.ckpt --input artifacts/image_testing --base_features 48   