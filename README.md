# Mini-Project-2B
# Emotion Recognition with Big Data and Large-Scale Neural Networks

## Project Overview

In Mini-Project 1B, you trained a small neural network on EEG data from a single subject. Your goal now is to scale this up: train larger neural networks on the complete dataset (28 subjects) using FABRIC's computational resources. You'll systematically increase model complexity, measure performance trade-offs, and learn how to work with real distributed computing systems.

**Important Note:** You will encounter errors and unfamiliar code. This is expected and intentional. Part of this project is learning how to use generative AI (like ChatGPT, Claude) to debug and understand code you didn't write.

## Background

- **Dataset**: Emotion Recognition using EEG and Computer Games (28 subjects, 2.4 GB)
- **Kaggle Link**: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions
- **Your Code**: Your working Experiment 2 code from Mini-Project 1B
- **Platform**: FABRIC for distributed computing
- **Goal**: Understand scaling laws—how model size, accuracy, and runtime relate

## Key Concepts

**Scaling Laws in Deep Learning:**
- Larger models can learn more complex patterns
- Larger models take longer to train
- More data helps—but doesn't help infinitely
- Eventually accuracy plateaus (diminishing returns)
- You'll observe these principles in practice

## Project Workflow

### Step 1: Set Up FABRIC Environment

**Create a FABRIC node:**
- Request a node with at least 8-16 CPU cores
- At least 10 GB storage (ideally 20+ GB for the full dataset)
- Note the storage location for later

**Set up conda environment:**
```bash
conda create -n emotion_big python=3.9
conda activate emotion_big
pip install -r requirements.txt
```

**Check your resources:**
```bash
nproc  # Number of CPU cores
df -h  # Disk space available
```

### Step 2: Download the Dataset

**Find the dataset:**
1. Go to Kaggle: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions
2. Click "Download"
3. Transfer to FABRIC (using scp, sftp, or Kaggle API)

**Dataset structure:**
```
emotion_dataset/
├── S01/
│   └── Preprocessed/
│       └── .csv format/
│           ├── 01_DE_emotion.csv
│           ├── 02_DE_emotion.csv
│           ├── ...
├── S02/
│   └── Preprocessed/
│       └── .csv format/
│           ├── 01_DE_emotion.csv
│           ├── 02_DE_emotion.csv
│           ├── ...
├── ... (28 subjects total)
```

**Emotion labels:** Check the Kaggle dataset description for which file number corresponds to which emotion (e.g., 01 = Happy, 02 = Sad, etc.)

**Download tips:**
- Dataset is 2.4 GB—allocate enough storage
- Use `wget` or `curl` on FABRIC for direct download
- Or use Kaggle API: `kaggle datasets download -d birdy654/eeg-brainwave-dataset-feeling-emotions`

### Step 3: Load and Explore the Dataset

**Your tasks:**
- Write code to load all subjects' data
- Understand the structure (features, labels, number of samples)
- Check data quality (missing values, outliers, class balance)
- Create train/test splits properly (keep data from different subjects separate if possible)

**Questions to answer:**
- How many total samples?
- How many features per sample?
- How many emotion classes?
- Is the dataset balanced (equal samples per emotion)?
- What are reasonable train/test/validation splits?

### Step 4: Modify Your Experiment 2 Code

**Add these capabilities:**

1. **Data Loading**
   - Load all 28 subjects' data
   - Combine into one large dataset
   - Preprocess (normalize/scale)

2. **Training Tracking**
   - Record **train accuracy** (accuracy on training data)
   - Record **test accuracy** (accuracy on held-out test data)
   - Record **train time** (seconds to train)
   - Record **test time** (seconds to evaluate on test data)

3. **Model Size Tracking**
   - Count number of parameters in model
   - Record model size for each experiment

**Example tracking code:**
```python
import time

# Before training
start_time = time.time()

# Train your model
model.fit(X_train, y_train, epochs=50, batch_size=32)

train_time = time.time() - start_time

# Evaluate
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

# Count parameters
num_params = sum(p.numel() for p in model.parameters())

# Record results
results.append({
    'model_name': 'Baseline',
    'num_layers': 2,
    'num_params': num_params,
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
    'train_time': train_time,
    'test_time': time.time()  # Similar for test
})
```

### Step 5: Systematically Increase Model Size

Train at least 6 models with increasing complexity:

**Model Progression Example:**
1. **Baseline** (Experiment 2): 2 layers, 64 units
   - `Dense(64) → Dense(32) → Output`

2. **Model 1**: 3 layers, 128 units
   - `Dense(128) → Dense(64) → Dense(32) → Output`

3. **Model 2**: 3 layers, 256 units
   - `Dense(256) → Dense(128) → Dense(64) → Output`

4. **Model 3**: 4 layers, 256 units
   - `Dense(256) → Dense(256) → Dense(128) → Dense(64) → Output`

5. **Model 4**: 5 layers, 512 units
   - `Dense(512) → Dense(512) → Dense(256) → Dense(128) → Dense(64) → Output`

6. **Model 5**: 5 layers, 1024 units
   - `Dense(1024) → Dense(1024) → Dense(512) → Dense(256) → Dense(128) → Output`

**Important:** Ensure you observe an increasing trend in runtime. If all models train equally fast, make them bigger.

### Step 6: Collect Results in a Table

Create a comprehensive table with your results:

| Model | Layers | Units | Parameters | Train Acc | Test Acc | Train Time (s) | Test Time (s) |
|-------|--------|-------|------------|-----------|----------|----------------|---------------|
| Baseline | 2 | 64 | ? | ? | ? | ? | ? |
| Model 1 | 3 | 128 | ? | ? | ? | ? | ? |
| Model 2 | 3 | 256 | ? | ? | ? | ? | ? |
| Model 3 | 4 | 256 | ? | ? | ? | ? | ? |
| Model 4 | 5 | 512 | ? | ? | ? | ? | ? |
| Model 5 | 5 | 1024 | ? | ? | ? | ? | ? |

## Expected Observations

**What you should see:**
- **Train accuracy**: Generally increases with model size (more capacity to memorize)
- **Test accuracy**: Increases initially, then may plateau or decrease (overfitting)
- **Train time**: Clearly increases with model size
- **Test time**: May increase slightly or stay similar

**What might surprise you:**
- Larger models don't always have better test accuracy
- There's often an "optimal" model size
- Overfitting becomes visible at large sizes

## Deliverables

**Word Document containing:**

1. **Comparative Table** (required)
   - At least 6 rows (baseline + 5 increments)
   - Columns: Model name, layers, units, parameters, train acc, test acc, train time, test time
   - Formatted professionally

2. **Four Plots** (required)
   - **Plot 1**: Train & Test Accuracy vs Model Size
   - **Plot 2**: Training Time vs Model Size
   - **Plot 3**: Test Time vs Model Size
   - **Plot 4**: Accuracy vs Number of Parameters
   - All with proper labels, titles, legends

3. **Discussion** (1-2 pages)
   - What trends did you observe?
   - Why does accuracy plateau?
   - At what point does overfitting start?
   - What's the trade-off between accuracy and runtime?
   - Which model would you recommend for production? Why?
   - What did you learn about scaling laws?

## Environment Setup

**Python packages needed:**
```bash
pip install -r requirements.txt
```

Includes: numpy, pandas, scikit-learn, tensorflow/keras, pytorch (choose one), matplotlib, etc.

## FABRIC Job Script Example

Save as `train_emotion_models.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=emotion_nn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=emotion_%j.out

module load anaconda3
conda activate emotion_big

cd /path/to/project
python train_models.py
```

Submit with: `sbatch train_emotion_models.sh`

## Next Steps

After this project, you'll understand:
- How to work with large datasets
- How to use distributed computing resources
- How scaling affects model performance
- How to measure and compare computational efficiency
- The trade-offs in deep learning (accuracy vs speed)
- How to use AI tools to solve programming problems
