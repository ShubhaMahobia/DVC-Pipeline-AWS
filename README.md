# 🚀 ML Pipeline with DVC and Experiment Tracking

A comprehensive Machine Learning pipeline for SMS Spam Detection using **DVC (Data Version Control)** for reproducible ML workflows and **DVC Live** for experiment tracking. This project demonstrates best practices for building scalable, maintainable ML pipelines.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Experiment Tracking](#experiment-tracking)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete ML pipeline for SMS spam detection using a Random Forest classifier. The pipeline is built with DVC for data versioning and reproducible workflows, ensuring that every experiment can be reproduced exactly as it was run.

### Key Highlights:
- **End-to-end ML Pipeline**: From data ingestion to model evaluation
- **DVC Integration**: Version control for data, models, and experiments
- **Experiment Tracking**: Real-time metrics and parameter tracking with DVC Live
- **Modular Architecture**: Clean separation of concerns across pipeline stages
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Parameter Management**: YAML-based configuration for easy hyperparameter tuning

## ✨ Features

- 🔄 **Reproducible Workflows**: DVC ensures every experiment can be reproduced
- 📊 **Experiment Tracking**: Real-time metrics visualization with DVC Live
- 🧹 **Data Preprocessing**: Text cleaning, stemming, and feature engineering
- 🤖 **Model Training**: Random Forest classifier with configurable parameters
- 📈 **Model Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, AUC)
- 📝 **Comprehensive Logging**: Detailed logs for each pipeline stage
- ⚙️ **Parameter Management**: YAML-based configuration system
- 🏗️ **Modular Design**: Clean, maintainable code structure

## 📁 Project Structure

```
DVC-Pipeline-AWS/
├── 📁 src/                          # Source code
│   ├── 📄 data_ingestion.py         # Data loading and splitting
│   ├── 📄 data_preprocessing.py     # Text preprocessing and cleaning
│   ├── 📄 feature_eng.py           # Feature engineering (TF-IDF)
│   ├── 📄 model_training.py        # Model training
│   └── 📄 model_evaluation.py      # Model evaluation and metrics
├── 📁 data/                         # Data storage
│   ├── 📁 raw/                     # Raw data files
│   ├── 📁 interim/                 # Intermediate processed data
│   └── 📁 processed/               # Final processed data
├── 📁 models/                      # Trained models
├── 📁 logs/                        # Log files
├── 📁 reports/                     # Evaluation reports
├── 📁 experiments/                 # Jupyter notebooks and datasets
├── 📄 dvc.yaml                     # DVC pipeline configuration
├── 📄 params.yaml                  # Hyperparameters and configuration
├── 📄 requirements.txt             # Python dependencies
└── 📄 README.md                    # This file
```

## 🔄 Pipeline Stages

The ML pipeline consists of 5 main stages, each managed by DVC:

### 1. **Data Ingestion** (`data_ingestion.py`)
- Downloads spam dataset from GitHub
- Splits data into train/test sets
- Saves raw data to `data/raw/`

### 2. **Data Preprocessing** (`data_preprocessing.py`)
- Text cleaning and normalization
- Label encoding for target variable
- Duplicate removal
- Saves processed data to `data/interim/`

### 3. **Feature Engineering** (`feature_eng.py`)
- TF-IDF vectorization
- Configurable feature extraction
- Saves feature-engineered data to `data/processed/`

### 4. **Model Training** (`model_training.py`)
- Random Forest classifier training
- Hyperparameter configuration
- Model serialization to `models/`

### 5. **Model Evaluation** (`model_evaluation.py`)
- Performance metrics calculation
- Experiment tracking with DVC Live
- Metrics logging to `reports/`

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **DVC**: Data version control and pipeline management
- **DVC Live**: Experiment tracking and visualization
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Pandas**: Data manipulation and analysis
- **NLTK**: Natural language processing
- **NumPy**: Numerical computing
- **PyYAML**: Configuration management
- **Matplotlib**: Data visualization

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Git
- DVC

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd DVC-Pipeline-AWS
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install DVC** (if not already installed)
   ```bash
   pip install dvc
   ```

4. **Initialize DVC** (if not already initialized)
   ```bash
   dvc init
   ```

## 🚀 Usage

### Running the Complete Pipeline

```bash
# Run the entire pipeline
dvc repro

# Run a specific stage
dvc repro data_ingestion
dvc repro model_training
```

### Running Individual Stages

```bash
# Data ingestion
python src/data_ingestion.py

# Data preprocessing
python src/data_preprocessing.py

# Feature engineering
python src/feature_eng.py

# Model training
python src/model_training.py

# Model evaluation
python src/model_evaluation.py
```

### Experiment Tracking

```bash
# View experiment results
dvc exp show

# Compare experiments
dvc exp diff

# View metrics
dvc metrics show
```

## ⚙️ Configuration

The pipeline is configured through `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.10

feature_eng:
  max_features: 45

model_training:
  n_estimators: 25
  random_state: 2
```

### Parameter Tuning

To experiment with different parameters:

1. **Modify `params.yaml`**
2. **Run the pipeline**: `dvc repro`
3. **Track results**: `dvc exp show`

## 📊 Experiment Tracking

This project uses **DVC Live** for comprehensive experiment tracking:

### Metrics Tracked
- **Accuracy**: Overall model performance
- **Precision**: Spam detection precision
- **Recall**: Spam detection recall
- **AUC**: Area under ROC curve

### Visualization
- Real-time metrics plotting
- Parameter tracking
- Experiment comparison

### Viewing Results
```bash
# View current metrics
dvc metrics show

# Compare experiments
dvc exp diff

# View experiment history
dvc exp show
```

## 🔧 DVC Pipeline Configuration

The pipeline is defined in `dvc.yaml`:

```yaml
stages:
  data_ingestion:
    cmd: python src/data_ingestion.py
    deps:
      - src/data_ingestion.py
    params:
      - data_ingestion.test_size
    outs:
      - data/raw

  # ... other stages
```

## 📈 Performance Metrics

The model evaluation stage calculates and tracks:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for spam detection
- **Recall**: Recall for spam detection  
- **AUC**: Area under the ROC curve

## 🧪 Experimentation

### Running Experiments

1. **Modify parameters** in `params.yaml`
2. **Run pipeline**: `dvc repro`
3. **Track results**: `dvc exp show`

### Example Experiment

```bash
# Modify test_size in params.yaml
data_ingestion:
  test_size: 0.20  # Changed from 0.10

# Run experiment
dvc repro

# View results
dvc exp show
```

## 📝 Logging

Each pipeline stage includes comprehensive logging:

- **Console output**: Real-time progress
- **File logging**: Detailed logs in `logs/` directory
- **Error handling**: Graceful error handling and reporting

