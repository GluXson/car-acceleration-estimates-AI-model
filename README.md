# 🚗 Fuel Consumption Prediction - Neural Network

A project for predicting vehicle fuel consumption (MPG) based on technical parameters using artificial neural networks.

## 📊 Project Description

Multi-Layer Perceptron (MLP) model trained on the Auto MPG dataset from UCI Machine Learning Repository. The project implements advanced optimization techniques:

- **Early Stopping** - automatic training termination when no improvement
- **Dropout Regularization** - preventing overfitting
- **Bias Correction** - automatic model calibration after detecting systematic over/underestimation
- **Fine-tuning** - retraining with lower learning rate for better precision

### Results

- **RMSE**: 2.3-2.8 MPG
- **R²**: 0.85-0.90
- **Maximum error**: <2 MPG on individual predictions

## 🛠️ Tech Stack

### Framework & Libraries

- **PyTorch 2.2.0** - deep learning framework
- **scikit-learn 1.4.0** - preprocessing and metrics
- **pandas 2.2.0** - data handling
- **matplotlib 3.8.0** - results visualization

### Model Architecture

- **Input layer**: 7 features (cylinders, displacement, horsepower, weight, acceleration, year, origin)
- **Hidden layers**: 64 → 32 neurons with ReLU activation
- **Dropout**: 10% on first hidden layer
- **Output layer**: 1 neuron (MPG prediction)
- **Loss function**: MSE (Mean Squared Error)
- **Optimizer**: Adam (lr=0.001)

## 🚀 Installation and Usage

### 1. Clone repository

git clone <repository-url>
cd fuel-prediction

### 2. Create virtual environment

python -m venv venv

### 3. Activate environment

Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

### 4. Install dependencies

pip install -r requirements.txt

### 5. Download data

python download_data.py

Auto MPG dataset will be automatically downloaded from UCI Repository (~30 KB).

### 6. Train model

python train.py

Training process:

- PHASE 1: Basic training with early stopping (max 200 epochs)
- PHASE 2: Bias analysis (checking for systematic over/underestimation)
- PHASE 3: Fine-tuning with bias correction (if bias >0.5 MPG)

Model will be saved to models/best_model.pth.

### 7. Evaluation

python evaluate.py

Available modes:

- Demo (y): Random 3 cars from dataset + comparison with actual values
- Custom file (n): Prediction for your own CSV file

#### Input file format (custom mode):

cyl,disp,hp,weight,acc,year,origin
4,97,52,2130,24.6,82,2
8,350,165,4142,11.5,71,1

Features:

- cyl - number of cylinders
- disp - engine displacement (cubic inches)
- hp - horsepower
- weight - vehicle weight (lbs)
- acc - acceleration 0-60 mph (seconds)
- year - model year (last 2 digits, e.g., 82 = 1982)
- origin - origin (1=USA, 2=Europe, 3=Japan)

## 📈 Example Results

Test RMSE: 2.29 MPG
Test R²: 0.90

Sample predictions:
🚗 Mercury Grand Marquis (8 cyl, 351 cc, 138 HP, 3955 lbs)
Actual MPG: 16.5
Predicted MPG: 16.5
Error: 0.0 MPG

🚗 Toyota Celica GT (4 cyl, 144 cc, 96 HP, 2665 lbs)
Actual MPG: 32.0
Predicted MPG: 30.3
Error: 1.7 MPG

## 📊 Visualizations

After training and evaluation, plots will be generated in results/plots/:

- loss_curves.png - training and validation loss over epochs
- predictions.png - scatter plot of predicted vs actual MPG values

## 🔬 Optimization Techniques

### 1. Early Stopping

- Monitors validation loss every epoch
- Stops training if no improvement for 25 epochs
- Saves best model checkpoint

### 2. Bias Correction

- Calculates systematic prediction bias after training
- If |bias| > 0.5 MPG, triggers fine-tuning phase
- Fine-tuning uses 10x lower learning rate (0.0001)

### 3. Dropout Regularization

- 10% dropout on hidden layer
- Prevents overfitting on small dataset

## 📝 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, CPU works fine)
- ~100 MB free disk space

## 📄 License

MIT License

## 👤 Author

Project created for Machine Learning course assignment.

## 🙏 Acknowledgments

- Dataset: UCI Machine Learning Repository - Auto MPG (https://archive.ics.uci.edu/ml/datasets/auto+mpg)
- Framework: PyTorch Team
