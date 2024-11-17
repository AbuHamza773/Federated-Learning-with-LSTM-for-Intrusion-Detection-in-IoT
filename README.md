
# Federated Learning with LSTM for Intrusion Detection in IoT

## Overview
This repository implements a Federated Learning (FL) framework integrated with Long Short-Term Memory (LSTM) networks for intrusion detection in IoT-based Wireless Sensor Networks (WSNs). The model ensures data privacy while achieving high detection accuracy for various types of network attacks.

## Features
- Federated Learning for decentralized training.
- LSTM-based model for handling sequential data.
- Support for multiple datasets: WSN-DS, CIC-IDS-2017, and UNSW-NB15.
- Comprehensive evaluation metrics: Accuracy, F1 Score, FPR, RMSE.
- Preprocessing, training, and visualization tools.

## Datasets
The project utilizes the following datasets:
1. **WSN-DS**: Contains Blackhole, Grayhole, Flooding, and Scheduling attacks.
2. **CIC-IDS-2017**: Includes DDoS, Brute Force, Web, and Infiltration attacks.
3. **UNSW-NB15**: Covers Fuzzers, Backdoors, Worms, and more.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-repo/federated-lstm-iot-ids.git
   cd federated-lstm-iot-ids
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
### Preprocessing
Prepare datasets by running:
```
python preprocess.py
```

### Model Training
Train the model locally or in a federated setting:
```
python training.py
```

### Evaluation
Evaluate model performance:
```
python evaluation.py
```

### Visualization
Analyze datasets and training results using Jupyter Notebooks:
```
jupyter notebook notebooks/data_exploration.ipynb
jupyter notebook notebooks/training_visualization.ipynb
```

## File Structure
```
federated-lstm-iot-ids/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── wsn-ds.csv
│   ├── cic-ids-2017.csv
│   ├── unsw-nb15.csv
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── federated_learning.py
│   ├── training.py
│   ├── evaluation.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── training_visualization.ipynb
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Authors of WSN-DS, CIC-IDS-2017, and UNSW-NB15 datasets.
- TensorFlow and Keras for the deep learning framework.
