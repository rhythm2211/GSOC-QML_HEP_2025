# GSoC 2025 Evaluation Tasks: QML-HEP

## Applicant: Rhythm Suthar

## Overview
This repository contains the code and results for the Google Summer of Code 2025 evaluation tasks for QML-HEP projects, completed within a Kaggle Notebooks environment. It demonstrates skills in quantum computing, machine learning (including GNNs and hybrid quantum-classical models), data handling, and debugging.

### Tasks Completed: I, II, III, XI, VI

*(See the main Jupyter Notebook `gsoc-2025-evaluation-tasks-qml-hep.ipynb` for detailed implementation and discussion.)*

## Environment & Key Libraries
- **Environment:** Kaggle Notebooks  
- **Libraries:** Pennylane, Pennylane-Lightning, PyTorch, PyTorch Geometric, NumPy, Scikit-learn, Matplotlib, h5py, torchvision, tqdm.  
  

## Tasks Summary & Results

### Task I: Quantum Computing Part
- **Goal:** Implement basic quantum circuits.
- **Method:** Used Pennylane to create a 5-qubit circuit with a standard gate sequence (H, CNOT, SWAP, RX) and a separate 5-qubit SWAP test circuit to compare two prepared states.
- **Result:** Circuits implemented and visualized successfully. Calculated squared fidelity from SWAP test P(0) was **0.50**.

### Task II: Classical Graph Neural Network (GNN)
- **Goal:** Classify Quark/Gluon jets using GCN and GAT models on ParticleNet data subset (100k jets).
- **Method:** Used PyTorch Geometric.
  - Implemented data loading, preprocessing, and manual k-NN graph construction (k=8, workaround for `torch-cluster` install issues).
  - Trained GCN and GAT models (3 layers each, with BatchNorm) on GPU using Adam, BCEWithLogitsLoss, LR Scheduling, and Early Stopping (Val AUC based, patience=10).
  - Saved best models.
- **Results:** Both models trained successfully, achieving similar performance. Best models loaded for test evaluation:

| Model Architecture | Test Loss | Test Accuracy | Test AUC | Best Validation AUC (Epoch Saved) |
|--------------------|----------|--------------|----------|-----------------------------------|
| **GCN** | 0.4567 | 0.7940 (79.40%) | 0.8716 | 0.8703 (Epoch 35) |
| **GAT** | 0.4620 | 0.7931 (79.31%) | 0.8699 | 0.8717 (Epoch 23) |

### Task III: Open Task
- **Goal:** Comment on QC/QML.
- **Method:** Provided a written discussion on Parameterized Quantum Circuits (PQCs), covering the hybrid approach, NISQ relevance, and key challenges (barren plateaus, encoding, noise).

### Task XI: Simple Quantum Embedding
- **Goal:** Implement MLP->PQC hybrid model for function approximation (*Y = sin(X_0)*cos(X_1)*) using MSE loss.
- **Method:**
  - Used a 3-layer MLP (PyTorch) to generate parameters for a 4-qubit PQC (Pennylane, depth 2).
  - Implemented manual batching in the forward pass due to `TorchLayer` issues.
  - Encountered persistent GPU dtype errors (*Expected Double*) during backpropagation.
- **Result:** Successfully trained the model by forcing execution onto the CPU using float64 precision. Achieved low MSE loss, indicating successful learning, visualized with prediction vs actual plot.

### Task VI: Quantum Representation Learning
- **Goal:** Train a PQC embedding for MNIST digits using a contrastive loss base and SWAP test fidelity.
- **Method:**
  - Used 4-qubit PQC with AmplitudeEmbedding and StronglyEntanglingLayers (24 params).
  - Implemented 9-qubit SWAP test QNode.
  - Created `FidelityCircuit` module (manual batching) and custom pairwise `DataLoader`.
  - Trained using contrastive loss and Adam optimizer. Used `lightning.qubit` backend on GPU.
- **Result:**
  - Pipeline implemented successfully and ran end-to-end.
  - However, training was extremely slow (~11 min/epoch), and minimal learning (little separation in classes) was observed in the 5 epochs run due to time/computational constraints.

## Running the Code

- The MNIST dataset (Task VI) downloads automatically.
- The Quark/Gluon dataset (Task II) requires separate download and placement (see path in notebook, e.g., `/kaggle/input/gsoc-data-task-2/`).
- **Execution:** Run the notebook cells sequentially. Task II and Task VI training can take significant time. Task XI training runs on the CPU.


## Contact
**Rhythm Suthar**  
📧 rhythmsuthar123@gmail.com
🔗 https://www.linkedin.com/in/rhythm-suthar-626b70220/

