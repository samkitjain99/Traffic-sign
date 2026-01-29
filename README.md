# 🚦 Traffic Sign Recognition using CNN
This project is a **Convolutional Neural Network (CNN)** model designed to recognize **traffic signs** from images using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. It can identify **43 different traffic sign classes**, which is essential for autonomous driving systems and driver assistance technologies.  
---

## 🔍 Why This Project Matters
Traffic sign recognition is a key component for:
- Autonomous vehicles
- Advanced Driver Assistance Systems (ADAS)
- Improving road safety

The goal of this project is to build a **robust deep learning model** that can accurately classify traffic signs under various conditions, such as different lighting, angles, and weather scenarios.  

---

## 📊 Dataset Overview
- **Source:** Kaggle – GTSRB (German Traffic Sign Recognition Benchmark)  
- **Classes:** 43 traffic sign categories  
- **Images:** RGB images of varying sizes  
- **Preprocessing Steps:**  
  - Resize images to **50 × 50 pixels**  
  - Normalize pixel values to **[0, 1]**  
  - Encode labels using **one-hot encoding**  

---

## 🧠 Model Architecture
The CNN model is built with **TensorFlow/Keras** and includes:  
- Multiple **convolutional layers** with ReLU activations  
- **MaxPooling** layers to reduce spatial dimensions  
- **Dropout layers (0.5)** to prevent overfitting  
- Fully connected **Dense layers**  
- A **softmax output layer** for multi-class classification  

**Loss function:** Categorical Crossentropy  
**Optimizer:** Adam  

---

## 📈 Training Details
- **Epochs:** Up to 10 (with Early Stopping)
- **Batch Size:** 128
- **Training/Validation Split:** 80/20
- **Early Stopping:** Monitors validation loss with a patience of 3 epochs and restores the best model weights

The model trains efficiently and quickly converges to a high validation accuracy.  

---

## 🧪 Results
| Metric | Value |
|--------|-------|
| Training Accuracy | ~93.39% |
| Validation Accuracy | **~99.12%** |

The results show the model generalizes well and can reliably predict traffic signs on unseen images.  

---

## 🔬 Model Evaluation
- Accuracy and loss curves were tracked during training  
- Model predictions were verified on test images to ensure reliable performance  

---

## 🛠 Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn  
- Kaggle API
- <img width="626" height="498" alt="Screenshot 2026-01-28 233745" src="https://github.com/user-attachments/assets/d875336d-dc71-420c-b163-ed046c3bd93a" />



