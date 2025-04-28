# Diabetic Retinopathy Detection App

This is a **Streamlit web application** that analyzes retinal images to detect the stages of **Diabetic Retinopathy (DR)**. It uses a pre-trained **EfficientNetB3** deep learning model trained on the **APTOS 2019 Blindness Detection** dataset.

---

## 📂 Project Structure
```
/diabetic-retinopathy-app
 ├── app.py                   # Main Streamlit app
 ├── efficientnetb3_retinopathy.h5  # Trained model
 └── requirements.txt        # Project dependencies
```

---

## 🚀 Features
- Upload and analyze **retinal fundus images** (.jpg, .png)
- Predicts DR stage:
  - No DR
  - Mild
  - Moderate
  - Severe
  - Proliferative DR
- Visualizes model confidence using a bar chart
- Displays side-by-side comparison of original and resized images
- Fast prediction thanks to model caching

---

## 📚 Technologies Used
- **Streamlit**
- **TensorFlow / Keras**
- **EfficientNetB3** architecture
- **NumPy**, **Pandas**, **Pillow (PIL)**

---

## 💪 Model Training Steps
1. **Dataset**: APTOS 2019 Blindness Detection dataset
2. **Preprocessing**:
   - Resized images to 512x512
   - Applied augmentations (rotation, zoom, brightness adjustment, flips)
   - Balanced minority classes through augmentation
3. **Model Building**:
   - Base Model: EfficientNetB3 with ImageNet weights
   - Added Global Average Pooling, Flatten, Dropout, Dense layer
   - Loss function: Categorical Crossentropy with Label Smoothing
4. **Training**:
   - Optimizer: Adam
   - Early stopping, Model checkpointing, Reduce LR on Plateau
5. **Saving the Model**:
   - Best model saved as `efficientnetb3_retinopathy.h5`

---

## 📊 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Parakh-Agrawal/Diabetic_rectenopathy_detection.git
cd diabetic-retinopathy-app
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Run the App Locally
```bash
streamlit run web.py
```

The app will open automatically in your web browser at [localhost:8501](http://localhost:8501/).

### 4. Deployment
Deploy easily on [Streamlit Community Cloud](https://streamlit.io/cloud) by connecting this repository.

---

## 📈 Results
| Metric             | Value    |
|--------------------|----------|
| **Validation Accuracy** | ~75%     |
| **Test Accuracy**       | ~85%     |
| **Model Size**           | ~41 MB  |

- Model generalizes well for unseen retinal images.
- Achieves reliable prediction for all five DR classes.

---

## ✨ Acknowledgments
- [APTOS 2019 Blindness Detection Dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
# Diabetic_rectenopathy_detection
## Deployed link (https://parakh-agrawal-diabetic-rectenopathy-detection-web-dkhid2.streamlit.app/)
