# 🔥 Fire Detection System

A Deep Learning based Fire Detection Web Application built using TensorFlow and Streamlit.

The system predicts whether an uploaded image contains fire or not.

---

# 🚀 Features

- Upload image from your computer
- Predict Fire / No Fire
- Confidence score display
- Simple and clean Streamlit UI
- CNN-based image classification

---

# 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

---

# 📂 Project Structure

```bash
fire_detection_project/
│
├── dataset/
│
├── models/
│   └── fire_detection_model.h5
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-link>
cd fire_detection_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser:

```bash
http://localhost:8501
```

---

# 🧠 Model Training

To retrain the model:

```bash
python train_model.py
```

The trained model will be saved as:

```bash
models/fire_detection_model.h5
```

---

# 📸 How It Works

1. Upload an image
2. Click the **Predict** button
3. The AI model analyzes the image
4. Displays:
   - 🔥 Fire Detected
   - ✅ No Fire Detected

---

# 📦 Requirements

- Python 3.9+
- TensorFlow
- Streamlit

---

# 🔮 Future Improvements

- Live webcam detection
- CCTV fire monitoring
- Alarm system
- Email alerts
- Mobile app integration

---

# 👨‍💻 Author

Team 2 
---