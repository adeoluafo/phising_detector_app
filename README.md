# 🛡️ Phishing Detection System
An intelligent email phishing detector that classifies emails as either "phishing" or "non-phising" using machine learning. It features a user-friendly GUI, real-time predictions, and supports CSV uploads for training and testing.
## 🎬 Demo

![Phishing Detector Demo](demo.gif)

## ✨ Features
- GUI built with Tkinter
- Email classification (Phishing vs. non-phising(legitimate))
- Upload and train on custom CSV datasets
- Choose model types: Logistic Regression, Random Forest, KNN
- View confusion matrix for performance
- Real-time predictions

## 🧰 Tech Stack

- Python
- Tkinter (GUI)
- scikit-learn
- Pandas
- Matplotlib
- BeautifulSoup
- Gmail API

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/adeoluafo/phising_detector_app.git
   cd phising_detector_app
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Add your Gmail API credentials:
   - Download `credentials.json` from Google Cloud Console.
   - Place it in the root directory (this file should not be pushed to GitHub).

## ▶️ Usage

1. Run the app:
   ```bash
   python phishing_detector_gui.py
   ```

2. Click **Run Full Pipeline** to fetch emails, train a model, and predict.

3. Or click **Run Prediction Only** to use existing trained models.

## 🔐 Gmail Authentication

When you click to fetch emails, a browser will open asking you to log into your Gmail account and authorize access. This uses Google OAuth2 and stores your token securely in `token.json` (excluded from Git).

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you’d like to change.

## 📄 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

Thanks to Google Developers for the Gmail API and OAuth2 tutorials.
