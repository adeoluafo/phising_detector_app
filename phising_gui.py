import sys
print(sys.executable)
print(sys.version)
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import pandas as pd
from CreateML_Model import train_model, predict_model, save_artifacts, load_artifacts

class PhishingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phishing Detection Pipeline")
        self.root.geometry("600x500")

        # Variables
        self.train_file_path = ""
        self.predict_file_path = ""
        self.model_type = tk.StringVar(value="RandomForest")
        self.pipeline_mode = tk.StringVar(value="train_predict")

        # UI Components
        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10, fill="x")

        # Model selection
        tk.Label(frame, text="Select Model:").grid(row=0, column=0, sticky="w")
        model_combo = ttk.Combobox(frame, textvariable=self.model_type, state="readonly",
                                   values=["KNN", "LogisticRegression", "RandomForest", "SVM"])
        model_combo.grid(row=0, column=1, sticky="ew")

        # Pipeline mode
        tk.Label(frame, text="Pipeline Mode:").grid(row=1, column=0, sticky="w")
        modes = [("Train + Predict", "train_predict"), ("Predict Only", "predict_only")]
        for idx, (text, mode) in enumerate(modes):
            tk.Radiobutton(frame, text=text, variable=self.pipeline_mode, value=mode).grid(row=1, column=1+idx, sticky="w")

        # Training file
        tk.Button(frame, text="Select Training CSV", command=self.load_train_file).grid(row=2, column=0, sticky="w")
        self.train_file_label = tk.Label(frame, text="No file selected")
        self.train_file_label.grid(row=2, column=1, columnspan=3, sticky="w")

        # Prediction file
        tk.Button(frame, text="Select Prediction CSV", command=self.load_predict_file).grid(row=3, column=0, sticky="w")
        self.predict_file_label = tk.Label(frame, text="No file selected")
        self.predict_file_label.grid(row=3, column=1, columnspan=3, sticky="w")

        # Run button
        tk.Button(frame, text="Run Pipeline", command=self.run_pipeline).grid(row=4, column=0, pady=10)

        # Fetch Emails
        tk.Button(frame, text="Fetch Emails", command=self.fetch_and_display_emails).grid(row=4, column=1, pady=10)


        # Log box
        self.log_box = scrolledtext.ScrolledText(self.root, height=15)
        self.log_box.pack(padx=10, pady=10, fill="both", expand=True)

        # Configure grid weights
        for i in range(4):
            frame.grid_columnconfigure(i, weight=1)

    def log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        self.root.update()

    def load_train_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.train_file_path = path
            self.train_file_label.config(text=path)
            self.log(f"Selected training file: {path}")

    def load_predict_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.predict_file_path = path
            self.predict_file_label.config(text=path)
            self.log(f"Selected prediction file: {path}")

    def run_pipeline(self):
        mode = self.pipeline_mode.get()
        model_type = self.model_type.get()

        if mode == "train_predict":
            if not self.train_file_path or not self.predict_file_path:
                messagebox.showerror("Error", "Please select both training and prediction CSV files.")
                return

            try:
                self.log(f"Loading training data from {self.train_file_path}...")
                train_df = pd.read_csv(self.train_file_path)

                self.log(f"Training {model_type} model...")
                model, scaler, tfidf_vectorizer, pca = train_model(train_df, model_type)
                self.log("Training complete.")

                save_artifacts(model, scaler, tfidf_vectorizer, pca, "phishing_model")
                self.log("Model and preprocessors saved.")

                self.log(f"Loading prediction data from {self.predict_file_path}...")
                predict_df = pd.read_csv(self.predict_file_path)

                self.log("Running prediction on new data...")
                preds = predict_model(model, scaler, tfidf_vectorizer, pca, predict_df)
                predict_df['Prediction'] = preds

                save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                         filetypes=[("CSV files", "*.csv")],
                                                         title="Save Prediction Results")
                if save_path:
                    predict_df.to_csv(save_path, index=False)
                    self.log(f"Predictions saved to {save_path}")
                else:
                    self.log("Prediction results not saved.")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
                self.log(f"Error: {e}")

        elif mode == "predict_only":
            if not self.predict_file_path:
                messagebox.showerror("Error", "Please select a prediction CSV file.")
                return

            try:
                self.log("Loading model and preprocessors from files...")
                model, scaler, tfidf_vectorizer, pca = load_artifacts("phishing_model")

                self.log(f"Loading prediction data from {self.predict_file_path}...")
                predict_df = pd.read_csv(self.predict_file_path)

                self.log("Running prediction with loaded model...")
                preds = predict_model(model, scaler, tfidf_vectorizer, pca, predict_df)
                predict_df['Prediction'] = preds

                save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                         filetypes=[("CSV files", "*.csv")],
                                                         title="Save Prediction Results")
                if save_path:
                    predict_df.to_csv(save_path, index=False)
                    self.log(f"Predictions saved to {save_path}")
                else:
                    self.log("Prediction results not saved.")

            except FileNotFoundError:
                messagebox.showerror("Error", "No trained model found. Please train a model first.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
                self.log(f"Error: {e}")
    def fetch_and_display_emails(self):
        try:
            self.log("Fetching emails from Gmail...")
            from gmail_fetcher import fetch_emails  # Make sure this import works
            emails = fetch_emails()

            if not emails:
                self.log("No emails fetched.")
                return

            for i, email in enumerate(emails[:5]):  # Display first 5 emails
                sender, recipient, subject, body = email
                self.log(f"\nEmail {i+1}:\nFrom: {sender}\nTo: {recipient}\nSubject: {subject}\nMessage: {body[:200]}...\n")

        except Exception as e:
            self.log(f"Error fetching emails: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PhishingApp(root)
    root.mainloop()
