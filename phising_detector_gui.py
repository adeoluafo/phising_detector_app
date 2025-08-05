import os
import base64
import threading 
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

vectorizer_cache = None
model_cache = None

selected_training_path = None  # Will hold your uploaded CSV path

def authenticate_gmail():
    # Always prompt user to log in with their own Gmail
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    return creds

def decode_body(body_data):
    if not body_data:
        return "[No body found]"
    body_data = body_data.replace("-", "+").replace("_", "/")
    decoded_bytes = base64.b64decode(body_data)
    soup = BeautifulSoup(decoded_bytes, "lxml")
    return soup.get_text()

import re

def clean_invisible_chars(text):
    if isinstance(text, str):
        return re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    return text

def fetch_emails(max_total=2000):
    creds = authenticate_gmail()
    service = build('gmail', 'v1', credentials=creds)

    email_data = []
    next_page_token = None
    fetched_count = 0

    while fetched_count < max_total:
        result = service.users().messages().list(
            userId='me',
            maxResults=500,
            pageToken=next_page_token
        ).execute()

        messages = result.get('messages', [])
        next_page_token = result.get('nextPageToken', None)

        for msg in messages:
            if fetched_count >= max_total:
                break
            msg_detail = service.users().messages().get(userId='me', id=msg['id']).execute()
            payload = msg_detail.get('payload', {})
            headers = payload.get('headers', [])
            sender = recipient = subject = ""
            for h in headers:
                name = h.get('name', '').lower()
                if name == 'from': sender = h.get('value', '')
                elif name == 'to': recipient = h.get('value', '')
                elif name == 'subject': subject = h.get('value', '')
            body = "[No body found]"
            parts = payload.get('parts', [])
            if parts:
                for part in parts:
                    if part.get('mimeType') == 'text/plain':
                        body = decode_body(part.get('body', {}).get('data'))
                        break
            else:
                body = decode_body(payload.get('body', {}).get('data'))
            email_data.append({
                'sender': sender,
                'recipient': recipient,
                'subject': subject,
                'body': body
            })
            fetched_count += 1

        if not next_page_token:
            break  # No more pages

    df = pd.DataFrame(email_data)
    df = df.applymap(clean_invisible_chars)  # Clean invisible characters
    df.to_csv("FetchedEmails.csv", index=False)
    return df


def load_vectorizer_and_model():
    global vectorizer_cache, model_cache
    if vectorizer_cache is None:
        vectorizer_cache = joblib.load("phishing_model_tfidf.pkl")
    if model_cache is None:
        model_cache = joblib.load("phishing_model_model.pkl")
    return vectorizer_cache, model_cache

def draw_confusion_matrix(y_true, y_pred):
    for widget in plot_frame.winfo_children():
        widget.destroy()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def predict_emails():
    vectorizer, model = load_vectorizer_and_model()
    df = pd.read_csv("PredictionEmails.csv", encoding="utf-8").fillna("")

    phishing_keywords = ['free', 'winner', 'urgent']
    df['link_count'] = df['body'].apply(lambda x: x.count('http') if isinstance(x, str) else 0)
    df['suspicious_keyword_count'] = df['body'].apply(
        lambda x: sum(1 for kw in phishing_keywords if isinstance(x, str) and kw in x.lower())
    )
    df['body_length'] = df['body'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    df['uses_public_email'] = df['sender'].apply(
        lambda x: 1 if isinstance(x, str) and ('@gmail.com' in x or '@yahoo.com' in x) else 0
    )
    df['domain_unknown'] = df['sender'].apply(
        lambda x: 1 if isinstance(x, str) and not any(d in x for d in ['trusted.com', 'safe.org']) else 0
    )

    texts = df['sender'].astype(str) + ' ' + df['subject'].astype(str) + ' ' + df['body'].astype(str)
    tfidf_matrix = vectorizer.transform(texts)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    manual_df = df[['link_count', 'suspicious_keyword_count', 'body_length', 'uses_public_email', 'domain_unknown']]
    full_features = pd.concat([manual_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    preds = model.predict(full_features)
    df['Prediction'] = ['Phishing' if p == 1 else 'Not Phishing' for p in preds]
    df[['sender', 'subject', 'Prediction']].to_csv("PredictedEmails.csv", index=False)
    show_prediction_table(df)

    if os.path.exists("TrainingEmails.csv"):
        train_df = pd.read_csv("TrainingEmails.csv")
        if 'label' in train_df.columns:
            phishing_keywords = ['free', 'winner', 'urgent']
            train_df['link_count'] = train_df['body'].apply(lambda x: x.count('http') if isinstance(x, str) else 0)
            train_df['suspicious_keyword_count'] = train_df['body'].apply(
                lambda x: sum(1 for kw in phishing_keywords if isinstance(x, str) and kw in x.lower())
            )
            train_df['body_length'] = train_df['body'].apply(lambda x: len(x) if isinstance(x, str) else 0)
            train_df['uses_public_email'] = train_df['sender'].apply(
                lambda x: 1 if isinstance(x, str) and ('@gmail.com' in x or '@yahoo.com' in x) else 0
            )
            train_df['domain_unknown'] = train_df['sender'].apply(
                lambda x: 1 if isinstance(x, str) and not any(d in x for d in ['trusted.com', 'safe.org']) else 0
            )
            texts_train = train_df['sender'].astype(str) + ' ' + train_df['subject'].astype(str) + ' ' + train_df['body'].astype(str)
            tfidf_train = vectorizer.transform(texts_train)
            tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), columns=vectorizer.get_feature_names_out())
            manual_train_df = train_df[['link_count', 'suspicious_keyword_count', 'body_length', 'uses_public_email', 'domain_unknown']]
            full_train = pd.concat([manual_train_df.reset_index(drop=True), tfidf_train_df.reset_index(drop=True)], axis=1)

            y_true = train_df['label']
            y_pred = model.predict(full_train)
            root.after(0, lambda: draw_confusion_matrix(y_true, y_pred))

def show_prediction_table(df):
    for widget in table_frame.winfo_children():
        widget.destroy()
    cols = ["No", "Sender", "Subject", "Prediction"]
    tree = ttk.Treeview(table_frame, columns=cols, show="headings")
    for c in cols:
        tree.heading(c, text=c)
        width = 50 if c == "No" else 200 if c == "Prediction" else 300
        tree.column(c, width=width, anchor='center' if c in ["No", "Prediction"] else 'w')
    for i, row in df.iterrows():
        tree.insert('', 'end', values=(i+1, row['sender'], row['subject'], row['Prediction']))
    vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    vsb.pack(side='right', fill='y')
    tree.pack(fill='both', expand=True)

def run_full_pipeline():
    progress_bar.start()
    threading.Thread(target=_run_full_pipeline, daemon=True).start()

def _run_full_pipeline():
    try:
        if selected_training_path:
            df = pd.read_csv(selected_training_path).fillna("")
            print(f"✅ Loaded custom file: {selected_training_path} with {len(df)} rows")
        else:
            fetch_emails()
            df = pd.read_csv("FetchedEmails.csv").fillna("")
            print(f"✅ Fetched from Gmail: {len(df)} rows")

        phishing_keywords = ["urgent", "free", "click here", "winner", "prize"]
        df["label"] = df["body"].apply(
            lambda text: 1 if isinstance(text, str) and any(kw in text.lower() for kw in phishing_keywords) else 0
        )

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * 0.7)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]

        df_train.to_csv("TrainingEmails.csv", index=False)
        df_test.to_csv("PredictionEmails.csv", index=False)

        print(f"✅ Split: {len(df_train)} train, {len(df_test)} test")

        texts_train = df_train['sender'].astype(str) + ' ' + df_train['subject'].astype(str) + ' ' + df_train['body'].astype(str)
        vectorizer = TfidfVectorizer(max_features=300)
        tfidf_train = vectorizer.fit_transform(texts_train)
        tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), columns=vectorizer.get_feature_names_out())

        for dataset in [df_train, df_test]:
            dataset['link_count'] = dataset['body'].apply(lambda x: x.count('http') if isinstance(x, str) else 0)
            dataset['suspicious_keyword_count'] = dataset['body'].apply(lambda x: sum(1 for kw in phishing_keywords if isinstance(x, str) and kw in x.lower()))
            dataset['body_length'] = dataset['body'].apply(lambda x: len(x) if isinstance(x, str) else 0)
            dataset['uses_public_email'] = dataset['sender'].apply(lambda x: 1 if isinstance(x, str) and ('@gmail.com' in x or '@yahoo.com' in x) else 0)
            dataset['domain_unknown'] = dataset['sender'].apply(lambda x: 1 if isinstance(x, str) and not any(d in x for d in ['trusted.com', 'safe.org']) else 0)

        manual_train_df = df_train[['link_count', 'suspicious_keyword_count', 'body_length', 'uses_public_email', 'domain_unknown']]
        X_train = pd.concat([manual_train_df.reset_index(drop=True), tfidf_train_df.reset_index(drop=True)], axis=1)
        y_train = df_train['label']

        model_type = model_choice.get()
        if model_type == "Logistic":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "KNN":
            model = KNeighborsClassifier()
        elif model_type == "RF":
            model = RandomForestClassifier()
        else:
            raise ValueError("Unknown model type selected.")

        model.fit(X_train, y_train)

        joblib.dump(model, "phishing_model_model.pkl")
        joblib.dump(vectorizer, "phishing_model_tfidf.pkl")

        predict_emails()
        root.after(0, lambda: messagebox.showinfo("Pipeline", f"✅ Full pipeline with {model_type} completed successfully."))

    except Exception as e:
        root.after(0, lambda e=e: messagebox.showerror("Error", str(e)))
    finally:
        root.after(0, progress_bar.stop)

def run_prediction_only():
    progress_bar.start()
    threading.Thread(target=_run_prediction_only, daemon=True).start()

def _run_prediction_only():
    try:
        fetch_emails()
        predict_emails()
    except Exception as e:
        root.after(0, lambda e=e: messagebox.showerror("Error", str(e)))
    finally:
        root.after(0, progress_bar.stop)

def choose_file():
    global selected_training_path, training_label
    filepath = filedialog.askopenfilename(
        filetypes=[("CSV Files", "*.csv")],
        title="Select a CSV file"
    )
    if filepath:
        selected_training_path = filepath
        training_label.config(text=os.path.basename(filepath), fg="black")

def setup_gui():
    global model_choice, training_label, table_frame, plot_frame, progress_bar
    root.title("Phishing Detection Pipeline")
    root.geometry("1400x800")
    top = tk.Frame(root)
    top.pack(side='top', fill='x')
    tk.Label(top, text="Phishing Detection System", font=("Helvetica", 16, "bold")).pack(pady=10)
    tk.Label(top, text="Select Model Type:", font=("Helvetica", 10)).pack()
    model_choice = ttk.Combobox(top, values=["Logistic", "RF", "KNN"])
    model_choice.set("Logistic")
    model_choice.pack()
    tk.Label(top, text="Optional: Upload Custom Training CSV", font=("Helvetica", 10)).pack(pady=(15, 0))
    tk.Button(top, text="Choose File", command=choose_file).pack()
    training_label = tk.Label(top, text="No custom file loaded", fg="gray")
    training_label.pack()
    tk.Button(top, text="Run Full Pipeline (Train + Predict)", command=run_full_pipeline, bg="#d17edd", width=40).pack(pady=(20, 5))
    tk.Button(top, text="Run Prediction Only (Skip Training)", command=run_prediction_only, bg="#cfe2ff", width=40).pack()
    progress_bar = ttk.Progressbar(top, mode="indeterminate")
    progress_bar.pack(pady=(10, 0), fill='x')
    middle = tk.Frame(root)
    middle.pack(side='top', fill='both', expand=True)
    plot_frame = tk.Frame(middle, width=600)
    plot_frame.pack(side='left', fill='both', expand=True)
    table_frame = tk.Frame(middle)
    table_frame.pack(side='left', fill='both', expand=True)
    root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    setup_gui()