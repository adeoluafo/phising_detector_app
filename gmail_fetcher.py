import os
import base64
import csv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from bs4 import BeautifulSoup

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def decode_body(body_data):
    if not body_data:
        return "[No body found]"
    body_data = body_data.replace("-", "+").replace("_", "/")
    decoded_bytes = base64.b64decode(body_data)
    soup = BeautifulSoup(decoded_bytes, "lxml")
    return soup.get_text()

def fetch_emails(max_results=50):
    creds = authenticate_gmail()
    service = build('gmail', 'v1', credentials=creds)

    result = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = result.get('messages', [])

    email_data = []

    for msg in messages:
        msg_detail = service.users().messages().get(userId='me', id=msg['id']).execute()
        payload = msg_detail.get('payload', {})
        headers = payload.get('headers', [])

        sender = recipient = subject = ""
        for h in headers:
            name = h.get('name', '').lower()
            if name == 'from':
                sender = h.get('value', '')
            elif name == 'to':
                recipient = h.get('value', '')
            elif name == 'subject':
                subject = h.get('value', '')

        body = "[No body found]"
        parts = payload.get('parts', [])
        if parts:
            for part in parts:
                if part.get('mimeType') == 'text/plain':
                    body = decode_body(part.get('body', {}).get('data'))
                    break
        else:
            # Sometimes body might be in payload directly
            body = decode_body(payload.get('body', {}).get('data'))

        email_data.append([sender, recipient, subject, body])

    return email_data

def save_emails_to_csv(email_data, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Sender', 'Recipient', 'Subject', 'Message'])
        writer.writerows(email_data)
