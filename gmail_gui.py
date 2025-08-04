import tkinter as tk
from tkinter import filedialog, messagebox
from gmail_fetcher import fetch_emails, save_emails_to_csv

class GmailDownloaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gmail Email Downloader")
        self.root.geometry("350x120")

        self.download_btn = tk.Button(root, text="Download Emails as CSV", command=self.download_emails)
        self.download_btn.pack(pady=30)

    def download_emails(self):
        try:
            self.download_btn.config(state='disabled')
            self.root.update()
            email_data = fetch_emails(max_results=50)

            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                     filetypes=[("CSV files", "*.csv")],
                                                     title="Save Emails CSV")
            if file_path:
                save_emails_to_csv(email_data, file_path)
                messagebox.showinfo("Success", f"Emails saved to {file_path}")
            else:
                messagebox.showinfo("Cancelled", "Save operation cancelled.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")
        finally:
            self.download_btn.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    app = GmailDownloaderApp(root)
    root.mainloop()
