import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

class CRMDataCleaner:
    def __init__(self, root):
        self.root = root
        self.root.title("AI CRM Data Cleaner")
        self.file_path = ""

        # GUI Components
        self.upload_btn = ttk.Button(root, text="Upload CSV File", command=self.upload_file)
        self.upload_btn.pack(pady=10)

        self.clean_btn = ttk.Button(root, text="Clean Data", command=self.clean_data)
        self.clean_btn.pack(pady=10)

        self.output_text = tk.Text(root, height=20, width=80)
        self.output_text.pack(pady=10)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path = file_path
            self.output_text.insert(tk.END, f"Uploaded File: {file_path}\n")

    def clean_data(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please upload a CSV file first.")
            return

        try:
            df = pd.read_csv(self.file_path)
            self.output_text.insert(tk.END, f"Original Data Shape: {df.shape}\n")

            # Drop duplicate entries
            df = df.drop_duplicates()
            self.output_text.insert(tk.END, f"After Dropping Duplicates: {df.shape}\n")

            # Impute missing values using KNN
            numeric_cols = df.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                imputer = KNNImputer(n_neighbors=3)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                self.output_text.insert(tk.END, f"Missing values in numeric columns handled.\n")

            # Clustering to detect anomalies
            if len(numeric_cols) >= 2:
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                clusters = dbscan.fit_predict(df[numeric_cols])
                df['Cluster'] = clusters
                df = df[df['Cluster'] != -1]  # Remove outliers
                df.drop(columns=['Cluster'], inplace=True)
                self.output_text.insert(tk.END, f"Outliers removed using DBSCAN.\n")

            # Save cleaned data
            cleaned_path = os.path.splitext(self.file_path)[0] + '_cleaned.csv'
            df.to_csv(cleaned_path, index=False)
            self.output_text.insert(tk.END, f"Cleaned data saved to: {cleaned_path}\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == '__main__':
    root = tk.Tk()
    app = CRMDataCleaner(root)
    root.mainloop()
