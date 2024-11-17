import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Fungsi untuk memuat dataset
def load_data():
    global df, X, y
    file_path = filedialog.askopenfilename()
    if file_path:
        df = pd.read_csv(file_path)
        if 'Timestamp' in df.columns:
            df.drop('Timestamp', axis=1, inplace=True)
        
        # Pilih fitur dan label yang relevan
        features_cfs = ['Init Bwd Win Byts', 'Dst Port', 'Fwd Pkt Len Max', 'Fwd Pkt Len Std', 'Fwd Seg Size Avg',
                        'Fwd Pkt Len Mean', 'ACK Flag Cnt', 'Pkt Len Mean', 'Pkt Len Max', 'Pkt Size Avg', 'PSH Flag Cnt',
                        'Pkt Len Std', 'Pkt Len Var', 'RST Flag Cnt', 'ECE Flag Cnt', 'Init Fwd Win Byts', 'Flow Byts/s',
                        'Bwd Seg Size Avg', 'Bwd Pkt Len Mean', 'Bwd Pkts/s', 'Tot Bwd Pkts', 'Subflow Bwd Pkts',
                        'Down/Up Ratio', 'Flow Pkts/s', 'Bwd Pkt Len Std', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd IAT Min',
                        'TotLen Bwd Pkts', 'Subflow Bwd Byts', 'Bwd Pkt Len Max']
        X = df[features_cfs]
        y = df['Label']

        messagebox.showinfo("Sukses", "Dataset berhasil dimuat!")

# Fungsi untuk menjalankan SVM dan menampilkan hasil
def run_svm():
    if 'X' not in globals() or 'y' not in globals():
        messagebox.showwarning("Error", "Silakan muat dataset terlebih dahulu!")
        return

    kernel = kernel_var.get()  # Ambil nilai kernel dari dropdown
    print(f"Training SVM dengan kernel {kernel}...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Normalisasi data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    svc = SVC(kernel=kernel)
    param_grid = {'C': [0.1, 1, 10, 100]}
    grid_search = GridSearchCV(svc, param_grid, cv=StratifiedKFold(3), scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Pemetaan hasil prediksi dan label asli
    y_pred_numeric = pd.Series(y_pred).map({'Benign': 0, 'DDOS attack-HOIC': 1})
    y_test_binary = y_test.map({'Benign': 0, 'DDOS attack-HOIC': 1})

    # Menghitung hasil akurasi dan laporan klasifikasi
    accuracy = accuracy_score(y_test_binary, y_pred_numeric)
    conf_matrix = confusion_matrix(y_test_binary, y_pred_numeric)
    class_report = classification_report(y_test_binary, y_pred_numeric)
    roc_auc = roc_auc_score(y_test_binary, best_model.decision_function(X_test))

    # Menampilkan hasil di GUI
    result_text.config(state=tk.NORMAL)
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, f"Kernel: {kernel}\n")
    result_text.insert(tk.END, f"Akurasi: {accuracy:.2f}\n")
    result_text.insert(tk.END, f"ROC AUC: {roc_auc:.2f}\n\n")
    result_text.insert(tk.END, "Matriks Kebingungan:\n")
    result_text.insert(tk.END, f"{conf_matrix}\n\n")
    result_text.insert(tk.END, "Laporan Klasifikasi:\n")
    result_text.insert(tk.END, f"{class_report}\n\n")
    
    # Menampilkan hasil klasifikasi per instance
    result_text.insert(tk.END, "\nHasil Klasifikasi per Instance:\n")
    for idx, pred in enumerate(y_pred_numeric):
        actual = y_test_binary.iloc[idx]
        actual_label = "Benign" if actual == 0 else "DDoS attack-HOIC"
        pred_label = "Benign" if pred == 0 else "DDoS attack-HOIC"
        result_text.insert(tk.END, f"Instance {idx+1}: Actual: {actual_label}, Predicted: {pred_label}\n")
    
    result_text.config(state=tk.DISABLED)

    # Menampilkan grafik ROC
    for widget in right_frame.winfo_children():
        widget.destroy()

    fpr, tpr, _ = roc_curve(y_test_binary, best_model.decision_function(X_test))
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5), dpi=80)
    ax_roc.plot(fpr, tpr, label=f"{kernel.capitalize()} Kernel (AUC = {roc_auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")

    canvas_roc = FigureCanvasTkAgg(fig_roc, master=right_frame)
    canvas_roc.get_tk_widget().pack(pady=10)
    canvas_roc.draw()

    # Menampilkan Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5), dpi=80)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title(f"Confusion Matrix for {kernel} Kernel")

    canvas_cm = FigureCanvasTkAgg(fig_cm, master=right_frame)
    canvas_cm.get_tk_widget().pack(pady=10)
    canvas_cm.draw()

    # Menampilkan Learning Curve
    for widget in learning_curve_frame.winfo_children():
        widget.destroy()

    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X_scaled, y, cv=StratifiedKFold(3, shuffle=True), scoring="accuracy", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    fig_lc, ax_lc = plt.subplots(figsize=(5, 4), dpi=80)
    ax_lc.plot(train_sizes, train_scores_mean, 'o-', label="Training Score")
    ax_lc.plot(train_sizes, test_scores_mean, 'o-', label="Validation Score")
    ax_lc.set_xlabel("Training Set Size")
    ax_lc.set_ylabel("Accuracy Score")
    ax_lc.set_title("Learning Curve")
    ax_lc.legend(loc="best")

    canvas_lc = FigureCanvasTkAgg(fig_lc, master=learning_curve_frame)
    canvas_lc.get_tk_widget().pack(pady=10)
    canvas_lc.draw()

# Inisialisasi GUI
root = tk.Tk()
root.title("SVM GUI untuk Deteksi DDoS")
root.geometry("1200x700")
root.configure(bg="#283593")

# Frame utama kiri dan kanan
left_frame = tk.Frame(root, bg="#283593")
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
left_frame.pack_propagate(False)

right_frame = tk.Frame(root, bg="#E8EAF6")
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
right_frame.pack_propagate(False)

# Sub-frame kiri atas untuk tombol dan hasil
button_result_frame = tk.Frame(left_frame, bg="#283593")
button_result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

# Sub-frame kiri bawah untuk grafik learning curve
learning_curve_frame = tk.Frame(left_frame, bg="#283593")
learning_curve_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)

# Frame untuk tombol dan dropdown
button_frame = tk.Frame(button_result_frame, bg="#283593")
button_frame.pack(pady=10)

# Tombol "Muat Dataset"
load_button = tk.Button(button_frame, text="Muat Dataset", command=load_data, width=15, bg="#1E88E5", fg="white", font=("Arial", 12))
load_button.pack(side=tk.LEFT, padx=10)

# Dropdown untuk memilih kernel
kernel_var = tk.StringVar(value="linear")
kernel_label = tk.Label(button_frame, text="Pilih Kernel:", fg="white", bg="#283593", font=("Arial", 12))
kernel_label.pack(side=tk.LEFT, padx=5)
kernel_options = ["linear", "poly", "rbf"]
kernel_dropdown = tk.OptionMenu(button_frame, kernel_var, *kernel_options)
kernel_dropdown.pack(side=tk.LEFT, padx=10)

# Tombol "Jalankan SVM"
run_button = tk.Button(button_frame, text="Jalankan SVM", command=run_svm, width=15, bg="#1E88E5", fg="white", font=("Arial", 12))
run_button.pack(side=tk.LEFT, padx=10)

# Kotak teks untuk menampilkan hasil
result_text = tk.Text(left_frame, height=15, width=40, wrap=tk.WORD, font=("Arial", 12))
result_text.pack(padx=5, pady=5)
result_text.config(state=tk.DISABLED)

root.mainloop()
