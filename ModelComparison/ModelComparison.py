import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
import os
import pickle
from sklearn.metrics import log_loss
import pandas as pd

mpl.rcParams['font.family'] = 'Times New Roman'

# Modelli usati
MODELS = ['decision_tree', 'random_forest', 'mlp']
LABELS = ['Decision Tree', 'Random Forest', 'MLP']

#Load data

def load_test_data(x_path="X_test.csv", y_path="y_test.csv"):
    X_test = pd.read_csv(x_path).values  # converte in numpy array
    y_test = pd.read_csv(y_path).values.ravel()  # .ravel() per avere un vettore 1D
    return X_test, y_test

# === Caricamento file JSON ===

def load_training_times(filepath="training_times.json"):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_classification_reports():
    reports = {}
    for model in MODELS:
        filename = f"classification_report_{model}_tuning.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                reports[model] = json.load(f)
        else:
            print(f"File non trovato: {filename}")
    return reports


# === Plot 1: Training Time ===

def plot_training_time(training_times):
    static_times = [training_times["static"][model] for model in MODELS]
    tuning_times = [training_times["tuning"][model] for model in MODELS]
    x = np.arange(len(MODELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, static_times, width, label='Static', 
                   color='#77dd77', edgecolor='black', linewidth=1)  
    bars2 = ax.bar(x + width/2, tuning_times, width, label='Tuning', 
                   color='#779ecb', edgecolor='black', linewidth=1)  

    ax.set_xlabel('Modello')
    ax.set_ylabel('Tempo di Addestramento (secondi)')
    ax.set_title('Confronto tra Tempi di Addestramento: Statico vs Tuning')
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("training_times.png", dpi=300)
    plt.show()


# === Plot 2: Macro Avg Metrics ===

def plot_avg_metrics(classification_reports, output_file="macro_avg_metrics_tuning.png"):
    metrics = ["precision", "recall", "f1-score"]
    values = {
        metric: [classification_reports[model]["macro avg"][metric] for model in MODELS]
        for metric in metrics
    }

    x = np.arange(len(MODELS))
    width = 0.2
    colors = {
        "precision": '#77dd77',  
        "recall": '#ff6961',     
        "f1-score": '#779ecb'    
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values[metric], width, label=metric.capitalize(),
                      color=colors[metric], edgecolor='black', linewidth=1)
        for xi, val in zip(x, values[metric]):
            ax.annotate(f'{val:.2f}', (xi + offset, val), textcoords="offset points",
                        xytext=(0, 3), ha='center', fontsize=8)

    ax.set_xlabel("Modelli")
    ax.set_ylabel("Valori")
    ax.set_title("Metriche di classificazione (macro average) per modello")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()


# === Plot 3: Log Loss ===

def plot_log_loss(X_test, y_test, model_dir='saved_models'):
    log_losses = []
    for model_name in MODELS:
        model_path = os.path.join(model_dir, f"{model_name}_tuning.pkl")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        y_proba = model.predict_proba(X_test)
        loss = log_loss(y_test, y_proba)
        log_losses.append(loss)

    y_pos = np.arange(len(MODELS))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(y_pos, log_losses, height=0.6, color='#ff6961', edgecolor='black', alpha=0.8)

    ax.set_xlabel('Log Loss', fontsize=12)
    ax.set_title('Confronto Log Loss tra Modelli', fontsize=14)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(LABELS, fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlim(0, max(log_losses) + 0.05)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.4f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=10)

    plt.tight_layout()
    plt.savefig("log_loss_tuning.png", dpi=300)
    plt.show()



# === MAIN ===

if __name__ == "__main__":
    training_times = load_training_times()
    classification_reports = load_classification_reports()

    plot_training_time(training_times)
    plot_avg_metrics(classification_reports)
    X_test, y_test = load_test_data("X_test.csv", "y_test.csv")
    plot_log_loss(X_test, y_test)
