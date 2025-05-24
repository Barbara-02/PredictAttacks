from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
import json

class EvaluatorAndAnalysis:
    def __init__(self, model,X_train, X_test, y_train, y_test, model_type, mode):

        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_type = model_type
        self.mode = mode

    #Calcola le metriche di classificazione 
    def evaluate_model(self):
    
        if self.model is None:
            raise ValueError("Nessun modello addestrato o caricato. Eseguire prima il training o il caricamento.")
        
        # Calcola il classification report
        predictions = self.model.predict(self.X_test)
        report_dict = classification_report(self.y_test, predictions, output_dict=True)
        
        # Stampa il report sul terminale
        print("\n Report di classificazione:")
        print(classification_report(self.y_test, predictions))
        
        filename = f"classification_report_{self.model_type}_{self.mode}.json"
        
        with open(filename, "w") as file:
            json.dump(report_dict, file, indent=4)
            print(f" Report salvato come {filename}")
        return report_dict        

    # Calcolo della matrice di confusione
    def plot_confusion_matrix(self):

        if self.model is None:
            raise ValueError("Nessun modello addestrato o caricato. Eseguire prima il training o il caricamento.")
        
         # Calcola la matrice di confusione
        predictions = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, predictions)
        
        # Crea la heatmap della matrice di confusione
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        heatmap_name=f"confusion_matrix_{self.model_type}_{self.mode}.png"
        plt.savefig(heatmap_name, bbox_inches="tight")
        print("Heatmap salvata come ", heatmap_name)
    from sklearn.model_selection import learning_curve

    #Calcolo delle curve di apprendimento
    def plot_learning_curves(self, X_train, y_train, scoring='accuracy', cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)):

        if self.model is None:
            raise ValueError("Nessun modello addestrato o caricato. Eseguire prima il training o il caricamento.")

        print("Calcolo delle learning curves...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=12,
            train_sizes=np.linspace(0.01, 1.0, 15),
            shuffle=True,
            random_state=42
        )

        # Media e deviazione standard
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        #Plot delle curve
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color='blue')

        plt.plot(train_sizes, val_scores_mean, 'o-', color='green', label='Validation score')
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                         val_scores_mean + val_scores_std, alpha=0.1, color='green')

        plt.title(f'Learning Curves - {self.model_type} ({self.mode})')
        plt.xlabel('Training set size')
        plt.ylabel(scoring.capitalize())
        plt.legend(loc='best')
        plt.grid()
        plt.tight_layout()

        filename = f"learning_curve_{self.model_type}_{self.mode}.png"
        plt.savefig(filename)
        print(f"Learning curve salvata come {filename}")

    #Calcolo della varianza spiegata
    def plot_pca_explained_variance(self):
        # Verifica dell'applicazione della PCA
        if hasattr(self.model, 'named_steps') and 'pca' in self.model.named_steps:
            pca_step = self.model.named_steps['pca']
            explained_variance = pca_step.explained_variance_ratio_
            n_components_set = pca_step.n_components
            if isinstance(n_components_set, float):
                threshold = n_components_set
            else:
                threshold = 0.90  # valore di default nel caso non sia stato messo un float

            # Plot della varianza spiegata cumulativa
            plt.figure(figsize=(8, 6))
            cumulative_variance = np.cumsum(explained_variance)
            n_components = len(cumulative_variance)

            plt.plot(cumulative_variance, marker='o', color='purple')
            plt.axhline(y=threshold, color='red', linestyle='--', label=f'Soglia {int(threshold*100)}%')
            plt.xlabel('Numero di Componenti')
            plt.ylabel('Varianza Spiegata Cumulativa')
            plt.title(f'PCA Explained Variance - {self.model_type} ({self.mode})')
            plt.legend()
            plt.grid(True)
    
            # Imposta i tick dell'asse X a intervalli regolari
            max_tick = max(25, threshold)
            xticks = np.arange(0, max_tick + 1, 5)
            plt.xticks(ticks=xticks)
    
            plt.tight_layout()
    
            pca_plot_name = f"pca_explained_variance_{self.model_type}_{self.mode}.png"
            plt.savefig(pca_plot_name)
            print(f" Grafico PCA salvato come {pca_plot_name}")
            plt.show()
        else:
            print(" Questo modello non utilizza PCA (nessun passo 'pca' trovato nel pipeline).")
            
    def run_all(self):

        self.evaluate_model()
        self.plot_confusion_matrix()
        self.plot_learning_curves(self.X_train,self.y_train)
        self.plot_pca_explained_variance()