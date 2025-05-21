import pandas as pd 
import numpy as np 
import ipaddress 
import seaborn as sns
import matplotlib.pyplot as plt 

class Preprocessor:
    def __init__(self, csv_path, output_path='preprocessed_dataset.csv'):
        self.csv_path = csv_path
        self.output_path = output_path
        self.df = None
        self.label_mapping_inverse= {}


    #Conversione indirizzo IP in intero decimale
    def _ip_to_int(self, ip):
        try:
            return int(ipaddress.IPv4Address(ip))
        except Exception:
            return np.nan

    #Caricamento del dataset in memoria
    def load_data(self):
        self.df = pd.read_csv(self.csv_path)

    #Step del preprocessing
    def clean(self):

        #Controllo dell'esistenza del DataFrame
        if self.df is None:
            self.load_data()

        # Rimozione dell'ultima colonna
        self.df = self.df.iloc[:, :-1]

        # Rimozione della colonna "Label" se esiste, prevenire data leakage
        if 'Label' in self.df.columns:
            self.df.drop(columns=['Label'], inplace=True)
        

        # Conversione IP (colonne 0 e 2)
        self.df.iloc[:, 0] = self.df.iloc[:, 0].apply(self._ip_to_int)
        self.df.iloc[:, 2] = self.df.iloc[:, 2].apply(self._ip_to_int)

        # Sostituzione infiniti con NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Rimozione dei record con NaN e duplicati
        self.df.dropna(inplace=True)

        #Limitazione dei valori numerici troppo grandi (es. sopra 1e9)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df[col] = np.clip(self.df[col], -1e9, 1e9)

        self.df.drop_duplicates(inplace=True)

    #Frequency encoding della colonna target
    def map_target_to_numeric(self):

        #Controllo dell'esistenza del DataFrame 
        if self.df is None:
            raise ValueError("Errore: caricare il dataset.")

        target_col = self.df.columns[-1]

        # Controllo tipologia feature. Se numerica, non è necessario l'encoding
        if pd.api.types.is_numeric_dtype(self.df[target_col]):
            print(f"La colonna target '{target_col}' è numerica.")
            return

        # Conteggio delle occorrenze e creazione mapping
        class_counts = self.df[target_col].value_counts()
        sorted_classes = class_counts.index.tolist()
        label_mapping = {label: idx for idx, label in enumerate(sorted_classes)}

        # Applicazione del mapping
        self.df[target_col] = self.df[target_col].map(label_mapping)
        print(f"Target '{target_col}' convertito in numerico con mapping: {label_mapping}")

        self.label_mapping_inverse = {v: k for k, v in label_mapping.items()}  


    #Rimozione delle features con correlazione superiore al 90%
    def remove_highly_correlated_features(self, threshold=0.9,
                                      corr_matrix_output='correlation_matrix.csv',
                                      show_heatmap=True,
                                      heatmap_before_output='correlation_heatmap_before.png',
                                      heatmap_after_output='correlation_heatmap_after.png'):

        if self.df is None:
            raise ValueError("Errore: caricare e processare il dataset.")

        # Seleziona tutte le colonne numeriche 
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("Non è stata trovata alcuna colonna numerica.")

        target_col = numeric_cols[-1]

        # Matrice di correlazione
        corr_matrix = self.df[numeric_cols].corr(method='pearson').abs()
        corr_matrix.to_csv(corr_matrix_output)

        # Heatmap prima della rimozione
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", square=True)
        plt.title("Initial correlation matrix")
        plt.tight_layout()
        plt.savefig(heatmap_before_output)
        if show_heatmap:
            plt.show()
        plt.close()

        # Matrice triangolare superiore (esclude la diagonale principale)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Identifica feature altamente correlate da rimuovere (escludendo il target)
        to_drop = []
        for col in upper.columns:
            for row in upper.index:
                if upper.loc[row, col] > threshold:
                    if col == target_col:
                        to_drop.append(row)
                    else:
                        to_drop.append(col)

        to_drop = list(set(to_drop) - {target_col})
        self.df.drop(columns=to_drop, inplace=True)

        print(f"Rimosse {len(to_drop)} feature altamente correlate: {to_drop}")

        # Heatmap dopo la rimozione
        final_numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        final_corr = self.df[final_numeric_cols].corr(method='pearson').abs()
        plt.figure(figsize=(12, 10))
        sns.heatmap(final_corr, cmap='coolwarm', annot=False, fmt=".2f", square=True)
        plt.title("Final correlation matrix")
        plt.tight_layout()
        plt.savefig(heatmap_after_output)
        if show_heatmap:
            plt.show()
        plt.close()
 
    #Pie chart della distribuzione della label
    def plot_label_distribution(self, stage='Initial', output_path=None):

        if self.df is None:
            raise ValueError("Il dataset deve essere caricato prima di poter generare il grafico.")

        target_col = 'Attack'
        if target_col not in self.df.columns:
            raise ValueError(f"La colonna '{target_col}' non è presente nel dataset.")

        # Conta il numero di occorrenze per ciascuna classe
        label_counts = self.df[target_col].value_counts()
        total = label_counts.sum()
        labels = label_counts.index.tolist()
        sizes = label_counts.values

        if self.label_mapping_inverse:
            class_labels = [f"Class {self.label_mapping_inverse.get(label, label)}" for label in labels]
        else:
            class_labels = [f"Class {label}" for label in labels]


        print(f"\n Distribuzione label ({stage}):\n")
        print(f"{'Class':<25} {'Count':>10} {'Percent':>10}")
        print("-" * 50)
        for label, count in zip(labels, sizes):
            print(f"{label:<25} {count:>10} {count / total * 100:>9.1f}%")
       
        # Grafico a torta 
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots(figsize=(6, 6))

        colors = plt.cm.tab10.colors
        wedges, _ = ax.pie(
            sizes,
            startangle=140,
            labels=None,
            colors=colors,
            wedgeprops=dict(edgecolor='w') 
        )

        for i, wedge in enumerate(wedges):
            ang = (wedge.theta2 + wedge.theta1) / 2.
            x = np.cos(np.deg2rad(ang))
            y = np.sin(np.deg2rad(ang))
            label_text = f'{sizes[i]/1_000_000:.1f} M'
            ax.annotate(
                label_text,
                xy=(x * 0.7, y * 0.7),
                xytext=(x * 1.23, y * 1.23),
                arrowprops=dict(arrowstyle='-', lw=1),
                ha='center',
                va='center',
                fontsize=10
            )

        ax.set_title(f"{stage} label distribution", fontsize=16, fontweight='bold')

        ax.legend(
            wedges,
            class_labels,
            title="Classes",
            loc='upper center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=2,
            fontsize=10,
            title_fontsize=11
        )

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        plt.show()
        plt.close()

    #Salvataggio del dataset processato
    def save(self):

        if self.df is not None:
            self.df.to_csv(self.output_path, index=False)
        else:
            raise ValueError("Il dataset non è stato ancora caricato o processato.")

    def run_all(self):

        self.load_data()
        self.plot_label_distribution(stage='Initial', output_path='label_distribution_before.png')
        self.clean()
        self.map_target_to_numeric()
        self.remove_highly_correlated_features()
        self.plot_label_distribution(stage='Post preprocessing', output_path='label_distribution_after.png')
        self.save()
