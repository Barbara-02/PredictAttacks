from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

class DatasetSplitter:
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    #Divisione del dataset in training (80%) e test (20%)
    def stratified_split(self, test_size=0.2, random_state=42):
        if self.df is None:
            raise ValueError("Errore: caricare il dataset.")

        if self.target_col not in self.df.columns:
            raise ValueError(f"La colonna target '{self.target_col}' non esiste nel DataFrame.")

        #Selezione features (X) e colonna target (y)
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.y_train = y_train
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test

        print(f"Dataset diviso con successo: {len(self.X_train)} campioni per il training, {len(self.X_test)} per il test.")

    
    #Undersampling delle classi maggioritarie (15%)
    def undersample_major_classes(self, reduction_rate=0.15):

        if self.X_train is None or self.y_train is None:
            raise ValueError("Esegui prima lo split del dataset.")

        # Combinazione di X_train e y_train in un unico DataFrame
        train_df = pd.concat([self.X_train, self.y_train], axis=1)

        # Individuazione delle 3 classi più frequenti
        class_counts = train_df[self.target_col].value_counts()
        print("Distribuzioni delle classi nel training set:")
        print(class_counts)
        top_3_classes = class_counts.head(3).index.tolist()

        # Applicazione di undersampling 
        dfs = []
        for label in train_df[self.target_col].unique():
            class_df = train_df[train_df[self.target_col] == label]
            if label in top_3_classes:
                n_samples = int(len(class_df) * reduction_rate)
                class_df = class_df.sample(n=n_samples, random_state=42)
            dfs.append(class_df)

        # Combinazione del training set bilanciato
        undersampled_df = pd.concat(dfs)

        self.X_train = undersampled_df.drop(columns=[self.target_col])
        self.y_train = undersampled_df[self.target_col]

        print("Undersampling completato sulle 3 classi maggioritarie.")
        print("Nuove distribuzioni delle classi nel training set:")
        print(self.y_train.value_counts())
 
        