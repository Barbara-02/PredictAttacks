from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
 
 
class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test, model_type, mode):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_type = model_type
        self.mode = mode  # 'static' o 'tuning'
        self.model = None
 
        # Salvataggio nella cartella saved_models
        os.makedirs('saved_models', exist_ok=True)
        self.model_filename = f"saved_models/{self.model_type}_{self.mode}.pkl"
 
    def save_model(self, filename=None):
        if filename is None:
            filename = self.model_filename
        joblib.dump(self.model, filename)
 
    def load_model(self, filename=None):
        if filename is None:
            filename = self.model_filename
        if os.path.exists(filename):
            print(f" Caricamento modello salvato da {filename}")
            self.model = joblib.load(filename)
        else:
            raise FileNotFoundError(f"Modello {filename} non trovato.")
 
    #Addestramento dei modelli in modalità statica
    def train_static(self):
        print("Training con parametri statici...")
 
        if self.model_type == 'random_forest':
 
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,      
                    max_depth=8,          
                    min_samples_leaf=30,
                    min_samples_split=10,
                    random_state=42
                ))
            ])
 
        elif self.model_type == 'decision_tree':
 
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.90)),
                ('classifier', DecisionTreeClassifier(
                    max_depth=8,                # Puoi personalizzare i parametri
                    min_samples_split=10,
                    min_samples_leaf=30,
                    random_state=42
                ))
            ])
 
        elif self.model_type == 'mlp':
 
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.90)),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(32,),
                     activation='relu',
                     alpha=1e-4,
                     max_iter=1000,
                     early_stopping=True,
                     random_state=42,
                     ))
            ])
        else:
            raise ValueError("Modello non supportato. Scegli tra: 'random_forest', 'lightgbm', 'mlp'")
 
        self.model = pipeline.fit(self.X_train, self.y_train)
        self.save_model()
 
    #Addestramento dei modelli con tuning degli iperparametri
    def tune_hyperparameters(self, n_iter=10):
        print("Training con tuining degli iperparametri...")
 
        if self.model_type == 'random_forest':
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('classifier', RandomForestClassifier())
            ])
            param_distributions = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, 30],
                'classifier__min_samples_split': [5, 10, 20], #prima era 2,5,10
                'classifier__min_samples_leaf': [2, 5, 10, 20], #priam era 1,2,4
                'classifier__max_features': ['sqrt', 'log2', None]

            }
 
        elif self.model_type == 'decision_tree':
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                 ('pca', PCA()),
                ('classifier', DecisionTreeClassifier(random_state=42))
            ])
            param_distributions = {
                'pca__n_components': stats.uniform(0.8, 0.15),
                'classifier__max_depth': [8, 10, 20, 30],
                'classifier__min_samples_split': [5, 10, 20],
                'classifier__min_samples_leaf': [2, 5, 10, 20],
                'classifier__max_features': ['sqrt', 'log2', None]
            }
        elif self.model_type == 'mlp':
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('classifier', MLPClassifier( random_state=42, verbose=True, early_stopping=True))
            ])
            param_distributions = {
                'pca__n_components': stats.uniform(0.8, 0.15),
                'classifier__hidden_layer_sizes': [(64,), (32,), (32, 16)],
                'classifier__max_iter': [1000, 2000, 3000],
                'classifier__activation': ['relu', 'tanh'],
                'classifier__alpha': stats.loguniform(1e-5, 1e-2)
            }
 
        else:
            raise ValueError("Modello non supportato. Scegli tra: 'random_forest', 'logistic_regression', 'mlp'")
 
        #Applicazione della cross-validazione
        search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=15,
            random_state=42
        )
 
        search.fit(self.X_train, self.y_train)
        self.model = search.best_estimator_
        print(f"Migliori parametri trovati: {search.best_params_}")
        self.save_model()
 