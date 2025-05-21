from data_preprocessing import Preprocessor  
from data_splitting_and_handling import DatasetSplitter
import pandas as pd 
from trainer import ModelTrainer
from evaluator_and_analysis import EvaluatorAndAnalysis
import time
import os
import json

def preprocess_and_split():
    print("\n Avvio Preprocessing e split del dataset...")

    # STEP 1: Preprocessing
    input_csv = 'input_dataset.csv'
    output_csv = 'preprocessed_dataset.csv'
    preprocessor = Preprocessor(csv_path=input_csv, output_path=output_csv)
    preprocessor.run_all()

    # STEP 2: Carica dataset processato
    df = pd.read_csv(output_csv)

    # STEP 3: Split + undersampling
    target_col = df.columns[-1]
    splitter = DatasetSplitter(df, target_col=target_col)
    splitter.stratified_split(test_size=0.2)
    splitter.undersample_major_classes(reduction_rate=0.15)    
    #X_train, X_test, y_train, y_test = splitter.get_split()
    X_train=splitter.X_train
    X_test=splitter.X_test
    y_train=splitter.y_train
    y_test=splitter.y_test

    # Salva split in formato csv
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)


def load_split():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").squeeze()
    y_test = pd.read_csv("y_test.csv").squeeze()
    
    assert isinstance(y_train, pd.Series), "y_train deve essere una Series"
    assert isinstance(y_test, pd.Series), "y_test deve essere una Series"
    return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test

def main():

    #Menù scelta preprocessing e split
    print("\n Eseguire preprocessing e split dei dati?")
    print("1. Sì (consigliato se hai modificato il dataset originale)")
    print("2. No (usa split già salvati)")
    prep_choice = input("Inserisci 1 o 2: ").strip()

    if prep_choice == '1':
        preprocess_and_split()
    elif prep_choice == '2':
        print("\n  Salto preprocessing.")
    else:
        print("Scelta non valida. Uscita dal programma.")
        return

    # Carica split
    try:
        X_train, X_test, y_train, y_test = load_split()
    except Exception as e:
        print(f"Errore nel caricamento dei file split: {e}")
        print("Assicurati di avere eseguito il preprocessing.")
        return
    
    model_map = {'1': 'random_forest', '2': 'decision_tree', '3': 'mlp'}

    #Dizionario per il salvataggio dei tempi di addestramento
    training_times = {
        "static": {},
        "tuning": {}    
    }
    while True:
        # Menù scelta modello
        print("\nScegli il modello da addestrare e/o valutare:")
        print("1. Random Forest")
        print("2. Decision Tree")
        print("3. MLP (MultiLayer Perceptron)")
        print("0. Esci dal programma")
        model_choice = input("Inserisci 0, 1, 2 o 3: ").strip()

        if model_choice == '0':
            print("\nUscita dal programma!")
            break

        if model_choice not in model_map:
            print("Scelta non valida.")
            continue

        model_type = model_map[model_choice]
        print(f"\n Hai scelto il modello: {model_type}")

        # Selezione modalità di addestramento
        print("\n Seleziona modalità di addestramento")
        print("1. Tuning degli iperparametri con cross-validazione")
        print("2. Addestramento statico")
        tuning_choice = input("Inserisci 1 o 2: ").strip()
  
        if tuning_choice == '1':
            mode = 'tuning'
        elif tuning_choice == '2':
            mode = 'static'
        else:
            print("Scelta non valida. Ritorno al menu principale.")
            continue

        filename = f"saved_models/{model_type}_{mode}.pkl"
        if os.path.exists(filename):

            print(f"\n Modello già addestrato trovato: {filename}")
            #Caricamento del modello
            trainer = ModelTrainer(X_train, y_train, X_test, y_test, model_type=model_type, mode=mode)
            trainer.load_model(filename)
            model=trainer.model

            #Valutazione e analisi del modello (varianza spiegata, metriche di valutazione, matrice di confusione e curve di apprendimento)
            evaluator=EvaluatorAndAnalysis(model, X_train, X_test, y_train, y_test, model_type=model_type, mode=mode)
            evaluator.run_all()
        else:
            print(f"\n Nessun modello trovato. Inizio addestramento per {model_type} in modalità {mode}...")

            #Addestramento del modello
            trainer = ModelTrainer(X_train, y_train, X_test, y_test, model_type=model_type, mode=mode)
            
            #Calcolo tempi di addestramento
            if mode == 'tuning':
                t_start = time.time()
                trainer.tune_hyperparameters()
                t_end = time.time()

                training_time = t_end - t_start
                print(f"\n Addestramento (con tuning) completato in {training_time:.2f} secondi")

                # Salva tempo nel dizionario
                training_times["tuning"][model_type] = training_time
            else:
                t_start = time.time()
                trainer.train_static()
                t_end = time.time()

                training_time = t_end - t_start
                print(f"\n Addestramento (statico) completato in {training_time:.2f} secondi")

                # Salva tempo nel dizionario
                training_times["static"][model_type] = training_time
            
            trainer.save_model(filename)
            print(f" Modello salvato in {filename}")
            model=trainer.model

            #Valutazione e analisi del modello (varianza spiegata, metriche di valutazione, matrice di confusione e curve di apprendimento)
            evaluator=EvaluatorAndAnalysis(model, X_train, X_test, y_train, y_test, model_type=model_type, mode=mode)
            evaluator.run_all()

        #Salvataggio tempi di addestramento in json
        if training_times:
            json_path = "training_times.json"
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = {"static": {}, "tuning": {}}
            else:
                existing_data = {"static": {}, "tuning": {}}

            # Aggiorna i dati esistenti
            for mode in training_times:
                existing_data.setdefault(mode, {})
                existing_data[mode].update(training_times[mode])

            # Salva il dizionario aggiornato
            with open(json_path, "w") as f:
                json.dump(existing_data, f, indent=4)
                print("\n Tempi di addestramento aggiornati in 'training_times.json'!")


        another = input("\n Vuoi addestrare o valutare un altro modello? (s/n): ").strip().lower()
        if another != 's':
            print("\nUscita dal programma!")
            break

if __name__ == "__main__":
    main()
