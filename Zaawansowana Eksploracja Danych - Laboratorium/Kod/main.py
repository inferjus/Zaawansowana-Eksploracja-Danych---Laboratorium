#main.py
import numpy as np

from data_processing import transform_matrix_M1
from clustering_tc import find_optimal_clusters
from methodology_m2 import create_internal_clusters, assign_new_user_M2

if __name__ == "__main__":
    
    np.random.seed(777)

    print("--- ROZPOCZECIE SYMULACJI METODOLOGII Z ARTYKULU ---")
    
###0. Konfiguracja i generowanie danych
    F_features = 57 #liczba cech fizjologicznych
    TC_classes = 2 #liczba klas emocji (0 - brak strachu, 1 - strach)
    
    #symulowanie danych: 30 uzytkownikow po 40 obserwacji
    N_train_users = 30
    O_obs_per_user = 40
    O_total_train_obs = N_train_users * O_obs_per_user
    
    #macierz D (obserwacje) z wartosciami symulujacymi odczyty z czujnikow
    X_D_matrix = np.random.rand(O_total_train_obs, F_features)
    
    #etykiety emocji dla kazdej obserwacji w X_D_matrix
    X_D_labels = np.random.randint(0, TC_classes, size=O_total_train_obs)
    
    #identyfikatory uzytkownikow  dla kazdej obserwacji
    X_D_user_ids = np.repeat(np.arange(N_train_users), O_obs_per_user)
    
    print(f"Konfiguracja: {N_train_users} ochotnikow, {O_obs_per_user} obs/ochotnika")
    print("-" * 40)
###

###1. Profilowanie (data_processing.py)
    #transformacja M1 macierzy D z obserwacjami na macierz D' z profilami uzytkownikow
    matrix_D_prime = transform_matrix_M1(X_D_matrix, X_D_user_ids, X_D_labels)

    X_for_clustering = matrix_D_prime.values #konwersja macierzy D' na tablicę NumPy
    user_id_map = matrix_D_prime.index.to_numpy() #tablica z ID uzytkownikow
    
    print("-" * 40)
###

###2. Klastrowanie TC (clustering_tc.py)
    #szukanie najlepszego sposobu podzialu uzytkownikow na typologie
    min_threshold_users = int(0.15 * N_train_users) #min. liczba uzytkownikow do utworzenia klastra
    best_tc_labels = find_optimal_clusters(X_for_clustering, min_threshold_users) #optymalne klastry
    
    print(f"Wynik klastrowania TC: {len(np.unique(best_tc_labels))} klastrow.")
    print("-" * 40)
###

###3. Konfiguracja M2 (methodology_m2.py) ---
    #mapa ID uzytkownikow i odpowiadajacych im klastrom wewnetrznym
    tc_label_map = {user_id: tc_label for user_id, tc_label in zip(user_id_map, best_tc_labels)}
    
    #wektor z obserwacjami i przypisanymi do nich klastrami
    obs_tc_labels = np.array([tc_label_map[user_id] for user_id in X_D_user_ids])
    
    #tworzenie klastrow wewnetrznych
    ic_map = create_internal_clusters(X_D_matrix, obs_tc_labels, k_internal=5)
    print("-" * 40)
###
    
###4: Wlaczanie nowego uzytkownika (methodology_m2.py)
    print("Symulacja: Pojawia sie nowy uzytkownik z 20 nieetykietowanymi obserwacjami...")
    O_new_user_obs = 20 #liczba obserwacji
    X_new_user = np.random.rand(O_new_user_obs, F_features) #cechy bez informacji o emocjach
    
    #przypisanie do klastra
    final_cluster = assign_new_user_M2(X_new_user, ic_map)

    print("\n--- ZAKONCZENIE SYMULACJI ---")
    print(f"Wynik: Nowy uzytkownik zostal przypisany do Klastra Typologii (TC) nr: {final_cluster}")
###