#methodology_m2.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

#Tworzenie klastrow wewnetrznych
#
#X_train_features = macierz surowych obserwacji treningowych (macierz D)
#obs_tc_labels = etykiety do ktorego klastra typologii (TC) naleza obserwacje
#k_internal = liczba klastrow wewnetrznych (IC) do utworzenia w klastrach typologii
def create_internal_clusters(X_train_features, obs_tc_labels, k_internal=5):
    #identyfikacja utworzonych klastrach wewnetrznych na etapie M1
    ic_centroids_map = {}
    unique_tcs = np.unique(obs_tc_labels)
    
    print(f"  [M2] Tworzenie Klastrow Wewnetrznych (IC) dla {len(unique_tcs)} TC...")
    
    #przetwarzanie klastrow typologii
    for tc_label in unique_tcs:
        #wyciaganie surowych obserwacji nalezace do aktualnego klastra
        observations_for_tc = X_train_features[obs_tc_labels == tc_label]
        
        #dzielenie dane jednego klastra na mniejsze podgrupy (klastry wewnetrzne)
        kmeans = KMeans(n_clusters=k_internal, random_state=42, n_init=10)
        kmeans.fit(observations_for_tc)
        
        #obliczanie srodkow nowych podgrup sluzacych do przyciagania danych
        #nowych uzytkownikow w fazie wlaczania
        ic_centroids_map[tc_label] = kmeans.cluster_centers_
            
    print("  [M2] Konfiguracja M2 zakończona. Mapa centroidów IC gotowa.")
    
    #slownik o kluczach w postaci numerow TC
    #i wartosciach w postaci tablic ze wspolrzednymi centroidow
    return ic_centroids_map 


#Wlaczanie nowego uzytkownika (bez etykietowanych danych)
#d najlepiej pasujacego klastra typologii (TC)
#
#X_new_user = nieetykietowane obserwacje nowego uzytkownika (macierz O x F)
#ic_centroids_map = centroidy klastrow wewnetrznych dla kazdego klastra typologii
def assign_new_user_M2(X_new_user, ic_centroids_map):
    total_distances = {}
    
    for tc_label, ic_centroids in ic_centroids_map.items():
        #stworzenie macierzy odleglosci poprzez porownanie
        #kazdej obserwacji nowego uzytkownika z kazdym
        #centroidem klastra wewnetrznego
        dists = pairwise_distances(X_new_user, ic_centroids)
        
        #znalezienie najlepszego dopasowania dla kazdej obserwacji
        min_dists_per_obs = dists.min(axis=1)
        
        #agregacja wynikow dla calego klastra typologii
        #reprezentujaca calkowity blad dopasowania uzytkownika
        #do danej typologii
        tc_total_dist = min_dists_per_obs.sum()
        total_distances[tc_label] = tc_total_dist

    #wybranie klastra typologii z najnizsza suma (najmniejsza laczna odlegloscia)
    assigned_tc_label = min(total_distances, key=total_distances.get)
    
    print("  [M2] Obliczone sumaryczne odleglosci do TC:")
    for tc, dist in total_distances.items():
        print(f"    -> TC {tc}: {dist:.2f}")
        
    return assigned_tc_label