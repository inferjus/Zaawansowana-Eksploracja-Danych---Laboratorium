#clustering_tc.py
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

#Obliczenie wskaznika Dunna sluzacego do oceny klastrowania
#uzywany do wybrania najlepszego podzialu na grupy sposrod testowanych opcji
#
#Zpoints = zbiór punktów danych|analizowana macierz profili uzytkownikow (X_D_prime)
def dunn_index(points, labels):
    unique = np.unique(labels)
    if len(unique) < 2:
        return -1
    
    #macierz odleglosci miedzy parami punktow
    D = squareform(pdist(points))
    
    #obliczanie mianownika (szukanie klastra z najwieksza odlegloscia miedzy najdalszymi punktami)
    max_intra = 0 #maksymalna srednica
    for lab in unique:
        pts = np.where(labels == lab)[0]
        if len(pts) > 1:
            intrad = D[np.ix_(pts, pts)] #wycinanie podmacierzy z odleglosciami miedzy punktami
            #wybranie najwiekszej odleglosci z gornego trojkata macierzy i porownanie z max_intra
            max_intra = max(max_intra, intrad[np.triu_indices(len(pts), k=1)].max())
            
    #obliczanie licznika (szukanie minimalnej odleglosci miedzyklastrowej
    min_inter = np.inf #minimalna 
    for i, lab_i in enumerate(unique):
        pts_i = np.where(labels == lab_i)[0] #indeksy rozpatrywanego klastra
        for j, lab_j in enumerate(unique): #wybranie klastra do porownania
            if i >= j: continue
            pts_j = np.where(labels == lab_j)[0]
            if len(pts_i) == 0 or len(pts_j) == 0: continue
            #wyciecie podmacierzy z odleglosciami miedzy wszystkimi punktami obu klastrow
            dist_ij = D[np.ix_(pts_i, pts_j)]
            #wybranie najmniejszej odleglosci
            min_inter = min(min_inter, dist_ij.min())
            
    return min_inter / max_intra if max_intra > 0 else np.inf



#Szukanie najlepszego sposobu podzialu uzytkownikow na typologie
#czyli wyszukiwanie optymalnych klastrow
#
#X_D_prime = przeksztalcona macierz profili uzytkownikow (D')
#min_percentage = minimalny prog wielkosci klastra
def find_optimal_clusters(X_D_prime, min_percentage=0.15):
    print("  [TC] Rozpoczecie wyszukiwania optymalnych Klastrow Typologii (TC)...")
    
    #prog wielkosci, tj. minimalna liczba uzytkownikow w klastrze
    threshold = int(min_percentage * len(X_D_prime))
    print(f"  [TC] Minimalny rozmiar klastra (15%): {threshold} uzytkownikow")
    
    best_dunn = -1
    best_labels = None
    
    #testowanie podzialow na 2,3...10 klastrow
    for k in range(2, 11):
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(X_D_prime)
        
        #identyfikowanie malych klastrow
        unique, counts = np.unique(labels, return_counts=True)
        labels_adj = labels.copy()
        small_clusters = [(lab, count) for lab, count in zip(unique, counts) if count > 0 and count < threshold]
        small_clusters.sort(key=lambda x: x[1])

        #"naprawianie" za malych klastrow
        for lab, count in small_clusters:
            #obliczanie srodka za malego klastra
            if np.sum(labels_adj == lab) == 0: continue 
            centroid = X_D_prime[labels_adj == lab].mean(axis=0)
            
            #szukanie najblizszeo duzego klastra
            centers = []
            remaining_unique = np.unique(labels_adj)
            for lab2 in remaining_unique:
                if lab2 != lab and np.sum(labels_adj == lab2) >= threshold:
                    centers.append((lab2, np.linalg.norm(centroid - X_D_prime[labels_adj == lab2].mean(axis=0))))
            
            #polaczenie obu klastrow
            if centers:
                nearest_tc_label = min(centers, key=lambda x: x[1])[0]
                labels_adj[labels_adj == lab] = nearest_tc_label

        #porzadkowanie etykiet po laczeniu klastrow
        unique_final = np.unique(labels_adj)
        label_map = {old: new for new, old in enumerate(unique_final)}
        labels_adj = np.array([label_map[l] for l in labels_adj])
        
        #ocena jakosci podzialu na grupy
        dunn = dunn_index(X_D_prime, labels_adj)
        
        #porownanie biezacego podzialu z najlepszym uzyskanym
        #i zapamietanie ukladu etykiet, jesli nowy podzial jest lepszy
        if dunn > best_dunn:
            best_dunn = dunn
            best_labels = labels_adj.copy()

    print(f"  [TC] Znaleziono optymalna liczbe Klastrew Typologii: {len(np.unique(best_labels))}")
    return best_labels