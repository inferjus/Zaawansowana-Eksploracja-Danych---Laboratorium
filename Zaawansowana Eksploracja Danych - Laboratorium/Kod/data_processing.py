#data_processing.py
import pandas as pd

#Transformacja macierzy D (O x F) do D' (N x M) 
#zgodnie z metodologia M1
#
#features_D = macierz D o wymiarach O x F (Obserwacje x Cechy)
#user_ids = identyfikatory uzytkownikow
#labels = etykiety emocji
def transform_matrix_M1(features_D, user_ids, labels):
    
    print(f"  [M1] Rozpoczecie transformacji D (O={len(features_D)}) do D'...")
    
    #tworzenie nazw kolumn i DataFrame
    feature_names = [f'feature_{i}' for i in range(features_D.shape[1])]
    df = pd.DataFrame(features_D, columns=feature_names)
    df['user_id'] = user_ids
    df['label'] = labels
    
    #liczenie sredniej i odchylenia standardowego wszystkich cech dla kazdej z klas docelowych
    #grupowanie i agregacja danych
    grouped = df.groupby(['user_id', 'label']).agg(['mean', 'std'])
    
    #splaszczenie do jednego wiersza na uzytkownika (N x M)
    matrix_D_prime = grouped.unstack(level='label', fill_value=0)
    
    #splaszczenie nazw kolumn
    matrix_D_prime.columns = ['_'.join(map(str, col)).strip('_') for col in matrix_D_prime.columns]
    
    #zamiana wartosci NaN na 0 w przypadku braku danych
    matrix_D_prime = matrix_D_prime.fillna(0)
    
    print(f"  [M1] Transformacja zakonczona. Ksztalt D': {matrix_D_prime.shape}")
    return matrix_D_prime #wyjciowa macierz D' o wymiarach N x M (Uzytkownicy x Metryki)


