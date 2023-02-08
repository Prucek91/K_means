# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import copy
from scipy.spatial import distance
import scipy
import random
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


# Dopasowanie rezultatu klasteryzacji do rzeczywistych klas decyzyjnych


def find_perm(clusters, Y_real, Y_pred):
    perm=[]
    for i in range(clusters):
        idx = Y_pred == i
        new_label=scipy.stats.mode(Y_real[idx])[0][0]
        perm.append(new_label)
    return [perm[label] for label in Y_pred]




def routine(X, Y_real, Y_pred, k):
    # dopasowanie
    res_ = find_perm(k, np.asarray(Y_real), np.asarray(Y_pred))
    # indeks Jaccarda
    #score = jaccard_score(Y, res_, average=None)
    #print(score)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    df = pd.DataFrame(X_reduced)
    df['Y'] = Y_real
    df2 = pd.DataFrame(X_reduced, columns=[ '1', '2'])
    df2['Y'] = res_
    points = []
    hulls = []
    for i in range(k):
        point = 0
        hull = 0
        point = df[[df.columns[0],df.columns[1]]].loc[df['Y']==i].values
        hull = ConvexHull(point)
        points.append(point)
        hulls.append(hull)
    plt.figure(figsize=(15,5), dpi=200)
    plt.subplot(1,3,1 )
    plt.scatter(X_reduced[:,0], X_reduced[:,1], c=Y_real)
    
    for i in range(k):
        plt.plot(points[i][:,0], points[i][:,1], 'o')
        for simplex in hulls[i].simplices:
            plt.plot(points[i][simplex, 0], points[i][simplex, 1], 'k-')
            
    plt.title('oryginal')
    
    plt.subplot(1,3,2)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=res_, cmap='Set2')
    points = []
    hulls = []
    for i in range(k):
        point = df2[['1','2']].loc[df2['Y']==i].values
        hull = ConvexHull(point)
        points.append(point)
        hulls.append(hull)
    
    for i in range(k):
        plt.plot(points[i][:,0], points[i][:,1], 'o')
        for simplex in hulls[i].simplices:
            plt.plot(points[i][simplex, 0], points[i][simplex, 1], 'k-')
    plt.title('rezultat klasteryzacji')
    plt.subplot(1,3,3)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c= res_==Y, cmap='Accent_r')
    
    
    plt.title('roznice')
    plt.suptitle('k means')
    plt.show()

    fig = plt.figure(figsize=(15,5), num=3)
    ax = fig.add_subplot(131, projection='3d')
    plt.title('oryginal')
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    ax.scatter(x,y,z, c=Y, s=20)
    ax = fig.add_subplot(132, projection='3d')
    plt.title('rezultat klasteryzacji')
    ax.scatter(x,y,z, c=res_, s=20)
    ax = fig.add_subplot(133, projection='3d')
    plt.title('roznice')
    ax.scatter(x,y,z, c=Y==res_, cmap='Dark2',s=20)
    plt.suptitle('K-Means' )
    plt.show()




from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
Y = iris.target
df = pd.DataFrame(X, columns=[ 'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
df['Y'] = Y




def k_means(X, k=3, n_it = 100):
    k = 3
    n_it = 10
    centroidy = []
    n_wymiarow = X.shape[1]
    # Wybranie poczatkowego połozenia centrów klastrów (centroidów)
    for i in range(k):
        id_ = np.random.randint(0, X.shape[0])
        coords = X[id_, :]
        centroidy.append(coords)
    decyzje = X.shape[0] * [0]
    # petla
    it = 0
    while it < n_it:
        prev_decyzje = copy.deepcopy(decyzje)
        odleglosci = []
        for i in range(X.shape[0]):
            tmp_odleglosci = []
            for j in range(k):
                tmp_odleglosci.append(distance.euclidean(centroidy[j] , X[i,:]))
            odleglosci.append(tmp_odleglosci)
        for i in range(len(decyzje)):
            decyzje[i] = np.argmin(np.asarray(odleglosci[i]))
            
        tmp_df = pd.DataFrame(X)
        tmp_df['Y'] = decyzje
        for i in range(len(centroidy)):
            tmp_df2 = tmp_df.loc[(tmp_df['Y'] == i)]
            centroidy[i] = np.average(tmp_df2.drop(columns=['Y']).values, axis=0)
        it += 1
        # brak zmian w klasteryzacji
        if prev_decyzje == decyzje:
            break
    return decyzje, centroidy




k = 3
d,c = k_means(X, k = k)




dopasowanie = find_perm(k, np.asarray(Y), np.asarray(d))


# Indeks Jaccarda dla danych pogrupowanych wzglęedem klasy decyzyjnej (sklearn.metrics.jaccard_score).




from sklearn.metrics import jaccard_score
score = jaccard_score(Y, dopasowanie, average=None)
'''
average{‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or None, default=’binary’
If None, the scores for each class are returned.
'''
print(score)




jaccarda = pd.DataFrame()
s = pd.Series(score)
s.name = 'k means'
jaccarda = jaccarda.append(s)


# Wizualizacja danych w przestrzeni 2Dprzy pomocy PCA, rzutując dane na dwie pierwsze składowe główne.
# 
# Wykresy
# - dane z podziałem na rzeczywiste klasy,
# - dane z podziałem na klasy decyzyjne obliczone w wyniku działania algorytmu,
# - wykres obrazujący, które dane zostały poprawnie sklasyfikowane, a które nie.


routine(X, Y, d, k)


# Dla następujących metod: - metody k-means (sklearn.cluster.KMeans) - metody
# GMM (sklearn.mixture.GaussianMixture) - aglomeracyjnych metod grupowania hierarchicznego
# sklearn.cluster.AgglomerativeClustering przy użyciu różnych kryteriów łączenia punktów w klastry (parametr linkage do konstruktora klasy AgglomerativeClustering) - metoda najbliższego sąsiedztwa
# - metoda średnich połączeń
# - metoda najdalszych połączeń
# - metoda Warda 
# Dendrogram (scipy.cluster.hierarchy.dendrogram) - obrazuje związki pomiędzy elementami dla wybranej metody aglomeracyjnej 

# sklearn cluster KMeans



from sklearn.cluster import KMeans
sklearn_kmeans = KMeans(n_clusters=k).fit(X)
#sklearn_kmeans.labels_
res_ = find_perm(clusters=k, Y_real=Y, Y_pred=sklearn_kmeans.labels_)




# indeks Jaccarda
from sklearn.metrics import jaccard_score
score = jaccard_score(Y, res_, average=None)
print(score)
s = pd.Series(score)
s.name = 'KMeans sklearn'
jaccarda = jaccarda.append(s)





routine(X,Y,sklearn_kmeans.labels_, k)


# GMM - sklearn mixture GaussianMixture



from sklearn.mixture import GaussianMixture
k = 3
gm = GaussianMixture(n_components=k, n_init=10).fit(X)
res = gm.predict(X)





# indeks Jaccarda
score = jaccard_score(Y, res_, average=None)
print(score)
s = pd.Series(score)
s.name = 'GMM'
jaccarda = jaccarda.append(s)






routine(X,Y,res, k)


# Metody aglomeracyjne



from sklearn.cluster import AgglomerativeClustering


# Metoda najbliższego sąsiedztwa



k = 3
linkages = ['ward', 'complete', 'average']#, 'single']
for L in linkages:
    ac = AgglomerativeClustering(n_clusters=k, linkage=L).fit(X)
    res = ac.labels_
    routine(X,Y,res, k)
    res_ = find_perm(k, Y, res)
    score = jaccard_score(Y, res_, average=None)
    s = pd.Series(score)
    s.name = L
    jaccarda = jaccarda.append(s)





k = 3
linkages = 'single'
'''
ac = AgglomerativeClustering(n_clusters=k, linkage='single').fit(X)
res = ac.labels_
action(X,Y,res, k)
res_ = find_perm(k, Y, res)
score = jaccard_score(Y, res_, average=None)
s = pd.Series(score)
s.name = linkages
jaccarda = jaccarda.append(s)
'''
# single nie zadzialalo dla k = 3


# Indeksy Jaccarda




print(jaccarda.sort_values([0,1,2], ascending=False))


# Dendrogram



from scipy.cluster.hierarchy import dendrogram




# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1 # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
        ).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
#iris = load_iris()
#X = iris.data
# setting distance_threshold=0 ensures we compute the full tree.
linkages = ['ward', 'complete', 'average', 'single']
for L in linkages:
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage=L)
    model = model.fit(X)
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.suptitle(L)
    plt.show()


