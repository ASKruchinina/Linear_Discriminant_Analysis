# -*- coding: utf-8 -*-
"""
Module regroupant des fonctions de vérification et modification de données 
ainsi que des fonctions permettant d'effectuer les calculs nécessaires pour la
pratique de l'analyse discriminante linéaire (lambda de Wilks, matrice de 
covariance, etc.).
"""

# ---- data analysis librairies
import pandas as pd
import numpy as np
import scipy.stats as stats


def verification_NA(inputData, targetValues):
    """Vérification pour les valeurs nulles.
    Les observations avec des valeurs nulles ne sont pas prises
    en compte pour l'analyse.
    La fonction affiche le nombre des observation supprimées.

    Paramètres
    ----------
    inputData : array-like of shape (n_samples, n_features)
        Input data.
    targetValues : array-like of shape (n_samples,) or (n_samples, n_targets) 
        Valeurs cibles.

    Sortie
    -------
    inputDataWithoutNA : 
        Input data sans NA.
    targetValuesWithoutNA : 
        Valeurs cibles sans NA.

    """
    n, p = inputData.shape
    # il faut concatener les inputs pour supprimer les lignes
    # avec des valeurs nulles
    df = pd.concat((inputData, targetValues), axis=1)
    df.dropna(axis=0, inplace=True)
    n_del = n - df.shape[0]
    inputDataWithoutNA = df.iloc[:, :-1]
    targetValuesWithoutNA = df.iloc[:, -1]
    print('Attention : ', n_del, ' observations ont été supprimées.')
    return inputDataWithoutNA, targetValuesWithoutNA


def recodification_var_expl(inputData):
    """La fonction recodification_expl_var() permet de convertir des 
    variables inputData en variables numériques. 
    Dans le cadre de notre projet LDA il suffit de lui donner en entrée 
    les variables (inputData) à convertir et il retourne 
    variable un dataframe de variables numériques.
    """

    d = dict()  # Creation d'un dictionnaire vide
    # Apply permet ici de faire une boucle comme avec R.
    # Test puis conversion de la variable en numérique si elle ne l'est pas
    d = inputData.apply(lambda s: pd.to_numeric(
        s, errors='coerce').notnull().all())
    # Renvoie False si la variable n'est pas numérique, True sinon.
    liste = d.values
    for i in range(len(inputData.columns)):
        # Conversion de toutes les variables qui ne sont pas numériques
        # en objet.
        if liste[i] == False:
            # Conversion des types "non-numeric" en "objet"
            inputData.iloc[:, i] = inputData.iloc[:, i].astype(object)

    # Recodage des colonnes (variables einputDataplicatives) grâce à la
    # fonction get_dummies de pandas
    for i in range(inputData.shape[1]):
        if inputData.iloc[:, i].dtype == object:
            dummy = pd.get_dummies(inputData.iloc[:, i], drop_first=True)
            for j in range(dummy.shape[1]):
                # Concatenation (rajout des variables recodees à inputData)
                # pour chaque colonne de dummy, avec inputData le
                # dataframe de base
                inputData = pd.concat(
                    [inputData, dummy.iloc[:, j]], axis=1)
    # Suppression des colonnes non numerics
    inputData = inputData._get_numeric_data()
    return inputData


def freq_relat(targetValues, n):
    """Calcul des fréquences relatives.

    Paramètres
    ----------
    targetValues : array-like of shape (n_samples,) or (n_samples, n_targets) 
        Target values.

    Sortie
    -------
    freqClassValues : numeric
        fréquences relatives par classe
    effClassValues : numeric
        nombre d'occurences par classe (effectifs)
    """
    # Nb nombre d'effectifs par classes
    effClassValues = np.unique(targetValues, return_counts=True)[1]
    freqClassValues = effClassValues / n
    return effClassValues, freqClassValues


def means_class(inputData, targetValues):
    """Calcul des moyennes conditionnelles selon le groupe d'appartenance.

    Paramètres
    ----------
    inputData : array-like of shape (n_samples, n_features)
                Input data.
    targetValues : array-like of shape (n_samples,) or (n_samples, n_targets) 
                   Target values.

    Sortie
    -------
    means_cls : moyennes conditionnelles par classe
    """
    # classValues: convertion des valeurs cibles en numérique
    classNames, classValues = np.unique(targetValues, return_inverse=True)
    # Nk nombre d'effectifs par classes
    effClassValues = np.bincount(classValues)
    # initiation d'une matrice remplie de 0
    means = np.zeros(shape=(len(classNames), inputData.shape[1]))
    np.add.at(means, classValues, inputData)
    means_cls = means / effClassValues[:, None]
    return means_cls


def cov_matrix(dataset):
    """Calcul de la matrice de covariance totale (V) ainsi que sa version 
    biaisée (Vb).

    Paramètres
    ----------
    dataset : DataFrame (pandas)
        Jeu de données

    Notes
    ----------
    Les DataFrame (pandas) permettent de garder un résultat avec le nom des 
    variables.
    """
    n = dataset.shape[0]  # taille de l'échantillon
    V = dataset.cov()  # matrice de covariace totale
    Vb = (n-1)/n * V  # matrice de covariance totale biaisée
    return (V, Vb)


def pooled_cov_matrix(dataset, className):
    """Calcul de la matrice de covariance intra-classe (W) ainsi que sa 
    version biaisée (Wb).

    Paramètres
    ----------
    dataset : DataFrame (pandas)
        Jeu de données
    className : string
        Nom de la colonne contenant les différentes classes du jeu de données

    Notes
    ----------
    Les DatFrame (pandas) permettent de garder un résultat avec le nom des 
    variables.
    """
    n = dataset.shape[0]  # taille de l'échantillon
    K = len(dataset[className].unique())  # nombre de classes
    W = 0  # initialisation de W
    for modalities in dataset[className].unique():
        Vk = dataset.loc[dataset[className] == modalities].cov()
        W += (dataset[className].value_counts()[modalities] - 1) * Vk
    W *= 1/(n-K)  # matrice de covariance intra-classes
    Wb = (n-K)/n * W  # matrice de covariance intra-classes biaisée
    return (W, Wb)


def wilks(Vb, Wb):
    """Calcul du Lambda de Wilks par le rapport entre les déterminants des 
    estimateurs biaisés des matrices de variance covariance intra-classes et 
    totales.

    Paramètres
    ----------
    Vb : Matrice Numpy / Pandas
        Matrice de covariance totale
    Wb : Matrice Numpy / Pandas
        Matrice de covariance intra-classes
    """
    # les paramètres d'entrée doivent être des matrices numpy ou
    # des DataFrame (pandas)
    detVb = np.linalg.det(Vb)  # dét. de la matrice de cov. totale biaisée
    detWb = np.linalg.det(Wb)  # dét. de la matrice de cov.
    # intra-classes biaisée
    return (detWb / detVb)


def wilks_log(Vb, Wb):
    """Calcul du Lambda de Wilks par le rapport entre les logarithmes naturels
    des déterminants des estimateurs biaisés des matrices de variance 
    covariance intra-classes et totales.
    Permet la gestion de bases de données avec beaucoup de variables (> 90).

    Paramètres
    ----------
    Vb : Matrice Numpy / Pandas
        Matrice de covariance totale
    Wb : Matrice Numpy / Pandas
        Matrice de covariance intra-classes
    """
    detVb = np.linalg.slogdet(Vb)  # log. nat. du dét. de la matrice de cov.
    # totale biaisée
    detWb = np.linalg.slogdet(Wb)  # log. nat. du dét. de la matrice de cov.
    # intra-classes biaisée
    return np.exp((detWb[0]*detWb[1])-(detVb[0]*detVb[1]))


def p_value(F, ddl1, ddl2):
    """Calcul de la p-value d'un test unilatéral de Fisher.

    Paramètres
    ----------
    F : numeric
        statistique de Fisher
    ddl1, ddl2 : integer
        degrés de liberté (int)
    """
    if (F < 1):
        return (1.0 - stats.f.cdf(1.0/F, ddl1, ddl2))
    return (1.0 - stats.f.cdf(F, ddl1, ddl2))
