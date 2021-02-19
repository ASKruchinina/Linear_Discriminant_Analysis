# -*- coding: utf-8 -*-
"""
Démonstration de l'application d'Analyse Discriminante Linéaire
"""

import pandas as pd
from discriminant_analysis import LinearDiscriminantAnalysis as LDA
from reporting import HTML, PDF

# Chargement des données
DTrain = pd.read_excel("data\Data_Illustration_Livre_ADL.xlsx",
                       sheet_name="DATA_2_TRAIN", engine="openpyxl")

# création d'un objet LDA permettant de faire une analyse discriminante 
# linéaire avec les données et le nom de la variable catégorielle en entrée
lda = LDA(DTrain, "TYPE")

help(lda)
# attempt to return a list of valid attributes for that object.
dir(lda)
lda.classNames

#---- Apprentissage, ~PROC DISCRIM de SAS
lda.fit()
print(lda.intercept_, '\n',lda.coef_)

# Prédiction
DP = pd.read_excel("data\Data_Illustration_Livre_ADL.xlsx",
                   sheet_name="DATA_2_TEST")

valeurs_a_predire = DP[DP.columns[1:]]
y_pred = lda.predict(valeurs_a_predire)
print(y_pred)

# Reporting PROC DISCRIM
HTML().discrim_html_output(lda, "output\discrim_results.html") # sortie html
PDF().discrim_pdf_output(lda, "output\discrim_results.pdf") # sortie pdf

# matrice de confusion, accuracy
y_true = DP.TYPE
lda.confusion_matrix(y_true, y_pred)
lda.accuracy_score(y_true, y_pred)
print('Confusion matrix: ', lda.confusionMatrix)
print('Accuracy score: ', lda.accuracy)

#---- Sélection de variables, ~PROC STEPDISC de SAS
# Approche ascendante
lda.stepdisc(0.01, "forward")
HTML().stepdisc_html_output(lda, "output\stepdisc_forward_results.html")
# Approche descendante
lda.stepdisc(0.01, "backward")
HTML().stepdisc_html_output(lda, "output\stepdisc_backward_results.html")

# Décroissance du lambda de Wilks
# chargement des données
DTrainLarge = pd.read_excel("data\Data_Illustration_Livre_ADL.xlsx",
                            sheet_name="WAVE_NOISE", engine="openpyxl")
lda_wave_noise = LDA(DTrainLarge, "classe") # création de l'objet
# Appel de la fonction
lda_wave_noise.wilks_decay()
# Affichage des valeurs du lambda
print("Valeurs du lambda de Wilks:", lda_wave_noise.infoWilksDecay)
# Affichage de la courbe
lda_wave_noise.figWilksDecay
lda_wave_noise.figWilksDecay.savefig("output\wilks_curve.png",dpi=200)

# vous pouvez également faire de la sélection de variables sur ce jeu de données ! ;)
lda_wave_noise.stepdisc(0.01, "forward")
HTML().stepdisc_html_output(lda_wave_noise, "output\stepdisc_wave_noise")
