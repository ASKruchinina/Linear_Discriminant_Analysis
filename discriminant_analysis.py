# -*- coding: utf-8 -*-
"""
Application créée dans le cadre d'un projet de L3 IDS à l'Université Lyon 2.

Groupe constitué de AbdDia, ASKruchinina et Valinquish
"""

# ---- data analysis librairies
import pandas as pd
import numpy as np
# ---- data visualization librairies
import matplotlib.pyplot as plt
import seaborn as sns
# ---- fonctions de vérification, modification et calculs nécessaires pour
# ---- l'analyse discriminante linéaire
import calculations


class LinearDiscriminantAnalysis:
    """Analyse Discriminante Linéaire Prédictive basée sur les méthodes 
    PROC DISCRIM et STEPDISC de SAS.
    """

    def __init__(self, dataset, classEtiquette, varNames=None):
        # jeu de données
        self.dataset = dataset
        # nom de la variable cible
        self.classEtiquette = classEtiquette
        # noms des valeurs prises pour la variable cible
        self.classNames = np.unique(dataset[classEtiquette])
        # nom des variables explicatives
        if varNames is None:
            varNames = list(dataset.columns)
            varNames.remove(classEtiquette)
            self.varNames = varNames
        else:
            self.varNames = list(varNames)
        self.n = dataset.shape[0]  # taille de l'échantillon
        self.p = len(varNames)  # nombre de variables explicatives
        self.K = len(dataset[classEtiquette].unique())  # nombre de classes
        self.V, self.Vb = calculations.cov_matrix(
            dataset)  # matrices de cov. totale
        # matrices de cov. intra-classes
        self.W, self.Wb = calculations.pooled_cov_matrix(
            dataset, classEtiquette)

    def _stats_dataset(self):
        """Informations de bases sur le jeu de données.
        """
        self.infoDataset = pd.DataFrame(
            [self.n, self.p, self.K, self.n - 1, self.n - self.K, self.K - 1],
            index=["Taille d'echantillon totale", "Variables", "Classes",
                   "Total DDL", "DDL dans les classes", "DDL entre les classes"],
            columns=["Valeur"])

    def _stats_pooled_cov_matrix(self):
        """Calcul des statistiques de la matrice de cov. intra-classes
        """
        # rang de la matrice de cov. intra-classes
        rangW = np.linalg.matrix_rank(self.W)
        # logarithme naturel du déterminant de la matrice de cov. intra-classes
        logDetW = np.linalg.slogdet(self.W)[0] * np.linalg.slogdet(self.W)[1]
        self.infoCovMatrix = pd. \
            DataFrame([rangW, logDetW],
                      index=["Rang de la mat. de cov. intra-classes",
                             "Log. naturel du det. de la mat. de cov. intra-classes"],
                      columns=["Valeurs"])

    def _stats_classes(self):
        """ Calcul des statistiques des classes
        """
        # effectifs et fréquences relatives des classes
        targetValues = self.dataset[self.classEtiquette]
        effClassValues, freqClassValues = calculations.freq_relat(
            targetValues, self.n)
        self.infoClasses = pd.DataFrame(
            [effClassValues, freqClassValues],
            columns=self.classNames,
            index=["Effectifs", "Frequences"]).transpose()

    def _stats_wilks(self):
        """Statistiques du Lambda de Wilks
        """
        L = calculations.wilks(self.Vb, self.Wb)  # lambda de Wilks
        ddlNum = self.p * (self.K - 1)  # ddl du numérateur
        # Calcul du ddl du dénominateur
        temp = self.p**2 + (self.K-1)**2 - 5
        # Evite de diviser par 0 dans le cas ou "temp" sera egal à 0
        if temp == 0:
            temp = np.where(temp > 0,
                            np.sqrt(((self.p**2) * ((self.K-1)**2) - 4)*1),
                            1)
        else:
            temp = np.where(temp > 0,
                            np.sqrt(((self.p**2) * ((self.K-1)**2) - 4)/temp),
                            1)
        ddlDenom = (2 * self.n - self.p - self.K - 2)/2*temp-(ddlNum - 2)/2
        # Fin calcul du ddl du dénominateur
        # Calcul de la F-statistique
        F_Rao = L ** (1 / temp)
        F_Rao = ((1 - F_Rao) / F_Rao) * (ddlDenom / ddlNum)
        # Fin calcul de la F-statistique
        p_val = calculations.p_value(
            F_Rao, ddlNum, ddlDenom)  # p-value du test
        self.infoWilksStats = pd.DataFrame([L, F_Rao, ddlNum, ddlDenom, p_val],
                                           index=["Valeur", "F-Valeur", "DDL num.",
                                                  "DDL den.", "p-value"],
                                           columns=["Lambda de Wilks"]).transpose()

    def fit(self):
        """Apprentissage d'un modèle d'analyse discrimnante linéiare.
        Calcul également des valeurs supplèmentaires pour l'affichage telles que
        la matrice de covariance intra-classe, le lambda de Wilks, la F-stat
        et la p-value.
        """
        #---- Données ----#
        # sélection des valeurs des variables explicatives
        inputValues = self.dataset.drop(self.classEtiquette, axis=1)
        # sélection des valeurs de la variable catégorielle
        targetValues = self.dataset[self.classEtiquette]
        # suppression des valeurs nulles de l'analyse
        inputValues, targetValues = calculations.verification_NA(
            inputValues, targetValues)
        # transformation des données en numpy object
        inputValues, targetValues = inputValues.values, targetValues.values

        #---- Fonction de classement ----#
        effClassValues, freqClassValues = calculations.freq_relat(
            targetValues, self.n)
        pi_k = pd.DataFrame(freqClassValues.reshape(1, self.K),
                            columns=self.classNames)
        means = calculations.means_class(inputValues, targetValues)  # moy cond

        if np.linalg.det(self.W) != 0:
            invW = np.linalg.inv(self.W)  # matrice inverse de W
            self.intercept_ = np.log(pi_k.values).reshape(1, self.K) - \
                0.5 * np.diagonal(means @ invW @ means.T)
            # coefficients associés aux variables de la fonction de classement
            self.coef_ = (means @ invW).T
            # récupération des valeurs de la fonction de classement
            self.infoFuncClassement = pd.concat(
                [pd.DataFrame(self.intercept_,
                              columns=self.classNames,
                              index=["Const"]),
                 pd.DataFrame(self.coef_,
                              columns=self.classNames,
                              index=self.varNames)])
        else:
            raise ValueError("Erreur : La matrice de variance-covariance "
                             "n'est pas inversible ! ")

    def predict(self, inputData):
        """Prédiction des classes sur des valeurs d'entrée.

        Paramètres
        ----------
        inputData : array-like of shape (n_samples, n_features)
            Valeurs à predire.  

        Renvoie
        ----------
        prediction : vecteur des classes predites, (n_samples,)
        """
        p = inputData.shape[1]  # nombre de descripteurs
        predictedValues = []  # liste contenant les valeurs prédites
        for i in range(inputData.shape[0]):
            omega = inputData.iloc[i].values
            x = omega.reshape(1, p) @ self.coef_ + self.intercept_
            predictedValues.append(np.argmax(x))
        prediction = np.array(self.classNames).take(predictedValues)
        return prediction

    def confusion_matrix(self, y_true, y_pred, graphShow=True):
        """Calcul d'une matrice de confusion.

        Paramètres
        ----------
        y_true : Series ou DataFrame
            Vraies valeurs de la variable cible.
        y_pred : array-like of shape (n_samples,) 
            Valeurs predites par le modèle de la variable cible.

        Renvoie
        ----------
        confMatrix : matrice de confusion
        """
        class_to_num = {cl: num for num, cl in enumerate(np.unique(y_true))}
        #conversion des variables cibles en numérique
        y_true = np.array(y_true.apply(lambda cl: class_to_num[cl]))
        y_pred = np.array(pd.Series(y_pred).apply(lambda cl: class_to_num[cl]))
        confMatrix = np.zeros((self.K, self.K))
        #parcours de la matrice zéro de dim(K,K) 
        #et calcul des cellules de la matrice de confusion
        for ind_p in range(self.K):
            for ind_t in range(self.K):
                confMatrix[ind_p, ind_t] = (
                    (np.sum((y_pred == ind_p) & (y_true == ind_t))))
        self.confusionMatrix = confMatrix
        
        #affichage en mode graphique grace à seaborn.heatmap
        if graphShow:
            infoConfusionMatrix = pd.DataFrame(self.confusionMatrix,
                                               index=self.classNames,
                                               columns=self.classNames)
            self.confusionMatrixGraph = plt.figure(figsize=(10, 7))
            sns.heatmap(infoConfusionMatrix, annot=True)
            
        return confMatrix

    def accuracy_score(self, y_true, y_pred):
        """Calcul du taux de précision.

        Paramètres
        ----------
        y_true : Series ou DataFrame
            Vraies valeurs de la variable cible.
        y_pred : array-like of shape (n_samples,) 
            Valeurs predites de la variable cible.

        Retour:
        ----------
        accuracy : (TP + TN) / (P + N)
        """
        classNumericValues = {cl: num for num,
                              cl in enumerate(self.classNames)}
        y_true = np.array(y_true.apply(lambda cl: classNumericValues[cl]))
        y_pred = np.array(pd.Series(y_pred).apply(
            lambda cl: classNumericValues[cl]))
        self.accuracy = np.sum(y_pred == y_true)/np.sum(self.confusionMatrix)
        return self.accuracy

    def wilks_decay(self, graphShow=True):
        """Calcul des différentes valeurs de chaque valeur du Lambda de Wilks 
        pour chaque q (allant de 1 à p (nombre de variables)).
        """
        wilksValues, varSelection = [], []
        for name in self.varNames:
            varSelection.append(name)
            # calcul du Lambda de Wilks
            L = calculations.wilks_log(self.Vb.loc[varSelection, varSelection],
                                       self.Wb.loc[varSelection, varSelection])
            # fin du calcul du Lambda de Wilks
            wilksValues.append(L)

        #---- Récupération des valeurs dans un DataFrame ----#
        self.infoWilksDecay = pd.DataFrame(
            wilksValues, index=range(1, self.p+1)).transpose()

        #---- Création et récupération du graphique ----#
        if graphShow:
            self.figWilksDecay = plt.figure()
            plt.title("Décroissance du Lambda de Wilks")
            plt.xlabel("Nombre de variables sélectionnées")
            plt.ylabel("Valeur du Lambda de Wilks")
            plt.xticks(range(1, self.p+1, 2))
            plt.plot(range(1, self.p+1), wilksValues,
                     '-cx', color='c', mfc='k', mec='k')
        #---- Fin création du graphique ----#

    # à retravailler / to rework
    def stepdisc(self, slentry, method):
        """Sélection de variables avec approche ascendante et descendante.

        Paramètres
        ----------
        slentry : float
            Risque alpha fixé.
        method : ["forward", "backward"], string
            Méthode de sélection de variables.

        Raises
        ------
        ValueError
            Si l'utilisateur tente d'utiliser une méthode non programmée, par 
            exemple "stepwise" (ou tout autre valeur), la fonction renverra 
            une erreur.
        """
        #---- GESTION D'ERREURS POTENTIELLES  ----#
        isMethodValid = ["forward", "backward"]
        method = method.lower()
        if method not in isMethodValid:
            raise ValueError(
                "Procédure STEPDISC : le paramètre METHOD doit être l'une des "
                "valeurs suivantes %r" % isMethodValid)
        #--------#
        #---- Variables ----#
        self._htmlStringOutput = ""
        varNames = self.varNames.copy()
        colNames = ["R-carre", "F-statistique", "p-value", "Lambda de Wilks"]
        # pour la sortie des résultats
        #--------#

        #---- DÉBUT DE L'APPROCHE ASCENDANTE ----#
        if method == "forward":
            enteredVarList = []
            enteredVarSummary = []
            L_initial = 1  # Valeur du Lambda de Wilks pour q = 0
            for q in range(self.p):
                infoVar = []
                for name in varNames:
                    # Calcul du Lambda de Wilks (q+1)
                    wilksVarSelect = enteredVarList+[name]
                    L = calculations.wilks_log(
                        self.Vb.loc[wilksVarSelect, wilksVarSelect],
                        self.Wb.loc[wilksVarSelect, wilksVarSelect])
                    # Fin du calcul du Lambda de Wilks
                    # Calcul des degrés de liberté
                    ddl1, ddl2 = self.K-1, self.n-self.K-q
                    # Calcul de la statistique F
                    F = ddl2/ddl1 * (L_initial/L-1)
                    R = 1-(L/L_initial)  # Calcul du R² partiel
                    # Calcul de la p-value du test
                    pval = calculations.p_value(F, ddl1, ddl2)
                    infoVar.append((R, F, pval, L))

                self.infoStepResults = pd \
                    .DataFrame(infoVar, index=varNames, columns=colNames) \
                    .sort_values(by=["F-statistique"], ascending=False)

                enteredVar = self.infoStepResults["Lambda de Wilks"].idxmin()

                #---- SORTIE HTML ----#
                self._htmlStringOutput += ("<h3>S&#233;lection ascendante : "
                                           "&#201;tape n&#176;%i</h3><h4>D&#233;tail des "
                                           "r&#233;sultats</h4><p>DF = %i, %i</p><div class='row "
                                           "justify-content-md-center'>%s</div>") % (
                    q+1, ddl1, ddl2, str(
                        self.infoStepResults.to_html(
                            classes="table table-striped",
                            float_format="%.6f", justify="center",
                            border=0)))
                #--------#

                if (self.infoStepResults.loc[enteredVar, "p-value"] > slentry):
                    self._htmlStringOutput += ("<p>La valeur de la p-value de la "
                                               "meilleure variable (%s) est sup&#233;rieure au risque "
                                               "fix&#233; (= %f).\rAucune variable ne peut &#234;tre "
                                               "choisie.</p>") % (enteredVar, slentry)
                    break
                else:
                    enteredVarList.append(enteredVar)
                    varNames.remove(enteredVar)
                    L_initial = self.infoStepResults.loc[enteredVar,
                                                         "Lambda de Wilks"]
                    enteredVarSummary.append(
                        list(self.infoStepResults.loc[enteredVar]))

                    self._htmlStringOutput += "<p>La variable %s est retenue.</p>" % (
                        enteredVar)

            self.stepdiscSummary = pd.DataFrame(
                enteredVarSummary, index=enteredVarList, columns=colNames)

            self._htmlStringOutput += ("<h3>Synth&#232;se de la proc&#233;dure "
                                       "STEPDISC : S&#233;lection ascendante</h3><p>Nombre de "
                                       "variables choisies : %i</p><h4>Variables retenues</h4><div "
                                       "class='row justify-content-md-center'>%s</div></div></body>"
                                       "</html>") % (len(enteredVarList), str(
                                           self.stepdiscSummary.to_html(
                                               classes="table table-striped", float_format="%.6f",
                                               justify="center", border=0)))
        #---- FIN DE L'APPROCHE ASCENDANTE ----#

        #---- DÉBUT DE L'APPROCHE DESCENDANTE ----#
        elif method == "backward":
            removedVarList = []
            removedVarSummary = []
            # Calcul du Lamba de Wilks pour q = p
            L_initial = calculations.wilks(self.Vb, self.Wb)
            for q in range(self.p, -1, -1):
                infoVar = []
                for name in varNames:
                    # calcul du Lambda de Wilks (q-1)
                    wilksVarSelect = [var for var in varNames if var != name]
                    L = calculations.wilks_log(
                        self.Vb.loc[wilksVarSelect, wilksVarSelect],
                        self.Wb.loc[wilksVarSelect, wilksVarSelect])
                    # fin du calcul du Lambda de Wilks
                    ddl1, ddl2 = self.K-1, self.n-self.K-q+1  # calcul des degrés de liberté
                    F = ddl2/ddl1*(L/L_initial-1)  # calcul de la statistique F
                    R = 1-(L_initial/L)  # calcul du R² partiel
                    # calcul de la p-value du test
                    pval = calculations.p_value(F, ddl1, ddl2)
                    infoVar.append((R, F, pval, L))

                self.infoStepResults = pd.DataFrame(
                    infoVar, index=varNames, columns=colNames) \
                    .sort_values(by=["F-statistique"])

                removedVar = self.infoStepResults["Lambda de Wilks"].idxmin()

                #---- SORTIE HTML ----#
                self._htmlStringOutput += ("<h3>S&#233;lection descendante : &#201;tape "
                                           "n&#176;%i</h3><h4>D&#233;tail des r&#233;sultats</h4>"
                                           "<p>DF = %i, %i</p><div class='row justify-content-md-"
                                           "center'>%s</div>") % (self.p-q+1, ddl1, ddl2, str(
                                               self.infoStepResults.to_html(
                                                   classes="table table-striped", float_format="%.6f",
                                                   justify="center", border=0)))

                if (self.infoStepResults.loc[removedVar, "p-value"] < slentry):
                    self._htmlStringOutput += ("<p>La valeur de la p-value de la pire "
                                               "variable (%s) est inf&#233;rieure au risque "
                                               "fix&#233; (= %f).\rAucune variable ne peut &#234;tre "
                                               "retir&#233;e.</p>") % (removedVar, slentry)
                    break
                else:
                    removedVarList.append(removedVar)
                    varNames.remove(removedVar)
                    removedVarSummary.append(
                        list(self.infoStepResults.loc[removedVar]))
                    L_initial = self.infoStepResults.loc[removedVar,
                                                         "Lambda de Wilks"]
                    self._htmlStringOutput += ("<p>La variable %s est &#233;limin"
                                               "&#233;e.</p>") % (removedVar)

            self.stepdiscSummary = pd.DataFrame(
                removedVarSummary, index=removedVarList, columns=colNames)

            self._htmlStringOutput += ("<h3>Synth&#232;se de la proc&#233;dure STEP"
                                       "DISC : S&#233;lection ascendante</h3><p>Nombre de variables "
                                       "&#233;limin&#233;es : %i</p><h4>Variables &#233;limin&#233;es"
                                       "</h4><div class='row justify-content-md-center'>%s</div></div>"
                                       "</body></html>") % (len(removedVarList), str(
                                           self.stepdiscSummary.to_html(
                                               classes="table table-striped", float_format="%.6f",
                                               justify="center", border=0)))
            #---- FIN DE L'APPROCHE DESCENDANTE ----#
