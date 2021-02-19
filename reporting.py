# -*- coding: utf-8 -*-
"""
Module regroupant les différentes classes pour la création de reporting
automatique, en format HTML ou PDF, dont les sorties ressemblent à celles 
des PROC DISCRIM et STEPDISC de SAS.
"""

# ---- data restitution librairies
from fpdf import FPDF
import datapane as dp


class HTML:
    """Pour reporting automatique en format HTML.
    """
    
    def create_html_head(self, proc):
        """Création automatisée du début du fichier html. Incorpore une feuille
        de style CSS populaire (Bootstrap) pour améliorer l'esthétique des
        résultats.
        """
        self._head = ("""<!DOCTYPE html>
        <html lang="fr" dir="ltr">
          <head>
            <title>R&#233;sultats : %s</title>
            <meta charset="utf-8" />
            <style></style>
            <link
              rel="stylesheet"
              href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
              integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z"
              crossorigin="anonymous"
            />
          </head>
          <body>
            <div class="container text-center">
                <h2>Proc&#233;dure %s</h2>""") % (proc, proc)
        
    def stepdisc_html_output(self, ProcStepdisc, fileName):
        """Création d'un reporting en format HTML pour la méthode stepdisc 
        de la classe LinearDiscriminantAnalysis
        (qui ressemble à PROC STEPDISC de SAS).
        
        Paramètres
        ----------
        ProcStepdisc : objet LinearDiscriminantAnalysis
            objet suite à appel de la méthode stepdisc() de la classe 
            LinearDiscriminantAnalysis
        fileName : string
            nom du fichier de sortie (avec ou sans .html)
        """
        if fileName[-5:] != ".html":
            fileName += ".html"
        
        self.create_html_head("STEPDISC")
        ProcStepdisc._stats_dataset()
        ProcStepdisc._stats_classes()
        
        with open(fileName, "w") as f:
            f.write(("""%s
                <h3>Informations sur le jeu de donn&#233;es</h3>
                    <div class='row justify-content-md-center'>%s</div>
                    <div class='row justify-content-md-center'>%s</div>
                %s""") % (
                self._head,
                str(ProcStepdisc.infoDataset.to_html(
                        classes="table table-striped",float_format="%.6f",
                        justify="center",border=0)),
                str(ProcStepdisc.infoClasses.to_html(
                        classes="table table-striped",float_format="%.6f",
                        justify="center",border=0)),
                ProcStepdisc._htmlStringOutput))
            f.close()
            
    def discrim_html_output(self, ProcDiscrim, fileName):
        """Création d'un reporting en format HTML grâce à la librairie datapane.
        
        Paramètres
        ----------
        ProcDiscrim : objet LinearDiscriminantAnalysis
            objet suite à appel de la méthode fit() et d'autres attributs
            de la classe LinearDiscriminantAnalysis
        fileName : string
            nom du fichier de sortie (avec ou sans .html)
        """
        if fileName[-5:] != ".html":
            fileName += ".html"
        
        self.create_html_head("DISCRIM")
        ProcDiscrim._stats_dataset()
        ProcDiscrim._stats_classes()
        ProcDiscrim._stats_pooled_cov_matrix()
        ProcDiscrim._stats_wilks()
        
        with open(fileName, "w") as f:
            f.write(("""%s
                <h3>General information about the data</h3>
                    <div class='row justify-content-md-center'>%s</div>
                    <div class='row justify-content-md-center'>%s</div>
                <h3>Informations on the covariance matrix</h3>
                    <div class='row justify-content-md-center'>%s</div>
                    <div class='row justify-content-md-center'>%s</div>
                <h3>Function of lda and its' intercept and coefficients</h3>
                    <div class='row justify-content-md-center'>%s</div>
                <h3>Statistics. Wilks' Lambda</h3>
                    <div class='row justify-content-md-center'>%s</div>
                """) % (
                self._head,
                str(ProcDiscrim.infoDataset.to_html(
                        classes="table table-striped",float_format="%.6f",
                        justify="center",border=0)),
                str(ProcDiscrim.infoClasses.to_html(
                        classes="table table-striped",float_format="%.6f",
                        justify="center",border=0)),
                str(ProcDiscrim.W.to_html(
                        classes="table table-striped",float_format="%.6f",
                        justify="center",border=0)),
                str(ProcDiscrim.infoCovMatrix.to_html(
                        classes="table table-striped",float_format="%.6f",
                        justify="center",border=0)),
                str(ProcDiscrim.infoFuncClassement.to_html(
                        classes="table table-striped",float_format="%.6f",
                        justify="center",border=0)),
                str(ProcDiscrim.infoWilksStats.to_html(
                        classes="table table-striped",float_format="%.6f",
                        justify="center",border=0))))
            f.close()
        
    def discrim_html_output_datapane(self, ProcDiscrim, fileName):
        """Création d'un reporting en format HTML pour la méthode PROC DISCRIM
        grâce à la librairie datapane.
        
        Paramètres
        ----------
        ProcDiscrim : objet LinearDiscriminantAnalysis
            objet suite à appel de la fonction fit() de la classe 
            LinearDiscriminantAnalysis
        fileName : string
            nom du fichier de sortie (avec ou sans .html)
        """
        if fileName[-5:] != ".html":
            fileName += ".html"
        
        ProcDiscrim._stats_dataset()
        ProcDiscrim._stats_classes()
        ProcDiscrim._stats_pooled_cov_matrix()
        ProcDiscrim._stats_wilks()
        report = dp.Report(
            dp.Text("# Linear Discriminant Analysis"),
            dp.Text("## General information about the data"),
            dp.Table(ProcDiscrim.infoDataset),
            dp.Table(ProcDiscrim.infoClasses),
            dp.Text("## Informations on the covariance matrix"),
            dp.Table(ProcDiscrim.W),
            dp.Table(ProcDiscrim.infoCovMatrix),
            dp.Text("## Function of lda and its' intercept "
                    "and coefficients"),
            dp.Table(ProcDiscrim.infoFuncClassement),
            dp.Text("## Statistics. Wilks' Lambda"),
            dp.Table(ProcDiscrim.infoWilksStats))
        
        report.save(path=fileName)


class PDF(FPDF):
    """Reporting automatique en format PDF pour la méthode fit() 
    et d'autres attributs de la classe LinearDiscriminantAnalysis
    ressemblant aux sorties de la PROC DISCRIM de SAS.
    """
    # Page footer
    def footer(self):
        """Facilite l'affichage du bas de page automatique dès la création d'une 
        instance PDF.
        Ref. : https://pyfpdf.readthedocs.io/en/latest/Tutorial/index.html
        """
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Text color in gray
        self.set_text_color(128)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
        
    def discrim_pdf_output(self, ProcDiscrim, fileName):
        """Création d'un reporting en format PDF grâce à la librairie FPDF.
        Les sorites ressemblent à celles de la procédure DISCRIM de SAS.
        
        Paramètres
        ----------
        ProcDiscrim : objet
            objet suite à appel de la fonction fit() de la classe du même nom
        fileName : string
            nom du fichier de sortie(avec ou sans .pdf)
            
        """
        if fileName[-4:] != ".pdf":
            fileName += ".pdf"
        
        ProcDiscrim._stats_dataset()
        ProcDiscrim._stats_classes()
        ProcDiscrim._stats_pooled_cov_matrix()
        ProcDiscrim._stats_wilks()
        # ---- Création du PDF
        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        # ---- Information du jeu de données
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(180, 10, 'General information about the data',
                 border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        #parcours de l'attribut de la classe LDA pour convertir
        #les données en string et les ajouter dans les cellules de pdf.
        for indx, elem in enumerate(ProcDiscrim.infoDataset.index):
            pdf.cell(180, 10, str(ProcDiscrim.infoDataset.index[indx]) + ': ' +
                     str(ProcDiscrim.infoDataset.iloc[indx, 0]), border=0, align='L')
            pdf.ln()
        pdf.ln()

        # ---- Statistiques des classes
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(180, 10, ' '*(len(max(ProcDiscrim.infoClasses.index, key=len))*2) +
                 'Frequences ' + 'Proportions', border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        j = 0
        for indx, elem in enumerate(ProcDiscrim.infoClasses.index):
            pdf.cell(80, 10, str(ProcDiscrim.infoClasses.index[indx]) + ': ' +
                     str(ProcDiscrim.infoClasses.iloc[indx, j]) + '   ' +
                     str(round(ProcDiscrim.infoClasses.iloc[indx, j+1], 4)),
                     border=0, align='L')
            pdf.ln()
        pdf.ln()
        # ----

        # ---- Matrice de covariance
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(180, 10, 'Informations on the covariance matrix',
                 border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        pdf.cell(180, 10, ' '*(len(max(ProcDiscrim.infoCovMatrix.index, key=len))*2) +
                 'Values ', border=0, align='L')
        pdf.ln()
        for indx, elem in enumerate(ProcDiscrim.infoCovMatrix.index):
            pdf.cell(180, 10, str(ProcDiscrim.infoCovMatrix.index[indx]) + ': ' +
                     str(ProcDiscrim.infoCovMatrix.iloc[indx, 0]), border=0, align='L')
            pdf.ln()
        pdf.ln()
        # ----

        # ---- Fonction de classement
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(
            180, 10, "Function of lda and its' intercept and coefficients",
            border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        # lign with column names
        my_str = ' '*(len(max(ProcDiscrim.infoFuncClassement.index, key=len))*2)
        for indx, elem in enumerate(ProcDiscrim.infoFuncClassement.columns):
            sub_str = str(ProcDiscrim.infoFuncClassement.columns[indx]) + ' '
            my_str += sub_str
        pdf.cell(180, 10, my_str, border=0, align='L')
        pdf.ln()

        my_str = ''
        for indx, elem in enumerate(ProcDiscrim.infoFuncClassement.index):
            #print(indx, elem)
            my_str = str(ProcDiscrim.infoFuncClassement.index[indx])
            # print(my_str)
            for j in range(len(ProcDiscrim.infoFuncClassement.columns)):
                # print(j)
                my_str += ' ' + str(round(
                    ProcDiscrim.infoFuncClassement.iloc[indx, j], 6))
                # print(my_str)
            if j == (len(ProcDiscrim.infoFuncClassement.columns)-1):
                pdf.cell(150, 10, my_str, border=0, align='L')
                pdf.ln()

         # ---- Lambda de Wilks
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(180, 10, "Statistics. Wilks' Lambda", border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
       
        for indx, elem in enumerate(ProcDiscrim.infoWilksStats.T.index):
            pdf.cell(180, 10, str(ProcDiscrim.infoWilksStats.T.index[indx]) + ': ' +
                     str(ProcDiscrim.infoWilksStats.T.iloc[indx, 0]), border=0,
                     align='L')
            pdf.ln()
        pdf.ln()

        # ---- Rendu du PDF
        pdf.set_compression(True)
        pdf.set_display_mode('fullpage')
        pdf.output(fileName, 'F')
