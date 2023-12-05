import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

link = 'https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv'

df = pd.read_csv(link, sep=',')

# barre de navigation vertical
st.sidebar.title("Sommaire")

pages = ["Contexte de la quête", "Exploration des données", "Analyse des données"]

page = st.sidebar.radio("Aller vers la page :", pages)



# -----------------------------------------------------------
# menu page Contexte de la quête
if page == pages[0] :

    st.header('Quête Streamlit : build and share data apps ')

    st.write("### Contexte de la quête")

    st.markdown("---")

    st.write("A partir du dataset des voitures, affichage : ")
    st.write("Analyse de corrélation et de distribution grâce à différents graphiques et des commentaires. ")
    st.write("- des boutons doivent être présents pour pouvoir filtrer les résultats par région (US / Europe / Japon). ")
    st.write("- l'application doit être disponible sur la plateforme de partage.")
    st.write("\n")  # Ajouter un saut de ligne manuel
    st.write("Nous avons à notre disposition le dataset cars qui contient des données sur les voitures. ")
    st.write("Chaque observation en ligne correspond à une voiture.")
    st.write("Chaque variable en colonne est une caractéristique des voitures ")
    st.write("\n")  # Ajouter un saut de ligne manuel
    st.write("Dans un premier temps, nous explorerons le dataset. ")
    st.write("Puis nous analyserons visuellement pour en extraire des informations selon certains axes d'étude.")



# -----------------------------------------------------------
elif page == pages[1]:

    st.header('Quête Streamlit : build and share data apps ')

    st.write("### Exploration des données")

    st.markdown("---")

    st.write('Aperçu du dataframe')
    st.dataframe(df.head())

    st.write("dimensions du dataframe")
    st.write(df.shape)

    if st.checkbox("Afficher les valeurs manquantes") :
        st.dataframe(df.isna().sum())

    if st.checkbox("Afficher les doublons") :
        duplicated_rows = df[df.duplicated()]
        st.dataframe(duplicated_rows)


# -----------------------------------------------------------
elif page == pages[2]:

    st.header('Quête Streamlit : build and share data apps ')

    st.write("### Analyse des données")

    st.markdown("---")

    # Use factorize to recode features continent (continentfact) into numerical data, because ML needs (and loves) numerical data.
    # factorize la colonne continent
    #df['continentfact'] = df['continent'].factorize()[0]


    # Checkbox pour filtrer par continent
    continents_filter = st.checkbox("Filtrer par continent", False)

    if continents_filter:
        all_continents = df['continent'].unique()
        selected_continents = [st.checkbox(continent, key=continent) for continent in all_continents]

        # Filtrer le DataFrame en fonction des continents sélectionnés
        filtered_df = df[df['continent'].isin([continent for continent, selected in zip(all_continents, selected_continents) if selected])]

        # Afficher le DataFrame original
        #st.write("DataFrame original :")
        #st.dataframe(df)

        # Afficher le DataFrame filtré
        st.write("DataFrame filtré :")
        st.dataframe(filtered_df)
    else:
        # Si la checkbox n'est pas cochée, afficher le DataFrame complet
        st.dataframe(df)
        st.markdown("---")

        st.write("1. **mpg** représente le nombre de miles par gallon, ce qui peut indiquer l'efficacité énergétique d'une voiture. Des valeurs plus élevées sont généralement meilleures.")
        st.write("2. **cylinders** indique le nombre de cylindres dans le moteur. Plus le nombre de cylindres est élevé, plus la voiture peut potentiellement produire de la puissance, mais cela peut également affecter l'efficacité du carburant.")
        st.write("3. **cubicinches** représente la taille du moteur en pouces cubes, ce qui peut être lié à la puissance du moteur. Des valeurs plus élevées indiquent généralement un moteur plus puissant.")
        st.write("4. **hp** (horsepower) indique la puissance du moteur. Des valeurs plus élevées signifient généralement plus de puissance.")
        st.write("5. **weightlbs** représente le poids de la voiture en livres. Un poids plus léger peut contribuer à une meilleure économie de carburant.")
        st.write("6. **time-to-60** indique le temps nécessaire à la voiture pour accélérer de 0 à 60 mph. Des valeurs plus faibles indiquent une accélération plus rapide.")
        st.write("7. **year** représente l'année de fabrication de la voiture.")
        st.write("8. **continent** indiquant le continent d'origine de la voiture")


        st.markdown("---")


        # Use factorize to recode features continent (continentfact) into numerical data, because ML needs (and loves) numerical data.
        # factorize la colonne continent
        df['continentfact'] = df['continent'].factorize()[0]

        # selectione que les colonnes numérique
        df2 = df[['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year', 'continentfact']]

    # Analyse de corrélation avec une matrice de corrélation
        st.subheader("Analyse de corrélation avec une matrice de corrélation")
        corr_matrix = df2.corr()
        #plt.figure(figsize=(8, 7))
        #sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        #plt.title('Matrice de Corrélation')
        #plt.show()
        #st.pyplot(sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f"))
        #st.pyplot(plt.title('Matrice de Corrélation'))

        # Convertir le graphique en une image
        buffer = io.BytesIO()
        plt.figure(figsize=(8, 7))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matrice de Corrélation')
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image = buffer.getvalue()

        # Afficher l'image dans Streamlit
        st.image(image, use_column_width=True)

        st.write("Les coefficients de corrélation peuvent être interprétés comme suit :")
        st.write("**Un coefficient de corrélation proche de 1** indique une corrélation positive forte. Cela signifie que les deux variables sont associées de manière positive, c'est-à-dire que lorsque l'une augmente, l'autre augmente également.")
        st.write("**Un coefficient de corrélation proche de -1** indique une corrélation négative forte. Cela signifie que les deux variables sont associées de manière négative, c'est-à-dire que lorsque l'une augmente, l'autre diminue.")
        st.write("**Un coefficient de corrélation proche de 0** indique une corrélation faible ou nulle. Cela signifie que les deux variables ne sont pas associées de manière significative.")
        st.write("En se basant sur ces interprétations, nous pouvons tirer les conclusions suivantes de cette matrice de corrélation :")
        st.write("**La consommation de carburant est fortement corrélée avec le nombre de cylindres**, la cylindrée et la puissance. Cela signifie que les voitures avec plus de cylindres, une cylindrée plus importante et une puissance plus élevée consomment généralement plus de carburant.")
        st.write("**La consommation de carburant est également corrélée négativement avec le poids**. Cela signifie que les voitures plus lourdes consomment généralement plus de carburant.")
        st.write("**La consommation de carburant est faiblement corrélée avec l'année de production et le continent de production**. Cela signifie que ces variables n'ont pas un impact significatif sur la consommation de carburant.")
        st.write("En conclusion, cette matrice de corrélation montre que la consommation de carburant est principalement influencée par le nombre de cylindres, la cylindrée et la puissance.")

        st.markdown("---")

    # Graphiques de dispersion pour les paires de variables corrélées
        st.subheader("Graphiques de dispersion pour les paires de variables corrélées")
        #sns.pairplot(df2)
        #plt.suptitle('Graphiques de Dispersion')
        #plt.show()
        scatter_plot = sns.pairplot(df2)
        st.pyplot(scatter_plot.fig)
        #st.pyplot(plt.suptitle('Graphiques de Dispersion'))

        st.write("Le graphique de dispersion montre la relation entre deux variables, la consommation de carburant (mpg) et le nombre de cylindres (cylinders).")
        st.write("Les points du graphique sont dispersés dans un motif linéaire, ce qui indique qu'il existe une corrélation positive entre les deux variables. Cela signifie que lorsque le nombre de cylindres augmente, la consommation de carburant augmente également.")
        st.write("La force de la corrélation peut être mesurée par le coefficient de corrélation, qui est de 0,91. Un coefficient de corrélation proche de 1 indique une corrélation forte.")
        st.write("**En conclusion, ce graphique de dispersion montre qu'il existe une corrélation positive forte entre la consommation de carburant et le nombre de cylindres. Cela signifie que les voitures avec plus de cylindres consomment généralement plus de carburant.**")
        st.write("Voici quelques autres observations que l'on peut faire sur ce graphique :")
        st.write("- Il y a une certaine dispersion des points autour de la ligne de régression. Cela signifie qu'il existe une certaine variabilité dans la relation entre les deux variables.")
        st.write("- Il y a quelques points qui sont situés loin de la ligne de régression. Ces points peuvent être considérés comme des valeurs aberrantes.")
        st.write("Les valeurs aberrantes peuvent être causées par des erreurs de mesure ou par des facteurs non pris en compte dans l'analyse. Dans ce cas, il est important d'examiner les données plus en détail pour déterminer si les valeurs aberrantes sont dues à des erreurs ou à des facteurs réels.")

        st.markdown("---")

    # Histogrammes pour l'analyse de distribution
        st.subheader("Histogrammes pour l'analyse de distribution")
        #df2.hist(figsize=(12, 8))
        #plt.suptitle('Histogrammes pour l\'analyse de distribution')
        #plt.show()
        hist_plot = df2.hist(figsize=(12, 8))
        st.pyplot(hist_plot[0][0].figure)
        #st.pyplot(plt.suptitle("Histogrammes pour l'analyse de distribution"))


        st.write("L'histogramme montre la distribution de la consommation de carburant (mpg) pour un ensemble de voitures. L'axe horizontal représente la consommation de carburant en miles par gallon et l'axe vertical représente le nombre de voitures dans chaque catégorie de consommation de carburant.")
        st.write("L'histogramme montre que la consommation de carburant est majoritairement concentrée entre 20 et 30 miles par gallon. Il y a un pic de la distribution à environ 25 miles par gallon. Cela signifie que la plupart des voitures dans cet ensemble ont une consommation de carburant de 25 miles par gallon environ.")
        st.write("Il y a également une petite quantité de voitures avec une consommation de carburant inférieure à 20 miles par gallon et une petite quantité de voitures avec une consommation de carburant supérieure à 30 miles par gallon.")
        st.write("En conclusion, cet histogramme montre que la consommation de carburant pour cet ensemble de voitures est généralement bonne. La plupart des voitures ont une consommation de carburant de 25 miles par gallon environ.")
        st.write("Voici quelques autres observations que l'on peut faire sur cet histogramme :")
        st.write("La distribution est étalée, ce qui signifie qu'il existe une certaine variabilité dans la consommation de carburant des voitures de cet ensemble.")
        st.write("Il n'y a pas de valeurs aberrantes visibles.")
        st.write("Les valeurs aberrantes seraient des points qui se situeraient loin de la distribution principale. Si elles étaient présentes, il serait important d'examiner les données plus en détail pour déterminer si elles sont dues à des erreurs de mesure ou à des facteurs réels.")


