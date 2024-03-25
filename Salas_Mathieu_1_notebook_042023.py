#!/usr/bin/env python
# coding: utf-8

# In[170]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import nbconvert


# # Calcul de la proportion de personnes en état de sous-nutrition en 2017
# Création du DataFrame population et nombre de personnes en sous-nutrition pour l'année 2017
# Comme indiqué par Julien dans ses notes, les valeurs de personnes en état de sous nutrition sont exprimées en million, pour la moyenne des 3 années, tandis que la population par pays est exprimée en millier.
# 
# Pour 2017, on peut donc conserver la moyenne des années 2016, 2017 et 2018, soit la tranche 2016-2018.

# In[2]:


aide_alimentaire = pd.read_csv("aidealimentaire.csv")
dispo_alimentaire = pd.read_csv("dispoalimentaire.csv")
population = pd.read_csv("population.csv")
sous_nutrition = pd.read_csv("sousnutrition.csv")


# In[ ]:


sousNutrition = pd.read_csv("sousnutrition.csv")

sousNutrition.loc[sousNutrition["Valeur"] == "<0.1", "Valeur"] = 0

sousNutrition["Valeur"] = sousNutrition["Valeur"].fillna(0)


# In[9]:


# DataFrame du nombre de personnes en sous-nutrition par pays en 2017
popMondiale = pd.read_csv("population.csv")

popMondialeParPays2017 = popMondiale[popMondiale["Année"] == 2017]


# In[10]:


# DataFrame du nombre de personnes en sous-nutrition par pays en 2017

sousNutrition = pd.read_csv("sousnutrition.csv")

sousNutrition2017 = sousNutrition[sousNutrition["Année"] == "2016-2018"]


# In[12]:


# Créer un DataFrame composé des populations et du nombre de personnes en sous-nutrition par pays en 2017

dfPopEtSousNutri = pd.merge(sousNutrition2017, popMondialeParPays2017, on="Zone", how="inner")

# Arrondir les valeurs <0.1 
dfPopEtSousNutri.loc[dfPopEtSousNutri["Valeur_x"] == "<0.1", "Valeur_x"] = 0

# Conserver et renommer les colonnes utiles
dfPopEtSousNutri = dfPopEtSousNutri[["Zone", "Valeur_x", "Valeur_y"]]
dfPopEtSousNutri = dfPopEtSousNutri.rename(columns={"Valeur_x": "Nombre personnes en sous-nutrition", \
                                                    "Valeur_y": "Population"})

# Traiter comme des valeur numériques
dfPopEtSousNutri["Nombre personnes en sous-nutrition"] = \
pd.to_numeric(dfPopEtSousNutri["Nombre personnes en sous-nutrition"]).fillna(0)


# Mettre les deux colonnes dans la même unité (nombre de personnes)
dfPopEtSousNutri["Nombre personnes en sous-nutrition"] *= 1000000
dfPopEtSousNutri["Population"] *= 1000

# Transformer le résultat en integer
dfPopEtSousNutri["Nombre personnes en sous-nutrition"] = dfPopEtSousNutri["Nombre personnes en sous-nutrition"].astype(int)
dfPopEtSousNutri["Population"] = dfPopEtSousNutri["Population"].astype(int)

display(dfPopEtSousNutri)


# # Calcul de la proportion de personnes en sous-nutrition en 2017

# Calcul du nombre de personnes en sous-nutrition en 2017

# In[13]:


nbPersSousNourrie2017 = dfPopEtSousNutri["Nombre personnes en sous-nutrition"].sum()

print("Nombre de personnes sous nourries dans le monde en 2017 = {:,.0f}".format(nbPersSousNourrie2017))


# Calcul de la population mondiale en 2017
# 

# In[14]:


popMondiale2017 = dfPopEtSousNutri["Population"].sum()

print("Population mondiale en 2017 = {:,.0f}".format(popMondiale2017))


# Calcul de la proportion de personnes en sous-nutrition en 2017

# In[20]:


proportionSousNutrition2017 = nbPersSousNourrie2017 / popMondiale2017 * 100

print("Proportion de la population mondiale en sous nutrition en 2017 = {:,.2f}%".format(proportionSousNutrition2017))

labels = ['Population mondiale', 'Personnes sous-alimentées']
sizes = [popMondiale2017, nbPersSousNourrie2017]
colors = ['#1f77b4', '#ff7f0e']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors=colors, labels=labels, autopct='%0.1f%%', startangle=90)
ax1.axis('equal')

plt.title('Proportion de la population mondiale en sous-nutrition en 2017')
plt.show()


# # Nombre théorique de personnes qui pourraient être nourries selon la disponibilité alimentaire mondiale

# In[ ]:


Calculer la disponibilité alimentaire en Kcal/jour/pers nous permet de savoir combien de personnes pourraient être théoriquement nourries par jour dans chaque pays.

On partira pour se faire du principe qu'un humain a besoin en moyenne de 2500 Kcal par jour, ce chiffre pouvant en pratique varier selon la taille, l'âge, le poids, le sexe de chaque individu


# Calcul de la disponibilité alimentaire pour chaque pays
# 

# In[60]:


# Calcul de la disponibilité alimentaire pour chaque pays

dispoAlimentaire = pd.read_csv("dispoalimentaire.csv")

dispoParPays = dispoAlimentaire.groupby("Zone").sum(numeric_only=True)

dispoParPays = dispoParPays[["Disponibilité alimentaire (Kcal/personne/jour)"]]

display(dispoParPays)


# Calcul du nombre de personnes théoriquement nourries par pays

# In[61]:


habNourrisParPays = pd.DataFrame({'Nombre de personne théoriquement nourries': \
                       dispoParPays['Disponibilité alimentaire (Kcal/personne/jour)']/2500},
                     index = dispoParPays.index)

habNourrisParPays = habNourrisParPays.reset_index()

display(habNourrisParPays)


# Pondération des résultats par la population de chaque pays
# 

# In[14]:


# Ajouter la colonne population à notre dataframe. 

df_ponderation = pd.merge(habNourrisParPays, popMondialeParPays2017, on='Zone')

# Garder les colonnes utiles 

df_ponderation = df_ponderation[['Zone', 'Nombre de personne théoriquement nourries', 'Valeur']]

# Renommer la colonne Valeur en 'Population en millier d'habitants'

df_ponderation = df_ponderation.rename(columns={'Valeur' : "Population en millier d'habitants"})

display(df_ponderation)


# In[15]:


# Calculer le coeficient de pondération

df_ponderation['Coeficient de Pondération'] = df_ponderation["Population en millier d'habitants"] / df_ponderation["Population en millier d'habitants"].sum()

display(df_ponderation)


#  Nombre de personnes qui pourraient être théoriquement nourries à l'échelle mondiale
# 

# In[16]:


# Utiliser la fonction 'average' de numpy pour faire la moyenne pondérée ('weight')

values = df_ponderation['Nombre de personne théoriquement nourries']
weights = df_ponderation['Coeficient de Pondération']

moyennePonderee = np.average(values, weights = weights)

print("On pourrait nourrir {:,.4f} fois la population mondiale en 2017 (soit {:,.2f}%)"\
      .format(moyennePonderee, moyennePonderee*100))


# In[17]:


nbPersTheoriquementNourriesMonde = float(popMondiale2017) * moyennePonderee

print("Le nombre de personnes que l'on pourrait théoriquement nourrir à l'échelle mondiale en 2017 est : {:,.0f}"\
      .format(nbPersTheoriquementNourriesMonde))


# # Nombre de personnes qui pourraient être nourries uniquement avec des aliments d'origine végétale

# Calcul de la disponibilité alimentaire des produits d'originie végétale pour chaque pays
# 

# In[58]:


dispoVege = dispoAlimentaire.loc[dispoAlimentaire["Origine"] == "vegetale"]

dispoVegeParPays = dispoVege.groupby("Zone").sum(numeric_only=True)

dispoVegeParPays = dispoVegeParPays[["Disponibilité alimentaire (Kcal/personne/jour)"]]

display(dispoVegeParPays)


# Calcul du nombre de personnes théoriquement nourries par pays, uniquement avec des produits d'origine végétale
# 

# In[19]:


habNourrisParPaysVege = pd.DataFrame({"Nombre de personne théoriquement nourries Vege": \
                       dispoVegeParPays['Disponibilité alimentaire (Kcal/personne/jour)']/2500},
                     index = dispoVegeParPays.index)

habNourrisParPaysVege = habNourrisParPaysVege.reset_index()

display(habNourrisParPaysVege)


# Pondération des résultats par la population de chaque pays
# 

# In[20]:


# Ajouter la colonne population à notre dataframe. 

df_ponderationVege = pd.merge(habNourrisParPaysVege, popMondialeParPays2017, on='Zone')

# Garder les colonnes utiles 

df_ponderationVege = df_ponderationVege[['Zone', 'Nombre de personne théoriquement nourries Vege', 'Valeur']]

# Renommer la colonne Valeur en 'Population en millier d'habitants'

df_ponderationVege = df_ponderationVege.rename(columns={'Valeur' : "Population en millier d'habitants"})

display(df_ponderationVege)


# In[21]:


# Calculer le coeficient de pondération

df_ponderationVege['Coeficient de Pondération'] = df_ponderationVege["Population en millier d'habitants"] / df_ponderationVege["Population en millier d'habitants"].sum()

display(df_ponderationVege)


# Nombre de personnes qui pourraient être théoriquement nourries, uniquement à partir de la disponibilité alimentaire d'origine végétale

# In[22]:


# Utiliser la fonction 'average' de numpy pour faire la moyenne pondérée ('weight')

values = df_ponderationVege['Nombre de personne théoriquement nourries Vege']
weights = df_ponderationVege['Coeficient de Pondération']

moyennePondereeVege = np.average(values, weights = weights)

print("On pourrait nourrir {:,.2f}% de la population mondiale en 2017, uniquement avec des produits\
 d'origine végétale".format(moyennePondereeVege*100))


# In[23]:


nbPersNourriesMondeVege = float(popMondiale2017) * moyennePondereeVege

print("Le nombre de personnes que l'on pourrait théoriquement nourrir à l'échelle mondiale en 2017, de cette manière\
 est : \n{:,.0f}".format(nbPersNourriesMondeVege))


# # Proportion de la disponibilité intérieure en fonction de l'usage

# Calcul de la disponibilité intérieure totale
# 

# In[24]:


dispoAlimentaire = pd.read_csv("dispoalimentaire.csv")

dispoInterieure = dispoAlimentaire["Disponibilité intérieure"].sum()

print("La disponibilité intérieure totale est égale à : {:,.0f} milliers de tonnes".format(dispoInterieure))


# Proportion de la disponibilité intérieure concrètement utilisée pour l'alimentation humaine
# 

# In[25]:


nourriture = dispoAlimentaire["Nourriture"].sum()
print ("Total de la disponibilité alimentaire utilisée comme nourriture : {:,.0f} milliers de tonnes".format(nourriture))


# In[8]:


# Calcul de la proportion d'utilisation de la nourriture
proportionNourriture = (nourriture / dispoInterieure) * 100

dfTemp = pd.DataFrame({'value': [proportionNourriture, 100 - proportionNourriture]},
                 index=['Nourriture', 'Autres usages'])

sns.set_theme(style='whitegrid', palette='pastel')
fig, ax = plt.subplots()
ax.pie(dfTemp['value'], labels=dfTemp.index, autopct='%1.1f%%', startangle = 90)
ax.axis('equal')
ax.set_title('Proportion de la disponibilité intérieure utilisée comme nourriture')

plt.show()


# Détail des proportions
# 

# In[28]:


# Calcul des proportions

categories = ["Nourriture", "Aliments pour animaux", "Pertes"]

proportions = []
for cat in categories:
    proportion = (dispoAlimentaire[cat].sum() / dispoInterieure) * 100
    proportions.append(proportion)

dfProportions = pd.DataFrame({'value': proportions},
                  index=['Nourriture Humaine', 'Nourriture Animaux', 'Pertes'])


# Même données mais en version "bar"

sns.set_theme(style='whitegrid', palette='pastel')
fig, ax = plt.subplots()

sns.barplot(x=dfProportions.index, y=dfProportions['value'], ax=ax)

# Ajouter la valeur sur chaque bar
for i, v in enumerate(dfProportions['value']):
    ax.text(i, v + 1, '{:,.1f}%'.format(v), ha='center', fontsize=12)


ax.set_title("Proportion d'usage de la disponibilité intérieure par catégorie")
ax.set_ylabel("Proportion (%)")
plt.xticks(rotation=45, ha="right")

plt.show()


# # Utilisation des céréales pour l'alimentation animale, par rapport à l'alimentation humaine

# Proportion en pourcentage de céréales utilisées pour l'alimentation humaines et l'alimentation animale

# In[29]:


cereales2 = ['Blé','Riz','Orge','Maïs','Seigle','Avoine','Millet','Sorgho','Céréales, Autres']

# Filtrer les lignes contenant une céréale dans la colonne "Produit"
dfCereales2 = dispoAlimentaire.loc[dispoAlimentaire['Produit'].isin(cereales2),:]

display(dfCereales2)


# In[30]:


proportionAnimaux = dfCereales2["Aliments pour animaux"].sum()/dfCereales2["Disponibilité intérieure"].sum()*100
print("Proportion d'alimentation animale : {:.2f}%".format(proportionAnimaux))

proportionHumain = dfCereales2["Nourriture"].sum()/dfCereales2["Disponibilité intérieure"].sum()*100
print("Proportion d'alimentation humaine : {:.2f}%".format(proportionHumain))


# Répartition selon les céréales
# 

# In[31]:


# Calcul de la quantité, pour chaque céréale, utilisée pour l'alimentation animale et pour l'alimentation humaine.

dfCereales2 = dfCereales2[["Produit", "Aliments pour animaux", "Nourriture"]]

dfCereales2 = dfCereales2.groupby(dfCereales2["Produit"]).sum()

# Ajout d'une colonne "Proportion (%)" représentant la part affectée à l'alimentation animale.

dfCereales2["Proportion d'usage pour les animaux par rapport à la nourriture (%)"] = \
dfCereales2["Aliments pour animaux"] / (dfCereales2["Aliments pour animaux"] + dfCereales2["Nourriture"]) * 100

dfCereales2 = dfCereales2.sort_values("Proportion d'usage pour les animaux par rapport à la nourriture (%)", ascending=False)

dfCereales2 = dfCereales2.reset_index().head(9)

display(dfCereales2)


# In[32]:


# Creation d'un graphique en barres 
ax = dfCereales2.plot(x='Produit', y=['Aliments pour animaux', 'Nourriture'], kind='bar')

# Titre et noms des axes
sns.set_theme(style='whitegrid', palette='pastel')
ax.set_title('Quantité utilisée pour les animaux et pour la nourriture humaine')
ax.set_xlabel('Céréale')
ax.set_ylabel('Quantité (millier de tonnes)')
plt.xticks(rotation=45, ha="right")

plt.show()


# In[33]:


cereals = ["Orge", "Maïs", "Avoine"]
df_cereals = dfCereales2[dfCereales2["Produit"].isin(cereals)]

# Extraire les valeurs pour chaque céréale
cereals_data = df_cereals.groupby("Produit")[["Aliments pour animaux", "Nourriture"]].sum()

# Créer un graphique à 3 colonnes
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Créer un graphique pour chacune des 3 céréales les plus utilisées pour l'alimentation animale
for i, cereal in enumerate(cereals):
    axs[i].pie(cereals_data.loc[cereal], labels=["Aliments pour animaux", "Nourriture"], autopct="%1.1f%%")
    axs[i].set_title(cereal)

# Display the plot
plt.show()


# # Les 10 Pays où la proportion personnes sous-alimentées est la plus élevée en 2017

# In[34]:


dfPopEtSousNutri['Proportion'] = dfPopEtSousNutri['Nombre personnes en sous-nutrition'] / dfPopEtSousNutri['Population']

dfPopEtSousNutri_sorted = dfPopEtSousNutri.sort_values('Proportion', ascending=False)

dfPopEtSousNutri_sorted = dfPopEtSousNutri_sorted[["Zone", "Proportion"]].head(10)

display(dfPopEtSousNutri_sorted)


# Les 10 Pays ayant le plus bénéficié de l'aide depuis 2013
# 

# In[59]:


aideAlimentaire = pd.read_csv("aidealimentaire.csv")

aideAlimentaireParPays = aideAlimentaire.groupby("Pays bénéficiaire").sum(numeric_only=True)

aideAlimentaireParPays_sorted = aideAlimentaireParPays.sort_values('Valeur', ascending=False)

aideAlimentaireParPays_sorted = aideAlimentaireParPays_sorted[["Valeur"]].head(10)

aideAlimentaireParPays_sorted = aideAlimentaireParPays_sorted.rename(columns={"Valeur" : "Quantité (tonnes)"})

display(aideAlimentaireParPays_sorted)


# # Disponibilité par habitant
# 

# Pays ayant le plus de disponibilité par habitant
# 

# In[36]:


topDispoParPays = dispoParPays.sort_values("Disponibilité alimentaire (Kcal/personne/jour)", ascending=False).head().reset_index()
display(topDispoParPays)


# Pays ayant le moins de disponibilité par habitant
# 

# In[38]:


lowDispoParPays = dispoParPays.sort_values("Disponibilité alimentaire (Kcal/personne/jour)", ascending=True).head(10).reset_index()
display(lowDispoParPays)


# # Export du manioc par la Thaïlande
# 
# 

# In[41]:


sousNutritionThailande = dfPopEtSousNutri[dfPopEtSousNutri["Zone"] == "Thaïlande"]
sousNutritionThailande = sousNutritionThailande.copy()
sousNutritionThailande['Proportion (%)'] = sousNutritionThailande['Nombre personnes en sous-nutrition'] / sousNutritionThailande['Population'] * 100
display(sousNutritionThailande)


# Proportion de manioc exporté par la Thaïlande par rapport à sa production
# 

# In[42]:


maniocGate = dispoAlimentaire[(dispoAlimentaire["Zone"]=="Thaïlande") & (dispoAlimentaire["Produit"].str.contains("Manioc"))]

maniocGate = maniocGate[["Zone", "Produit", "Disponibilité intérieure", "Exportations - Quantité", "Importations - Quantité", "Nourriture", "Pertes", "Production"]]

display(maniocGate)


# In[43]:


export = maniocGate["Exportations - Quantité"].sum()
production = maniocGate["Production"].sum()
nourriture = maniocGate["Nourriture"].sum()

proportionExport = (export / production) * 100
proportionNourriture  = (nourriture / production) * 100

print("Proportion de manioc exporté : {:,.2f}%".format(proportionExport))
print("Proportion de manioc utilisée pour l'alimentation : {:,.2f}%".format(proportionNourriture))


# In[44]:


# Création d'un graphique pour visualiser les résultats

dfManioc = pd.DataFrame({'value': [proportionExport, 100 - proportionExport - proportionNourriture, proportionNourriture]},
                  index=['Export', 'Autre', 'Nourriture'])

sns.set_theme(style='whitegrid', palette='pastel')
fig, ax = plt.subplots()
ax.pie(dfManioc['value'], labels=dfManioc.index, autopct='%1.2f%%')
ax.axis('equal')
ax.set_title('Proportion du Manioc exporté et utilisé comme nourriture, sur la production totale en Thaïlande ')

plt.show()


# # Apport supplémentaire

# Création d'un DataFrame avec la proportion de personnes en sous-nutrition pour chaque année
# 

# In[47]:


# Créer un mapping pour faire correspondre les années et les périodes entre population.csv et sous_nutrition.csv
year_mapping = {
    '2012-2014': 2013,
    '2013-2015': 2014,
    '2014-2016': 2015,
    '2015-2017': 2016,
    '2016-2018': 2017,
    '2017-2019': 2018
}

# Appliquer le mapping sur sousNutritionTemp sans affecter sousNutrition
sousNutritionTemp = sousNutrition.copy()
sousNutritionTemp['Année'] = sousNutritionTemp['Année'].map(year_mapping)

# Merge les Dataframe
evolutionSousNutrition = pd.merge(sousNutritionTemp, popMondiale, on=['Zone', 'Année'])

# Renommer les colonnes, unifier les unités et traiter comme des integer

evolutionSousNutrition = evolutionSousNutrition.rename(columns={"Valeur_x": "Nombre personnes en sous-nutrition", \
                                                    "Valeur_y": "Population"})

evolutionSousNutrition["Nombre personnes en sous-nutrition"] = \
pd.to_numeric(evolutionSousNutrition["Nombre personnes en sous-nutrition"]).fillna(0)

evolutionSousNutrition["Nombre personnes en sous-nutrition"] *= 1000000
evolutionSousNutrition["Nombre personnes en sous-nutrition"] = evolutionSousNutrition["Nombre personnes en sous-nutrition"].astype(int)
evolutionSousNutrition["Population"] *= 1000
evolutionSousNutrition["Population"] = evolutionSousNutrition["Population"].astype(int)

evolutionSousNutrition["Proportion (%)"] = evolutionSousNutrition["Nombre personnes en sous-nutrition"]/evolutionSousNutrition["Population"]*100

display(evolutionSousNutrition)


# Sélectionner uniquement les 5 pays les plus en difficulté en 2018
# 

# In[48]:


evolutionSousNutrition_2018 = evolutionSousNutrition.loc[evolutionSousNutrition['Année'] == 2018]

evolutionSousNutrition_sorted = evolutionSousNutrition_2018.sort_values("Proportion (%)",ascending=False)

top_5 = list(evolutionSousNutrition_sorted["Zone"].head(5))

print("Les 5 pays les plus en difficulté sont : ", top_5)


# In[49]:


# Création d'un dataframe avec toutes les valeurs pour chaque année et pour chaque pays en difficulté
evolutionTop5 = evolutionSousNutrition[evolutionSousNutrition["Zone"].isin(top_5)]


# Visualisation de l'évolution de la proportion de chaque pays en sous-nutrition
# 

# In[50]:


# Create a figure and axis object
fig, ax = plt.subplots(figsize=(8,6))

# Iterate over the top 5 countries and plot their data
for pays in top_5:
    # filter data for the current country
    pays_data = evolutionTop5[evolutionTop5['Zone'] == pays]
    
    # plot the data for the current country
    ax.plot(pays_data['Année'], pays_data['Proportion (%)'], label=pays)

# Add labels and legend
ax.set_xlabel('Année')
ax.set_ylabel('Proportion en sous-nutrition (%)')
ax.set_title('Evolution de la sous-nutrition dans les pays les plus touchés')
legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

# Show the plot
plt.show()


# 
# Ce graphique en courbe pose plusieurs courbes questions qui peuvent êtes discutées par l'organisation et dépendent des priorité que l'on souhaite donner, en effet on peut constater que :
# 
# La situation en Haïti reste la plus critique en termes de proportion de la population touchée par la sous-nutrition
# La situation en République populaire démocratique de Corée et à Madagascar se sont aggravées ces dernières années

# In[168]:


Visualisation de l'évolution du nombre de personne en sous-nutrition


# In[51]:


# Create a figure and axis object
fig, ax = plt.subplots(figsize=(8,6))

# Iterate over the top 5 countries and plot their data
for pays in top_5:
    # filter data for the current country
    pays_data = evolutionTop5[evolutionTop5['Zone'] == pays]
    
    # plot the data for the current country
    ax.plot(pays_data['Année'], pays_data['Nombre personnes en sous-nutrition'], label=pays)

# Add labels and legend
ax.set_xlabel('Année')
ax.set_ylabel('Nombre personnes sous nourries (dizaine en million)')
ax.set_title('Evolution de la sous-nutrition dans les pays les plus touchés')
legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

# Show the plot
plt.show()


# Ce second graphique met en évidence qu'en termes de nombre de personnes touchées par la sous-nutrition les deux pays à la fois les plus touchés mais aussi où la situation s'est le plus aggravées sont :
# 
# La République populaire démocratique de Corée
# Madagascar

# In[ ]:




