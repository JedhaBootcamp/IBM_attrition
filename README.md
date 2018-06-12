# Prédire le turnover en entreprise
### Construction d'un modèle de régression logistique pour déterminer si un employé va rester ou partir de l'entreprise

## Introduction

Le turnover en entreprise est le phénomène de rotation des employés dans l'entreprise : certains s'en vont d'autres restent. Les départements RH s'intéressent énormément à ce taux car il permet de savoir rapidement si les employés sont heureux ou non dans l'entreprise. Le Machine Learning peut aider à analyser et prédire ce taux et donc prendre les bonnes décisions. C'est pour cela que nous avons décidé de construire un modèle qui va permettre de prédire le turnover.

Pour cela nous avons récupérer un base de données d'[IBM](https://www.kaggle.com/shuchirb/ibm-attrition-analysis) sur laquelle nous allons essayer de prédire si un employé d'IBM va rester dans la boite ou plutôt la quitter.

Ce type de modélisation est d'autant plus utile qu'il ne se limite pas uniquement au turnover en entreprise. Il peut aussi permettre de prédire si un client va acheter ou non un produit, prédire aussi l'attrition de vos clients dans le cadre d'un business model basé sur de l'abonnement, prédire si un email et un spam ou non ou encore prédire la présence ou l'absence d'une maladie.

## Choix du modèle

Dans le cadre de notre problématique, nous cherchons à classer un employé dans une catégorie (Catégorie 1: Il est partie d'IBM, catégorie 2: Il est resté chez IBM). Nous sommes donc face à un problème de classification.

La regression logistique est un modèle tout à fait adapté pour ce genre de problèmes et est simple à mettre en place. L'idée générale de ce modèle est qu'il va calculer une probabilité pour chaque employé de partir de l'entreprise. Si cette probabilité est supérieure à 50%, alors le modèle considèrera que la personne va partir.

Il y a d'autres modèles de classification que nous pourrions utiliser et comparer pour voir lequel est le plus précis. Parmi les plus prometteurs il y aurait :

  * Random Forest Classification
  * Naive Bayes
  * K-Nearest Neighbors (KNN)
  * Kernel SVM

Restons pour le moment sur la regression logistique qui fera déjà un très bon boulot.

## Exploration des données

Commencons par importer le dataset pour voir ce qui s'y trouve

```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Nous avons ici importé les librairies ```Numpy``` et ```Pandas``` qui vont nous permettre de faire nos calculs et manipulation de données ainsi que ```matplotlib```et ```seaborn```qui vont nous permettre de faire notre data Visualisation.

Importons donc le dataset :

```Python
dataset = pd.read_csv("dataset.csv")
dataset.info()
```

On peut voir ici que nous avons un base de données comportant 35 colonnes. Nous avons donc 34 variables indépendantes qui vont nous mettre de prédire la colonne Attrition de notre dataset.

Voici la liste des 34 Variables :

  Age                         
  Attrition                   
  BusinessTravel              
  DailyRate                   
  Department                  
  DistanceFromHome            
  Education                   
  EducationField              
  EmployeeCount               
  EmployeeNumber              
  EnvironmentSatisfaction     
  Gender                      
  HourlyRate                  
  JobInvolvement              
  JobLevel                    
  JobRole                     
  JobSatisfaction             
  MaritalStatus               
  MonthlyIncome               
  MonthlyRate                 
  NumCompaniesWorked          
  Over18                      
  OverTime                    
  PercentSalaryHike           
  PerformanceRating           
  RelationshipSatisfaction    
  StandardHours               
  StockOptionLevel            
  TotalWorkingYears           
  TrainingTimesLastYear       
  WorkLifeBalance             
  YearsAtCompany              
  YearsInCurrentRole          
  YearsSinceLastPromotion     
  YearsWithCurrManager

Avant de les regarder, commencons par voir l'attrition générale chez IBM. Utilisons Matplotlib et Seaborn pour cela :

```Python
sns.countplot(x=dataset.Attrition, data= dataset, palette='hls')
plt.show()
```

On peut voir un fort déséquilibre entre les personnes qui restent et celles qui partent. Ce qui est plutôt bon signe pour IBM. Mais, de notre côté, le fait de ne pas avoir assez de personnes qui soient parties peut avoir des conséquences négatives sur notre modèle puisqu'il n'aura pas assez de données sur lesquelles s'entrainer.

Cependant, fort heureusement pour nous, 200 'Yes' sera assez pour que notre modèle puisse s'entrainer.

Regardons donc maintenant quelques variables pour déterminer ce qui pourraient avoir un fort impact sur le turnover des employés d'IBM.

Commencons par le département dans lequel ils travaillent. En effet, il est assez commun de dire que les Sales ont un turnover plus fort que les autres employés de manière générale. Voyons si cela est vrai chez IBM.

```Python
pd.crosstab(dataset.Department,dataset.Attrition).plot(kind='bar')
plt.title('Attrition par Departement')
plt.xlabel('Department')
plt.ylabel('Fréquence')
plt.show()
```

On peut voir ici que le nombre de personnes total dans chaque département est franchement inégal. Ce qui ne nous permet pas vraiment de donner de conclusion. Essayons de regarder l'attrition de manière relative.

```python
table1 = pd.crosstab(dataset.Department, dataset.Attrition)
table1.div(table1.sum(1).astype(float), axis=0).plot(kind='bar', stacked = True)
plt.title('Attrition par Departement')
plt.xlabel('Department')
plt.ylabel('Fréquence')
plt.show()
```

Ce graphique est beaucoup mieux. Et d'ailleurs, il nous permet de voir que l'attrition en Sales n'est pas plus élevé qu'en RH. Le département R&D est celui qui a le moins de personnes qui partent en relatif. Cependant au global, on peut dire que le département n'a pas tant d'influence que cela puisque les trois sont relativement équilibrés. Tentons donc de regarder d'autres variables


La variable `Job Satisfaction` parait être un bon candidat. Voyons plutôt :

```Python
table2 = pd.crosstab(dataset.JobSatisfaction, dataset.Attrition)
table2.div(table2.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.title("Attrition en fonction de la satisfation au travail")
plt.xlabel("Satisfaction au travail")
plt.ylabel("Proportion d'employés")
plt.show()
```

Sans grande surprise, les personnes qui sont le moins satisfaites au travail sont celles qui partent le plus.

Essayons de regarder des variables un peu moins évidentes comme le nombre d'années après la dernière promotion et l'équilibre vie perso/travail

```python
table3 = pd.crosstab(dataset.YearsSinceLastPromotion, dataset.Attrition)
table3.div(table3.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.title("Attrition en fonction de la dernière promotion")
plt.xlabel("nombre d'années après la dernière promotion")
plt.ylabel("Proportion d'employés")
plt.show()
```

```python
table4 = pd.crosstab(dataset.WorkLifeBalance, dataset.Attrition)
table4.div(table4.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.title("Attrition en fonction de l'equilibre Vie Perso / Travail")
plt.xlabel("Equilibre Vie Perso / Travail")
plt.ylabel("Proportion d'employés")
plt.show()
```

D'après ces deux graphiques, on peut voir que les personnes qui ont le moins quitté l'entreprise sont celles qui ont un équilibre vie perso / travail relativement élevé (3 / 4) et que leur dernière promotion remonte à 4 ou 5 ans. Pour ce qui est de l'équilibre vie perso / travail, il faudrait voir comment celui-ci est calculé mais cela paraît vraisemblable.

Pour ce qui est de la promotion, on peut effectivement penser que les personnes qui n'ont pas eu de promotion depuis 4 ou 5 ans vont en avoir une nouvelle bientôt et c'est pour cela qu'ils restent.


Nous pourrions continuer à regarder les variables une à une de cette façon mais cela peut rendre l'exercice un peu fastidieux. Implémentons notre modèle et nous vous montrerons un moyen de sélectionner les meilleures variables de manière automatique.

## Préparation des données
### Séparation des variables indépendantes de la variable dépendante

Dans ce dataset, nous n'avons pas de valeurs manquantes à gérer, cela nous facilite donc grandement le travail. Passons donc directement à l'étape de séparer nos variables indépendantes de notre variable dépendante. Voici un moyen très simple d'y parvenir :

```Python
X, y = dataset.loc[:, dataset.columns !="Attrition"], dataset.loc[:, "Attrition"]
```

Ici, nous avons créé deux variables X et y dont la première va contenir toutes les colonnes sauf celle qui se nomme ``Attrition`` et la variable y, va contenir uniquement celle contenant la colonne ``Attrition``

### Gestion des variables catégoriques

Nous avons pas mal de variables catégoriques dans ce dataset comme EducationField ou encore Department. Occupons nous en avec ``get_dummies``

```Python
X = pd.get_dummies(X, drop_first= True)
X.head()
```

Faisons de même avec y

```python
y = pd.get_dummies(y, drop_first= True)
y.head()
```

NB : Très important de ne pas oublier de mettre le paramètre `drop_first = True` sinon vous tomberez dans le piège des variables factices et biaiserez votre modèle.

Nous nous retrouvons donc avec 47 Colonnes pour X et toujours 1 colonne pour y.

## Implémentation du modèle

Maintenant que nos données sont prêtes, implémentons notre modèle. Commencons d'abord par choisir les meilleures variables à mettre dans notre modèle

### Sélection des variables (Feature Selection)

Sklearn nous aide beaucoup car la librairie inclue le module `feature_selection` avec `SelectKBest` grâce auquel nous allons pouvoir sélectionner les variables qui ont le plus d'influence dans notre modèle.

```Python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X_new = SelectKBest(chi2, k=40).fit_transform(X,y)
```

Ici nous avons donc importé ```SelectKBest``` et ```chi2``` qui nous servira d'indicateur pour sélectionner nos variables.

Nous avons choisi k = 40 c'est à dire que nous garderons 40 variables sur les 47. Vous pouvez tenter de choisir encore moins de variables et voir ce qui est le plus optimal dans votre modèle. D'après nos essais, à partir de 35 variables, la qualité prédictive du modèle baisse.

Il serait intéressant aussi de voir quelles variables ont été gardées. Pour cela, nous pouvons utiliser `get_support` qui nous donnera les indices des colonnes que nous avons gardé

```Python
SelectKBest(chi2, k=40).fit(X,y).get_support(indices=True)
```

Nous obtenons

```
array([ 0,  1,  2,  3,  5,  6,  8,  9, 10, 11, 12, 13, 16, 18, 19, 20, 21,
       22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39,
       40, 42, 43, 44, 45, 46])
```

### Séparation du dataset en un Training set et un Test set

Avant de séparer notre modèle. Il faudra simplement que nous changions y en une array à 1 dimension. Ceci n'impactera pas vraiment le modèle mais cela nous évitera de recevoir des warnings

```Python
y = np.ravel(y)
```

Une fois ceci fait, séparons notre dataset

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.25, random_state = 0)
```

### Application de la normalisation

Lorsqu'on applique des modèles de classification, il est bon de normaliser les données. C'est à dire de les mettre à la même échelle.

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### Application du modèle de régression logistique

Nous sommes presque au bout, nous allons pouvoir maintenant appliquer un modèle de régression logistique sur nos données d'entraînement.

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
```
Nous avons appliqué les paramètres par défaut du modèle. Nous verrons dans la partie Optimisation du modèle que nous pouvons le booster un peu

### Prédiction des résultats

Maintenant passons à la partie la plus fun : la prédiction des résultats

```python
y_pred = classifier.predict(X_test)
```

Et créeons une matrice de confusion pour voir la performance de notre modèle

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

Notre modèle a obtenu 325 prédictions justes. Ce résultat peut être différent sur votre code car nous n'avons pas sélectionné un `random_state` dans la régression logistique.

### Evaluation du modèle

Un bon moyen d'évaluer notre modèle est de le faire tourner plusieurs fois sur des données différentes et voir sa performance moyenne.

Avec python, nous pouvons utiliser le module `cross_val_score` qui va nous permettre de faire tourner notre modèle automatiquement plusieurs fois :

```Python
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
```

Ici, nous avons fait tourner notre modèle 10 fois et on a obtenu une précision moyenne de 86%. Ce qui est pas trop mal, notre modèle voit juste dans 86% des cas.

## Optimisation du modèle

Il y a un moyen assez simple de booster notre modèle qui est par le module `GridSearchCV`. Ce module va nous permettre de trouver les paramètres optimaux à mettre dans notre régression logistique et améliorer la performance de notre modèle

```python
from sklearn.model_selection import GridSearchCV
grid = {"C": np.arange(0.3,0.4,0.01),
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "max_iter": [200]
       }

classifier_opt = GridSearchCV(classifier, grid, scoring = 'accuracy', cv=10)
classifier_opt.fit(X_train,y_train)
print("Tuned_parameter k : {}".format(classifier_opt.best_params_))
print("Best Score: {}".format(classifier_opt.best_score_))

```

Nous avons entré les paramètres principaux qui auront un impact sur notre modèle et GridSearch va trouver ceux qui ont la meilleure performance.

NB : Les intervalles que vous voyez dans chaque paramètre résulte de plusieurs essais sur des intervalles plus grands avant d'avoir les paramètres optimum que nous avons mis dans le code.

Nous avons donc trouvé que C optimum était de 0.35 et le solver optimum était ```newton-cg```. Mettons les dans notre modèle.

```Python
classifier = LogisticRegression(C=0.35000000000000003, solver="newton-cg", max_iter=200)
classifier.fit(X_train, y_train)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

```

La précision de notre modèle est passée à 88,4%. Belle progression!

## Visualisation

Nous finirons par la visualisation du modèle de régression logistique. La façon dont la performance de notre modèle est représentée est généralement en la comparant avec un modèle aléatoire.

Nous utiliserons la courbe ROC pour voir nos résultats :

```python
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
```

Plus notre courbe est éloignée de la droite, meilleur est notre modèle. Dans notre cas, nous avons quelque chose de plutôt puissant. Donc nous pouvons être content de nous puisque nous avons construit un modèle plutôt robuste et précis. 


Merci d'avoir lu jusqu'à la fin ce projet. Nous en avons d'autres sur notre [blog](jedha.co/blog)

Si vous êtes intéressé à l'idée d'apprendre les Data Sciences, regardez notre site : [Jedha Bootcamp](jedha.co)
