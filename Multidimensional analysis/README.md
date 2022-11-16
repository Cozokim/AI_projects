# Multivariate Data Analysis
:dart: The goal of this project is to perform multivariate data analysis (incl. exploration, cleaning and visualisation), to help the agents of a Public Health Agency to identify relevant insights from *cleaned* Open Food Facts dataset.

# Dataset
<img src="./pictures\openfoodfacts_logo.png">

[Open Food Facts](https://world.openfoodfacts.org/) dataset, *made by everyone, for everyone*.


# Criteria for qualifying healthy food 
:green_salad: Is a green salad always healthy ?

To demonstrate nutritional quality of a specific food, the Public Health Agency has created an indicator: the Nutri Score.
We are gonna see that this nutriscore is really imperfect : it only look after nutritional values like kcal but never about if a product is sane or not. 

<img src="./pictures\nutri_score_logo.png">

We're gonna add some personal scoring and personal criteria to make our analysis more accurate

<img src="./pictures\personal_scoring_criteria.png">

We will try to explore the dataset with those criterias in mind.

And then do some analysis and dimensional reduction to have some graphical representation

<img src="./pictures\acp_image.png">
# Dependencies
:computer: Pandas, Numpy, Matplotlib, Seaborn, Plotly, Scipy, Scikit-learn, mlxtend, ipywidgets, wordcloud, Collections, Voilà!

# References
:pushpin: 
- [Nutri-score - Santé Publique France](https://www.santepubliquefrance.fr/determinants-de-sante/nutrition-et-activite-physique/articles/nutri-score)
- [Réglement d'usage du nutri-score - Santé Publique France (PDF)](https://www.santepubliquefrance.fr/media/files/02-determinants-de-sante/nutrition-et-activite-physique/nutri-score/reglement-usage)