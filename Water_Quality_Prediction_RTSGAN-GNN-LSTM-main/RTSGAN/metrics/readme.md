### Explication du code `discriminative_metrics.py` 

Ce script Python calcule un **score discriminatif** pour comparer des données temporelles originales et synthétiques. Ce score est obtenu en entraînant un réseau récurrent (RNN) post-hoc pour classifier si un échantillon est réel (issu des données originales) ou synthétique. Le score discriminatif mesure la divergence entre les distributions des données réelles et synthétiques.

---

#### **Sections principales du code**

1. **Initialisation**
   - `tf.reset_default_graph()` réinitialise le graphe par défaut de TensorFlow pour éviter les conflits lors d'exécutions multiples.
   - Les dimensions des données (`no`, `seq_len`, `dim`) sont extraites à partir des données d'entrée (`ori_data`).
   - Les longueurs de séquences et les longueurs maximales sont calculées à l'aide de la fonction utilitaire `extract_time` pour gérer les séquences de longueur variable.

2. **Discriminateur RNN post-hoc**
   - Un RNN (basé sur une cellule GRU) est conçu pour classifier les séries temporelles en données réelles ou synthétiques.
   - **Entrées du discriminateur** :
     - `X` et `X_hat` : Données réelles et synthétiques.
     - `T` et `T_hat` : Longueurs des séquences pour chaque ensemble de données.
   - **Sorties du discriminateur** :
     - `y_hat_logit` : Logits (sorties brutes) des prédictions.
     - `y_hat` : Probabilités (après transformation sigmoid) indiquant si l'entrée est réelle ou synthétique.

3. **Fonction de perte et optimisation**
   - Deux composantes de la perte :
     - `d_loss_real` : Pénalise le modèle si des échantillons réels sont mal classés comme synthétiques.
     - `d_loss_fake` : Pénalise le modèle si des échantillons synthétiques sont mal classés comme réels.
   - La perte totale (`d_loss`) est la somme des deux.
   - L'optimiseur Adam est utilisé pour minimiser la perte du discriminateur.

4. **Processus d'entraînement**
   - Les données sont divisées en ensembles d'entraînement et de test à l'aide de `train_test_divide`.
   - Pendant l'entraînement, des mini-lots (batches) sont générés avec `batch_generator`, et le discriminateur est mis à jour au cours de 2000 itérations.

5. **Test et calcul de la précision**
   - Après l'entraînement, le discriminateur est évalué sur les données de test.
   - Les probabilités prédites (`y_pred_real_curr` et `y_pred_fake_curr`) sont calculées pour les ensembles de test réels et synthétiques.
   - Les prédictions finales sont concaténées, et la précision est calculée par rapport aux étiquettes de vérité.
   - Le score discriminatif est défini comme \( \text{abs}(0.5 - \text{accuracy}) \), ce qui indique à quel point le discriminateur peut distinguer entre les données réelles et synthétiques. Un score bas indique une meilleure similitude entre les distributions réelles et synthétiques.

---

#### **Fonctions et utilitaires**
- **`discriminative_score_metrics`**
  - La fonction principale qui calcule le score discriminatif.
  - Elle initialise le réseau discriminateur, l'entraîne et évalue la précision de classification sur les données de test.
  
- **Fonctions utilitaires**
  - `train_test_divide` : Divise les données en ensembles d'entraînement et de test pour les données réelles et synthétiques.
  - `extract_time` : Détermine les longueurs des séquences pour des séries temporelles de longueur variable.
  - `batch_generator` : Crée des lots de données pour l'entraînement.

---

### **Flux de travail global**
1. **Entrées** :
   - Données originales (`ori_data`) et données synthétiques (`generated_data`).
   
2. **Sortie** :
   - **Score discriminatif** : Une mesure de la capacité à distinguer les distributions des données réelles et synthétiques.

3. **Étapes** :
   - Extraction des propriétés des séries temporelles.
   - Entraînement d’un discriminateur RNN post-hoc pour classifier les échantillons réels et synthétiques.
   - Évaluation du discriminateur sur un ensemble de test.
   - Calcul de la précision de classification et dérivation du score discriminatif.

---

### **Importance**
- Ce script est souvent utilisé dans le cadre de l'évaluation de générateurs de données synthétiques comme les GANs (Generative Adversarial Networks). Un score discriminatif bas signifie que les données synthétiques ressemblent étroitement aux données originales, rendant leur distinction plus difficile pour un classifieur.




=======================================================================================


### Explication du code `predictive_metrics.py`

Ce code évalue les données synthétiques générées en termes de **capacité prédictive** en entraînant un RNN (Réseau de Neurones Récurrents) post-hoc. Le réseau prédit une étape à l'avance (la dernière caractéristique) pour chaque série temporelle, et sa performance est mesurée par l'erreur absolue moyenne (MAE) sur les données originales.

---

#### **Objectif**

L'objectif est de comparer les données synthétiques avec les données originales en testant si un modèle prédictif entraîné sur les données synthétiques peut bien généraliser sur les données originales. Un faible score MAE indique que les dynamiques des données synthétiques sont proches de celles des données originales.

---

### **Détails du code**

#### 1. **Initialisation**
- **Réinitialisation du graphe TensorFlow** :  
  `tf.reset_default_graph()` réinitialise le graphe pour éviter les conflits avec d'autres modèles ou exécutions précédentes.
  
- **Dimensions des données** :  
  Les dimensions des données (`no`, `seq_len`, `dim`) sont extraites à partir des données originales (`ori_data`).

- **Longueur des séquences** :  
  `extract_time` calcule la longueur de chaque séquence temporelle et détermine la séquence maximale pour gérer des longueurs variables.

---

#### 2. **Réseau prédictif (RNN)**
Un réseau RNN avec cellule GRU est utilisé pour prédire une étape à l'avance.

- **Entrées** :
  - `X`: Les caractéristiques de la série temporelle sans la dernière colonne (entrée pour la prédiction).
  - `T`: Les longueurs des séquences.
  - `Y`: La dernière caractéristique (cible à prédire).

- **Sorties** :
  - `y_hat`: Prédictions du modèle.
  - `p_vars`: Variables du réseau (pour l'optimisation).

- **Architecture** :
  - La cellule GRU extrait les dynamiques temporelles.
  - La sortie est passée par une couche dense avec une seule sortie (prédiction de la caractéristique cible).

---

#### 3. **Fonction de perte et optimisation**
- **Perte** :  
  `p_loss = tf.losses.absolute_difference(Y, y_pred)` calcule l'erreur absolue entre les prédictions et les valeurs réelles.

- **Optimisation** :  
  L'optimiseur Adam ajuste les paramètres du réseau pour minimiser la perte.

---

#### 4. **Entraînement**
Le modèle est entraîné sur les données **synthétiques**.

- Les mini-lots de données sont créés en :
  - Sélectionnant aléatoirement des indices d'échantillons.
  - Préparant `X`, `T`, et `Y` pour chaque lot.
  
- La perte est calculée et le modèle est mis à jour pour 5000 itérations.

---

#### 5. **Évaluation**
Après l'entraînement, le modèle est testé sur les données **originales**.

- Les prédictions (`pred_Y_curr`) sont calculées pour les données originales.
- La MAE est calculée pour chaque série temporelle individuelle, puis moyennée sur tous les échantillons.

---

#### **Fonction principale : `predictive_score_metrics`**
- **Entrées** :
  - `ori_data`: Données originales.
  - `generated_data`: Données synthétiques.

- **Sortie** :
  - `predictive_score`: MAE moyen sur les données originales.

---

### **Étapes principales**

1. **Préparation des données** :
   - Diviser les séries temporelles en caractéristiques d'entrée (`X`) et cibles (`Y`).

2. **Entraînement** :
   - Entraîner le réseau RNN sur les données synthétiques.

3. **Test** :
   - Tester le modèle entraîné sur les données originales.

4. **Calcul de la performance** :
   - Calculer le score prédictif (MAE moyen).

---

### **Signification du résultat**

- Un score prédictif bas (MAE faible) signifie que les données synthétiques reproduisent bien les relations temporelles présentes dans les données originales. Cela indique que les données synthétiques peuvent être utilisées efficacement pour des tâches prédictives similaires.

================================================================================================================
### Explication du code `visualization_metrics.py`

Ce script permet de visualiser les données **originales** et **synthétiques** en projetant leurs caractéristiques dans un espace 2D à l'aide des techniques de **PCA (Principal Component Analysis)** ou **t-SNE (t-Distributed Stochastic Neighbor Embedding)**. Cela permet de comparer les distributions des deux ensembles de données pour évaluer leur similarité.

---

### **Objectif**

- Fournir une **visualisation intuitive** des relations entre les données synthétiques et originales.
- Utiliser **PCA** ou **t-SNE** pour réduire les dimensions des données tout en préservant les structures importantes.

---

### **Structure du code**

#### 1. **Échantillonnage**

- Le nombre d'échantillons analysés est limité à 1000 pour accélérer le calcul :
  ```python
  anal_sample_no = min([1000, len(generated_data)])
  idx = np.random.permutation(len(generated_data))[:anal_sample_no]
  ```
- Les données originales et synthétiques sont réduites à cette taille échantillonée.

---

#### 2. **Prétraitement des données**

- Chaque série temporelle est transformée en une **moyenne par étape temporelle** (pour chaque dimension) :
  ```python
  prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1, seq_len])
  ```
- Ceci convertit chaque série 3D de forme `(seq_len, dim)` en une représentation 2D de forme `(seq_len,)`.

- Les données originales et synthétiques prétraitées sont ensuite concaténées pour faciliter l'analyse.

---

#### 3. **PCA (Principal Component Analysis)**

- **But** :
  - PCA réduit les dimensions des données tout en préservant le maximum de variance.

- **Étapes** :
  - La PCA est appliquée séparément aux données originales et synthétiques.
    ```python
    pca = PCA(n_components=2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)
    ```
  - Les résultats transformés sont ensuite tracés sur un graphique.

- **Visualisation** :
  - Les points rouges représentent les données originales.
  - Les points bleus représentent les données synthétiques.

---

#### 4. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**

- **But** :
  - t-SNE est une technique non linéaire de réduction de dimensionnalité qui préserve les structures locales.

- **Étapes** :
  - Les données originales et synthétiques sont combinées dans une seule matrice :
    ```python
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
    ```
  - t-SNE est appliqué à cette matrice pour obtenir des représentations en 2D :
    ```python
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(prep_data_final)
    ```
  - Les résultats sont divisés et tracés en fonction des deux ensembles de données.

- **Visualisation** :
  - Rouge pour les données originales.
  - Bleu pour les données synthétiques.

---

#### 5. **Visualisation des résultats**

- **PCA Plot** :
  - Affiche les projections des données originales et synthétiques sur deux composantes principales.
  - Axes : `x-pca` et `y-pca`.

- **t-SNE Plot** :
  - Montre les relations de voisinage des points en 2D après une réduction de dimension non linéaire.
  - Axes : `x-tsne` et `y-tsne`.

---

### **Fonction principale : `visualization`**

- **Entrées** :
  - `ori_data`: Données originales.
  - `generated_data`: Données synthétiques.
  - `analysis`: Choix de la méthode (`'pca'` ou `'tsne'`).

- **Sortie** :
  - Un graphique montrant la distribution des données originales et synthétiques.

---

### **Signification des résultats**

1. **Distributions similaires** :
   - Si les points rouges et bleus se mélangent bien, cela indique que les données synthétiques reproduisent correctement les caractéristiques des données originales.

2. **Distributions distinctes** :
   - Si les points rouges et bleus forment des clusters séparés, cela suggère que les données synthétiques ne capturent pas bien les relations des données originales.

---

### **Application**

Ce code est utile pour :
- Évaluer des générateurs de données synthétiques (comme les GANs).
- Identifier des différences entre deux ensembles de données en termes de distribution.