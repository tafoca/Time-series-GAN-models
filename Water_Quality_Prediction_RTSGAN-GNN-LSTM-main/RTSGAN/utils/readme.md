### Explication du code : Gestion de journalisation et création de chemin

Ce code définit des fonctions pour :
1. **Créer des répertoires** : S'assurer que le chemin spécifié existe, sinon le créer.
2. **Configurer un logger** : Initialiser un système de journalisation qui écrit les messages dans un fichier et les affiche sur la console.

---

### **Fonctions**

#### 1. `make_sure_path_exists(path)`

- **Objectif** :  
  Garantir que le chemin spécifié (dossier) existe, sinon le créer.

- **Détails** :
  - `os.makedirs(path)` : Crée le répertoire, y compris les répertoires parents s'ils n'existent pas.
  - Si une erreur `OSError` est levée, elle est ignorée si le chemin existe déjà (`errno.EEXIST`).  
    Pour d'autres erreurs, l'exception est propagée (`raise`).

- **Cas d'utilisation** :  
  Créer un chemin où les journaux ou d'autres fichiers seront enregistrés.

---

#### 2. `init_logger(root_dir)`

- **Objectif** :  
  Configurer un système de **journalisation** qui :
  - Écrit les messages dans un fichier de log (`info.log`) dans le dossier spécifié.
  - Affiche simultanément les messages sur la console.

- **Étapes principales** :
  1. **Création du répertoire** :
     - `make_sure_path_exists(root_dir)` garantit que le répertoire `root_dir` existe.

  2. **Configuration du logger** :
     - Crée un **logger** principal.
     - Définit un format de message avec `logging.Formatter` (dans ce cas, c'est juste `"%(message)s"`).

  3. **Gestionnaires de journalisation** :
     - **FileHandler** : Écrit les messages dans un fichier `info.log` dans le dossier spécifié.  
       - Mode `'w'` : Écrase le contenu précédent du fichier.
     - **StreamHandler** : Affiche les messages sur la console (stdout).

  4. **Ajout au logger principal** :
     - Les gestionnaires sont attachés au logger.
     - Niveau de journalisation est configuré sur `logging.INFO` (inclut les messages d'information et plus graves).

- **Sortie** :
  - Retourne l'objet logger pour permettre l'envoi de messages de log.

---

### **Utilisation du logger**

```python
# Initialiser le logger dans un répertoire
log_dir = "./logs"
logger = init_logger(log_dir)

# Envoyer des messages
logger.info("This is an informational message.")
logger.warning("This is a warning.")
logger.error("This is an error message.")
```

- Les messages apparaîtront :
  1. Dans le fichier `./logs/info.log`.
  2. Simultanément sur la console.

---

### **Résumé des fonctions**

| Fonction                   | Utilité                                                |
|----------------------------|-------------------------------------------------------|
| `make_sure_path_exists()`  | Créer un répertoire si nécessaire.                     |
| `init_logger()`            | Configurer un logger pour enregistrer dans un fichier et afficher sur la console. |

---

### **Applications**

- **Systèmes de production** : Suivre les erreurs, avertissements, et informations importantes.
- **Débogage** : Afficher les messages directement sur la console tout en conservant un historique.
- **Automatisation** : S'assurer que les répertoires nécessaires pour l'application existent avant d'écrire des fichiers (comme des journaux, modèles, ou résultats).