# CreaCube

L’ordinateur sur lequel a été développé le projet était composé d’un processeur Intel Xeon CPU E5-1650 v3, de 16 Go de mémoire vive et d’une carte graphique Nvidia Geforce GTX 1070.

## Installation sous Windows 10

La procédure suivante décrit l'installation de toutes les dépendances requises dans un environnement Anaconda sur un ordinateur fonctionnant sur Windows 10.

### 1. Installation d’Anaconda

Veuillez suivre ce [guide](https://docs.anaconda.com/anaconda/install/windows/) afin d’installer Anaconda sur Windows.

### 2. Installer Cuda

Si vous n'avez pas de carte graphique de marque Nvidia ou que vous ne souhaitez pas utiliser cette dernière pour l'inférence, passez cette étape.

Afin de pouvoir profiter de la carte graphique pendant l’inférence, mais surtout pendant l’apprentissage, il faut installer l’API CUDA de Nvidia. Pour cela veuillez télécharger l'éxécutable [ici](https://developer.nvidia.com/cuda-11-6-2-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) et suivre ce [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

### 3. Télécharger le projet depuis le git

Clonez ce git à l'aide de la commande suivante :
```
git clone https://github.com/TheoCarme/CreaCube.git
```
Puis téléchargez [ici](https://drive.google.com/drive/folders/183w4dgVz06fKd1bsC_wdGqFAW9lKTLOe?usp=sharing) les fichiers nécessaire selon vos besoins :
  1. Le dossier "ONNX_Models" contient le modèle pour l'inférence.
  2. Le dossier "YOLOX_outputs" contient les checkpoints correspondant à l'apprentissage des différentes tailles de modèles YOLOX.
  3. Le dossier "COCO" contient le jeu de données au format COCO.

### 4. Créer l’environement Anaconda adéquat

Pour créer l'environement Anaconda adéquat ouvrez l'Anaconda Powershell Prompt et entrez les lignes suivantes :
```
conda create -n "myenv" python=3.9.12
conda activate myenv
```
```
conda install -c conda-forge pycocotools
```
Si vous avez une carte graphique Nvidia et que vous avez installé VUDA, vous pouvez maintenant installer CUDA pour python. Pour cela entrez la commande qui correspond à la version de CUDA que vous avez installé précédement.
```
conda install cuda -c nvidia/label/cuda-11.6.2
```
Installez maintenant torch. La première commande est pour l'installation avec support GPU et la deuxième pour l'installation CPU.
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
```
pip install torch torchvision torchaudio
```
Ensuite déplacer vous dans le répertoire où vous avez téléchargé le git puis dans le dossier nommé "YOLOX" et entrez les lignes suivantes :
```
cd path\to\the\project\YOLOX_0.3.0
```
```
pip install -r requirements.txt
pip install -v -e .
```
Il se peut que lors de l'exécution de la dernière ligne il soit retourné une erreur comme quoi la version de de quelque chose en rapport avec C++ n'est pas installé (je préciserai le quelque chose dès que j'aurai pu reproduire l'erreur).
Dans ce cas télécharger les Outils de génération Microsoft C++ [ici](https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/) et lors de l'intalation cochez uniquement la case correspondant au C++. Ceci fait réexécuter la dernière ligne.

## Installation sous Ubuntu

Voir [ici](https://github.com/TheoCarme/CreaCube/blob/TheoCarme-YOLOX-0.3.0/README_ubuntu.md)

## Utilisation

### 1. Aide sur la commande

Ouvrez l'Anaconda Powershell Prompt et entrez les lignes suivantes
```
conda activate myenv
cd path\to\the\project\
python combined_detection.py --help
```
La dernière ligne affiche les différends arguments que peut accepter le programme. Ces arguments sont les suivants :  
| Argument | Description |
| :------: | :---------- |
| -m / --model | Permet de préciser le chemin du modèle ONNX à utiliser. |
| -v / --video_path | Permet de donner le chemin de la vidéo à analyser. |
| -o / --output_dir | Permet de donner le chemin du dossier là où sera crée si demandé le dossier contenant la vidéo annotée et le fichier csv. |
| -s / --score_thr | Permet de fixer un seuil au score de confiance des cubes, en dessous duquel les cubes ne seront pas dessinés sur la vidéo. Ce seuil est par défaut fixé à 0.3. |
| -S / --start | Permet d'indiquer un nombre de seconde à ignorer au début de la vidéo. |
| -E / --end | Permet d'indiquer un nombre de seconde à ignorer à la fin de la vidéo. |
| -n / --num_hands | Permet d'indiquer le nombre maximum de main à détecter sur la vidéo. Ce maximum est par défaut fixé à 4. |
| -c / --csv | Permet de demander au programme d'écrire le fichier csv contenant les traces des objets suivis. |
| -w / --write_video | Permet de demander au programme d'écrire la vidéo avec les objets suivis dessinés dessus. |
| -d / --display | Permet de demander au programme d'afficher chaque trame traitée avec les objets suivis dessinés dessus. |

La ligne pour éxecuter le programme pourra donc ressembler à cela :
```
python .\combined_detection.py -m ..\Project_Data\ONNX_Models\yolox_s.onnx -v path/to/the/video -o ..\Results -n 2 -c -w -d
```

Entrer à chaque fois les chemin vers le modèle, la vidéo et dossier des résultats peut vite devenir rébarbatif. Pour remédier à cela vous pouvez aller modifier les valeurs par défaut dans la première fonction nommée "make_parser" dans le fichier "combined_detection.py"

### 2. Exemple

J'ai ajouté 3 sous-dossiers (ignorés par git) videos, onyx_models et yolox_outputs, contenant :
- videos
  - une vidéo de test p359.mp4 (téléchargé [ici](https://creamaker.univ-cotedazur.fr/))
- onyx_models
  - yolox_s.onnx (téléchargé [ici](https://drive.google.com/drive/folders/1Wp4CAXRcb4OCIt-8F-fsc5UlX-_H_CzD))
- yolox_outputs
  - dossier vide avant de faire tourner la commande

Nous pouvons alors générer la sortie du modèle pour une des vidéos :
```
python combined_detection.py -m onyx_models/yolox_s.onnx -v videos/p359.mp4 -o yolox_outputs -n 2 -c -w -d
```

## Création/extension du jeu de données

### 1. Extraction de trames depuis une vidéo

Pour ce faire utilisez le programme Manage_Dataset/extract_frames.py dont les options sont les suivantes :
| Argument | Description |
| :------: | :---------- |
| -i / --input_video | Permet d'indiquer le chemin vers le fichier de la vidéo dont seront extraites les images. |
| -o / --output_dir | Permet d'indiquer le chemin vers le dossier où seront stockées les images extraites. |
| -n / --nb_frames | Permet de fixer le nombre de trames à extraire de la vidéo. |

Les images extraites par ce programme seront extraites à intervalle régulier, intervalle défini selon le nombre de trames demandées.

### 2. Annotations des images

Vous aurez besoin pour cela d'installer labelme comme expliqué [ici](https://github.com/wkentaro/labelme#installation), vous n'avez pas besoin de créer un nouvel environment et pouvez installer tout ça dans l'environment déjà crée.

Ensuite lancer labelme avec la commande suivante :
```
labelme --nodata --autosave
```
Une fois la fenêtre labelme ouverte cliquez sur "Fichier" puis "Change Output Dir" et choisissez le dossier dans lequel vous voulez enregistrer  
Ensuite cliquez sur "Fichier" puis "Open Dir" et choisissez le dossier contenant les images à annoter.  
Maintenant utilisez l'outil "Create Polygons" (disponible sur la barre d'outils à gauche) pour dessiner des polygônes englobants chaqun des cubes.  
A la finalisation de chaque polygône choisissez le label a lui associer ou entrez le nom d'un nouveau label.  
En cas d'erreur de manipulation, au lieu de supprimer puis recréer un nouveau polygône, vous pouvez éditer les polygônes avec l'outil "Edit Polygons" (toujours sur la barre d'outils à gauche)  
Dans le cas où deux images successives sont très similaires, l'outil "Duplicate Polygons" (toujours sur la barre d'outils à gauche) peut être pratique.  

### 3. Mette le jeu de données au format COCO

Premièrement installez le paquet qui permet la conversion :
```
pip install -U labelme2coco
```
Ensuite effectuer la conversion en entrant cette commande en y indiquant le chemin du dossier contenant les fichiers json crées par labelme et le taux de partage désiré entre le jeu d'entraînement et celui de validation :
```
labelme2coco path/to/labelme/dir --train_split_rate 0.85
```
Les deux fichiers json au format seront déposés dans le dossier path/to/labelme/dir/run/labelme2coco  
  
Ensuite déplacez ces deux fichiers à "CreaCube/YOLOX_0.3.0/dataset/COCO/annotations/" et renommez les "instances_train2017.json" et "instances_val2017.json"
Enfin pour déplacer les images afin qu'elle correspondent à la [structure](https://github.com/chlMercier/CreaCubeXlim/blob/main/YOLOX/datasets/README.md) attendue pour un jeu de données au format COCO exécuter le programme "split_images.py" qui se trouve dans le dossier "CreaCube/Manage_Dataset". Si vous n'avez pas toucher à la structure du dossier les paramètres par défaut fonctionneront très bien, sinon tapez la commande suivante et modifiez les paramètres en conséquence :
```
python split_images.py --help
```

### 4. Changer les dimensions des images (besoin de debugging)

Il se peut que vous vouliez changez les dimensions des images afin par exemple que l'apprentissage prenne moins de temps (pas sûr de ça) ou qu'il soit moins gourmand en mémoire. Il est plus pratique de faire cela après les précédentes étapes afin de pouvoir profiter d'une bonne résolution d'image lors de l'étape d'annotation.  
Pour changer ces dimensions exécutez le programme "change_image_dim.py" se trouvant dans le dossier "CreaCube/Manage_Dataset". Auparavant lancez la commande suivante afin de savoir paramètrer le programme :
```
python change_image_dim.py --help
```

### 5. Extension du jeu de données

Pour ce faire installez le paquet adéquat :
```
git clone https://github.com/mohamadmansourX/Merge_COCO_FILES.git
cd Merge_COCO_FILES
```
Ensuite exécuter le programme "Merge_COCO_FILES/merge.py" pour fusionner les fichier json "train" et "val" de chaque jeu de données :
```
python merge.py Json1.json Json2.json OUTPUt_JSON.json
```
Faites attention à ce que les classes (cube_bleu...) à la fin des des fichiers json à fusionner soient dans le même ordre.
Ensuite il ne reste plus qu'à rassembler les images d'entraînement et de validation dans deux dossiers uniques.

## Apprentissage

Je n'ai pas encore réussi à finir un nouvel entraînement, j'ai toujours une erreur en cachant une autre qui fait s'échouer l'apprentissage à la fin de la première époque, ce qui est déjà mieux que lors de mes précédents essais.  
Il ressort de mes différentes lectures sur la section "issues" du github de YOLOX qu'il se peut que l'apprentissage fontcionne mieux avec une installation sous linux ou une ne passant pas par Anaconda.

Une fois le jeu de données prêt, l'apprentissage sur des données personnelles requiert seulement la création d'un fichier exp personnalisé.
Celui que j'ai crée se trouve au chemin suivant : "CreaCube/YOLOX_0.3.0/exps/example/custom/yolox_s.py"
Si vous avez besoin de le personnaliser plus avant, vous trouverez la classe dont il hérite au chemin suivant : "CreaCube/YOLOX_0.3.0/yolox/exp/yolox_base.py".  
Vous trouverez tous les paramètre que vous pouvez modifier dans la fonction "__init__" de la classe, il suffit de les redéfinir dans la même fonction "__init__" du fichier personnalisé.

Ensuite vous pouvez prendre connaissances des différentes options qu'accepte la fonction d'entraînement de YOLOX avec la commande suivante :
```
python tools/train.py --help
```
Voici pour exemple la commande que j'ai tapé qui a crée le moins d'ereurs :
```
python .\tools\train.py -f .\exps\example\custom\yolox_s.py -d 1 -b 1 --fp16 -c .\YOLOX_outputs\yolox_s\best_ckpt.pth
```
