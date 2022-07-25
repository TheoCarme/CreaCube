# CreaCube

Les instructions qui suivent ont été testées sur un ordinateur fonctionnant sous Ubuntu 18, doté d'un processeur CPU Intel® Core™ i5-10310U 1.70GHz × 8, de 32 Go de mémoire vive et d’une carte graphique intégrée Intel® UHD (CML GT2).

## Installation

### 1. Installation de pipenv et Python 3.9

Le tutoriel suivant utilise [Pipenv](https://pypi.org/project/pipenv/) pour gérer les environnements virtuels. Si vous préférez utiliser Anaconda, veuillez vous reporter aux instructions pour Windows.

Les librairies ont été testées sous Python 3.9, il est donc recommandé d'utiliser la même version et de l'installer si vous ne l'avez pas déjà (différentes versions de python peuvent être installées en parallèle sans interférence grâce aux environnements virtuels) :
```
sudo apt install python 3.9
```

Pipenv peut être installé de cette manière (ou voir [ici](https://pipenv.pypa.io/en/latest/#install-pipenv-today) pour plus d'informations)
```
sudo apt install pipenv
```

### 2. Installer Cuda

N'ayant pas de carte graphique de marque Nvidia j'ai passé cette étape. Si besoin, des instructions sont disponibles [ici](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

### 3. Télécharger le projet depuis le git

Clonez ce git à l'aide de la commande suivante :
```
git clone https://github.com/TheoCarme/CreaCube.git
```

### 4. Installation des dépendances

- Si le Pipfile n'est pas déjà créé:

```
cd path\to\the\project
pipenv install pycocotools alive_progress
pipenv install torch torchvision torchaudio
```
Cette dernière ligne est à adapter si vous souhaitez utiliser une carte graphique Nvidia.

Puis, pour installer Yolox:
```
cd YOLOX
pipenv install -r requirements.txt
pipenv shell # activer l'environnement virtuel
pip install -v -e .
```

- Si le Pipfile est déjà créé:
```
cd path\to\the\project
pipenv install 
cd YOLOX
pipenv shell # activer l'environnement virtuel
pip install -v -e .
```

## Utilisation

### 1. Aide sur la commande

```
cd path\to\the\project
pipenv shell # activer l'environnement virtuel
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
| -c / --csv | Permet de demander au programme de écrire le fichier csv contenant les traces des objets suivis. |
| -w / --write_video | Permet de demander au programme d'écrire la vidéo avec les objets suivis dessinés dessus. |
| -d / --display | Permet de demander au programme d'afficher chaque trame traitée avec les objets suivis dessinés dessus. |

La ligne pour éxecuter le programme pourra donc ressembler à cela :
```
python .\combined_detection.py -m ..\Project_Data\ONNX_Models\yolox_s.onnx -v C:\Users\theon\Documents\Stage_XLIM\20220330T072838Z\scenevideo.mp4 -o ..\Results -n 2 -c -w -d
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


🔴 Traceback (most recent call last):
  File "/home/chloe/Projects/xlim-creacube/combined_detection.py", line 262, in <module>
    cubes_detector = Cubes_Detector(args.model, input_shape)
  File "/home/chloe/Projects/xlim-creacube/cubes_detector.py", line 21, in __init__
    self.session = onnxruntime.InferenceSession(model)
  File "/home/chloe/.local/share/virtualenvs/xlim-creacube-iwzZKUKy/lib/python3.9/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 347, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "/home/chloe/.local/share/virtualenvs/xlim-creacube-iwzZKUKy/lib/python3.9/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 375, in _create_inference_session
    raise ValueError(
ValueError: This ORT build has ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] enabled. Since ORT 1.9, you are required to explicitly set the providers parameter when instantiating InferenceSession. For example, onnxruntime.InferenceSession(..., providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'], ...)
```

Résolu en ajoutant les providers à la ligne 21 de cubes_detector.py

`self.session = onnxruntime.InferenceSession(model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])`