# CreaCube

L’ordinateur sur lequel a été développé le projet était composé d’un processeur Intel Xeon CPU E5-1650 v3, de 16 Go de mémoire vive et d’une carte graphique Nvidia Geforce GTX 1070.

## Installation

La procédure suivante décrit l'installation de toutes les dépendances requises dans un environnement Anaconda sur un ordinateur fonctionnant sur Windows 10.

### 1. Installation d’Anaconda

Veuillez suivre ce guide afin d’installer Anaconda sur Windows.

### 2. Installer Cuda

Si vous n'avez pas de carte graphique de marque Nvidia ou que vous ne souhaitez pas utiliser cette dernière pour l'inférence, passez cette étape.

Afin de pouvoir profiter de la carte graphique pendant l’inférence, mais surtout pendant l’apprentissage, il faut installer l’API CUDA de Nvidia. Pour cela veuillez télécharger l'éxécutable [ici](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) et suivre ce [guide].

### 3. Télécharger le projet depuis le git

Clonez ce git et télécharger [ici](https://user-images.githubusercontent.com/55946370/180219702-0046b8ee-8824-46c8-a823-bb14a8e2eb41.png) les fichiers nécessaire selon vos besoins :
  1. Le dossier "ONNX_Models" contient le modèle pour l'inférence.
  2. Le dossier "YOLOX_outputs" contient les checkpoints correspondant à l'apprentissage des différentes tailles de modèles YOLOX.
  3. Le dossier "COCO" contient le jeu de données au format COCO.

### 4. Créer l’environement Anaconda adéquat

Pour créer l'environement Anaconda adéquat ouvrez l'Anaconda Powershell Prompt et entrez les lignes suivantes :
```
conda create -n "myenv" python=3.9.12
conda activate myenv
conda install -c conda-forge pycocotools
```
Si vous avez une carte graphique Nvidia et que vous avez installé VUDA, vous pouvez maintenant installer CUDA pour python. Pour cela entrez la commande qui correspond à la version de CUDA que vous avez installé précédement.
```
conda install cuda -c nvidia/label/cuda-11.X.X
```
Installez maintenant torch. Les 2 premières commandes sont pour l'installation avec support GPU et la dernière pour les installations CPU.
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

pip install torch torchvision torchaudio
```
Ensuite déplacer vous dans le répertoire où vous avez téléchargé le git puis dans le dossier nommé "YOLOX" et entrez les lignes suivantes :
```
cd path\to\the\project\YOLOX
pip install -r requirements.txt
pip install -v -e .
```
Il se peut que lors de l'exécution de la dernière ligne il soit retourné une erreur comme quoi la version de
Dans ce cas télécharger les Outils de génération Microsoft C++ [ici](https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/) et lors de l'intalation cochez uniquement la case correspondant au C++. Ceci fait réexécuter la dernière ligne.

## Utilisation

### 1. Inférence

Ouvrez l'Anaconda Powershell Prompt et entrez les lignes suivantes
```
conda activate myenv
cd path\to\the\project\
python --help
```

