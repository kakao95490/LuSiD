import kagglehub
import os
import shutil
from sklearn.model_selection import train_test_split

def dataset_download():

    # Download latest version
    path = kagglehub.dataset_download("rhtsingh/google-universal-image-embeddings-128x128")

    print("Path to dataset files:", path)


    # Chemin du dossier d'entrée contenant les images
    input_dir = "./1/128x128"
    output_dir = "dataset"

    # Proportions pour train, val et test
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Vérification que les proportions totalisent 1
    assert train_ratio + val_ratio + test_ratio == 1, "Les proportions doivent totaliser 1."

    # Récupérer toutes les images de tous les sous-dossiers
    images = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                images.append(os.path.join(root, file))

    # Mélanger et diviser les images en train, val et test
    train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

    # Fonction pour copier les images dans les répertoires de sortie
    def copy_images(image_list, output_subdir):
        output_path = os.path.join(output_dir, output_subdir)
        os.makedirs(output_path, exist_ok=True)
        for image_path in image_list:
            shutil.copy(image_path, os.path.join(output_path, os.path.basename(image_path)))

    # Copier les images dans les répertoires train, val et test
    copy_images(train_images, "train")
    copy_images(val_images, "val")
    copy_images(test_images, "test")

    print("Séparation des datasets terminée avec succès !")
    
    print(f"Nombre d'images dans l'ensemble train : {len(train_images)}")
    print(f"Nombre d'images dans l'ensemble validation : {len(val_images)}")
    print(f"Nombre d'images dans l'ensemble test : {len(test_images)}")
        
    return train_images,val_images,test_images
