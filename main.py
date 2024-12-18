import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # Couche 1 : extraction de caractéristiques
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()

        # Couche 2 : non-linéarité et mapping
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.relu2 = nn.ReLU()

        # Couche 3 : reconstruction
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

def upscale_and_predict(image_path, model_path, output_path):
    """
    Upscale une image x3, la passe dans le modèle SRCNN, et sauvegarde l'image prédite.
    
    :param image_path: Chemin vers l'image d'entrée.
    :param model_path: Chemin vers le fichier .pth du modèle.
    :param output_path: Chemin pour sauvegarder l'image prédite.
    """
    # Charger l'image
    img = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
    original_size = img.size
    
    # Upscaling de l'image (x3) avec interpolation bicubique
    upscale_size = (original_size[0] * 3, original_size[1] * 3)
    img_upscaled = img.resize(upscale_size, Image.BICUBIC)
    
    # Transformer l'image en tenseur
    transform = transforms.ToTensor()
    img_tensor = transform(img_upscaled).unsqueeze(0)  # Ajouter une dimension batch
    
    # Charger le modèle
    model = SRCNN()  # Supposons que la classe SRCNN est déjà définie
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Envoyer le tenseur dans le modèle
    with torch.no_grad():
        predicted_tensor = model(img_tensor)
    
    # Convertir la sortie en image
    predicted_img = predicted_tensor.squeeze().clamp(0, 1).numpy()  # Supprimer les dimensions inutiles et normaliser
    predicted_img = (predicted_img * 255).astype('uint8')  # Convertir en échelle de 0-255
    predicted_img = Image.fromarray(predicted_img)
    
    # Sauvegarder l'image prédite
    predicted_img.save(output_path)
    print(f"Image prédite sauvegardée à : {output_path}")

# Exemple d'utilisation
upscale_and_predict('./input.jpg', './srcnn_model.pth', './output.jpg')
