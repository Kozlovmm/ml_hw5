# src/embed_database.py
import os
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import datasets
from PIL import Image
import torch
from tqdm import tqdm

DATA_DIR = 'data/my_dataset'
SAVE_PATH = 'models/embeddings.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_face_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return None
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device))
    return embedding.squeeze().cpu()

def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    embeddings = {}

    for person in tqdm(os.listdir(DATA_DIR)):
        person_path = os.path.join(DATA_DIR, person)
        if not os.path.isdir(person_path):
            continue
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            emb = extract_face_embedding(img_path)
            if emb is not None:
                embeddings[img_path] = emb

    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Saved {len(embeddings)} embeddings to {SAVE_PATH}")

if __name__ == '__main__':
    main()
