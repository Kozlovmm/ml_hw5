# src/inference.py
import os
import pickle
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import json

EMB_PATH = 'models/embeddings.pkl'
ANCHOR_PATH = 'validation/anchor.jpg'
TEST_DIR = 'validation/test'
RESULT_PATH = 'predictions/test_results.json'
THRESHOLD = 0.9  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return None
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device))
    return embedding.squeeze().cpu()

def cosine_distance(a, b):
    return 1 - torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def main():
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    anchor_emb = get_embedding(ANCHOR_PATH)
    if anchor_emb is None:
        print("No face found in anchor.")
        return

    results = []

    for file in os.listdir(TEST_DIR):
        img_path = os.path.join(TEST_DIR, file)
        emb = get_embedding(img_path)
        if emb is None:
            continue
        distance = cosine_distance(anchor_emb, emb)
        results.append({
            'image': file,
            'distance': round(distance, 4),
            'is_match': distance < THRESHOLD
        })

    with open(RESULT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {RESULT_PATH}")

if __name__ == '__main__':
    main()
