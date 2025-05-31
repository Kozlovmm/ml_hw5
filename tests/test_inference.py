import os
from src.inference import infer

def test_inference_runs():
    anchor_path = 'validation/anchor.jpg'
    test_dir = 'validation/test'
    output_path = 'predictions/test_results.json'

    infer(anchor_path, test_dir, output_path=output_path, threshold=0.6)

    assert os.path.exists(output_path), "Файл результатов не создан"

    import json
    with open(output_path, 'r') as f:
        results = json.load(f)
    
    assert isinstance(results, dict), "Результаты не в формате словаря"
    assert len(results) > 0, "Результаты пусты"
