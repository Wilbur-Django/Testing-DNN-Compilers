import os

mutants_dir = "/export/d2/dwxiao/mutants"

removed = []

for model_name in os.listdir(mutants_dir):
    for seed_number in os.listdir(os.path.join(mutants_dir, model_name)):
        models_dir = os.path.join(mutants_dir, model_name, seed_number, "hybrid", "models")
        for model_id in os.listdir(models_dir):
            if model_id[-6] != "1" and model_id != "seed.onnx":
                removed.append(model_id)
                os.remove(os.path.join(models_dir, model_id))
