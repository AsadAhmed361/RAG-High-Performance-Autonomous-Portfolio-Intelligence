from sentence_transformers import SentenceTransformer
import os

# 1. Model ka naam
model_name = 'all-MiniLM-L6-v2'

# 2. Model download karein
model = SentenceTransformer(model_name)

# 3. Project folder mein aik directory banayein aur save karein
save_path = "./my_local_model"
model.save(save_path)

print(f"✅ Model save ho gaya hai yahan: {os.path.abspath(save_path)}")