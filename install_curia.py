from transformers import AutoModel
import os

out_dir = "./curia_model"

model = AutoModel.from_pretrained(
    "raidium/curia",
    cache_dir=out_dir
)

print(f"Model downloaded in: {os.path.abspath(out_dir)}")