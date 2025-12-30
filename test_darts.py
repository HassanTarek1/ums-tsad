from darts.models import NHiTSModel
from darts import models
model = NHiTSModel(input_chunk_length=24, output_chunk_length=12)
print("✅ Darts is working. Model created:", model)
print(models)
