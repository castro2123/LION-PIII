from transformers import AutoModel, AutoTokenizer

# Pasta onde quer salvar o modelo
cache_dir = "checkpoints/bert-base-uncased"

# Baixa o modelo e o tokenizer
model = AutoModel.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=cache_dir)

print("BERT baixado com sucesso!")
