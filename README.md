# Usage
Run example with:
```bash
OMP_NUM_THREADS=8 uv run main.py test_data/dataset_c4h6_finetune.pkl 
```

or with GPU support:
```bash
uv run --extra cuda main.py test_data/dataset_c4h6_finetune.pkl --gpu 0
```

To split a dataset into shards:
```bash
uv run make_shards.py test_data/dataset_c4h6_finetune.pkl 10
```