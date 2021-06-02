<h2>MS-GAT: </h2>

**Data**

1. Download from
   [OneDrive](https://1drv.ms/u/s!AufZP2YDvxUDjhJBFIUK-PcyhufH?e=y3gyf9)
   [Google Drive](https://drive.google.com/drive/folders/1uOPQRiYGqKTVxI6zJkvmYuxDln-MXV8g?usp=sharing)

2. Unzip and move to `./data/`

**Usage**

1. Create directory: `mkdir checkpoints && mkdir checkpoints/pems{d3,d4,d7,d8,-bay}`
2. Train PEMSD3/D4/D7/D8/-BAY:

```bash
# PEMSD3
python3 ./main.py --data data/pemsd3/pemsd3.npz --adj data/pemsd3/pemsd3.csv --checkpoints checkpoints/pemsd3 --nodes 358 --channels 1 --batch 32 --out-timesteps 12 --workers 4 --gpu 0
# PEMSD4
python3 ./main.py --data data/pemsd4/pemsd4.npz --adj data/pemsd4/pemsd4.csv --checkpoints checkpoints/pemsd4 --nodes 307 --channels 3 --batch 64 --out-timesteps 12 --workers 4 --gpu 0
# PEMSD7
python3 ./main.py --data data/pemsd7/pemsd7.npz --adj data/pemsd7/pemsd7.csv --checkpoints checkpoints/pemsd7 --nodes 883 --channels 1 --batch 16 --out-timesteps 12 --workers 4 --gpu 0
# PEMSD8
python3 ./main.py --data data/pemsd8/pemsd8.npz --adj data/pemsd8/pemsd8.csv --checkpoints checkpoints/pemsd8 --nodes 170 --channels 3 --batch 64 --out-timesteps 12 --workers 4 --gpu 0
# PEMS-BAY
python3 ./main.py --data data/pems-bay/pems-bay.npz --adj data/pems-bay/pems-bay.csv --checkpoints checkpoints/pems-bay --nodes 358 --channels 1 --batch 32 --out-timesteps 12 --workers 8 --gpu 0
```

3. Train on GPUs: `python3 ./main.py --data data/pemsd4/pemsd4.npz --adj data/pemsd4/pemsd4.csv --checkpoints checkpoints/pemsd4 --nodes 307 --channels 3 --batch 128 --out-timesteps 12 --workers 8 --gpus 0,1,2,3`

**Citation**

...
