<h2 align="center">Multi-relational Graph Attention Networks for Traffic Interaction Modeling and Prediction</h2>

**Data**

1. _Download from_ [<img src="https://img.shields.io/badge/Onedrive-0078D4?&style=flat-square&logo=Microsoft+OneDrive&logoColor=white"/>](https://1drv.ms/u/s!AufZP2YDvxUDlg5G8bGu7Ay7vzhX?e=U6Kmt4) [<img src="https://img.shields.io/badge/Google_Drive-4285F4?style=flat-square&logo=Google+Drive&logoColor=white"/>](https://drive.google.com/file/d/1oXSKwV71olfoeyt4dgoVXSdIN_S17hsL/view?usp=sharing)
2. _Unzip and move to_ `./data/`

**Usage**

1. _Create directories:_

    ```bash
    mkdir checkpoints && mkdir checkpoints/pems{d3,d4,d7,d8,-bay}
    ```

2. _Train on a single GPU:_

    ```bash
    # PEMSD3
    python3 ./main.py --data data/pemsd3/pemsd3.npz --adj data/pemsd3/pemsd3.csv \
        --checkpoints checkpoints/pemsd3 --nodes 358 --channels 1 --batch 64 --workers 4 --gpu 0

    # PEMSD4
    python3 ./main.py --data data/pemsd4/pemsd4.npz --adj data/pemsd4/pemsd4.csv \
        --checkpoints checkpoints/pemsd4 --nodes 307 --channels 3 --batch 64 --workers 4 --gpu 0

    # PEMSD7
    python3 ./main.py --data data/pemsd7/pemsd7.npz --adj data/pemsd7/pemsd7.csv \
        --checkpoints checkpoints/pemsd7 --nodes 883 --channels 1 --batch 24 --workers 4 --gpu 0

    # PEMSD8
    python3 ./main.py --data data/pemsd8/pemsd8.npz --adj data/pemsd8/pemsd8.csv \
        --checkpoints checkpoints/pemsd8 --nodes 170 --channels 3 --batch 64 --workers 4 --gpu 0

    # PEMS-BAY
    python3 ./main.py --data data/pems-bay/pems-bay.npz --adj data/pems-bay/pems-bay.csv \
        --checkpoints checkpoints/pems-bay --nodes 325 --channels 1 --batch 64 --delta 20 \
        --workers 4 --gpu 0
    ```

3. _Train on multiple GPUs:_

    ```bash
    # PEMSD3
    python3 ./main.py --data data/pemsd3/pemsd3.npz --adj data/pemsd3/pemsd3.csv \
    --checkpoints checkpoints/pemsd3 --nodes 358 --channels 1 --batch 64 --workers 4 --gpus 0,1,2,3

    # PEMSD4
    python3 ./main.py --data data/pemsd4/pemsd4.npz --adj data/pemsd4/pemsd4.csv \
    --checkpoints checkpoints/pemsd4 --nodes 307 --channels 3 --batch 64 --workers 4 --gpus 0,1,2,3

    # PEMSD7
    python3 ./main.py --data data/pemsd7/pemsd7.npz --adj data/pemsd7/pemsd7.csv \
    --checkpoints checkpoints/pemsd7 --nodes 883 --channels 1 --batch 48 --workers 4 --gpus 0,1,2,3

    # PEMSD8
    python3 ./main.py --data data/pemsd8/pemsd8.npz --adj data/pemsd8/pemsd8.csv \
    --checkpoints checkpoints/pemsd8 --nodes 170 --channels 3 --batch 64 --workers 4 --gpus 0,1,2,3

    # PEMS-BAY
        python3 ./main.py --data data/pems-bay/pems-bay.npz --adj data/pems-bay/pems-bay.csv \
        --checkpoints checkpoints/pems-bay --nodes 325 --channels 1 --batch 64 --delta 20 \
        --workers 4 --gpus 0,1,2,3
    ```

\***Checkpoints**

-   _PEMSD4, 2080TI-11G:_ [<img src="https://img.shields.io/badge/PEMSD4-MAE=19.49_MAPE=13.67_RMSE=31.66-4285F4?style=flat-square&logo=Pytorch"/>](https://drive.google.com/file/d/1UEE1YJuA2RGhnL8R_XjBzrY03QJ6z1Vs/view?usp=sharing)
