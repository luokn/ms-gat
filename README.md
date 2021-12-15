<h2 align="center">Multi-relational Graph Attention Networks for Traffic Signal Coupling Learning and Prediction</h2>

**Data**

1. _Download from:_ [<img src="https://img.shields.io/badge/Google_Drive-4285F4?style=flat-square&logo=Google+Drive&logoColor=white"/>](https://drive.google.com/file/d/1oXSKwV71olfoeyt4dgoVXSdIN_S17hsL/view?usp=sharing) [<img src="https://img.shields.io/badge/Onedrive-0078D4?&style=flat-square&logo=Microsoft+OneDrive&logoColor=white"/>](https://1drv.ms/u/s!AufZP2YDvxUDlg5G8bGu7Ay7vzhX?e=X0asLx)
2. _Unzip and move to_ `./data/`

**Usage**

1.  _Train_

    -   single GPU:

        ```bash
        # PEMSD4
        python3 ./main.py train -a data/pemsd4/pemsd4.csv -d data/pemsd4/pemsd4.npz -o checkpoints/pemsd4/debug \
            -n 307 -c 3 -b 64 -j 4
        ```

    -   multiple GPUs:

        ```bash
        # PEMSD4
        python3 ./main.py train -a data/pemsd4/pemsd4.csv -d data/pemsd4/pemsd4.npz -o checkpoints/pemsd4/debug \ -n 307 -c 3 -b 64 -j 4 --gpus 0,1,2,3
        ```

2.  _Evaluate_

    ```bash
    # PEMSD4
    python3 ./main.py eval -a data/pemsd4/pemsd4.csv -d data/pemsd4/pemsd4.npz -o checkpoints/pemsd4/debug \
        -n 307 -c 3 -b 64 -j 4 -f checkpoints/pemsd4/debug/xx_loss\=xxx.xx.pkl
    ```

**_Checkpoints_**

-   _PEMSD3, 2080TI-11G:_ [<img src="https://img.shields.io/badge/PEMSD3-MAE=15.60_MAPE=16.36_RMSE=26.36-4EAA25?style=flat-square"/>](https://drive.google.com/file/d/16bUCaI4p23vTGdMOXRRT45TNqci7VLCi/view?usp=sharing)
-   _PEMSD4, 2080TI-11G:_ [<img src="https://img.shields.io/badge/PEMSD4-MAE=19.58_MAPE=13.52_RMSE=31.72-4EAA25?style=flat-square"/>](https://drive.google.com/file/d/1CzS1-OCZXP6g8jM_CAMZGWlHRDyRxDBH/view?usp=sharing)
-   _PEMSD7, 2080TI-11G:_ [<img src="https://img.shields.io/badge/PEMSD7-MAE=20.44_MAPE=8.85_RMSE=34.11-4EAA25?style=flat-square">](https://drive.google.com/file/d/1a9VdvFOaMGU9-JyeRlDUDlzjHdrsEKSr/view?usp=sharing)
-   _PEMSD8, 2080TI-11G:_ [<img src="https://img.shields.io/badge/PEMSD8-MAE=14.58_MAPE=10.10_RMSE=23.94-4EAA25?style=flat-square"/>](https://drive.google.com/file/d/18_mJtL0G6KQZF8QxSLQu9THFg-h_46q-/view?usp=sharing)
