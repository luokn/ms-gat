<h2 align="center">Multi-relational Graph Attention Networks for Traffic Signal Coupling Learning and Prediction</h2>

**Data**

1. _Download from:_ [<img src="https://img.shields.io/badge/Google_Drive-4285F4?style=flat-square&logo=Google+Drive&logoColor=white"/>](https://drive.google.com/file/d/1oXSKwV71olfoeyt4dgoVXSdIN_S17hsL/view?usp=sharing) [<img src="https://img.shields.io/badge/Onedrive-0078D4?&style=flat-square&logo=Microsoft+OneDrive&logoColor=white"/>](https://1drv.ms/u/s!AufZP2YDvxUDlg5G8bGu7Ay7vzhX?e=X0asLx)
2. _Unzip and move to_ `./data/`

**Usage**

1.  _Train_

    -   Single GPU:

        ```bash
        # PEMSD3
        python3 main.py -d pemsd3 -o checkpoints/pemsd3/exp0 -j 16 -i 1,2,3,24 --gpu-ids 0
        # PEMSD4
        python3 main.py -d pemsd4 -o checkpoints/pemsd4/exp0 -j 16 --gpu-ids 0
        # PEMSD7
        python3 main.py -d pemsd7 -o checkpoints/pemsd7/exp0 -j 16 --gpu-ids 0
        # PEMSD8
        python3 main.py -d pemsd8 -o checkpoints/pemsd8/exp0 -j 16 --gpu-ids 0
        ```

    -   Multiple GPUs:

        ```bash
        # PEMSD7
        python3 main.py -d pemsd7 -o checkpoints/pemsd7/exp0 -j 16 -b 128 --gpu-ids 0,1,2,3
        ```

2.  _Evaluate_

    ```bash
    # PEMSD4
    python3 main.py -d pemsd4 -o checkpoints/pemsd8/exp0 -c checkpoints/pemsd4/exp0/xxx.pkl -j 16 --eval
    ```

**_Checkpoints_**

-   _PEMSD3:_ [<img src="https://img.shields.io/badge/PEMSD3-MAE=15.60_MAPE=16.36%_RMSE=26.36-4EAA25?style=flat-square"/>](https://drive.google.com/file/d/16bUCaI4p23vTGdMOXRRT45TNqci7VLCi/view?usp=sharing)
-   _PEMSD4:_ [<img src="https://img.shields.io/badge/PEMSD4-MAE=19.59_MAPE=13.34%_RMSE=31.58-4EAA25?style=flat-square"/>](https://drive.google.com/file/d/1i3H6GuqBvCOZ_DdPRReKECwb14zvQzY3/view?usp=sharing)
-   _PEMSD7:_ [<img src="https://img.shields.io/badge/PEMSD7-MAE=20.44_MAPE=8.85%_RMSE=34.11-4EAA25?style=flat-square">](https://drive.google.com/file/d/1a9VdvFOaMGU9-JyeRlDUDlzjHdrsEKSr/view?usp=sharing)
-   _PEMSD8:_ [<img src="https://img.shields.io/badge/PEMSD8-MAE=14.58_MAPE=10.10%_RMSE=23.94-4EAA25?style=flat-square"/>](https://drive.google.com/file/d/18_mJtL0G6KQZF8QxSLQu9THFg-h_46q-/view?usp=sharing)
