<h2 align="center">Learning Multiaspect Traffic Couplings by Multirelational Graph Attention Networks for Traffic Prediction</h2>

**Data**

1. _Download from:_ [<img src="https://img.shields.io/badge/Google_Drive-4285F4?style=flat-square&logo=Google+Drive&logoColor=white"/>](https://drive.google.com/file/d/1oXSKwV71olfoeyt4dgoVXSdIN_S17hsL/view?usp=sharing) [<img src="https://img.shields.io/badge/Onedrive-0078D4?&style=flat-square&logo=Microsoft+OneDrive&logoColor=white"/>](https://1drv.ms/u/s!AufZP2YDvxUDlg5G8bGu7Ay7vzhX?e=X0asLx)
2. _Unzip and move to_ `./data`

**Usage**

1.  _Train_

    -   Docker container (recommended)

        ```sh
        # PEMSD3
        docker run -it --rm --gpus=all --shm-size=512m -v /path/to/checkpoints:/ms-gat/checkpoints luokn/ms-gat -d pemsd3 -i 1,2,3,24 -w 8
        # PEMSD4
        docker run -it --rm --gpus=all --shm-size=512m -v /path/to/checkpoints:/ms-gat/checkpoints luokn/ms-gat -d pemsd4 -w 8
        # PEMSD7
        docker run -it --rm --gpus=all --shm-size=512m -v /path/to/checkpoints:/ms-gat/checkpoints luokn/ms-gat -d pemsd7 -b 32 -w 8
        # PEMSD8
        docker run -it --rm --gpus=all --shm-size=512m -v /path/to/checkpoints:/ms-gat/checkpoints luokn/ms-gat -d pemsd8 -w 8
        ```

    -   Physical machine:

        ```sh
        # PEMSD3
        python3 src/main.py -d pemsd3 -o checkpoints/pemsd3 -i 1,2,3,24 -w 8
        # PEMSD4
        python3 src/main.py -d pemsd4 -o checkpoints/pemsd4 -w 8
        # PEMSD7
        python3 src/main.py -d pemsd7 -o checkpoints/pemsd7 -b 32 -w 8
        # PEMSD8
        python3 src/main.py -d pemsd8 -o checkpoints/pemsd8 -w 8
        ```

2.  _Evaluate_

    ```sh
    python3 src/main.py --eval -d pemsd4 -o checkpoints/pemsd4 -c checkpoints/pemsd4/xx_xxx.xx.pkl
    ```

**Checkpoints**

-   PEMSD3: [_MAE = 15.60 MAPE = 16.36% RMSE = 26.36_](https://drive.google.com/file/d/16bUCaI4p23vTGdMOXRRT45TNqci7VLCi/view?usp=sharing)
-   PEMSD4: [_MAE = 19.59 MAPE = 13.34% RMSE = 31.58_](https://drive.google.com/file/d/1i3H6GuqBvCOZ_DdPRReKECwb14zvQzY3/view?usp=sharing)
-   PEMSD7: [_MAE = 20.44 MAPE = 8.85% RMSE = 34.11_](https://drive.google.com/file/d/1a9VdvFOaMGU9-JyeRlDUDlzjHdrsEKSr/view?usp=sharing)
-   PEMSD8: [_MAE = 14.58 MAPE = 10.10% RMSE = 23.94_](https://drive.google.com/file/d/18_mJtL0G6KQZF8QxSLQu9THFg-h_46q-/view?usp=sharing)

**Citation**

```tex
    @ARTICLE{9780244,
        author    ={Huang, Jing and Luo, Kun and Cao, Longbing and Wen, Yuanqiao and Zhong, Shuyuan},
        journal   ={IEEE Transactions on Intelligent Transportation Systems},
        title     ={Learning Multiaspect Traffic Couplings by Multirelational Graph Attention Networks for Traffic Prediction},
        year      ={2022},
        volume    ={},
        number    ={},
        pages     ={1-15},
        doi       ={10.1109/TITS.2022.3173689}
    }
```
