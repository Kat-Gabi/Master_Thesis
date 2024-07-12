# Enhancing Face Recognition Models with Synthetic Child Data: A Fairness Assessment

## Project structure 

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed/       <- The final, data sets for modeling and testing. (ask authors for access due to storage limitation)
│   ├── raw/             <- The original, immutable data dump. (ask authors for access due to storage limitation)
│   └── image_info_csv/  <- Directory with csv files for data analysis
│
├── AdaFace-master       <- Folder with AdaFace requirements as well as fine-tuning scripts and saved models (ask authors for access due to storage limitation)
│   │
│   ├── predict_adaface_fine_tuning.py       <- Fine-tuning script
│   │
│   ├── scripts/         <- Directory with scripts for setting hyper-parameters and run in cluster
│   │
│   └── Master_thesis_data_prep/          <- Directory for model-specific data processing
│
├── MagFace-main         <- Folder with MagFace requirements as well as fine-tuning scripts and saved models (ask authors for access due to storage limitation)
│   │
│   ├── predict_adaface_fine_tuning.py       <- Fine-tuning script
│   │
│   ├── run/             <- Directory with scripts for setting hyper-parameters and run in cluster
│   │
│   └── Master_thesis_data_prep/          <- Directory for model-specific data processing
│
├── figures               <- Saved figures for report
│
├── notebooks            <- Jupyter notebooks for different types of analysis
│
├── utils                <- Functions (Data processing, plotting, OFIQ, DET etc.) to use inside jupyter notebooks and other scripts
│
├── pyproject.toml       <- Project configuration file
│
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
│
└── LICENSE              <- Open-source license: MIT
```

## Data loading 
YLFW data is downloaded through https://github.com/JessyFrish/YLFW_Links, YLFW_benchmark 
RFW are downloaded through http://whdeng.cn/RFW/testing.html by asking for access. 
Age estimation of the face images has been done using Cognitec FaceVACS. 
Preprocessing of the images are done through data_loading_ylfw.ipynb and data_loading_rfw.ipynb, which creates the csv files raw_ylfw_df.csv and raw_rfw_df.csv. 
To move these csvs to folders AdaFace and MagFace can work with, move_data function in Data_proc_funcs.py is used. 
The balancing of the two data sets to create the final children and adults data sets are done in balance_data functions in Data_proc_funcs.py. 
To use the respective balanced datasets for training MagFace, use make_image_lits_magface.py. 
