# Ai-Lab-Project

Struttura del progetto:


# Project Structure

```
Ai-Lab-Project/
├── data
│   ├── processed
│   ├── raw
│   └── labels.json
├── datasets
│   ├── ASL_Citizen
│   │   └── .gitkeep
│   ├── ASLLVD_non_va_bene_per_noi
│   │   └── .gitkeep
│   ├── MS_ASL
│   │   └── .gitkeep
│   └── WLASL
│       └── .gitkeep
├── models
├── src
│   ├── neural_network
│   │   ├── bulk_train.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train.py
│   ├── neural-network
│   ├── parsers
│   │   ├── ASLCitizen_parser.py
│   │   ├── MSASL_parser.py
│   │   └── WLASL_parser.py
│   ├── config.py
│   ├── download_datasets.py
│   ├── evaluate_model.py
│   └── extract_features.py
├── marimba_bloop.mp3
└── README.md
```


Comando per addestrare i modelli (Ai-Lab-Project/src): python train.py --modalita {SOLO_MANI, MANI_VOLTO}