# Ai-Lab-Project

Struttura del progetto:


AI-LAB-PROJECT:.
│   .gitignore
│   README.md
│   
├───data
│   │   labels.json
│   │   
│   ├───processed
│   │       .gitkeep
│   │       
│   ├───raw
│   │       .gitkeep
│   │       
│   └───target_videos
│           .gitkeep
│               
│       
├───models
│       .gitkeep
│       
└───src
        config.py
        dataset.py
        extract_features.py
        model.py
        train.py


Comando per addestrare i modelli (Ai-Lab-Project/src): python train.py --modalita {SOLO_MANI, MANI_VOLTO}