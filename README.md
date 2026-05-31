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


Comandi: 
- python src/train.py --modalita {SOLO_MANI, MANI_VOLTO}
- python src/bulk_train.py --modalita {SOLO_MANI, MANI_VOLTO}
- python src/evaluate_model.py --model_path models/nome_del_modello.pth
- python src/predict_single.py --video_path data/raw/nome_del_video.mp4
- python src/predict_single.py --video_path data/raw/nome_del_video.mp4 --modalita {SOLO_MANI, MANI_VOLTO}
- python src/webcam_inference.py
- python src/webcam_inference.py --modalita {SOLO_MANI, MANI_VOLTO}
- python src/webcam_inference.py --camera_index 1