import kagglehub
import argparse
from pathlib import Path

# calcola il percorso assoluto della cartella Ai-Lab-Project (in questo modo possiamo runnare il coice da qualsiasi cartella senza problemi, sia che ci troviamo in src/ sia che ci troviamo in Ai-Lab-Project/)
BASE_DIR = Path(__file__).resolve().parent.parent

paths = {
    "asllvd": BASE_DIR / "datasets" / "ASLLVD",
    "asl": BASE_DIR / "datasets" / "ASL_Citizen",
    "ms": BASE_DIR / "datasets" / "MS_ASL",
    "wlasl": BASE_DIR / "datasets" / "WLASL",
}

urls = {
    "asllvd": "https://www.kaggle.com/datasets/resolutebelief/asl-video-dataset",
    "asl": "https://www.kaggle.com/datasets/abd0kamel/asl-citizen",
    "ms": "https://www.kaggle.com/datasets/saurabhshahane/american-sign-language-dataset",
    "wlasl": "https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed",
}

handles = {
    "asllvd": "resolutebelief/asl-video-dataset",
    "asl": "abd0kamel/asl-citizen",
    "ms": "saurabhshahane/american-sign-language-dataset",
    "wlasl": "risangbaskoro/wlasl-processed",
}


def download_dataset(name: str):
    if name not in urls:
        print(f"Dataset '{name}' non trovato. Scegli tra: {list(urls.keys())}")
        return

    url = urls[name]
    target_path = paths[name]
    handle = handles[name]
    output_dir = str(target_path / "raw")

    print(f"Scaricando il dataset '{name}' da Kaggle...")
    kagglehub.dataset_download(handle, output_dir=output_dir)
    print(f"Dataset '{name}' scaricato e salvato in '{target_path}/raw'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scarica un dataset di segni ASL da Kaggle"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(urls.keys())
        + ["all"], 
        help=f"Scegli quale dataset scaricare: {list(urls.keys())} oppure 'all' per scaricarli tutti",
        required=True,
    )

    args = parser.parse_args()

    if args.dataset == "all":
        for ds in urls.keys():
            download_dataset(ds)
    else:
        download_dataset(args.dataset)
