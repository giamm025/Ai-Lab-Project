import kagglehub

paths = {
    "asllvd": "../datasets/ALLVD",
    "asl": "../datasets/ASL_Citizen",
    "ms": "../datasets/MS_ASL",
    "wlasl": "../datasets/WLASL",
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

    print(f"Scaricando il dataset '{name}' da Kaggle...")

    # FIX: Removed 'unzip=True' and changed 'path' to 'output_dir'
    kagglehub.dataset_download(handle, output_dir=target_path + "/raw")

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
        + ["all"],  # Limita le scelte per evitare errori di battitura
        # Se non scrivi nulla nel terminale termina con un messaggio chiaro che mostra le opzioni disponibili
        help=f"Scegli quale dataset scaricare: {list(urls.keys())} oppure 'all' per scaricarli tutti",
        required=True,  # Rendi questo argomento obbligatorio
    )

    args = parser.parse_args()

    if args.dataset == "all":
        for ds in urls.keys():
            download_dataset(ds)
    else:
        download_dataset(args.dataset)
