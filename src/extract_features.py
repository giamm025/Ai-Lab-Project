import os
import sys
import cv2
import mediapipe as mp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import PROCESSED_DIR, RAW_DIR, LABEL_MAP

# ------------------------------------------------------------- MEDIA PIPE -------------------------------------------------------------
mp_holistic = mp.solutions.holistic


# funzione per processare un intero video: apre il video, lo divide in frame ed estrae i keypoints per ogni frame
# alla fine salva il risultato in un file NumPy pronto per essere utilizzato del modello LSTM
def process_video(video_path, save_path):

    # apriamo il video con OpenCV
    cap = cv2.VideoCapture(video_path)

    # prepariamo una lista vuota dove salvare le coordinate di ogni frame. ogni elemento di questa lista sarà un array
    # di 1530 elementi. Ogni elemento è una lista di coordinate  [x, y, z] di cui 63 per MANO SX, 63 per MANO DX e 1404 per VOLTO
    frames_keypoints = []

    # avviamo il modello Holistic di MediaPipe (il un modello pre-addestrato per riconoscere le coordinate di mani, volto e corpo)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        # finche il video è aperto
        while cap.isOpened():

            # leggiamo un frame alla volta (ret è un bool che ci dice se il frame è stato letto correttamente;
            # frame è l'immagine vera e propria). se ret è False, significa che il video è finito o c'è stato
            # un errore nella lettura del frame... in entrambi i casi, usciamo dal ciclo
            ret, frame = cap.read()
            if not ret:
                break

            # solita conversione (openCV) BGR -> RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # estriamo i keypoints da MediaPipe
            mp_keypoints = holistic.process(frame_rgb)

            # estraiamo l'array di 1530 numeri e lo salviamo nella lista
            keypoints = convert_keypoints(mp_keypoints)
            frames_keypoints.append(keypoints)

        # una volta finito, rilasciamo il video per liberare le risorse
        cap.release()

    # trasformiamo la lista in un Array NumPy e lo salviamo in un file .npy pronto per essere utilizzato dal modello LSTM
    numpy_data = np.array(frames_keypoints)
    np.save(save_path, numpy_data)


# funzione per convertire i risultati di MediaPipe in un array di 1530 numeri (di cui 63*3 MANO SX + 63*3 MANO DX + 468*3 VOLTO)
def convert_keypoints(mp_keypoints):

    # estraiamo i keypoints per ogni parte del corpo
    left_hand = mp_keypoints.left_hand_landmarks
    right_hand = mp_keypoints.right_hand_landmarks
    face = mp_keypoints.face_landmarks

    # formiamo la lista di coordinare per la MANO SINISTRA
    temp_list = []
    if left_hand:
        for res in left_hand.landmark:
            temp_list.append([res.x, res.y, res.z])
        lh = np.array(temp_list).flatten()
    else:
        lh = np.zeros(21 * 3)

    # formiamo la lista di coordinare per la MANO DESTRA
    temp_list = []
    if right_hand:
        for res in right_hand.landmark:
            temp_list.append([res.x, res.y, res.z])
        rh = np.array(temp_list).flatten()
    else:
        rh = np.zeros(21 * 3)

    # formiamo la lista di coordinare per il VOLTO
    temp_list = []
    if face:
        for res in face.landmark:
            temp_list.append([res.x, res.y, res.z])
        face = np.array(temp_list).flatten()
    else:
        face = np.zeros(468 * 3)

    return np.concatenate([lh, rh, face])


# ------------------------------------------------------ MAIN ------------------------------------------------------
if __name__ == "__main__":

    # creiamo la cartella processed se non esiste (la useremo per salvare i file .npy con i dati estratti da ogni video)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("\n--- INIZIO ESTRAZIONE MASSIVA (DA TUTTI I DATASET) ---")
    video_processati = 0
    video_saltati = 0

    # Leggiamo TUTTI i file mp4 presenti nella cartella data/raw/ (indipendentemente da quale dataset provengano)
    for video_filename in os.listdir(RAW_DIR):
        if video_filename.endswith(".mp4"):

            # dal filename estriamo la parola (il formato del nostro dataet è sempre parola_dataset_id.mp4)
            word = video_filename.split("_")[0]

            # creiamo il percorso da cui leggere il video (data/raw/parola_dataset_id.mp4)
            # creiamo il percorso in cui salvare il file .npy con le coordinate estratte (data/processed/parola_dataset_id.npy)
            video_path = os.path.join(RAW_DIR, video_filename)
            save_filename = video_filename.replace(".mp4", ".npy")
            save_path = os.path.join(PROCESSED_DIR, save_filename)

            # se il video è gia stato processato in passato => next
            if os.path.exists(save_path):
                video_saltati += 1
            else:
                print(f"Elaborazione: [{word}] -> {video_filename}...")
                process_video(video_path, save_path)
                video_processati += 1

    print("\n--- RESOCONTO FINALE ESTRAZIONE ---")
    print(f"Video processati: {video_processati}")
    print(f"Video già esistenti (saltati): {video_saltati}")
    print(
        f"Totale tensori pronti per la Rete Neurale: {video_processati + video_saltati}"
    )
