
import json
import os

import cv2
import mediapipe as mp
import numpy as np

# ------------------------------------------------------ CONFIGURAZIONE INIZIALE ------------------------------------------------------

# salviamo il percorso del json e dei video. Poi scegliamo le 5 parole che ci interessano
JSON_PATH = '../data/WLASL_v0.3.json'
VIDEO_DIR = '../data/raw/'
TARGET_WORDS = ['hello', 'book', 'computer', 'deaf', 'fine']

# funzione per filtrare il dataset e prendere solo i video relativi alle parole scelte prima (TARGET_WORDS)
def filter_dataset():

    with open(JSON_PATH, 'r') as f:
        wlasl_data = json.load(f)
    
    filtered_videos = {word: [] for word in TARGET_WORDS}
    video_count = 0

    # per ogni entry nel dataset
    for entry in wlasl_data:

        # prendiamo la parola associata a quell'entry e vediamo se è una delle nostre parole target
        word = entry['gloss']
        
        # se la parola è una delle nostre TARGET_WORDS
        if word in TARGET_WORDS:

            # prendiamo i video (instances) associati a quella parola
            for instance in entry['instances']:
                
                # estriamo i campi necessari
                video_id = instance['video_id']
                video_filename = f"{video_id}.mp4"
                video_path = os.path.join(VIDEO_DIR, video_filename)
                
                # Controllo di sicurezza: verifichiamo che il video .mp4 esista davvero sul PC
                if os.path.exists(video_path):
                    filtered_videos[word].append(video_filename)
                    video_count += 1
                    
    for word, videos in filtered_videos.items():
        print(f"Parola '{word}': trovati {len(videos)} video.")
        
    print(f"\nTotale video utili da processare: {video_count}")
    return filtered_videos


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
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
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
        
    # trasformiamo la lista in un Array NumPy
    numpy_data = np.array(frames_keypoints)
    
    # salviamo l'array in un file .npy pronto per essere utilizzato dal modello LSTM
    np.save(save_path, numpy_data)


# funzione per convertire i risultati di MediaPipe in un array di 1530 numeri 
# (di cui 63*3 MANO SX + 63*3 MANO DX + 468*3 VOLTO)
def convert_keypoints(mp_keypoints):

    # estraiamo i keypoints per ogni parte del corpo
    left_hand = mp_keypoints.left_hand_landmarks
    right_hand = mp_keypoints.right_hand_landmarks
    face = mp_keypoints.face_landmarks

    # formiamo la lista di coordinare per la MANO SINISTRA
    temp_list = []
    if left_hand:
        for res in left_hand.landmark:
            coordinates = [res.x, res.y, res.z]
            temp_list.append(coordinates)
        lh = np.array(temp_list).flatten()
    else:
        lh = np.zeros(21*3)
        
    # formiamo la lista di coordinare per la MANO DESTRA
    temp_list = []
    if right_hand:
        for res in right_hand.landmark:
            coordinates = [res.x, res.y, res.z]
            temp_list.append(coordinates)
        rh = np.array(temp_list).flatten()
    else:
        rh = np.zeros(21*3)
        
    # formiamo la lista di coordinare per il VOLTO
    temp_list = []
    if face:
        for res in face.landmark:
            coordinates = [res.x, res.y, res.z]
            temp_list.append(coordinates)
        face = np.array(temp_list).flatten()

    else:
        face = np.zeros(468*3)
        
    return np.concatenate([lh, rh, face])


if __name__ == "__main__":

    # estriamo solo i video che ci interessano
    my_videos = filter_dataset()
    
    # creiamo la cartella processed se non esiste (la useremo per salvare i file .npy con i dati estratti da ogni video)
    PROCESSED_DIR = '../data/processed/'
    os.makedirs(PROCESSED_DIR, exist_ok=True)
        
    # prendiamo la prima parola ("hello")
    test_word = TARGET_WORDS[0]
    
    if len(my_videos[test_word]) == 0:
        print(f"nessun video trovato per la parola {test_word}.")
        
    else:

        # prendiamo il primo video associato alla test_word
        test_video_filename = my_videos[test_word][0]
        
        # creiamo un nome intelligente per il salvataggio: es. "hello_00123.npy"
        video_path = os.path.join(VIDEO_DIR, test_video_filename)
        save_filename = f"{test_word}_{test_video_filename.replace('.mp4', '.npy')}"
        save_path = os.path.join(PROCESSED_DIR, save_filename)
                
        # estriamo i keypoints dal video e salviamo il risultato in un file .npy
        process_video(video_path, save_path)
        
        # controlliamo che il file sia stato salvato correttamente e che abbia la forma giusta (Frame, 1530 coordinate)
        if os.path.exists(save_path):

            dati_estratti = np.load(save_path)
            print(f"\nSUCCESSO! File salvato correttamente in: {save_path}")
            print(f"Formato dell'array (Frame, Coordinate): {dati_estratti.shape}")
            # il formato dovrebbe essere (Frame, 1530) ... se è diverso c'è un errore