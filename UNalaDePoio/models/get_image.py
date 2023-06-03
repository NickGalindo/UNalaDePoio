import cv2

# Ruta del video de entrada
video_path = r'C:\Users\USUARIO\Desktop\codefest\UNalaDePoio\data\codefest_data\codefest_data\Videos\VideoCodefest_008-46min.mpg'  # Reemplaza 'video.mp4' con la ruta de tu archivo de video

# Iniciar la captura de video
cap = cv2.VideoCapture(video_path)

# Contador para controlar los fotogramas tomados
frame_count = 0

# Número de fotogramas a saltar antes de tomar una imagen
frames_to_skip = 100  # Reemplaza con el número deseado de fotogramas a saltar

# Procesar cada fotograma del video
while cap.isOpened():
    # Leer el siguiente fotograma
    ret, frame = cap.read()

    if ret:
        # Incrementar el contador de fotogramas
        frame_count += 1

        # Verificar si es el momento de tomar una imagen
        if frame_count % frames_to_skip == 0:
            # Guardar la imagen en disco
            cv2.imwrite(fr'C:\Users\USUARIO\Desktop\codefest\UNalaDePoio\data\imagesJacobo\imagen_{frame_count}.jpg', frame)

        # Mostrar el fotograma en una ventana
        cv2.imshow('Video', frame)

        # Esperar la tecla 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()