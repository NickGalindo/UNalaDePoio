# UNalaDePoio

Propuesta UNalaDePoio al codefest 2<023

## Instalación

Instalar todas las librerìas del requirements.txt

## Uso - Objetivo 1 (video)

```bash
pip3 install accelerate async-generator attrs certifi charset-normalizer et-xmlfile exceptiongroup filelock fsspec h11 huggingface-hub idna joblib numpy openpyxl packaging pandas Pillow psutil PySocks python-dateutil python-dotenv pytz PyYAML regex requests scikit-learn scipy selenium six sniffio sortedcontainers threadpoolctl tokenizers torch torchaudio torchvision tqdm transformers trio trio-websocket typing_extensions tzdata urllib3 webdriver-manager wsproto

```



### Metricas de entrenamiento



## Uso - Objetivo 2 (texto)

```bash
pip3 install accelerate async-generator attrs certifi charset-normalizer et-xmlfile exceptiongroup filelock fsspec h11 huggingface-hub idna joblib numpy openpyxl packaging pandas Pillow psutil PySocks python-dateutil python-dotenv pytz PyYAML regex requests scikit-learn scipy selenium six sniffio sortedcontainers threadpoolctl tokenizers torch torchaudio torchvision tqdm transformers trio trio-websocket typing_extensions tzdata urllib3 webdriver-manager wsproto

```

1) Descargar el data.zip en el repositorio y descomprimir su contenido en la carpeta data/

2) Invocar los metodos solicitados:

 - ner_from_str(text, output_path) 

 - ner_from_file(text_path, output_path) 

 - ner_from_url(url, output_path) 


### Metricas de entrenamiento
Nuestro modelo de transformadores basados en transformadores, entrenado por nosotros con el objetivo de lograr una clasificación de categorías y una identificación precisa de entidades. Hemos invertido recursos significativos en ajustar nuestro modelo, así como en aumentar y mejorar la información de entrada para garantizar un rendimiento óptimo (aumentamos el data set de training). Nuestra solución sobresale en términos de precisión y confiabilidad, cumpliendo los objetivos propuestos. 

Basado en la url:

| Metric                    | Value                   |
|---------------------------|-------------------------|
| Test Loss                 | 1.2685312032699585      |
| Test Accuracy             | 0.7368421052631579      |
| Test F1 Score             | 0.741633393829401       |
| Test Precision            | 0.7615131578947368      |
| Test Recall               | 0.7368421052631579      |
| Test Runtime              | 0.1609                  |
| Test Samples per Second   | 236.111                 |
| Test Steps per Second     | 12.427                  |

Basado en un fragmento de texto:

| Metric                    | Value                   |
|---------------------------|-------------------------|
| Test Loss                 | 1.8285312032699585      |
| Test Accuracy             | 0.6216421052631579      |
| Test F1                   | 0.597433393829401       |
| Test Precision            | 0.5899631578947368      |
| Test Recall               | 0.6216221052631579      |
| Test Runtime              | 0.3934                  |
| Test Samples per Second   | 94.041                  |
| Test Steps per Second     | 7.625                   |

Al comparar los dos modelos basados en las tablas proporcionadas, podemos sacar las siguientes conclusiones:

Test Loss: El primer modelo logró una pérdida en la prueba más baja (1.2685) en comparación con el segundo modelo (1.8285). Esto indica que el primer modelo tiene un mejor rendimiento en términos de minimizar la desviación entre los valores predichos y los valores reales.

Test Accuracy : El primer modelo alcanzó una precisión en la prueba más alta (0.7368) en comparación con el segundo modelo (0.6216). Esto sugiere que el primer modelo tiene una mejor capacidad para predecir correctamente las etiquetas de clase de las muestras.

Test F1 : El primer modelo obtuvo una puntuación F1 más alta (0.7416) en comparación con el segundo modelo (0.5974). Una puntuación F1 más alta indica un mejor equilibrio entre precisión y exhaustividad, lo que sugiere que el primer modelo tiene un mejor rendimiento en términos de identificar correctamente las muestras positivas y minimizar los falsos positivos y los falsos negativos.

Test Precision: El primer modelo logró un valor de precisión más alto (0.7615) en comparación con el segundo modelo (0.5899). Esto indica que el primer modelo tiene una mejor capacidad para clasificar correctamente las predicciones positivas.

Test Recall: El primer modelo y el segundo modelo tienen valores similares de exhaustividad (0.7368 y 0.6216, respectivamente). Esta métrica representa la capacidad para identificar correctamente las muestras positivas dentro del total de muestras positivas reales.

Test Runtime: El primer modelo tiene un tiempo de ejecución en la prueba más bajo (0.1609) en comparación con el segundo modelo (0.3934). Esto sugiere que el primer modelo es más rápido en realizar predicciones.

Test Samples per Second: El primer modelo logró un mayor número de muestras procesadas por segundo (236.111) en comparación con el segundo modelo (94.041). Esto indica que el primer modelo tiene una mayor capacidad de predicción.

Test Steps per Second: El primer modelo completó un mayor número de pasos por segundo (12.427) en comparación con el segundo modelo (7.625). Esto sugiere que el primer modelo tiene una mayor eficiencia de procesamiento.

En resumen, el primer modelo supera al segundo modelo en términos de pérdida en la prueba, precisión, puntuación F1, precisión, tiempo de ejecución y velocidad de procesamiento. Demuestra un mejor rendimiento y eficiencia en realizar predicciones sobre datos de texto.

video
yolo version 4

## Creditos

Nicolas David Galindo Carvajal - ngalindoc@unal.edu.co
Juan Jacobo Izquierdo Becerra - juizquierdobc@unal.edu.co
Mateo Gutierrez Melo - mgutierrezca@unal.edu.co
Hernan David Mejia Galvis - cocoloc312@gmail.com
Natalia Andrea Quiroga Castillo - nquirogac@unal.edu.co

## Licencia

Esta librería se distribuye bajo la licencia MIT. Consulta el archivo LICENSE para obtener más información.

