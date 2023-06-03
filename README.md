# UNalaDePoio

Propuesta UNalaDePoio al codefest 2023

## Instalación

1) Clonar y entrar al proyecto

```bash
git clone https://github.com/NickGalindo/UNalaDePoio.git
cd UNalaDePoio
```

2) Inicializar virtualenv:

```bash
python3 -m venv env
source env/bin/activate
```

3) Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

#### NOTE: Instalar la version adecuada de pytorch segun su sistema operativo y hardware. Para mas informacion visitar: https://pytorch.org/get-started/locally/

4) Del bucket de s3 copiar los modelos y descromprimirlos en sus respectivas carpetas

```bash
# Descargar modelo de texto
aws s3 cp s3://codefest-team-unaladepoio/data_text.zip ./text/
unzip ./text/data_text.zip

# Descargar modelo de video
aws s3 cp s3://codefest-team-unaladepoio/data_video.zip ./video/
unzip ./video/data_video.zip
```

5) Instalar el UNalaDePoio package

```bash
python -m pip install -e .
```

6) Para cualquier uso futuro basta con tener el virtual env que se creó activado e importar el package como se muestra a continuación:

```python
import UNalaDePoio
```

## Uso - Objetivo 1 (video)

Importar la libreria y e invocar el metodo solicitado:


```python
from UNalaDePoio import *
detect_objects_in_video(video_path, output_path)
```

### Descripciòn general

Para el procesamiento de videos se uso un modelo OpenSource preentrenado llamado YoloV4, en donde cada 30 frames se procesan las imagenes para detectar los objetos, vehiculos, construcciones, etc. El modelo es capaz de detectar más de 80 objetos, asi mismo se uso el modelo easyocr el cual detecta las letras y números de cada frame que contiene un objeto. Finalmente, se guardan las imagenes de cada frame clasificado y se genera un archivo .csv con la información anterior y la hora (militar)


¡Feliz procesamiento de videos!


## Uso - Objetivo 2 (texto)

Importar la libreria y e invocar los metodo solicitados:


```python
from UNalaDePoio import *
# A partir de un texto
ner_from_str(text, output_path) 
# A partir de un archivo
ner_from_file(text_path, output_path) 
# A partir de una url
ner_from_url(url, output_path)
```

### Descripciòn general

Para el análisis de textos se utilizó un modelo basado en transformadores, entrenado por nosotros con el objetivo de lograr una clasificación de categorías y una identificación precisa de entidades; invirtiendo recursos significativos para ajustar nuestro modelo, así como en aumentar y mejorar la información de entrada para garantizar un rendimiento óptimo (aumentamos el data set de training). Nuestra solución sobresale en términos de precisión y confiabilidad, cumpliendo los objetivos propuestos. 

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
| Test Loss                 | 1.1285312032699585      |
| Test Accuracy             | 0.7116421052631579      |
| Test F1                   | 0.727433393829401       |
| Test Precision            | 0.6299631578947368      |
| Test Recall               | 0.6416221052631579      |
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

#### NOTA: En una prueba el modelo se entreno usando los dos datasets (urls, textos) y se obtuvo un accuracy de 0.78, sin embargo, por requerimientos de la prueba se decidio entrenar el modelo con los datasets aparte.


## Creditos

Nicolas David Galindo Carvajal - ngalindoc@unal.edu.co
Juan Jacobo Izquierdo Becerra - juizquierdobc@unal.edu.co
Mateo Gutierrez Melo - mgutierrezca@unal.edu.co
Hernan David Mejia Galvis - cocoloc312@gmail.com
Natalia Andrea Quiroga Castillo - nquirogac@unal.edu.co

## Licencia

Esta librería se distribuye bajo la licencia MIT. Consulta el archivo LICENSE para obtener más información.
