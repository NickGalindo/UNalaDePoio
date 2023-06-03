from bs4 import BeautifulSoup
from lxml import etree
import requests

def __getInfo(url):
    """
    Obtener información de texto de una URL.

    Args:
        url (str): La URL de la página web.

    Returns:
        str: El texto extraído de la página web o "NOT FOUND" si no se encontró texto o se produjo un error.
    """
    try:
        HEADERS = ({'User-Agent':
                        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
                    (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36', 'Accept-Language': 'en-US, en;q=0.5'})

        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.content, 'html.parser')
        dom = etree.HTML(str(soup))
        xpath_str = '//h1|//h2|//h3|//p'  # The XPath to get titles an paragrpahs
        texto = ""
        for x in dom.xpath(xpath_str):
            textToAdd = ' '.join(x.itertext())
            if (textToAdd != None or textToAdd != ""):
                texto += textToAdd + " "
        if texto == "":
            return "NOT FOUND"
        return texto
    except:
        return "NOT FOUND"


def readUrlText(url: str):
    """
    Leer el texto de una URL.

    Args:
        url (str): La URL de la página web.

    Returns:
        str: El texto extraído de la página web o "NOT FOUND" si no se encontró texto o se produjo un error.
    """
    text = __getInfo(url)

    return text

