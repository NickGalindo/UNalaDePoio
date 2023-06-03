from bs4 import BeautifulSoup
from lxml import etree
import requests

def __getInfo(url):
    """
    Obtener información de texto de una URL.

    Args:
        url (str): The URL of the web page.

    Returns:
        str: The extracted text from the web page or "NOT FOUND" if no text was found or an error occurred.
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
    Read the text from a URL.

    Args:
        url (str): The URL of the web page.

    Returns:
        str: The extracted text from the web page or "NOT FOUND" if no text was found or an error occurred.
    """
    text = __getInfo(url)

    return text

