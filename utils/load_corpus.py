import requests
from loguru import logger

class LoadCorpus:
    """Loads a text file from a URL and reads it into a text file."""
    def __init__(self, url):
        logger.info("Loading tinyshakespeare.txt")
        self.write_file(url)
        self.text = self.read_file()
        
    def write_file(self, url):
        response = requests.get(url)
        with open("tinyshakespeare.txt", "wb") as file:
            file.write(response.content)
    
    def read_file(self):
        with open("tinyshakespeare.txt", "r", encoding="utf-8") as file:
            text = file.read()
        return text