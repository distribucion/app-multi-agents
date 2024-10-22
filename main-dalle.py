from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
import openai
import os
import getpass as gp
import requests
import shutil
from PIL import Image

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]


client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="un dragon medieval",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
print(image_url)

# guardar la imagen
foto = requests.get(image_url, stream=True)
print(foto)

if foto.status_code == 200:
    with open('foto ejemplo.png', 'wb')as f:
        shutil.copyfileobj(foto.raw, f)
else:
    print('Error al acceder a la imagen')
