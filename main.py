import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
nlp = spacy.load("en_core_web_lg")


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    vector = nlp(request.word).vector
    return {"word": request.word, "embedding": vector.tolist()}


# ---- CNN image classifier ----
import io
import torch
from fastapi import UploadFile, File
from PIL import Image
from torchvision import transforms
from app.cnn import CNN

class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

cnn = CNN()
cnn.load_state_dict(torch.load("app/cnn.pth", map_location="cpu"))
cnn.eval()

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = image_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = cnn(x)
    predicted = output.argmax(dim=1).item()
    return {"class": class_names[predicted]}


import io
import torch
from fastapi.responses import StreamingResponse
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from app.gan import Generator

gan_generator = Generator()
gan_generator.load_state_dict(torch.load("app/gan_generator.pth", map_location="cpu"))
gan_generator.eval()


@app.get("/generate_digits")
def generate_digits(n: int = 16):
    with torch.no_grad():
        z = torch.randn(n, 100)
        imgs = (gan_generator(z) + 1) / 2
    grid = make_grid(imgs, nrow=4)
    buf = io.BytesIO()
    to_pil_image(grid).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
