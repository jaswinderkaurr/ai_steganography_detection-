from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch, torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import io, os

app = FastAPI()

# Allow React frontend (Vite default runs on 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join("models", "resnet18.pt")
_model = None
_tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model weights not found.")
        m = resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Linear(m.fc.in_features, 2)
        state = torch.load(MODEL_PATH, map_location="cpu")
        m.load_state_dict(state)
        m.eval()
        _model = m
    return _model

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x = _tfm(img).unsqueeze(0)
        model = load_model()
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=1)[0].numpy()
        label = "stego" if float(prob[1]) > float(prob[0]) else "cover"
        return {"label": label, "p_cover": float(prob[0]), "p_stego": float(prob[1])}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not found at models/resnet18.pt. Train your model and place the file there.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")
