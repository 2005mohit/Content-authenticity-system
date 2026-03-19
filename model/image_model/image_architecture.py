device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = 'image_model.pth'

state_dict = torch.load(MODEL_PATH, map_location=device)
classifier_shape = state_dict['classifier.1.weight'].shape[0]

def build_model(neurons):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3 if neurons == 256 else 0.4),
        nn.Linear(in_features, neurons),
        nn.ReLU(),
        nn.Dropout(p=0.2 if neurons == 256 else 0.3),
        nn.Linear(neurons, 2)
    )
    return model.to(device)

model = build_model(classifier_shape)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded!")

# ── Transform ────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
