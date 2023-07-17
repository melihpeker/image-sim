from transformers import AutoFeatureExtractor, AutoModel
import torchvision.transforms as T
import torch


model_ckpt = "google/vit-base-patch16-224"
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
hidden_dim = model.config.hidden_size

device = "cpu"  #"mps"

# Data transformation chain.
transformation_chain = T.Compose(
    [
        # We first resize the input image to 256x256 and then we take center crop.
        T.Resize(int((256 / 224) * extractor.size["height"])),
        T.CenterCrop(extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
    ]
)


def extract_embeddings(model: torch.nn.Module):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        try:
            images = batch["image"]
            image_batch_transformed = torch.stack(
                [transformation_chain(image) for image in images]
            )
            new_batch = {"pixel_values": image_batch_transformed.to(device)}
            with torch.no_grad():
                embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
            return {"embeddings": embeddings}
        except:
            return {"embeddings": torch.zeros([64,768])}

    return pp
