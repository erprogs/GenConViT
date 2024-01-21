import os
import numpy as np
import cv2
import torch
import dlib
import face_recognition
from torchvision import transforms
from tqdm import tqdm
from dataset.loader import normalize_data
from .config import load_config
from .genconvit import GenConViT
from decord import VideoReader, cpu

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_genconvit(net, fp16):
    config = load_config()
    model = GenConViT(
        config,
        ed="genconvit_ed_inference",
        vae="genconvit_vae_inference",
        net=net,
        fp16=fp16
    )

    model.to(device)
    model.eval()
    if fp16:
        model.half()

    return model


def face_rec(frames, p=None, klass=None):
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0
    mod = "cnn" if dlib.DLIB_USE_CUDA else "hog"

    for _, frame in tqdm(enumerate(frames), total=len(frames)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(
            frame, number_of_times_to_upsample=0, model=mod
        )

        for face_location in face_locations:
            if count < len(frames):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                face_image = cv2.resize(
                    face_image, (224, 224), interpolation=cv2.INTER_AREA
                )
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                temp_face[count] = face_image
                count += 1
            else:
                break

    return ([], 0) if count == 0 else (temp_face[:count], count)


def preprocess_frame(frame):
    df_tensor = torch.tensor(frame, device=device).float()
    df_tensor = df_tensor.permute((0, 3, 1, 2))

    for i in range(len(df_tensor)):
        df_tensor[i] = normalize_data()["vid"](df_tensor[i] / 255.0)

    return df_tensor


def pred_vid(df, model):
    with torch.no_grad():
        return max_prediction_value(torch.sigmoid(model(df).squeeze()))

def max_prediction_value3(y_pred):
    if y_pred.dim() == 1 and y_pred.size(0) == 2:
        # When y_pred is a 1D tensor with two elements, 
        # no need to take the mean across the frames
        pred_label = torch.argmax(y_pred).item()
        pred_val = y_pred[pred_label].item()
        return (pred_label, pred_val)
    else:
        # Compute the mean value across the batch dimension (dim=0)
        mean_val = torch.mean(y_pred, dim=0)
        # Still, check if mean_val is not a 0-dimensional tensor, just to be safe
        if mean_val.dim() == 0:
            mean_val_val = mean_val.item()
            return (0, mean_val_val) if mean_val_val > 0.5 else (1, 1 - mean_val_val)
        pred_label = torch.argmax(mean_val).item()
        pred_val = mean_val[pred_label].item()
        return (pred_label, pred_val)

def real_or_fake(prediction):
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]


def extract_frames(video_file, frames_nums=15):
    vr = VideoReader(video_file, ctx=cpu(0))
    step_size = max(1, len(vr) // frames_nums)  # Calculate the step size between frames
    return vr.get_batch(
        list(range(0, len(vr), step_size))[:frames_nums]
    ).asnumpy()  # seek frames with step_size


def df_face(vid, num_frames, net):
    img = extract_frames(vid, num_frames)
    face, count = face_rec(img)
    return preprocess_frame(face) if count > 0 else []


def is_video(vid):
    return os.path.isfile(vid) and vid.endswith(
        tuple([".avi", ".mp4", ".mpg", ".mpeg", ".mov"])
    )


def set_result():
    return {
        "video": {
            "name": [],
            "pred": [],
            "klass": [],
            "pred_label": [],
            "correct_label": [],
        }
    }


def store_result(
    result, filename, y, y_val, klass, correct_label=None, compression=None
):
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(klass.lower())
    result["video"]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)

    if compression is not None:
        result["video"]["compression"].append(compression)

    return result
