import os
import argparse
import json
from time import perf_counter
from datetime import datetime
from model.pred_func import *
from model.config import load_config

config = load_config()
print('CONFIG')
print(config)
def vids(
    ed_weight, vae_weight, root_dir="sample_prediction_data", dataset=None, num_frames=15, net=None, fp16=False
):
    result = set_result()
    r = 0
    f = 0
    count = 0
    
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    for filename in os.listdir(root_dir):
        curr_vid = os.path.join(root_dir, filename)

        try:
            if is_video(curr_vid):
                result, accuracy, count, pred = predict(
                    curr_vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "uncategorized",
                    count,
                )
                f, r = (f + 1, r) if "FAKE" == real_or_fake(pred[0]) else (f, r + 1)
                print(
                    f"Prediction: {pred[1]} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}"
                )
            else:
                print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return result


def faceforensics(
    ed_weight, vae_weight, root_dir="FaceForensics\\data", dataset=None, num_frames=15, net=None, fp16=False
):
    vid_type = ["original_sequences", "manipulated_sequences"]
    result = set_result()
    result["video"]["compression"] = []
    ffdirs = [
        "DeepFakeDetection",
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    ]

    # load files not used in the training set, the files are appended with compression type, _c23 or _c40
    with open(os.path.join("json_file", "ff_file_list.json")) as j_file:
        ff_file = list(json.load(j_file))

    count = 0
    accuracy = 0
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    for v_t in vid_type:
        for dirpath, dirnames, filenames in os.walk(os.path.join(root_dir, v_t)):
            klass = next(
                filter(lambda x: x in dirpath.split(os.path.sep), ffdirs),
                "original",
            )
            label = "REAL" if klass == "original" else "FAKE"
            for filename in filenames:
                try:
                    if filename in ff_file:
                        curr_vid = os.path.join(dirpath, filename)
                        compression = "c23" if "c23" in curr_vid else "c40"
                        if is_video(curr_vid):
                            result, accuracy, count, _ = predict(
                                curr_vid,
                                model,
                                fp16,
                                result,
                                num_frames,
                                net,
                                klass,
                                count,
                                accuracy,
                                label,
                                compression,
                            )
                        else:
                            print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

                except Exception as e:
                    print(f"An error occurred: {str(e)}")

    return result


def timit(ed_weight, vae_weight, root_dir="DeepfakeTIMIT", dataset=None, num_frames=15, net=None, fp16=False):
    keywords = ["higher_quality", "lower_quality"]
    result = set_result()
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    count = 0
    accuracy = 0
    i = 0
    for keyword in keywords:
        keyword_folder_path = os.path.join(root_dir, keyword)
        for subfolder_name in os.listdir(keyword_folder_path):
            subfolder_path = os.path.join(keyword_folder_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                # Loop through the AVI files in the subfolder
                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".avi"):
                        curr_vid = os.path.join(subfolder_path, filename)
                        try:
                            if is_video(curr_vid):
                                result, accuracy, count, _ = predict(
                                    curr_vid,
                                    model,
                                    fp16,
                                    result,
                                    num_frames,
                                    net,
                                    "DeepfakeTIMIT",
                                    count,
                                    accuracy,
                                    "FAKE",
                                )
                            else:
                                print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

                        except Exception as e:
                            print(f"An error occurred: {str(e)}")

    return result


def dfdc(
    ed_weight,
    vae_weight,
    root_dir="deepfake-detection-challenge\\train_sample_videos",
    dataset=None,
    num_frames=15,
    net=None,
    fp16=False,
):
    result = set_result()
    if os.path.isfile(os.path.join("json_file", "dfdc_files.json")):
        with open(os.path.join("json_file", "dfdc_files.json")) as data_file:
            dfdc_data = json.load(data_file)

    if os.path.isfile(os.path.join(root_dir, "metadata.json")):
        with open(os.path.join(root_dir, "metadata.json")) as data_file:
            dfdc_meta = json.load(data_file)
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    count = 0
    accuracy = 0
    for dfdc in dfdc_data:
        dfdc_file = os.path.join(root_dir, dfdc)

        try:
            if is_video(dfdc_file):
                result, accuracy, count, _ = predict(
                    dfdc_file,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "dfdc",
                    count,
                    accuracy,
                    dfdc_meta[dfdc]["label"],
                )
            else:
                print(f"Invalid video file: {dfdc_file}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return result


def celeb(ed_weight, vae_weight, root_dir="Celeb-DF-v2", dataset=None, num_frames=15, net=None, fp16=False):
    with open(os.path.join("json_file", "celeb_test.json"), "r") as f:
        cfl = json.load(f)
    result = set_result()
    ky = ["Celeb-real", "Celeb-synthesis"]
    count = 0
    accuracy = 0
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    for ck in cfl:
        ck_ = ck.split("/")
        klass = ck_[0]
        filename = ck_[1]
        correct_label = "FAKE" if klass == "Celeb-synthesis" else "REAL"
        vid = os.path.join(root_dir, ck)

        try:
            if is_video(vid):
                result, accuracy, count, _ = predict(
                    vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    klass,
                    count,
                    accuracy,
                    correct_label,
                )
            else:
                print(f"Invalid video file: {vid}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred x: {str(e)}")

    return result


def predict(
    vid,
    model,
    fp16,
    result,
    num_frames,
    net,
    klass,
    count=0,
    accuracy=-1,
    correct_label="unknown",
    compression=None,
):
    count += 1
    print(f"\n\n{str(count)} Loading... {vid}")

    df = df_face(vid, num_frames, net)  # extract face from the frames
    if fp16:
        df.half()
    y, y_val = (
        pred_vid(df, model)
        if len(df) >= 1
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )
    result = store_result(
        result, os.path.basename(vid), y, y_val, klass, correct_label, compression
    )

    if accuracy > -1:
        if correct_label == real_or_fake(y):
            accuracy += 1
        print(
            f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy/count}"
        )

    return result, accuracy, count, [y, y_val]


def gen_parser():
    parser = argparse.ArgumentParser("GenConViT prediction")
    parser.add_argument("--p", type=str, help="video or image path")
    parser.add_argument(
        "--f", type=int, help="number of frames to process for prediction"
    )
    parser.add_argument(
        "--d", type=str, help="dataset type, dfdc, faceforensics, timit, celeb"
    )
    parser.add_argument(
        "--s", help="model size type: tiny, large.",
    )
    parser.add_argument(
        "--e", nargs='?', const='genconvit_ed_inference', default='genconvit_ed_inference', help="weight for ed.",
    )
    parser.add_argument(
        "--v", '--value', nargs='?', const='genconvit_vae_inference', default='genconvit_vae_inference', help="weight for vae.",
    )
    
    parser.add_argument("--fp16", type=str, help="half precision support")

    args = parser.parse_args()
    path = args.p
    num_frames = args.f if args.f else 15
    dataset = args.d if args.d else "other"
    fp16 = True if args.fp16 else False

    net = 'genconvit'
    ed_weight = 'genconvit_ed_inference'
    vae_weight = 'genconvit_vae_inference'

    if args.e and args.v:
        ed_weight = args.e
        vae_weight = args.v
    elif args.e:
        net = 'ed'
        ed_weight = args.e
    elif args.v:
        net = 'vae'
        vae_weight = args.v
    
        
    print(f'\nUsing {net}\n')  
    

    if args.s:
        if args.s in ['tiny', 'large']:
            config["model"]["backbone"] = f"convnext_{args.s}"
            config["model"]["embedder"] = f"swin_{args.s}_patch4_window7_224"
            config["model"]["type"] = args.s
    
    return path, dataset, num_frames, net, fp16, ed_weight, vae_weight


def main():
    start_time = perf_counter()
    path, dataset, num_frames, net, fp16, ed_weight, vae_weight = gen_parser()
    result = (
        globals()[dataset](ed_weight, vae_weight, path, dataset, num_frames, net, fp16)
        if dataset in ["dfdc", "faceforensics", "timit", "celeb"]
        else vids(ed_weight, vae_weight, path, dataset, num_frames, net, fp16)
    )

    curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
    file_path = os.path.join("result", f"prediction_{dataset}_{net}_{curr_time}.json")

    with open(file_path, "w") as f:
        json.dump(result, f)
    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()
