import torch
import argparse
import os


def clean_and_save_weights(tar_path):
    checkpoint = torch.load(tar_path, map_location="cpu")

    if "model_state_dict" not in checkpoint or not isinstance(checkpoint["model_state_dict"], dict):
        print("The .tar file is not a standard PyTorch checkpoint containing model_state_dict.")
        return

    original_state_dict = checkpoint["model_state_dict"]
    cleaned_state_dict = {}

    for key, value in original_state_dict.items():
        if key.startswith("projector"):
            continue
        # elif key.startswith("backbone."):
        #     new_key = key[len("backbone."):]
        #     cleaned_state_dict[new_key] = value
        else:
            cleaned_state_dict[key] = value

    # # new checkpoint
    # new_checkpoint = {
    #     "model_state_dict": cleaned_state_dict
    # }

    dirname = os.path.dirname(tar_path)
    new_filename = os.path.join(dirname, "backbone.pth")

    torch.save(cleaned_state_dict, new_filename)
    print(f"\n✅ Saved clean weights to: {new_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and save ResNet weights from a .tar checkpoint file")
    parser.add_argument("path", type=str, help="Path to the .tar file")
    args = parser.parse_args()

    clean_and_save_weights(args.path)