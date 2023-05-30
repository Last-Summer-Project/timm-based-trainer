from model import TimmBasedClassifierModel
from cfg import Config
import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description='Predict an image by model.')
    parser.add_argument("--model", type=str, help="Model", required=True)
    args = parser.parse_args()

    model = TimmBasedClassifierModel().load_from_checkpoint(args.model)
    print(model)
    filepath = "model.onnx"
    model.to_onnx(filepath,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                  input_sample=torch.zeros(4, 3, 500, 500))


if __name__ == "__main__":
    main()
