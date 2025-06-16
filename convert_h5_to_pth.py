import os
import h5py
import torch
import tf2onnx  # add fallback for ONNX conversion
import tensorflow as tf  # required for ONNX conversion  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from model.deblurgan import DeblurGANv2Generator

def convert(h5_path="fpn_mobilenet.h5", pth_path="model/weights.pth"):
    # Debug: print header bytes to help diagnose format issues
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"HDF5 file '{h5_path}' not found.\n"
            "Please download 'fpn_mobilenet.h5' from DeblurGANv2's Pre-trained models and place it here."
        )
    with open(h5_path, 'rb') as fh:
        header = fh.read(8)
    print(f"Header bytes of '{h5_path}': {header}")
    # Check for Python pickle signature (.pth saved as .h5)
    if header.startswith(b'\x80\x02'):
        # Fallback: file is a PyTorch pickle, load weights directly
        print(f"Detected PyTorch pickle format in '{h5_path}'. Loading state dict...")
        model = DeblurGANv2Generator()
        state = torch.load(h5_path, map_location='cpu')
        # If nested under 'generator' key, unwrap
        if isinstance(state, dict) and 'generator' in state:
            sd = state['generator']
        elif isinstance(state, dict):
            sd = state
        else:
            raise RuntimeError(f"Unrecognized pickle content in '{h5_path}'.")
        model.load_state_dict(sd, strict=False)
        return model
    # Validate HDF5 magic number
    if header != b'\x89HDF\r\n\x1a\n':
        raise RuntimeError(
            f"Invalid HDF5 signature: {header}. The file may be corrupted or not HDF5.\n"
            "Re-download 'fpn_mobilenet.h5' and ensure it's a valid Keras H5 checkpoint."
        )
    # Open with h5py
    f = h5py.File(h5_path, 'r')
    # 2. Instantiate the PyTorch generator
    model = DeblurGANv2Generator()
    # 3. Build a new state_dict by mapping layer names.
    #    You’ll need to inspect f['model_weights'] to see the layer names,
    #    but for example:
    sd = {}
    for name, param in model.named_parameters():
        # strip prefixes to match f.keys(), e.g. name="conv_first.0.weight" → "conv/0"
        keras_key = name.replace('.', '/')
        if keras_key in f['model_weights']:
            w = f['model_weights'][keras_key][...]
            sd[name] = torch.from_numpy(w).permute(3,2,0,1)  # H5 is (H,W,in_c,out_c)
    # 4. Load into the PyTorch model (allowing missing keys)
    model.load_state_dict(sd, strict=False)
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', choices=['pth','onnx'], default='pth', help='Output format')
    parser.add_argument('--h5_path', type=str, default='fpn_mobilenet.h5', help='Input HDF5 file')
    parser.add_argument('--output_dir', type=str, default='model', help='Output directory')
    args = parser.parse_args()

    h5_path = args.h5_path
    output_dir = args.output_dir

    if args.format == 'onnx':
        # Load Keras model for ONNX conversion
        keras_model = load_model(h5_path)
        onnx_path = os.path.join(output_dir, 'fpn_mobilenet.onnx')
        print(f"Converting to ONNX at {onnx_path}...")
        spec = (tf.TensorSpec(keras_model.input.shape, tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, output_path=onnx_path)
        print("ONNX model saved.")
        return

    else:
        # PyTorch conversion to .pth
        pth_model = convert(h5_path)
        pth_path = os.path.join(output_dir, 'weights.pth')
        os.makedirs(os.path.dirname(pth_path), exist_ok=True)
        torch.save({'generator': pth_model.state_dict()}, pth_path)
        print(f"Converted {h5_path} → {pth_path}")

if __name__ == "__main__":
    main()