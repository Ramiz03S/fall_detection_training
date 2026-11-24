import os
from dotenv import load_dotenv
import numpy as np
from pathlib import Path
from ai_edge_litert.interpreter import Interpreter

def wrap(items, per_line=10, indent="  "):                  
    lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return indent + (",\n" + indent).join(lines)

def floats(a):                                              
    return [f"{float(v):.9e}f" for v in np.asarray(a, dtype=np.float32).ravel()]

def ints(a):                                                
    return [str(int(v)) for v in np.asarray(a).ravel()]

def main():
    load_dotenv() 
    norm_stats_path = Path(os.environ.get("BEST_MODEL_NORM_STATS_PATH"))
    quantize_export_path = Path(os.environ.get("QUANTIZE_EXPORT_PATH"))
    esp32_export_path = Path(os.environ.get("ESP32_EXPORT_PATH"))
    esp32_export_path.mkdir(parents=True, exist_ok=True)

    norm_stats = np.load(norm_stats_path)
    mean = norm_stats['mean'] # (1, 6, 1, 1)
    std = norm_stats['std'] # (1, 6, 1, 1)

    m = mean.ravel()
    s = std.ravel()  

    model_bytes = (quantize_export_path/"tflite_format"/"int8.tflite").read_bytes()          
    hexes = [f"0x{b:02x}" for b in model_bytes]                     

    (esp32_export_path / "model_data.h").write_text(                             
        "#pragma once\n"
        "extern const unsigned char g_model[];\n"
        "extern const unsigned int g_model_len;\n")

    (esp32_export_path / "model_data.cc").write_text(                             
        '#include "model_data.h"\n\n'
        "alignas(16) const unsigned char g_model[] = {\n"
        + wrap(hexes, per_line=12) + "\n};\n"
        f"const unsigned int g_model_len = {len(model_bytes)};\n")

    (esp32_export_path / "norm_params.h").write_text(
        "#pragma once\n"
        f"static constexpr float kMean[6] = {{\n{wrap(floats(m))}\n}};\n"
        f"static constexpr float kStd[6] = {{\n{wrap(floats(s))}\n}};\n")


if __name__ == "__main__":
    main()
