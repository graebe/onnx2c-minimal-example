# Export Pytorch model as ONNX file
python export_pytorch_graph_to_onnx.py

# Convert to C Code with onnx2c
mkdir exports/c
onnx2c exports/onnx/model.onnx > exports/c/model.c
