# Export Pytorch model as ONNX file
python models/CNN_3layers_onnxexport.py

# Convert to C Code with onnx2c
mkdir -p exports/c >/dev/null 2>&1
onnx2c exports/onnx/CNN_3layers.onnx > exports/c/CNN_3layers.c

# Omit sample usage
python /workspaces/onnx2c-minimal-example/src/omit_lines_of_CNN_3layers.py

# Make C-Code of ONNX model (doesn't use freshly converted model)
cp /workspaces/onnx2c-3layers-example/exports/c/CNN_3layers_lib.c /workspaces/onnx2c-3layers-example/clib/CNN_3layers/CNN_3layers_lib.c
cd clib/CNN_3layers
make

# Execute binary and print runtime
./CNN_3layers
