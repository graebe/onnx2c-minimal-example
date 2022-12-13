# Export Pytorch model as ONNX file
python models/FFNN_onnxexport.py

# Convert to C Code with onnx2c
mkdir -p exports/c >/dev/null 2>&1
onnx2c exports/onnx/FFNN.onnx > exports/c/FFNN.c

# Omit sample usage
python /workspaces/onnx2c-minimal-example/src/omit_lines_of_FFNN.py

# Make C-Code of ONNX model (doesn't use freshly converted model)
cp /workspaces/onnx2c-minimal-example/exports/c/FFNN_lib.c /workspaces/onnx2c-minimal-example/clib/FFNN/FFNN_lib.c
cd clib/FFNN
make

# Execute binary and print runtime
./FFNN
