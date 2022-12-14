# Export Pytorch model as ONNX file
python models/CNN_minimal_onnxexport.py

# Convert to C Code with onnx2c
mkdir -p exports/c >/dev/null 2>&1
onnx2c exports/onnx/CNN_minimal.onnx > exports/c/CNN_minimal.c

# Omit sample usage
python /workspaces/onnx2c-minimal-example/src/omit_lines_of_CNN_minimal.py

# Make C-Code of ONNX model (doesn't use freshly converted model)
cp /workspaces/onnx2c-minimal-example/exports/c/CNN_minimal_lib.c /workspaces/onnx2c-minimal-example/clib/CNN_minimal/CNN_minimal_lib.c
cd clib/CNN_minimal
make

# Execute binary and print runtime
./CNN_minimal
