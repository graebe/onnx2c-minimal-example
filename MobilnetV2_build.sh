# Export Pytorch model as ONNX file
python models/MobilnetV2_onnxexport.py

# Convert to C Code with onnx2c
mkdir -p exports/c >/dev/null 2>&1
onnx2c exports/onnx/MobilnetV2.onnx > exports/c/MobilnetV2.c

# Omit sample usage
python /workspaces/onnx2c-minimal-example/src/omit_lines_of_MobilnetV2.py

# Make C-Code of ONNX model (doesn't use freshly converted model)
cp /workspaces/onnx2c-minimal-example/exports/c/MobilnetV2_lib.c /workspaces/onnx2c-minimal-example/clib/MobilnetV2/MobilnetV2_lib.c
cd clib/MobilnetV2
make

# Execute binary and print runtime
./MobilnetV2
