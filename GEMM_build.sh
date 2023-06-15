# Export Pytorch model as ONNX file
python models/GEMM_onnxexport.py

# Convert to C Code with onnx2c
mkdir -p exports/c >/dev/null 2>&1
onnx2c exports/onnx/GEMM.onnx > exports/c/GEMM.c

# Omit sample usage
python /workspaces/onnx2c-minimal-example/src/omit_lines_of_GEMM.py

# Make C-Code of ONNX model (doesn't use freshly converted model)
cp /workspaces/onnx2c-minimal-example/exports/c/GEMM_lib.c /workspaces/onnx2c-minimal-example/clib/GEMM/GEMM_lib.c
cd clib/GEMM
make

# Execute binary and print runtime
./GEMM
