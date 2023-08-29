# Export Pytorch model as ONNX file
python models/LSTM_onnxexport.py

# Convert to C Code with onnx2c
mkdir -p exports/c >/dev/null 2>&1
onnx2c exports/onnx/LSTM.onnx > exports/c/LSTM.c

# Omit sample usage
python /workspaces/onnx2c-minimal-example/src/omit_lines_of_LSTM.py

# Make C-Code of ONNX model (doesn't use freshly converted model)
cp /workspaces/onnx2c-minimal-example/exports/c/LSTM_lib.c /workspaces/onnx2c-minimal-example/clib/LSTM/LSTM_lib.c
cd clib/LSTM
make

# Execute binary and print runtime
./LSTM
