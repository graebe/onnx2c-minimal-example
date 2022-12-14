def omit_lines_of_file(sourcefile, targetfile, N):
    with open(sourcefile) as f1:
        lines = f1.readlines()
    with open(targetfile, 'w') as f2:
        f2.writelines(lines[:-N])

omit_lines_of_file(sourcefile="/workspaces/onnx2c-minimal-example/exports/c/MobilnetV2.c",
                   targetfile="/workspaces/onnx2c-minimal-example/exports/c/MobilnetV2_lib.c",
                   N=173)