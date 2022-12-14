CC=gcc
CFLAGS=-lm

DIR=/workspaces/onnx2c-minimal-example/clib/CNN_3layers
_OBJ=CNN_3layers_main.c
OBJ = $(patsubst %,$(DIR)/%,$(_OBJ))

DIRDEP=/workspaces/onnx2c-minimal-example/exports/c
_DEPS = CNN_3layers_lib.c
DEPS = $(patsubst %,$(DIRDEP)/%,$(_DEPS))

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	$(CC) $(OBJ) $(CFLAGS) $(DEPS) -o CNN_3layers
