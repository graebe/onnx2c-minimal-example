CC=gcc
CFLAGS=-lm

DIR=/workspaces/onnx2c-minimal-example/clib/MobilnetV2
_OBJ=MobilnetV2_main.c
OBJ = $(patsubst %,$(DIR)/%,$(_OBJ))

DIRDEP=/workspaces/onnx2c-minimal-example/exports/c
_DEPS = MobilnetV2_lib.c
DEPS = $(patsubst %,$(DIRDEP)/%,$(_DEPS))

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	$(CC) $(OBJ) $(CFLAGS) $(DEPS) -o MobilnetV2
