#include <stdio.h>
#include <time.h>
#include "LSTM_lib.c"

void entry(const float tensor_input[10][1][5], const float tensor_h_in[1][1][3], const float tensor_c_in[1][1][3], float tensor_output[10][1][3], float tensor_h_out[1][1][3], float tensor_c_out[1][1][3]);

int main () {
    clock_t tstart=0;
    clock_t tend=0;
    
    const float tensor_input[10][1][5] = {
        {{1.0, 2.0, 3.0, 4.0, 5.0}},
        {{1.0, 2.0, 3.0, 4.0, 5.0}},
        {{1.0, 2.0, 3.0, 4.0, 5.0}},
        {{1.0, 2.0, 3.0, 4.0, 5.0}},
        {{1.0, 2.0, 3.0, 4.0, 5.0}},
        {{1.0, 2.0, 3.0, 4.0, 5.0}},
        {{1.0, 2.0, 3.0, 4.0, 5.0}},
        {{1.0, 2.0, 3.0, 4.0, 5.0}},
        {{1.0, 2.0, 3.0, 4.0, 5.0}},
        {{1.0, 2.0, 3.0, 4.0, 5.0}}
    };
    const float tensor_h_in[1][1][3] = {{{1.0, 1.0, 1.0}}};
    const float tensor_c_in[1][1][3] = {{{1.0, 1.0, 1.0}}};
    float tensor_output[10][1][3] = {
        {{1.0, 2.0, 3.0}},
        {{1.0, 2.0, 3.0}},
        {{1.0, 2.0, 3.0}},
        {{1.0, 2.0, 3.0}},
        {{1.0, 2.0, 3.0}},
        {{1.0, 2.0, 3.0}},
        {{1.0, 2.0, 3.0}},
        {{1.0, 2.0, 3.0}},
        {{1.0, 2.0, 3.0}},
        {{1.0, 2.0, 3.0}},
        };
    float tensor_h_out[1][1][3] = {{{1.0, 1.0, 1.0}}};
    float tensor_c_out[1][1][3] = {{{1.0, 1.0, 1.0}}};
    printf("\n------------------------------------\n");
    printf("| Executing C-Code Binary for GEMM |\n");
    printf("------------------------------------\n");
    printf("   Values before execution:\n");
    printf("      Input: %lf\n", tensor_input[0][0][0]);
	printf("      Output: %lf\n", tensor_output[0][0][0]);
    tstart = clock();
    entry(tensor_input, tensor_h_in, tensor_c_in, tensor_output, tensor_h_out, tensor_c_out);
    tend = clock();
    printf("   Values after execution:\n");
    printf("      Input: %lf\n", tensor_input[0][0][0]);
	printf("      Output: %lf\n", tensor_output[0][0][0]);
    printf("    Wall Time: %lf mus\n\n",(double)(tend-tstart)/CLOCKS_PER_SEC*1000000);
}


void entry(const float tensor_input[10][1][5], const float tensor_h_in[1][1][3], const float tensor_c_in[1][1][3], float tensor_output[10][1][3], float tensor_h_out[1][1][3], float tensor_c_out[1][1][3]) {
	node__l_LSTM( tensor_input, tensor_onnx__LSTM_89, tensor_onnx__LSTM_90, tensor_onnx__LSTM_91, tensor_h_in, tensor_c_in, tu0.tensor__l_LSTM_output_0, tensor_h_out, tensor_c_out);
	node__l_Constant( tensor__l_Constant_output_0);
	node__l_Squeeze( tu0.tensor__l_LSTM_output_0, tensor_output);
}
