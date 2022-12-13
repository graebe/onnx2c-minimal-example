#include <stdio.h>
#include <time.h>
#include "FFNN_lib.c"

void entry(const float tensor_input[1][1], float tensor_output[1][1]);

int main () {
    clock_t tstart=0;
    clock_t tend=0;
    float input[1][1] = {{3}};
    float output[1][1] = {{0}};
    
    printf("\n-----------------------------------\n");
    printf("| Executing C-Code Binary for FFNN |\n");
    printf("-----------------------------------\n");
    //printf("   Values before execution:\n");
    //printf("      Input: %f, Output: %f\n", input[0][0], output[0][0]);
    tstart = clock();
    entry(input, output);
    tend = clock();
    //printf("   Values after execution:\n");
    //printf("      Input: %f, Output: %f\n", input[0][0], output[0][0]);
    printf("    Wall Time: %lf mus\n\n",(double)(tend-tstart)/CLOCKS_PER_SEC*1000000);
}

void entry(const float tensor_input[1][1], float tensor_output[1][1]) {
	node__l1_Gemm( tensor_input, tensor_l1_weight, tensor_l1_bias, tu0.tensor__l1_Gemm_output_0);
	node__a1_Tanh( tu0.tensor__l1_Gemm_output_0, tu1.tensor__a1_Tanh_output_0);
	node__l2_Gemm( tu1.tensor__a1_Tanh_output_0, tensor_l2_weight, tensor_l2_bias, tensor_output);
}
