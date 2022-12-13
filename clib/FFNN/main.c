#include <stdio.h>
#include "model.c"

void entry(const float tensor_input[1][1], float tensor_output[1][1]);

int main () {
    float input[1][1] = {{3}};
    float output[1][1] = {{0}};
    
    printf("Executing NN.\n");
    printf("   Values before execution:\n");
    printf("      Input: %f, Output: %f\n", input[0][0], output[0][0]);
    entry(input, output);
    printf("   Values after execution:\n");
    printf("      Input: %f, Output: %f\n", input[0][0], output[0][0]);
}

void entry(const float tensor_input[1][1], float tensor_output[1][1]) {
	node__l1_Gemm( tensor_input, tensor_l1_weight, tensor_l1_bias, tu0.tensor__l1_Gemm_output_0);
	node__a1_Tanh( tu0.tensor__l1_Gemm_output_0, tu1.tensor__a1_Tanh_output_0);
	node__l2_Gemm( tu1.tensor__a1_Tanh_output_0, tensor_l2_weight, tensor_l2_bias, tensor_output);
}
