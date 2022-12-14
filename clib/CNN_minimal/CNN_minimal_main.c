#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "CNN_minimal_lib.c"

void entry(const float tensor_input[1][1][16][10], float tensor_output[1][1]);

int main () {
    clock_t tstart=0;
    clock_t tend=0;
	float output[1][1] = {{-1.0}};
    float input[1][1][16][10];
    for (int i=0; i<1; i++) {
        for (int j=0; j<16; j++) {
            for (int k=0; k<10; k++) {
                input[0][i][j][k] = rand();
            }
        }
    }
    
    printf("\n------------------------------------------\n");
    printf("| Executing C-Code Binary for MobilnetV2 |\n");
    printf("------------------------------------------\n");
    printf("   Values before execution:\n");
    printf("      Input: %lf\n", input[0][0]);
	printf("      Output: %lf\n", output[0][0]);
    tstart = clock();
    entry(input, output);
    tend = clock();
    printf("   Values after execution:\n");
    printf("      Input: %lf\n", input[0][0]);
	printf("      Output: %lf\n", output[0][0]);
    printf("    Wall Time: %lf mus\n\n",(double)(tend-tstart)/CLOCKS_PER_SEC*1000000);
}


void entry(const float tensor_input[1][1][16][10], float tensor_output[1][1]) {
	node__main_main_0_Conv( tensor_input, tensor_main_0_weight, tensor_main_0_bias, tu0.tensor__main_main_0_Conv_output_0);
	node__main_main_1_Relu( tu0.tensor__main_main_0_Conv_output_0, tu1.tensor__main_main_1_Relu_output_0);
	node__main_main_2_Flatten( tu1.tensor__main_main_1_Relu_output_0, tu0.tensor__main_main_2_Flatten_output_0);
	node__main_main_3_Gemm( tu0.tensor__main_main_2_Flatten_output_0, tensor_main_3_weight, tensor_main_3_bias, tensor_output);
}
