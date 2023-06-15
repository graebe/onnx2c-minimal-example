#include <stdio.h>
#include <time.h>
#include "GEMM_lib.c"

void entry(const float tensor_input[1][5], float tensor_output[1][3]);

int main () {
    clock_t tstart=0;
    clock_t tend=0;
    
    float input[1][5] = {{1.0, 2.0, 3.0, 4.0, 5.0}};
    float output[1][3] = {{0.0, 0.0, 0.0}};
    
    printf("\n------------------------------------\n");
    printf("| Executing C-Code Binary for GEMM |\n");
    printf("------------------------------------\n");
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

void entry(const float tensor_input[1][5], float tensor_output[1][3]) {
	node__l1_Gemm( tensor_input, tensor_l1_weight, tensor_l1_bias, tensor_output);
}
