#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "MobilnetV2_lib.c"

void entry(const float tensor_input[1][3][512][120], float tensor_output[1][1]);

int main () {
    clock_t tstart=0;
    clock_t tend=0;
	float output[1][1] = {{-1.0}};
    float input[1][3][512][120];
    for (int i=0; i<3; i++) {
        for (int j=0; j<512; j++) {
            for (int k=0; k<120; k++) {
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

void entry(const float tensor_input[1][3][512][120], float tensor_output[1][1]) {
	node__features_features_0_features_0_0_Conv( tensor_input, tensor_onnx__Conv_538, tensor_onnx__Conv_539, tu0.tensor__features_features_0_features_0_0_Conv_output_0);
	node__features_features_0_features_0_2_Constant( tensor__features_features_0_features_0_2_Constant_output_0);
	node__features_features_0_features_0_2_Constant_1( tensor__features_features_0_features_0_2_Constant_1_output_0);
	node__features_features_0_features_0_2_Clip( tu0.tensor__features_features_0_features_0_0_Conv_output_0, tensor__features_features_0_features_0_2_Constant_output_0, tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_0_features_0_2_Clip_output_0);
	node__features_features_1_conv_conv_0_conv_0_0_Conv( tu1.tensor__features_features_0_features_0_2_Clip_output_0, tensor_onnx__Conv_541, tensor_onnx__Conv_542, tu0.tensor__features_features_1_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_1_conv_conv_0_conv_0_2_Constant( tensor__features_features_1_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_1_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_1_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_1_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_1_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_1_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_1_conv_conv_0_conv_0_2_Constant_1_output_0, tu1.tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_1_conv_conv_1_Conv( tu1.tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_544, tensor_onnx__Conv_545, tu0.tensor__features_features_1_conv_conv_1_Conv_output_0);
	node__features_features_2_conv_conv_0_conv_0_0_Conv( tu0.tensor__features_features_1_conv_conv_1_Conv_output_0, tensor_onnx__Conv_547, tensor_onnx__Conv_548, tu1.tensor__features_features_2_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_2_conv_conv_0_conv_0_2_Constant( tensor__features_features_2_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_2_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_2_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_2_conv_conv_0_conv_0_2_Clip( tu1.tensor__features_features_2_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_2_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_2_conv_conv_0_conv_0_2_Constant_1_output_0, tu0.tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_2_conv_conv_1_conv_1_0_Conv( tu0.tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_550, tensor_onnx__Conv_551, tu1.tensor__features_features_2_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_2_conv_conv_1_conv_1_2_Constant( tensor__features_features_2_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_2_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_2_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_2_conv_conv_1_conv_1_2_Clip( tu1.tensor__features_features_2_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_2_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_2_conv_conv_1_conv_1_2_Constant_1_output_0, tu0.tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_2_conv_conv_2_Conv( tu0.tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_553, tensor_onnx__Conv_554, tu1.tensor__features_features_2_conv_conv_2_Conv_output_0);
	node__features_features_3_conv_conv_0_conv_0_0_Conv( tu1.tensor__features_features_2_conv_conv_2_Conv_output_0, tensor_onnx__Conv_556, tensor_onnx__Conv_557, tu0.tensor__features_features_3_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_3_conv_conv_0_conv_0_2_Constant( tensor__features_features_3_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_3_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_3_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_3_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_3_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_3_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_3_conv_conv_0_conv_0_2_Constant_1_output_0, tu2.tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_3_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_559, tensor_onnx__Conv_560, tu0.tensor__features_features_3_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_3_conv_conv_1_conv_1_2_Constant( tensor__features_features_3_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_3_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_3_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_3_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_3_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_3_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_3_conv_conv_1_conv_1_2_Constant_1_output_0, tu2.tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_3_conv_conv_2_Conv( tu2.tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_562, tensor_onnx__Conv_563, tu0.tensor__features_features_3_conv_conv_2_Conv_output_0);
	node__features_features_3_Add( tu1.tensor__features_features_2_conv_conv_2_Conv_output_0, tu0.tensor__features_features_3_conv_conv_2_Conv_output_0, tu2.tensor__features_features_3_Add_output_0);
	node__features_features_4_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_3_Add_output_0, tensor_onnx__Conv_565, tensor_onnx__Conv_566, tu0.tensor__features_features_4_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_4_conv_conv_0_conv_0_2_Constant( tensor__features_features_4_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_4_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_4_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_4_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_4_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_4_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_4_conv_conv_0_conv_0_2_Constant_1_output_0, tu1.tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_4_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_568, tensor_onnx__Conv_569, tu0.tensor__features_features_4_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_4_conv_conv_1_conv_1_2_Constant( tensor__features_features_4_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_4_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_4_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_4_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_4_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_4_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_4_conv_conv_1_conv_1_2_Constant_1_output_0, tu1.tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_4_conv_conv_2_Conv( tu1.tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_571, tensor_onnx__Conv_572, tu0.tensor__features_features_4_conv_conv_2_Conv_output_0);
	node__features_features_5_conv_conv_0_conv_0_0_Conv( tu0.tensor__features_features_4_conv_conv_2_Conv_output_0, tensor_onnx__Conv_574, tensor_onnx__Conv_575, tu1.tensor__features_features_5_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_5_conv_conv_0_conv_0_2_Constant( tensor__features_features_5_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_5_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_5_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_5_conv_conv_0_conv_0_2_Clip( tu1.tensor__features_features_5_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_5_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_5_conv_conv_0_conv_0_2_Constant_1_output_0, tu2.tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_5_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_577, tensor_onnx__Conv_578, tu1.tensor__features_features_5_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_5_conv_conv_1_conv_1_2_Constant( tensor__features_features_5_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_5_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_5_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_5_conv_conv_1_conv_1_2_Clip( tu1.tensor__features_features_5_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_5_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_5_conv_conv_1_conv_1_2_Constant_1_output_0, tu2.tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_5_conv_conv_2_Conv( tu2.tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_580, tensor_onnx__Conv_581, tu1.tensor__features_features_5_conv_conv_2_Conv_output_0);
	node__features_features_5_Add( tu0.tensor__features_features_4_conv_conv_2_Conv_output_0, tu1.tensor__features_features_5_conv_conv_2_Conv_output_0, tu2.tensor__features_features_5_Add_output_0);
	node__features_features_6_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_5_Add_output_0, tensor_onnx__Conv_583, tensor_onnx__Conv_584, tu0.tensor__features_features_6_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_6_conv_conv_0_conv_0_2_Constant( tensor__features_features_6_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_6_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_6_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_6_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_6_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_6_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_6_conv_conv_0_conv_0_2_Constant_1_output_0, tu1.tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_6_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_586, tensor_onnx__Conv_587, tu0.tensor__features_features_6_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_6_conv_conv_1_conv_1_2_Constant( tensor__features_features_6_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_6_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_6_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_6_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_6_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_6_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_6_conv_conv_1_conv_1_2_Constant_1_output_0, tu1.tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_6_conv_conv_2_Conv( tu1.tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_589, tensor_onnx__Conv_590, tu0.tensor__features_features_6_conv_conv_2_Conv_output_0);
	node__features_features_6_Add( tu2.tensor__features_features_5_Add_output_0, tu0.tensor__features_features_6_conv_conv_2_Conv_output_0, tu1.tensor__features_features_6_Add_output_0);
	node__features_features_7_conv_conv_0_conv_0_0_Conv( tu1.tensor__features_features_6_Add_output_0, tensor_onnx__Conv_592, tensor_onnx__Conv_593, tu0.tensor__features_features_7_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_7_conv_conv_0_conv_0_2_Constant( tensor__features_features_7_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_7_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_7_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_7_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_7_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_7_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_7_conv_conv_0_conv_0_2_Constant_1_output_0, tu1.tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_7_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_595, tensor_onnx__Conv_596, tu0.tensor__features_features_7_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_7_conv_conv_1_conv_1_2_Constant( tensor__features_features_7_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_7_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_7_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_7_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_7_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_7_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_7_conv_conv_1_conv_1_2_Constant_1_output_0, tu1.tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_7_conv_conv_2_Conv( tu1.tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_598, tensor_onnx__Conv_599, tu0.tensor__features_features_7_conv_conv_2_Conv_output_0);
	node__features_features_8_conv_conv_0_conv_0_0_Conv( tu0.tensor__features_features_7_conv_conv_2_Conv_output_0, tensor_onnx__Conv_601, tensor_onnx__Conv_602, tu1.tensor__features_features_8_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_8_conv_conv_0_conv_0_2_Constant( tensor__features_features_8_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_8_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_8_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_8_conv_conv_0_conv_0_2_Clip( tu1.tensor__features_features_8_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_8_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_8_conv_conv_0_conv_0_2_Constant_1_output_0, tu2.tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_8_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_604, tensor_onnx__Conv_605, tu1.tensor__features_features_8_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_8_conv_conv_1_conv_1_2_Constant( tensor__features_features_8_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_8_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_8_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_8_conv_conv_1_conv_1_2_Clip( tu1.tensor__features_features_8_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_8_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_8_conv_conv_1_conv_1_2_Constant_1_output_0, tu2.tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_8_conv_conv_2_Conv( tu2.tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_607, tensor_onnx__Conv_608, tu1.tensor__features_features_8_conv_conv_2_Conv_output_0);
	node__features_features_8_Add( tu0.tensor__features_features_7_conv_conv_2_Conv_output_0, tu1.tensor__features_features_8_conv_conv_2_Conv_output_0, tu2.tensor__features_features_8_Add_output_0);
	node__features_features_9_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_8_Add_output_0, tensor_onnx__Conv_610, tensor_onnx__Conv_611, tu0.tensor__features_features_9_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_9_conv_conv_0_conv_0_2_Constant( tensor__features_features_9_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_9_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_9_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_9_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_9_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_9_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_9_conv_conv_0_conv_0_2_Constant_1_output_0, tu1.tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_9_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_613, tensor_onnx__Conv_614, tu0.tensor__features_features_9_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_9_conv_conv_1_conv_1_2_Constant( tensor__features_features_9_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_9_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_9_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_9_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_9_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_9_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_9_conv_conv_1_conv_1_2_Constant_1_output_0, tu1.tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_9_conv_conv_2_Conv( tu1.tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_616, tensor_onnx__Conv_617, tu0.tensor__features_features_9_conv_conv_2_Conv_output_0);
	node__features_features_9_Add( tu2.tensor__features_features_8_Add_output_0, tu0.tensor__features_features_9_conv_conv_2_Conv_output_0, tu1.tensor__features_features_9_Add_output_0);
	node__features_features_10_conv_conv_0_conv_0_0_Conv( tu1.tensor__features_features_9_Add_output_0, tensor_onnx__Conv_619, tensor_onnx__Conv_620, tu0.tensor__features_features_10_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_10_conv_conv_0_conv_0_2_Constant( tensor__features_features_10_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_10_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_10_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_10_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_10_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_10_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_10_conv_conv_0_conv_0_2_Constant_1_output_0, tu2.tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_10_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_622, tensor_onnx__Conv_623, tu0.tensor__features_features_10_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_10_conv_conv_1_conv_1_2_Constant( tensor__features_features_10_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_10_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_10_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_10_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_10_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_10_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_10_conv_conv_1_conv_1_2_Constant_1_output_0, tu2.tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_10_conv_conv_2_Conv( tu2.tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_625, tensor_onnx__Conv_626, tu0.tensor__features_features_10_conv_conv_2_Conv_output_0);
	node__features_features_10_Add( tu1.tensor__features_features_9_Add_output_0, tu0.tensor__features_features_10_conv_conv_2_Conv_output_0, tu2.tensor__features_features_10_Add_output_0);
	node__features_features_11_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_10_Add_output_0, tensor_onnx__Conv_628, tensor_onnx__Conv_629, tu0.tensor__features_features_11_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_11_conv_conv_0_conv_0_2_Constant( tensor__features_features_11_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_11_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_11_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_11_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_11_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_11_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_11_conv_conv_0_conv_0_2_Constant_1_output_0, tu1.tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_11_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_631, tensor_onnx__Conv_632, tu0.tensor__features_features_11_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_11_conv_conv_1_conv_1_2_Constant( tensor__features_features_11_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_11_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_11_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_11_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_11_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_11_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_11_conv_conv_1_conv_1_2_Constant_1_output_0, tu1.tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_11_conv_conv_2_Conv( tu1.tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_634, tensor_onnx__Conv_635, tu0.tensor__features_features_11_conv_conv_2_Conv_output_0);
	node__features_features_12_conv_conv_0_conv_0_0_Conv( tu0.tensor__features_features_11_conv_conv_2_Conv_output_0, tensor_onnx__Conv_637, tensor_onnx__Conv_638, tu1.tensor__features_features_12_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_12_conv_conv_0_conv_0_2_Constant( tensor__features_features_12_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_12_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_12_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_12_conv_conv_0_conv_0_2_Clip( tu1.tensor__features_features_12_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_12_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_12_conv_conv_0_conv_0_2_Constant_1_output_0, tu2.tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_12_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_640, tensor_onnx__Conv_641, tu1.tensor__features_features_12_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_12_conv_conv_1_conv_1_2_Constant( tensor__features_features_12_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_12_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_12_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_12_conv_conv_1_conv_1_2_Clip( tu1.tensor__features_features_12_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_12_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_12_conv_conv_1_conv_1_2_Constant_1_output_0, tu2.tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_12_conv_conv_2_Conv( tu2.tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_643, tensor_onnx__Conv_644, tu1.tensor__features_features_12_conv_conv_2_Conv_output_0);
	node__features_features_12_Add( tu0.tensor__features_features_11_conv_conv_2_Conv_output_0, tu1.tensor__features_features_12_conv_conv_2_Conv_output_0, tu2.tensor__features_features_12_Add_output_0);
	node__features_features_13_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_12_Add_output_0, tensor_onnx__Conv_646, tensor_onnx__Conv_647, tu0.tensor__features_features_13_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_13_conv_conv_0_conv_0_2_Constant( tensor__features_features_13_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_13_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_13_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_13_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_13_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_13_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_13_conv_conv_0_conv_0_2_Constant_1_output_0, tu1.tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_13_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_649, tensor_onnx__Conv_650, tu0.tensor__features_features_13_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_13_conv_conv_1_conv_1_2_Constant( tensor__features_features_13_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_13_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_13_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_13_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_13_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_13_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_13_conv_conv_1_conv_1_2_Constant_1_output_0, tu1.tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_13_conv_conv_2_Conv( tu1.tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_652, tensor_onnx__Conv_653, tu0.tensor__features_features_13_conv_conv_2_Conv_output_0);
	node__features_features_13_Add( tu2.tensor__features_features_12_Add_output_0, tu0.tensor__features_features_13_conv_conv_2_Conv_output_0, tu1.tensor__features_features_13_Add_output_0);
	node__features_features_14_conv_conv_0_conv_0_0_Conv( tu1.tensor__features_features_13_Add_output_0, tensor_onnx__Conv_655, tensor_onnx__Conv_656, tu0.tensor__features_features_14_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_14_conv_conv_0_conv_0_2_Constant( tensor__features_features_14_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_14_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_14_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_14_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_14_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_14_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_14_conv_conv_0_conv_0_2_Constant_1_output_0, tu1.tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_14_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_658, tensor_onnx__Conv_659, tu0.tensor__features_features_14_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_14_conv_conv_1_conv_1_2_Constant( tensor__features_features_14_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_14_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_14_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_14_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_14_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_14_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_14_conv_conv_1_conv_1_2_Constant_1_output_0, tu1.tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_14_conv_conv_2_Conv( tu1.tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_661, tensor_onnx__Conv_662, tu0.tensor__features_features_14_conv_conv_2_Conv_output_0);
	node__features_features_15_conv_conv_0_conv_0_0_Conv( tu0.tensor__features_features_14_conv_conv_2_Conv_output_0, tensor_onnx__Conv_664, tensor_onnx__Conv_665, tu1.tensor__features_features_15_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_15_conv_conv_0_conv_0_2_Constant( tensor__features_features_15_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_15_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_15_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_15_conv_conv_0_conv_0_2_Clip( tu1.tensor__features_features_15_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_15_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_15_conv_conv_0_conv_0_2_Constant_1_output_0, tu2.tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_15_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_667, tensor_onnx__Conv_668, tu1.tensor__features_features_15_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_15_conv_conv_1_conv_1_2_Constant( tensor__features_features_15_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_15_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_15_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_15_conv_conv_1_conv_1_2_Clip( tu1.tensor__features_features_15_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_15_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_15_conv_conv_1_conv_1_2_Constant_1_output_0, tu2.tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_15_conv_conv_2_Conv( tu2.tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_670, tensor_onnx__Conv_671, tu1.tensor__features_features_15_conv_conv_2_Conv_output_0);
	node__features_features_15_Add( tu0.tensor__features_features_14_conv_conv_2_Conv_output_0, tu1.tensor__features_features_15_conv_conv_2_Conv_output_0, tu2.tensor__features_features_15_Add_output_0);
	node__features_features_16_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_15_Add_output_0, tensor_onnx__Conv_673, tensor_onnx__Conv_674, tu0.tensor__features_features_16_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_16_conv_conv_0_conv_0_2_Constant( tensor__features_features_16_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_16_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_16_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_16_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_16_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_16_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_16_conv_conv_0_conv_0_2_Constant_1_output_0, tu1.tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_16_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_676, tensor_onnx__Conv_677, tu0.tensor__features_features_16_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_16_conv_conv_1_conv_1_2_Constant( tensor__features_features_16_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_16_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_16_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_16_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_16_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_16_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_16_conv_conv_1_conv_1_2_Constant_1_output_0, tu1.tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_16_conv_conv_2_Conv( tu1.tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_679, tensor_onnx__Conv_680, tu0.tensor__features_features_16_conv_conv_2_Conv_output_0);
	node__features_features_16_Add( tu2.tensor__features_features_15_Add_output_0, tu0.tensor__features_features_16_conv_conv_2_Conv_output_0, tu1.tensor__features_features_16_Add_output_0);
	node__features_features_17_conv_conv_0_conv_0_0_Conv( tu1.tensor__features_features_16_Add_output_0, tensor_onnx__Conv_682, tensor_onnx__Conv_683, tu0.tensor__features_features_17_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_17_conv_conv_0_conv_0_2_Constant( tensor__features_features_17_conv_conv_0_conv_0_2_Constant_output_0);
	node__features_features_17_conv_conv_0_conv_0_2_Constant_1( tensor__features_features_17_conv_conv_0_conv_0_2_Constant_1_output_0);
	node__features_features_17_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_17_conv_conv_0_conv_0_0_Conv_output_0, tensor__features_features_17_conv_conv_0_conv_0_2_Constant_output_0, tensor__features_features_17_conv_conv_0_conv_0_2_Constant_1_output_0, tu1.tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_17_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_685, tensor_onnx__Conv_686, tu0.tensor__features_features_17_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_17_conv_conv_1_conv_1_2_Constant( tensor__features_features_17_conv_conv_1_conv_1_2_Constant_output_0);
	node__features_features_17_conv_conv_1_conv_1_2_Constant_1( tensor__features_features_17_conv_conv_1_conv_1_2_Constant_1_output_0);
	node__features_features_17_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_17_conv_conv_1_conv_1_0_Conv_output_0, tensor__features_features_17_conv_conv_1_conv_1_2_Constant_output_0, tensor__features_features_17_conv_conv_1_conv_1_2_Constant_1_output_0, tu1.tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_17_conv_conv_2_Conv( tu1.tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_688, tensor_onnx__Conv_689, tu0.tensor__features_features_17_conv_conv_2_Conv_output_0);
	node__features_features_18_features_18_0_Conv( tu0.tensor__features_features_17_conv_conv_2_Conv_output_0, tensor_onnx__Conv_691, tensor_onnx__Conv_692, tu1.tensor__features_features_18_features_18_0_Conv_output_0);
	node__features_features_18_features_18_2_Constant( tensor__features_features_18_features_18_2_Constant_output_0);
	node__features_features_18_features_18_2_Constant_1( tensor__features_features_18_features_18_2_Constant_1_output_0);
	node__features_features_18_features_18_2_Clip( tu1.tensor__features_features_18_features_18_0_Conv_output_0, tensor__features_features_18_features_18_2_Constant_output_0, tensor__features_features_18_features_18_2_Constant_1_output_0, tu0.tensor__features_features_18_features_18_2_Clip_output_0);
	node__GlobalAveragePool( tu0.tensor__features_features_18_features_18_2_Clip_output_0, tu1.tensor__GlobalAveragePool_output_0);
	node__Flatten( tu1.tensor__GlobalAveragePool_output_0, tu0.tensor__Flatten_output_0);
	node__classifier_classifier_1_Gemm( tu0.tensor__Flatten_output_0, tensor_classifier_1_weight, tensor_classifier_1_bias, tensor_output);
}
