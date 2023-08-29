// This file is computer-generated by onnx2c 
// (TODO: add creating command line here)
// (TODO: print creation date here )

// ONNX model:
// produced by pytorch, version 2.0.1
// ONNX IR version: 14
// Model documentation: 
/*

*/

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#define MAX(X,Y) ( X > Y ? X : Y)
#define MIN(X,Y) ( X < Y ? X : Y)
#define CLIP(X,L) ( MAX(MIN(X,L), -L) )

static const float tensor_onnx__LSTM_89[1][12][5] = 
{
  {
    {-0.55399370193481445312f, 0.32715484499931335449f, -0.10812743008136749268f, 0.34738039970397949219f, -0.20202718675136566162f},
    {-0.53157353401184082031f, -0.29621130228042602539f, -0.35109347105026245117f, 0.56237089633941650391f, -0.27638715505599975586f},
    {-0.49065104126930236816f, -0.078796751797199249268f, 0.33557778596878051758f, 0.38834294676780700684f, 0.18878163397312164307f},
    {-0.31522092223167419434f, -0.35194009542465209961f, -0.42676293849945068359f, -0.55994832515716552734f, -0.24563674628734588623f},
    {0.14770877361297607422f, 0.22948360443115234375f, -0.12404119223356246948f, -0.25148367881774902344f, -0.49919909238815307617f},
    {-0.32781869173049926758f, 0.39034107327461242676f, 0.45046889781951904297f, 0.066140286624431610107f, 0.15847563743591308594f},
    {0.37269711494445800781f, -0.46338498592376708984f, -0.30457711219787597656f, -0.41335380077362060547f, -0.22127437591552734375f},
    {0.15957732498645782471f, -0.094365559518337249756f, -0.47974988818168640137f, 0.48022726178169250488f, 0.55979484319686889648f},
    {0.41396901011466979980f, -0.14590051770210266113f, 0.45673036575317382812f, -0.12429963797330856323f, -0.38209193944931030273f},
    {0.30816072225570678711f, 0.52557820081710815430f, -0.46692875027656555176f, 0.48881950974464416504f, -0.31027182936668395996f},
    {-0.022238625213503837585f, 0.34496793150901794434f, 0.53290617465972900391f, -0.23961912095546722412f, 0.059091109782457351685f},
    {0.43275260925292968750f, 0.55166214704513549805f, 0.25944128632545471191f, -0.056564249098300933838f, 0.49175348877906799316f}
  }
};
static const float tensor_onnx__LSTM_90[1][12][3] = 
{
  {
    {0.52353674173355102539f, -0.51553642749786376953f, -0.21105255186557769775f},
    {0.24122840166091918945f, -0.079966366291046142578f, 0.42574238777160644531f},
    {0.54705882072448730469f, -0.13853076100349426270f, 0.47342801094055175781f},
    {0.15558199584484100342f, -0.25672549009323120117f, 0.11205846816301345825f},
    {0.43732413649559020996f, 0.011295918375253677368f, -0.29688373208045959473f},
    {0.17684495449066162109f, 0.45969215035438537598f, 0.45262554287910461426f},
    {0.51188796758651733398f, 0.29247823357582092285f, -0.56238746643066406250f},
    {0.21198368072509765625f, 0.40736019611358642578f, 0.42191484570503234863f},
    {0.54875814914703369141f, -0.52350771427154541016f, -0.37563058733940124512f},
    {-0.31436631083488464355f, -0.26434364914894104004f, -0.39344647526741027832f},
    {0.13153168559074401855f, 0.52092772722244262695f, -0.18840049207210540771f},
    {0.0058881603181362152100f, 0.37932437658309936523f, -0.011741908267140388489f}
  }
};
static const float tensor_onnx__LSTM_91[1][24] = 
{
  {-0.23570233583450317383f, 0.52185195684432983398f, 0.23598878085613250732f, 0.40169921517372131348f, 0.52260780334472656250f, -0.26196959614753723145f, -0.31339937448501586914f, 0.0039857542142271995544f, 0.016573389992117881775f, 0.091871120035648345947f, 0.13055346906185150146f, 0.34198895096778869629f, -0.26913446187973022461f, 0.40813592076301574707f, -0.058767352253198623657f, -0.45270952582359313965f, 0.22087325155735015869f, 0.30768024921417236328f, 0.38294303417205810547f, -0.086650840938091278076f, -0.49063113331794738770f, -0.46155029535293579102f, 0.22299638390541076660f, -0.50542837381362915039f}
};
static float tensor_h_out[1][1][3] = 
{
  {
    {0.0000000000000000000f, 0.0000000000000000000f, 0.0000000000000000000f}
  }
};
static float tensor_c_out[1][1][3] = 
{
  {
    {0.0000000000000000000f, 0.0000000000000000000f, 0.0000000000000000000f}
  }
};
static const int64_t tensor__l_Constant_output_0[1] = 
{1};
union tensor_union_0 {
float tensor__l_LSTM_output_0[10][1][1][3];
};
static union tensor_union_0 tu0;


static inline void node__l_LSTM( const float X[10][1][5], const float W[1][12][5], const float R[1][12][3], const float B[1][24], const float initial_h[1][1][3], const float initial_c[1][1][3], float Y[10][1][1][3], float Y_h[1][1][3], float Y_c[1][1][3] )
{
	/* LSTM 
	 * inputs: 
	 *   X = tensor_input
	 *   W = tensor_onnx__LSTM_89
	 *   R = tensor_onnx__LSTM_90
	 *   B = tensor_onnx__LSTM_91
	 *   sequence_lens = 
	 *   initial_h = tensor_h_in
	 *   initial_c = tensor_c_in
	 *   P = 
	 * outputs: 
	 *   Y = tensor__l_LSTM_output_0
	 *   Y_h = tensor_h_out
	 *   Y_c = tensor_c_out
	 * attributes:
	 *   activations: Sigmoid Tanh Tanh 
	 * clip: off
	 * layout: 0
	 * (rest TBD):
	 */
	int hs = 3;
	int ds = 5;
	int bs = 1;
	int iidx = 0;
	int oidx = hs;
	int fidx = 2*hs;
	int cidx = 3*hs;
	int Rb = 4*hs;
	int sequence_lenght = 10;
	/* Forget gate */
	float ft[bs][hs];
	/* Input gate */
	float it[bs][hs];
	/* Cell gate */
	float ct[bs][hs];
	/* Output gate */
	float ot[bs][hs];

	memcpy(Y_h, initial_h, sizeof(*initial_h));
	memcpy(Y_c, initial_c, sizeof(*initial_c));

	for( int s=0; s<sequence_lenght; s++) {

		/* Forward lane */
		for( int b=0; b<bs; b++)
		for( int h=0; h<hs; h++) {
			ft[b][h]=0;
			it[b][h]=0;
			ct[b][h]=0;
			for( int i=0; i<ds; i++) {
				ft[b][h] += X[s][b][i]*W[0][fidx+h][i];
				it[b][h] += X[s][b][i]*W[0][iidx+h][i];
				ct[b][h] += X[s][b][i]*W[0][cidx+h][i];
			}
			for( int k=0; k<hs; k++) {
				ft[b][h] += Y_h[0][b][k]*R[0][fidx+h][k];
				ct[b][h] += Y_h[0][b][k]*R[0][cidx+h][k];
				it[b][h] += Y_h[0][b][k]*R[0][iidx+h][k];
			}
			ft[b][h] += B[0][fidx+h];
			ft[b][h] += B[0][Rb+fidx+h];
			it[b][h] += B[0][iidx+h];
			it[b][h] += B[0][Rb+iidx+h];
			ct[b][h] += B[0][cidx+h];
			ct[b][h] += B[0][Rb+cidx+h];
			ft[b][h] =1.0f/(1+expf(-ft[b][h]));
			it[b][h] =1.0f/(1+expf(-it[b][h]));
			ct[b][h] =tanh(ct[b][h]);
		}
		for( int b=0; b<bs; b++)
		for( int h=0; h<hs; h++) {
			/* Cell state */
			Y_c[0][b][h] = Y_c[0][b][h]*ft[b][h] + it[b][h]*ct[b][h];
			/* Output gate */
			ot[b][h]=0;
			for( int i=0; i<ds; i++)
				ot[b][h] += X[s][b][i]*W[0][oidx+h][i];
			for( int k=0; k<hs; k++)
				ot[b][h] += Y_h[0][b][k]*R[0][oidx+h][k];
			ot[b][h] += B[0][oidx+h];
			ot[b][h] += B[0][Rb+oidx+h];
			ot[b][h] =1.0f/(1+expf(-ot[b][h]));
		}
		/* Hidden state */
		for( int b=0; b<bs; b++)
		for( int h=0; h<hs; h++) {
			Y_h[0][b][h] = ot[b][h] * tanh(Y_c[0][b][h]);
			Y[s][0][b][h]= Y_h[0][b][h];
		}

	} /* sequences */
}

static inline void node__l_Constant( const int64_t output[1] )
{
	/* Constant */
	/* The output is generated as a global tensor */
	(void)output;
}

static inline void node__l_Squeeze( const float tensor__l_LSTM_output_0[10][1][1][3], float tensor_output[10][1][3] )
{
	/*Squeeze*/
	float *data = (float*)tensor__l_LSTM_output_0;
	float *squeezed= (float*)tensor_output;
	for( uint32_t i=0; i<30; i++ )
		squeezed[i] = data[i];

}