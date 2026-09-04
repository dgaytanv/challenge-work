#include "ap_fixed.h"
typedef ap_fixed<16,6> T;
void topf(T a[8], T w[8], T &y){
#pragma HLS PIPELINE II=1
  T acc=0;
  for(int i=0;i<8;i++) acc += a[i]*w[i];
  y=acc;
}
