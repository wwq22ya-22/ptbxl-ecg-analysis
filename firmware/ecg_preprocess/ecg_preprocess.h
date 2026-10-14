#ifndef ECG_PREPROCESS_H
#define ECG_PREPROCESS_H
#include <stdint.h>
/* 12 leads x 1000 points at 100 Hz, raw mV. Static workspace, main loop only.
 * Full 10-second record required for forward/backward filtering. */
int ecg_preprocess_lead(const float *raw_mv, unsigned lead, float *normalized);
int ecg_preprocess_to_int8(const float *raw_mv, int8_t *input);
unsigned ecg_preprocess_workspace_bytes(void);
const char *ecg_preprocess_contract_sha256(void);
#endif
