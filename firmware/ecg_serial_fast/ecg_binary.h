#ifndef ECG_BINARY_H
#define ECG_BINARY_H
#include <stdint.h>
#include <stddef.h>
#define ECG_FAST_SAMPLES 12000u
#define ECG_FAST_PAYLOAD 1024u
#define ECG_FAST_FRAME (16u + ECG_FAST_PAYLOAD + 4u)
typedef void (*ecg_binary_send)(const uint8_t *, size_t, void *);
typedef int (*ecg_binary_run)(float[5], uint32_t *, void *);
typedef struct {
    int8_t *input;
    float scale;
    int zero_point;
    const float *thresholds;
    const char *model_sha256;
    ecg_binary_send send;
    ecg_binary_run run;
    void *context;
    uint32_t sequence, received, crc, used, needed;
    int active;
    uint8_t frame[ECG_FAST_FRAME];
} ecg_binary;
/* Main-loop parser; UART ISR only buffers bytes. */
void ecg_binary_byte(ecg_binary *, uint8_t);
void ecg_binary_abort(ecg_binary *, const char *);
#endif
