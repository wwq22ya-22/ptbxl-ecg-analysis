#ifndef ECG_PROTOCOL_H
#define ECG_PROTOCOL_H
#include <stddef.h>
#include <stdint.h>

#define ECG_SAMPLE_COUNT 12000u
#define ECG_CHUNK_COUNT 64u
#define ECG_LINE_CAPACITY 2048u

typedef void (*ecg_send_fn)(const char *line, void *context);
typedef int (*ecg_run_fn)(float logits[5], uint32_t *elapsed_ms, void *context);
typedef struct {
    int8_t *input;
    float scale;
    int zero_point;
    const float *thresholds;
    const char *model_sha256;
    ecg_send_fn send;
    ecg_run_fn run;
    void *context;
    uint32_t sequence, received, crc;
    int active;
} ecg_protocol;

/* Call in the main loop, never inside an interrupt. Modifies the line buffer. */
void ecg_protocol_line(ecg_protocol *protocol, char *line);
void ecg_protocol_abort(ecg_protocol *protocol, const char *code);
#endif
