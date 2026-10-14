#include "ecg_binary.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
static uint32_t crc_bytes(uint32_t crc, const uint8_t *bytes, size_t n) {
    size_t j; int bit;
    for (j = 0; j < n; ++j) {
        crc ^= bytes[j];
        for (bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return crc;
}
static uint32_t get32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static void put32(uint8_t *p, uint32_t v) {
    int i; for (i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (i * 8));
}
static void reply(ecg_binary *p, uint8_t kind, uint32_t offset, const char *json) {
    uint8_t header[16] = {'E','C','G','3',3,0,0,0}, tail[4];
    size_t length = json ? strlen(json) : 0;
    uint32_t crc;
    header[5] = kind; header[6] = (uint8_t)length; header[7] = (uint8_t)(length >> 8);
    put32(header + 8, p->sequence); put32(header + 12, offset);
    crc = crc_bytes(0xFFFFFFFFu, header, sizeof(header));
    crc = crc_bytes(crc, (const uint8_t *)json, length);
    put32(tail, crc ^ 0xFFFFFFFFu);
    p->send(header, sizeof(header), p->context);
    if (length) p->send((const uint8_t *)json, length, p->context);
    p->send(tail, sizeof(tail), p->context);
}
void ecg_binary_abort(ecg_binary *p, const char *code) {
    char json[96];
    p->active = 0; p->used = 0; p->needed = 0;
    snprintf(json, sizeof(json), "{\"type\":\"error\",\"code\":\"%s\"}", code);
    reply(p, 0xFF, p->received, json);
}
static void execute(ecg_binary *p) {
    uint8_t *f = p->frame, *data = f + 16;
    uint32_t seq = get32(f + 8), offset = get32(f + 12), elapsed = 0, i;
    unsigned length = f[6] | (unsigned)f[7] << 8, mask = 0;
    char json[640];
    float logits[5], probabilities[5];
    if (get32(data + length) != (crc_bytes(0xFFFFFFFFu, f, 16 + length) ^ 0xFFFFFFFFu)) {
        p->sequence = seq; ecg_binary_abort(p, "frame_crc"); return;
    }
    if (f[5] == 1) {
        p->sequence = seq; p->active = 0;
        if (length || offset) { ecg_binary_abort(p, "format"); return; }
        snprintf(json, sizeof(json),
            "{\"type\":\"hello\",\"protocol\":3,\"ready\":%s,\"model\":\"%s\","
            "\"input_shape\":[12,1000],\"preprocessing\":\"host\",\"input_kind\":\"int8\","
            "\"input_scale\":%.9g,\"input_zero_point\":%d,\"chunk_bytes\":1024,"
            "\"labels\":[\"NORM\",\"MI\",\"STTC\",\"CD\",\"HYP\"],"
            "\"thresholds\":[%.9g,%.9g,%.9g,%.9g,%.9g]}",
            (p->input && p->run) ? "true" : "false", p->model_sha256, p->scale, p->zero_point,
            p->thresholds[0], p->thresholds[1], p->thresholds[2], p->thresholds[3], p->thresholds[4]);
        reply(p, 0x81, 0, json); return;
    }
    if (f[5] == 2) {
        p->sequence = seq; p->active = 0;
        if (!p->input || !p->run) { ecg_binary_abort(p, "ai_not_ready"); return; }
        if (length != 4 || offset || get32(data) != ECG_FAST_SAMPLES) {
            ecg_binary_abort(p, "sample_count"); return;
        }
        p->received = 0; p->crc = 0xFFFFFFFFu; p->active = 1;
        reply(p, 0x82, 0, NULL); return;
    }
    if (seq != p->sequence) { p->sequence = seq; ecg_binary_abort(p, "sequence"); return; }
    if (!p->active) { ecg_binary_abort(p, "no_transfer"); return; }
    if (f[5] == 3) {
        if (!length || offset != p->received || length > ECG_FAST_SAMPLES - p->received) {
            ecg_binary_abort(p, "offset_or_count"); return;
        }
        memcpy(p->input + offset, data, length);
        p->crc = crc_bytes(p->crc, data, length); p->received += length;
        reply(p, 0x83, p->received, NULL); return;
    }
    if (f[5] == 4) {
        if (length != 4 || offset != ECG_FAST_SAMPLES || p->received != ECG_FAST_SAMPLES
                || get32(data) != (p->crc ^ 0xFFFFFFFFu)) {
            ecg_binary_abort(p, "crc_or_incomplete"); return;
        }
        p->active = 0;
        if (!p->run(logits, &elapsed, p->context)) { ecg_binary_abort(p, "ai_run"); return; }
        for (i = 0; i < 5; ++i) {
            if (!isfinite(logits[i])) { ecg_binary_abort(p, "nonfinite_output"); return; }
            probabilities[i] = logits[i] >= 0 ? 1.0f / (1.0f + expf(-logits[i]))
                : expf(logits[i]) / (1.0f + expf(logits[i]));
            if (probabilities[i] >= p->thresholds[i]) mask |= 1u << i;
        }
        snprintf(json, sizeof(json),
            "{\"type\":\"result\",\"seq\":%lu,\"probabilities\":[%.9g,%.9g,%.9g,%.9g,%.9g],"
            "\"mask\":%u,\"inference_ms\":%lu}", (unsigned long)seq,
            probabilities[0], probabilities[1], probabilities[2], probabilities[3], probabilities[4],
            mask, (unsigned long)elapsed);
        reply(p, 0x84, p->received, json); return;
    }
    ecg_binary_abort(p, "command");
}
void ecg_binary_byte(ecg_binary *p, uint8_t byte) {
    static const uint8_t magic[] = {'E','C','G','3'};
    if (p->used < 4) {
        if (byte == magic[p->used]) p->frame[p->used++] = byte;
        else { p->used = byte == magic[0] ? 1u : 0u; if (p->used) p->frame[0] = byte; }
        return;
    }
    p->frame[p->used++] = byte;
    if (p->used == 16) {
        unsigned length = p->frame[6] | (unsigned)p->frame[7] << 8;
        if (p->frame[4] != 3 || length > ECG_FAST_PAYLOAD) {
            p->sequence = get32(p->frame + 8); ecg_binary_abort(p, "header"); return;
        }
        p->needed = 16 + length + 4;
    }
    if (p->needed && p->used == p->needed) { execute(p); p->used = 0; p->needed = 0; }
}
