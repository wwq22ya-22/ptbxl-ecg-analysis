#include "ecg_protocol.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static uint32_t crc_bytes(uint32_t crc, const void *data, size_t length) {
    const unsigned char *bytes = (const unsigned char *)data;
    size_t j; int bit;
    for (j = 0; j < length; ++j) {
        crc ^= bytes[j];
        for (bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return crc;
}

void ecg_protocol_abort(ecg_protocol *p, const char *code) {
    char reply[128];
    p->active = 0;
    snprintf(reply, sizeof(reply), "{\"type\":\"error\",\"seq\":%lu,\"code\":\"%s\"}\r\n",
             (unsigned long)p->sequence, code);
    p->send(reply, p->context);
}

static char *token(char **cursor) {
    char *begin = *cursor, *end;
    if (!begin) return NULL;
    end = strchr(begin, ',');
    if (end) { *end = 0; *cursor = end + 1; } else *cursor = NULL;
    return begin;
}

static int unsigned_token(char **cursor, uint32_t *value) {
    char *part = token(cursor), *end;
    unsigned long parsed;
    if (!part || !*part || *part == '-' || *part == '+') return 0;
    errno = 0;
    parsed = strtoul(part, &end, 10);
    if (*end || errno || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

void ecg_protocol_line(ecg_protocol *p, char *line) {
    char reply[640], *cursor = line, *command = token(&cursor);
    uint32_t sequence, offset, count, crc, i, elapsed = 0;
    float chunk[ECG_CHUNK_COUNT], logits[5], probabilities[5];
    unsigned mask = 0;
    if (command && !strcmp(command, "$P") && !cursor) {
        snprintf(reply, sizeof(reply),
            "{\"type\":\"hello\",\"protocol\":1,\"ready\":%s,\"model\":\"%s\","
            "\"input_shape\":[12,1000],\"preprocessing\":\"host\","
            "\"labels\":[\"NORM\",\"MI\",\"STTC\",\"CD\",\"HYP\"],"
            "\"thresholds\":[%.9g,%.9g,%.9g,%.9g,%.9g]}\r\n",
            p->input ? "true" : "false", p->model_sha256,
            p->thresholds[0], p->thresholds[1], p->thresholds[2], p->thresholds[3], p->thresholds[4]);
        p->send(reply, p->context); return;
    }
    if (!command || !unsigned_token(&cursor, &sequence)) {
        ecg_protocol_abort(p, "format"); return;
    }
    if (!strcmp(command, "$B")) {
        p->sequence = sequence;
        p->active = 0;
        if (!p->input || !isfinite(p->scale) || p->scale <= 0) {
            ecg_protocol_abort(p, "ai_not_ready"); return;
        }
        if (!unsigned_token(&cursor, &count) || count != ECG_SAMPLE_COUNT || cursor) {
            ecg_protocol_abort(p, "sample_count"); return;
        }
        p->received = 0; p->crc = 0xFFFFFFFFu; p->active = 1;
        snprintf(reply, sizeof(reply), "{\"type\":\"begin\",\"seq\":%lu}\r\n", (unsigned long)sequence);
        p->send(reply, p->context); return;
    }
    if (sequence != p->sequence) {
        /* Tag the failure with the request ID so the host need not time out. */
        p->sequence = sequence; ecg_protocol_abort(p, "sequence"); return;
    }
    if (!p->active) { ecg_protocol_abort(p, "no_transfer"); return; }
    if (!strcmp(command, "$D")) {
        if (!unsigned_token(&cursor, &offset) || !unsigned_token(&cursor, &count)
                || offset != p->received || count == 0 || count > ECG_CHUNK_COUNT
                || count > ECG_SAMPLE_COUNT - p->received) {
            ecg_protocol_abort(p, "offset_or_count"); return;
        }
        for (i = 0; i < count; ++i) {
            char *part = token(&cursor), *end;
            if (!part || !*part) { ecg_protocol_abort(p, "float"); return; }
            chunk[i] = strtof(part, &end);
            if (*end || !isfinite(chunk[i])) { ecg_protocol_abort(p, "float"); return; }
        }
        if (cursor) { ecg_protocol_abort(p, "extra_values"); return; }
        for (i = 0; i < count; ++i) {
            /* Round to nearest, ties to even, matching ONNX Runtime's quantizer. */
            float q = nearbyintf(chunk[i] / p->scale) + (float)p->zero_point;
            q = fminf(127.0f, fmaxf(-128.0f, q));
            p->input[offset + i] = (int8_t)q;
            p->crc = crc_bytes(p->crc, &chunk[i], sizeof(float));
        }
        p->received += count;
        snprintf(reply, sizeof(reply), "{\"type\":\"ack\",\"seq\":%lu,\"next\":%lu}\r\n",
                 (unsigned long)sequence, (unsigned long)p->received);
        p->send(reply, p->context); return;
    }
    if (!strcmp(command, "$R")) {
        if (!unsigned_token(&cursor, &crc) || cursor || p->received != ECG_SAMPLE_COUNT
                || crc != (p->crc ^ 0xFFFFFFFFu)) { ecg_protocol_abort(p, "crc_or_incomplete"); return; }
        p->active = 0;
        if (!p->run(logits, &elapsed, p->context)) { ecg_protocol_abort(p, "ai_run"); return; }
        for (i = 0; i < 5; ++i) {
            if (!isfinite(logits[i])) { ecg_protocol_abort(p, "nonfinite_output"); return; }
            probabilities[i] = logits[i] >= 0 ? 1.0f / (1.0f + expf(-logits[i]))
                : expf(logits[i]) / (1.0f + expf(logits[i]));
            if (probabilities[i] >= p->thresholds[i]) mask |= 1u << i;
        }
        snprintf(reply, sizeof(reply),
            "{\"type\":\"result\",\"seq\":%lu,\"probabilities\":[%.9g,%.9g,%.9g,%.9g,%.9g],"
            "\"mask\":%u,\"inference_ms\":%lu}\r\n",
            (unsigned long)sequence, probabilities[0], probabilities[1], probabilities[2],
            probabilities[3], probabilities[4], mask, (unsigned long)elapsed);
        p->send(reply, p->context); return;
    }
    ecg_protocol_abort(p, "command");
}
