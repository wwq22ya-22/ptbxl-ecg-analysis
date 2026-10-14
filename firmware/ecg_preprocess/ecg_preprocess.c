#include "ecg_preprocess.h"
#include "ecg_preprocess_config.h"
#include <math.h>
#include <stddef.h>
#define SAMPLES 1000
#define PAD 27
#define EXTENDED (SAMPLES + 2 * PAD)
#define SECTIONS 4
typedef struct {
    float median_a[SAMPLES];
    float median_b[SAMPLES];
    float window[61];
    double filter[EXTENDED];
} ecg_workspace;
static ecg_workspace workspace;
unsigned ecg_preprocess_workspace_bytes(void) { return sizeof(workspace); }
const char *ecg_preprocess_contract_sha256(void) { return ECG_PREPROCESS_SHA256; }
static int reflect(int index) {
    if (index < 0) return -index - 1;
    if (index >= SAMPLES) return 2 * SAMPLES - index - 1;
    return index;
}
static float median(float *values, int count) {
    int low = 0, high = count - 1, middle = count / 2;
    while (low < high) {
        float pivot = values[middle];
        int i = low, j = high;
        do {
            while (values[i] < pivot) ++i;
            while (values[j] > pivot) --j;
            if (i <= j) {
                float temp = values[i]; values[i] = values[j]; values[j] = temp;
                ++i; --j;
            }
        } while (i <= j);
        if (j < middle) low = i;
        if (middle < i) high = j;
    }
    return values[middle];
}
static void median_pass(const float *input, float *output, int width) {
    int n, k, half = width / 2;
    for (n = 0; n < SAMPLES; ++n) {
        for (k = 0; k < width; ++k) workspace.window[k] = input[reflect(n + k - half)];
        output[n] = median(workspace.window, width);
    }
}
static void sos_pass(double *values) {
    double state[SECTIONS][2];
    int section, n;
    for (section = 0; section < SECTIONS; ++section) {
        state[section][0] = ECG_SOS_ZI[section][0] * values[0];
        state[section][1] = ECG_SOS_ZI[section][1] * values[0];
    }
    for (n = 0; n < EXTENDED; ++n) {
        double value = values[n];
        for (section = 0; section < SECTIONS; ++section) {
            const double *s = ECG_SOS[section];
            double next = s[0] * value + state[section][0];
            state[section][0] = s[1] * value - s[4] * next + state[section][1];
            state[section][1] = s[2] * value - s[5] * next;
            value = next;
        }
        values[n] = value;
    }
}
static void reverse_filter(void) {
    int n;
    for (n = 0; n < EXTENDED / 2; ++n) {
        double value = workspace.filter[n];
        workspace.filter[n] = workspace.filter[EXTENDED - 1 - n];
        workspace.filter[EXTENDED - 1 - n] = value;
    }
}
int ecg_preprocess_lead(const float *raw_mv, unsigned lead, float *normalized) {
    int n;
    if (!raw_mv || !normalized || lead >= 12) return 0;
    for (n = 0; n < SAMPLES; ++n) if (!isfinite(raw_mv[n])) return 0;
    /* scipy.ndimage median_filter(mode="reflect"), 21 then 61 points. */
    median_pass(raw_mv, workspace.median_a, 21);
    median_pass(workspace.median_a, workspace.median_b, 61);
    for (n = 0; n < SAMPLES; ++n)
        workspace.median_b[n] = (float)((double)raw_mv[n] - (double)workspace.median_b[n]);
    /* scipy.signal.sosfiltfilt: odd extension, padlen=27, steady-state zi. */
    for (n = 0; n < PAD; ++n)
        workspace.filter[n] = 2.0 * (double)workspace.median_b[0] - (double)workspace.median_b[PAD - n];
    for (n = 0; n < SAMPLES; ++n) workspace.filter[PAD + n] = workspace.median_b[n];
    for (n = 0; n < PAD; ++n)
        workspace.filter[PAD + SAMPLES + n] = 2.0 * (double)workspace.median_b[SAMPLES - 1] - (double)workspace.median_b[SAMPLES - 2 - n];
    sos_pass(workspace.filter);
    reverse_filter();
    sos_pass(workspace.filter);
    reverse_filter();
    for (n = 0; n < SAMPLES; ++n) {
        /* Python rounds bandpass to float32 before train_global normalization. */
        float filtered = (float)workspace.filter[PAD + n];
        normalized[n] = (float)(((double)filtered - ECG_TRAIN_MEAN[lead]) / ECG_TRAIN_STD[lead]);
        if (!isfinite(normalized[n])) return 0;
    }
    return 1;
}
int ecg_preprocess_to_int8(const float *raw_mv, int8_t *input) {
    unsigned lead, n;
    if (!raw_mv || !input) return 0;
    for (lead = 0; lead < 12; ++lead) {
        if (!ecg_preprocess_lead(raw_mv + lead * SAMPLES, lead, workspace.median_b)) return 0;
        for (n = 0; n < SAMPLES; ++n) {
            float q = nearbyintf(workspace.median_b[n] / ECG_PREPROCESS_INPUT_SCALE) + (float)ECG_PREPROCESS_INPUT_ZERO;
            q = fminf(127.0f, fmaxf(-128.0f, q));
            input[lead * SAMPLES + n] = (int8_t)q;
        }
    }
    return 1;
}
