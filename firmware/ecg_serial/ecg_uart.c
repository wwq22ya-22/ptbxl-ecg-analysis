/* STM32Cube HAL adapter: UART work and inference both run in the main loop. */
#include "ecg_uart.h"
#include "ecg_protocol.h"
#include "ecg_deployment_config.h"
#include "ecg_network.h"
#include "ecg_network_data.h"
#include <string.h>

static AI_ALIGNED(32) ai_u8 activations[AI_ECG_NETWORK_DATA_ACTIVATIONS_SIZE];
static ai_handle network = AI_HANDLE_NULL;
static ai_buffer *inputs, *outputs;
static UART_HandleTypeDef *serial_uart;
static ecg_protocol protocol;
static char line[ECG_LINE_CAPACITY];
static size_t line_size;
static int discarding;
static uint32_t last_byte;

static void send_line(const char *message, void *context) {
    (void)context;
    HAL_UART_Transmit(serial_uart, (uint8_t *)message, (uint16_t)strlen(message), 1000);
}

static int infer(float logits[5], uint32_t *elapsed, void *context) {
    uint32_t start = HAL_GetTick();
    ai_i32 batches;
    (void)context;
    batches = ai_ecg_network_run(network, inputs, outputs);
    *elapsed = HAL_GetTick() - start;
    if (batches != 1) return 0;
    memcpy(logits, outputs[0].data, sizeof(float) * 5);
    return 1;
}

int ECG_UART_Init(UART_HandleTypeDef *uart) {
    ai_handle pools[] = { activations };
    ai_error error;
    serial_uart = uart;
    memset(&protocol, 0, sizeof(protocol));
    protocol.scale = ECG_INPUT_SCALE;
    protocol.zero_point = ECG_INPUT_ZERO_POINT;
    protocol.thresholds = ECG_THRESHOLDS;
    protocol.model_sha256 = ECG_MODEL_SHA256;
    protocol.send = send_line;
    protocol.run = infer;
    if (AI_ECG_NETWORK_IN_1_FORMAT != AI_BUFFER_FORMAT_S8
        || AI_ECG_NETWORK_OUT_1_FORMAT != AI_BUFFER_FORMAT_FLOAT
        || AI_ECG_NETWORK_IN_1_SIZE != ECG_SAMPLE_COUNT
        || AI_ECG_NETWORK_OUT_1_SIZE != 5) return 0;
    error = ai_ecg_network_create_and_init(&network, pools, NULL);
    if (error.type != AI_ERROR_NONE) return 0;
    inputs = ai_ecg_network_inputs_get(network, NULL);
    outputs = ai_ecg_network_outputs_get(network, NULL);
    if (!inputs || !outputs || !inputs[0].data || !outputs[0].data) return 0;
    protocol.input = (int8_t *)inputs[0].data;
    line_size = 0; discarding = 0; last_byte = HAL_GetTick();
    return 1;
}

void ECG_UART_Poll(void) {
    uint8_t byte;
    unsigned budget = 256;
    while (budget-- && HAL_UART_Receive(serial_uart, &byte, 1, 1) == HAL_OK) {
        last_byte = HAL_GetTick();
        if (byte == '\r') continue;
        if (byte == '\n') {
            if (discarding) ecg_protocol_abort(&protocol, "line_too_long");
            else if (line_size) { line[line_size] = 0; ecg_protocol_line(&protocol, line); }
            line_size = 0; discarding = 0;
        } else if (line_size < sizeof(line) - 1 && !discarding) line[line_size++] = (char)byte;
        else discarding = 1;
    }
    if ((protocol.active || line_size || discarding) && HAL_GetTick() - last_byte > 5000u) {
        ecg_protocol_abort(&protocol, "receive_timeout"); line_size = 0; discarding = 0;
    }
}
