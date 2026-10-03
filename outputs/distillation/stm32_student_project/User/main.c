/* Replacement main.c for the user's STM32Test standard-peripheral-library project. */
#include "stm32f4xx.h"
#include "delay.h"
#include "usart.h"
#include "ecg_protocol.h"
#include "ecg_tick.h"
#include "ecg_deployment_config.h"
#include "ecg_network.h"
#include "ecg_network_data.h"
#include <stdio.h>
#include <string.h>

static AI_ALIGNED(32) ai_u8 activations[AI_ECG_NETWORK_DATA_ACTIVATIONS_SIZE];
static ai_handle network = AI_HANDLE_NULL;
static ai_buffer *inputs, *outputs;
static ecg_protocol protocol;
static char line[ECG_LINE_CAPACITY];
static volatile uint32_t milliseconds;

void ecg_tick_1ms(void) { ++milliseconds; }

static void transmit(const char *message, void *context) {
    (void)context;
    while (*message) {
        while (USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET) {}
        USART_SendData(USART1, (uint8_t)*message++);
    }
}

static int infer(float logits[5], uint32_t *elapsed, void *context) {
    uint32_t start = milliseconds;
    ai_i32 batches;
    (void)context;
    batches = ai_ecg_network_run(network, inputs, outputs);
    *elapsed = milliseconds - start;
    if (batches != 1) return 0;
    memcpy(logits, outputs[0].data, sizeof(float) * 5);
    return 1;
}

static void initialize_ai(void) {
    ai_handle pools[] = { activations };
    ai_error error;
    protocol.scale = ECG_INPUT_SCALE;
    protocol.zero_point = ECG_INPUT_ZERO_POINT;
    protocol.thresholds = ECG_THRESHOLDS;
    protocol.model_sha256 = ECG_MODEL_SHA256;
    protocol.send = transmit;
    protocol.run = infer;
    if (AI_ECG_NETWORK_IN_1_FORMAT != AI_BUFFER_FORMAT_S8
        || AI_ECG_NETWORK_OUT_1_FORMAT != AI_BUFFER_FORMAT_FLOAT
        || AI_ECG_NETWORK_IN_1_SIZE != ECG_SAMPLE_COUNT
        || AI_ECG_NETWORK_OUT_1_SIZE != 5) return;
    error = ai_ecg_network_create_and_init(&network, pools, NULL);
    if (error.type != AI_ERROR_NONE) return;
    inputs = ai_ecg_network_inputs_get(network, NULL);
    outputs = ai_ecg_network_outputs_get(network, NULL);
    if (inputs && outputs && inputs[0].data && outputs[0].data)
        protocol.input = (int8_t *)inputs[0].data;
}

int main(void) {
    uint32_t last_change, last_count = 0;
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
    delay_init();
    uart_init(115200);
    SysTick_Config(SystemCoreClock / 1000u);
    initialize_ai();
    last_change = milliseconds;
    while (1) {
        uint32_t status = USART_RX_STA;
        uint32_t count = status & 0x3FFFu;
        if (count != last_count) { last_change = milliseconds; last_count = count; }
        if (status & 0x8000u) {
            if (count >= sizeof(line)) ecg_protocol_abort(&protocol, "line_too_long");
            else if (count) {
                memcpy(line, USART_RX_BUF, count); line[count] = 0;
                /* Sender waits for this reply, so this buffer is no longer needed. */
                USART_RX_STA = 0;
                ecg_protocol_line(&protocol, line);
            }
            USART_RX_STA = 0;
            last_count = 0; last_change = milliseconds;
        } else if ((protocol.active || count)
                && milliseconds - last_change > 5000u) {
            USART_RX_STA = 0; last_count = 0; last_change = milliseconds;
            ecg_protocol_abort(&protocol, "receive_timeout");
        }
    }
}
