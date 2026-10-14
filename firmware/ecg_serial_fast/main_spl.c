#include "stm32f4xx.h"
#include "delay.h"
#include "usart.h"
#include "ecg_binary.h"
#include "ecg_tick.h"
#include "ecg_deployment_config.h"
#include "ecg_network.h"
#include "ecg_network_data.h"
#include <string.h>
static AI_ALIGNED(32) ai_u8 activations[AI_ECG_NETWORK_DATA_ACTIVATIONS_SIZE];
static ai_handle network = AI_HANDLE_NULL;
static ai_buffer *inputs, *outputs;
static ecg_binary protocol;
static volatile uint32_t milliseconds;
extern int ecg_uart_pop(uint8_t *);
extern int ecg_uart_fault(void);
void ecg_tick_1ms(void) { ++milliseconds; }
static void transmit(const uint8_t *bytes, size_t n, void *context) {
    size_t i; (void)context;
    for (i = 0; i < n; ++i) {
        while (USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET) {}
        USART_SendData(USART1, bytes[i]);
    }
}
static int infer(float logits[5], uint32_t *elapsed, void *context) {
    uint32_t start = milliseconds;
    ai_i32 batches; (void)context;
    batches = ai_ecg_network_run(network, inputs, outputs);
    *elapsed = milliseconds - start;
    if (batches != 1) return 0;
    memcpy(logits, outputs[0].data, sizeof(float) * 5); return 1;
}
static void initialize_ai(void) {
    ai_handle pools[] = { activations };
    ai_error error;
    protocol.scale = ECG_INPUT_SCALE; protocol.zero_point = ECG_INPUT_ZERO_POINT;
    protocol.thresholds = ECG_THRESHOLDS; protocol.model_sha256 = ECG_MODEL_SHA256;
    protocol.send = transmit; protocol.run = infer;
    if (AI_ECG_NETWORK_IN_1_FORMAT != AI_BUFFER_FORMAT_S8
            || AI_ECG_NETWORK_OUT_1_FORMAT != AI_BUFFER_FORMAT_FLOAT
            || AI_ECG_NETWORK_IN_1_SIZE != ECG_FAST_SAMPLES || AI_ECG_NETWORK_OUT_1_SIZE != 5) return;
    error = ai_ecg_network_create_and_init(&network, pools, NULL);
    if (error.type != AI_ERROR_NONE) return;
    inputs = ai_ecg_network_inputs_get(network, NULL); outputs = ai_ecg_network_outputs_get(network, NULL);
    if (inputs && outputs && inputs[0].data && outputs[0].data) protocol.input = (int8_t *)inputs[0].data;
}
int main(void) {
    uint32_t last_byte; uint8_t byte;
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2); delay_init(); uart_init(115200);
    SysTick_Config(SystemCoreClock / 1000u); initialize_ai(); last_byte = milliseconds;
    while (1) {
        if (ecg_uart_fault()) { ecg_binary_abort(&protocol, "uart_receive"); last_byte = milliseconds; }
        if (ecg_uart_pop(&byte)) { last_byte = milliseconds; ecg_binary_byte(&protocol, byte); }
        else if ((protocol.active || protocol.used) && milliseconds - last_byte > 5000u) {
            ecg_binary_abort(&protocol, "receive_timeout"); last_byte = milliseconds;
        }
    }
}
