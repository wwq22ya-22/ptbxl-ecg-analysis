/* Included by the fresh project's usart.c, replacing its text-line ISR. */
#define RX_RING_SIZE 2048u
static volatile uint8_t rx_ring[RX_RING_SIZE];
static volatile uint32_t rx_head, rx_tail;
static volatile int rx_fault;
int ecg_uart_pop(uint8_t *byte) {
    uint32_t tail = rx_tail;
    if (tail == rx_head) return 0;
    *byte = rx_ring[tail]; rx_tail = (tail + 1u) & (RX_RING_SIZE - 1u); return 1;
}
int ecg_uart_fault(void) {
    int fault;
    __disable_irq(); fault = rx_fault;
    if (fault) { rx_tail = rx_head; rx_fault = 0; }
    __enable_irq(); return fault;
}
void USART1_IRQHandler(void) {
    uint32_t status = USART1->SR;
    if (status & (USART_SR_RXNE | USART_SR_ORE | USART_SR_FE | USART_SR_NE | USART_SR_PE)) {
        uint8_t byte = (uint8_t)USART1->DR;
        uint32_t next = (rx_head + 1u) & (RX_RING_SIZE - 1u);
        if (status & (USART_SR_ORE | USART_SR_FE | USART_SR_NE | USART_SR_PE)) rx_fault = 1;
        else if (status & USART_SR_RXNE) {
            if (next == rx_tail) rx_fault = 1;
            else { rx_ring[rx_head] = byte; rx_head = next; }
        }
    }
}
