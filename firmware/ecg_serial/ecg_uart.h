#ifndef ECG_UART_H
#define ECG_UART_H
#include "stm32f4xx_hal.h"
int ECG_UART_Init(UART_HandleTypeDef *uart);
void ECG_UART_Poll(void);
#endif
