#ifndef __USART_H
#define __USART_H
#include "stm32f4xx.h"
#include "stdint.h"
#include "stdio.h"

#define USART_REC_LEN 4096
extern u8  USART_RX_BUF[USART_REC_LEN]; //���ջ���,���USART_REC_LEN���ֽ�.ĩ�ֽ�Ϊ���з� 
extern volatile u16 USART_RX_STA;         		//����״̬���	

void uart_init(u32 bound);
void USART1_IRQHandler(void) ;               	//����1�жϷ������
#endif


