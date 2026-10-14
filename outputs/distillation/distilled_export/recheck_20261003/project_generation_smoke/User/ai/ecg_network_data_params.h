/**
  ******************************************************************************
  * @file    ecg_network_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-10-03T14:36:40+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef ECG_NETWORK_DATA_PARAMS_H
#define ECG_NETWORK_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_ECG_NETWORK_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_ecg_network_data_weights_params[1]))
*/

#define AI_ECG_NETWORK_DATA_CONFIG               (NULL)


#define AI_ECG_NETWORK_DATA_ACTIVATIONS_SIZES \
  { 24000, }
#define AI_ECG_NETWORK_DATA_ACTIVATIONS_SIZE     (24000)
#define AI_ECG_NETWORK_DATA_ACTIVATIONS_COUNT    (1)
#define AI_ECG_NETWORK_DATA_ACTIVATION_1_SIZE    (24000)



#define AI_ECG_NETWORK_DATA_WEIGHTS_SIZES \
  { 112276, }
#define AI_ECG_NETWORK_DATA_WEIGHTS_SIZE         (112276)
#define AI_ECG_NETWORK_DATA_WEIGHTS_COUNT        (1)
#define AI_ECG_NETWORK_DATA_WEIGHT_1_SIZE        (112276)



#define AI_ECG_NETWORK_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_ecg_network_activations_table[1])

extern ai_handle g_ecg_network_activations_table[1 + 2];



#define AI_ECG_NETWORK_DATA_WEIGHTS_TABLE_GET() \
  (&g_ecg_network_weights_table[1])

extern ai_handle g_ecg_network_weights_table[1 + 2];


#endif    /* ECG_NETWORK_DATA_PARAMS_H */
