/**
  ******************************************************************************
  * @file    ecg_network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-10-03T14:36:40+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "ecg_network.h"
#include "ecg_network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_ecg_network
 
#undef AI_ECG_NETWORK_MODEL_SIGNATURE
#define AI_ECG_NETWORK_MODEL_SIGNATURE     "0x4d573327af164355881fc61b405482f0"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-10-03T14:36:40+0800"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_ECG_NETWORK_N_BATCHES
#define AI_ECG_NETWORK_N_BATCHES         (1)

static ai_ptr g_ecg_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_ecg_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  ecg_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 12000, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  ecg_Transpose_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12000, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12168, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8000, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8000, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8256, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7936, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3968, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _pool_GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 5, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5760, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14336, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40960, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 49152, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 640, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 5, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6288, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8448, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 153, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_0_features_0_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07849705219268799f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_0_features_0_2_Relu_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2008364498615265f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_0_features_0_2_Relu_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07849705219268799f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_0_features_0_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007860472542233765f, 0.0009523332701064646f, 0.0017498944653198123f, 0.0009232732700183988f, 0.0023506616707891226f, 0.0007751870434731245f, 0.0017640634905546904f, 0.0034570018760859966f, 0.0010202117264270782f, 0.0006682389066554606f, 0.0010551030281931162f, 0.001311505795456469f, 0.001216635457240045f, 0.0010396282887086272f, 0.0007185094873420894f, 0.0019593338947743177f, 0.0017750844126567245f, 0.001295924186706543f, 0.0014112204080447555f, 0.0010100086219608784f, 0.0010714514646679163f, 0.0015890791546553373f, 0.0020354653242975473f, 0.0017862380482256413f, 0.0005530036287382245f, 0.0019199217204004526f, 0.0013006365625187755f, 0.0007690562633797526f, 0.0012230586726218462f, 0.0018603663193061948f, 0.00121885200496763f, 0.0016755389515310526f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_1_features_1_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09007448703050613f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_1_features_1_2_Relu_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07849705219268799f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_1_features_1_2_Relu_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09007448703050613f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_1_features_1_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0014987654285505414f, 0.0016391758108511567f, 0.0011325109517201781f, 0.0012908527860417962f, 0.0024633617140352726f, 0.0015653364825993776f, 0.0019026421941816807f, 0.0022058163303881884f, 0.00180481665302068f, 0.0017395184841006994f, 0.0013762026792392135f, 0.0010680723935365677f, 0.0012330978643149137f, 0.0018240011995658278f, 0.0018653097795322537f, 0.0023970173206180334f, 0.0015040802536532283f, 0.002673681592568755f, 0.0012872680090367794f, 0.0012843600707128644f, 0.0028088227845728397f, 0.0020689689554274082f, 0.0022431588731706142f, 0.0028376805130392313f, 0.0013239661930128932f, 0.0022580528166145086f, 0.002108433283865452f, 0.0035877805203199387f, 0.001815019641071558f, 0.0017067274311557412f, 0.0014186182525008917f, 0.0014356679748743773f, 0.0011945986188948154f, 0.0021588916424661875f, 0.00257489993236959f, 0.001544193015433848f, 0.0014297693269327283f, 0.0020485990680754185f, 0.0018490080256015062f, 0.0015233285957947373f, 0.0031455333810299635f, 0.0015432051150128245f, 0.0024428570177406073f, 0.002043617656454444f, 0.0015007234178483486f, 0.002175740897655487f, 0.0023117228411138058f, 0.001855496782809496f, 0.0015869785565882921f, 0.0012695254990831017f, 0.0016216333024203777f, 0.00175845914054662f, 0.0013875499134883285f, 0.00245688040740788f, 0.0019198119407519698f, 0.0037672885227948427f, 0.0013945867540314794f, 0.001706837909296155f, 0.0014096972299739718f, 0.0018550991080701351f, 0.0022892197594046593f, 0.0022230688482522964f, 0.0014714765129610896f, 0.0027602133341133595f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_2_features_2_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.034804072231054306f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_2_features_2_2_Relu_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09007448703050613f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_2_features_2_2_Relu_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.034804072231054306f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_2_features_2_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017393530579283834f, 0.001693917904049158f, 0.0017716551665216684f, 0.0017825172981247306f, 0.0025092903524637222f, 0.0017531627090647817f, 0.001940308604389429f, 0.0021381862461566925f, 0.001899084192700684f, 0.002459464129060507f, 0.002443933393806219f, 0.003647448029369116f, 0.001405220478773117f, 0.002611246658489108f, 0.0018945545889437199f, 0.0024335598573088646f, 0.0021366800647228956f, 0.0018588576931506395f, 0.0026267284993082285f, 0.002081032609567046f, 0.002654736628755927f, 0.001794306794181466f, 0.0017823901725932956f, 0.0025570213329046965f, 0.0016053440049290657f, 0.0013112513115629554f, 0.0017098867101594806f, 0.002041027881205082f, 0.0018658009357750416f, 0.0027318906504660845f, 0.0018228784902021289f, 0.0015434019733220339f, 0.0013047736138105392f, 0.0016918308101594448f, 0.00244954414665699f, 0.0019062473438680172f, 0.0023552333004772663f, 0.002831720281392336f, 0.0019656361546367407f, 0.001757725840434432f, 0.0017980339471250772f, 0.001732790027745068f, 0.0016213159542530775f, 0.0018423936562612653f, 0.002427077619358897f, 0.0024342078249901533f, 0.0018179926555603743f, 0.002945608226582408f, 0.001661369577050209f, 0.002155203837901354f, 0.0017003763932734728f, 0.0024259304627776146f, 0.003161417553201318f, 0.0021978330332785845f, 0.0029125178698450327f, 0.0021241481881588697f, 0.0018591397674754262f, 0.00231439177878201f, 0.002366001019254327f, 0.0016784202307462692f, 0.0018716517370194197f, 0.0016022206982597709f, 0.002377367578446865f, 0.001417536404915154f, 0.0019332884112372994f, 0.001792392460629344f, 0.0013950139982625842f, 0.0018772619077935815f, 0.001946986303664744f, 0.002412034897133708f, 0.0022118554916232824f, 0.0016078376211225986f, 0.0015243868110701442f, 0.0022125830873847008f, 0.001749983406625688f, 0.001812894712202251f, 0.0025075434241443872f, 0.0015276814810931683f, 0.0019357482669875026f, 0.001622047508135438f, 0.002062735380604863f, 0.0019566183909773827f, 0.0015118269948288798f, 0.002155599882826209f, 0.0023349463008344173f, 0.0017466866411268711f, 0.0019945832900702953f, 0.0024543721228837967f, 0.0021154091227799654f, 0.0018347142031416297f, 0.0013804201735183597f, 0.001837861374951899f, 0.0020738118328154087f, 0.0026021553203463554f, 0.001851959154009819f, 0.0028742486611008644f, 0.0017611575312912464f, 0.00343881594017148f, 0.0012565577635541558f, 0.0016196074429899454f, 0.0025457569863647223f, 0.0015720858937129378f, 0.0020738434977829456f, 0.002960068406537175f, 0.0020179597195237875f, 0.001716351485811174f, 0.002045529894530773f, 0.002698351629078388f, 0.0019046268425881863f, 0.001622152398340404f, 0.002186096739023924f, 0.002165332669392228f, 0.0022216516081243753f, 0.0017452915199100971f, 0.001821943442337215f, 0.0014349151169881225f, 0.0019235318759456277f, 0.0018469219794496894f, 0.0024601442273706198f, 0.002360381418839097f, 0.0020500225946307182f, 0.0021497472189366817f, 0.001971213147044182f, 0.002103165490552783f, 0.0020488817244768143f, 0.0024678027257323265f, 0.0027573422994464636f, 0.002330929506570101f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_3_features_3_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033715471625328064f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_3_features_3_2_Relu_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.034804072231054306f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_3_features_3_2_Relu_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033715471625328064f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_3_features_3_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0021943130996078253f, 0.0015887751942500472f, 0.0017525117145851254f, 0.0013784898910671473f, 0.0013915540184825659f, 0.002133510774001479f, 0.0020067370496690273f, 0.002435792703181505f, 0.0023543518036603928f, 0.0014735215809196234f, 0.00199244380928576f, 0.0016001329058781266f, 0.0012944018235430121f, 0.00161577423568815f, 0.0014937338419258595f, 0.0016402722103521228f, 0.0016080621862784028f, 0.0013713851803913713f, 0.0018181907944381237f, 0.0015802002744749188f, 0.0013194751227274537f, 0.0014663080219179392f, 0.0017133879009634256f, 0.0016616194043308496f, 0.001327612902969122f, 0.0016947259427979589f, 0.001790711423382163f, 0.0015740542439743876f, 0.0017313103890046477f, 0.0017723578494042158f, 0.0015257844934239984f, 0.00212808302603662f, 0.0018249540589749813f, 0.0015432647196576f, 0.0016771235968917608f, 0.0018697184277698398f, 0.001592727261595428f, 0.0015323556726798415f, 0.0018808150198310614f, 0.0014669790398329496f, 0.0018646807875484228f, 0.0020810002461075783f, 0.0014599676942452788f, 0.0016088290140032768f, 0.001891105086542666f, 0.001550813321955502f, 0.0015047000488266349f, 0.001807601423934102f, 0.0015948913060128689f, 0.0020993000362068415f, 0.002038425300270319f, 0.0018585034413263202f, 0.0017173768719658256f, 0.0026258083526045084f, 0.001704207737930119f, 0.002557438565418124f, 0.0015522041358053684f, 0.0012570105027407408f, 0.0013719935668632388f, 0.0015101676108315587f, 0.002163857687264681f, 0.0016477727331221104f, 0.001668185112066567f, 0.0014191020745784044f, 0.0024012126959860325f, 0.001366192358545959f, 0.0020538275130093098f, 0.0030731495935469866f, 0.001714852754957974f, 0.0012574344873428345f, 0.002408142201602459f, 0.0015281549422070384f, 0.0020973393693566322f, 0.0016202825354412198f, 0.0015933047980070114f, 0.003097914857789874f, 0.0014808331616222858f, 0.0016435632715001702f, 0.0015465555479750037f, 0.001994614489376545f, 0.0017453241162002087f, 0.0014085480943322182f, 0.0015886977780610323f, 0.002124902792274952f, 0.0016373989637941122f, 0.001234055613167584f, 0.001854118425399065f, 0.001732101198285818f, 0.0016951916040852666f, 0.0015188258839771152f, 0.001373962382785976f, 0.001594320172443986f, 0.001710425247438252f, 0.0019405479542911053f, 0.001880154013633728f, 0.001526975422166288f, 0.0022515940945595503f, 0.0016220753313973546f, 0.001921280287206173f, 0.0017625043401494622f, 0.001866213628090918f, 0.0017386202234774828f, 0.0013040931662544608f, 0.0015129651874303818f, 0.0021225677337497473f, 0.0023857441265136003f, 0.0019847978837788105f, 0.0017467867583036423f, 0.0020312892738729715f, 0.0019228076562285423f, 0.001600910909473896f, 0.0016768630594015121f, 0.0018116900464519858f, 0.0015322658000513911f, 0.0013134273467585444f, 0.0016225670697167516f, 0.0021165485959500074f, 0.001343216747045517f, 0.0024524873588234186f, 0.0017901335377246141f, 0.0015643095830455422f, 0.0015963782789185643f, 0.0019381535239517689f, 0.0015674388268962502f, 0.001757779042236507f, 0.0016148455906659365f, 0.0013085890095680952f, 0.001944344723597169f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_pool_GlobalAveragePool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013872058130800724f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(ecg_Transpose_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2008364498615265f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(ecg_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2008364498615265f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08291391283273697f),
    AI_PACK_INTQ_ZP(27)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 5,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002510672202333808f, 0.0023800814524292946f, 0.0022254232317209244f, 0.0022969304118305445f, 0.0023309793323278427f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_features_features_0_features_0_2_Relu_output_0_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 250), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_features_features_0_features_0_2_Relu_output_0_output_array, &_features_features_0_features_0_2_Relu_output_0_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_pad_before_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 1, 1014), AI_STRIDE_INIT(4, 1, 1, 12, 12),
  1, &_features_features_0_features_0_2_Relu_output_0_pad_before_output_array, &_features_features_0_features_0_2_Relu_output_0_pad_before_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_scratch0, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 6288, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6288, 6288),
  1, &_features_features_0_features_0_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_scratch1, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 2), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_features_features_0_features_0_2_Relu_output_0_scratch1_array, &_features_features_0_features_0_2_Relu_output_0_scratch1_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_weights, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 12, 1, 15, 32), AI_STRIDE_INIT(4, 1, 12, 384, 384),
  1, &_features_features_0_features_0_2_Relu_output_0_weights_array, &_features_features_0_features_0_2_Relu_output_0_weights_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_bias, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_features_features_1_features_1_2_Relu_output_0_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_output, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 125), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_features_features_1_features_1_2_Relu_output_0_output_array, &_features_features_1_features_1_2_Relu_output_0_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_pad_before_output, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 256), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_features_features_1_features_1_2_Relu_output_0_pad_before_output_array, &_features_features_1_features_1_2_Relu_output_0_pad_before_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_scratch0, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 6912, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6912, 6912),
  1, &_features_features_1_features_1_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_scratch1, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 2), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_features_features_1_features_1_2_Relu_output_0_scratch1_array, &_features_features_1_features_1_2_Relu_output_0_scratch1_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_weights, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 32, 1, 7, 64), AI_STRIDE_INIT(4, 1, 32, 2048, 2048),
  1, &_features_features_1_features_1_2_Relu_output_0_weights_array, &_features_features_1_features_1_2_Relu_output_0_weights_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_features_features_2_features_2_2_Relu_output_0_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 62), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &_features_features_2_features_2_2_Relu_output_0_output_array, &_features_features_2_features_2_2_Relu_output_0_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_pad_before_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 129), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_features_features_2_features_2_2_Relu_output_0_pad_before_output_array, &_features_features_2_features_2_2_Relu_output_0_pad_before_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_scratch0, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 8192, 1, 1), AI_STRIDE_INIT(4, 1, 1, 8192, 8192),
  1, &_features_features_2_features_2_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_scratch1, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 2), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &_features_features_2_features_2_2_Relu_output_0_scratch1_array, &_features_features_2_features_2_2_Relu_output_0_scratch1_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_weights, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 5, 128), AI_STRIDE_INIT(4, 1, 64, 8192, 8192),
  1, &_features_features_2_features_2_2_Relu_output_0_weights_array, &_features_features_2_features_2_2_Relu_output_0_weights_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_features_features_3_features_3_2_Relu_output_0_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 31), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &_features_features_3_features_3_2_Relu_output_0_output_array, &_features_features_3_features_3_2_Relu_output_0_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_pad_before_output, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 64), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &_features_features_3_features_3_2_Relu_output_0_pad_before_output_array, &_features_features_3_features_3_2_Relu_output_0_pad_before_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_scratch0, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 8448, 1, 1), AI_STRIDE_INIT(4, 1, 1, 8448, 8448),
  1, &_features_features_3_features_3_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_scratch1, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 2), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &_features_features_3_features_3_2_Relu_output_0_scratch1_array, &_features_features_3_features_3_2_Relu_output_0_scratch1_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_weights, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 128, 1, 3, 128), AI_STRIDE_INIT(4, 1, 128, 16384, 16384),
  1, &_features_features_3_features_3_2_Relu_output_0_weights_array, &_features_features_3_features_3_2_Relu_output_0_weights_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _pool_GlobalAveragePool_output_0_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &_pool_GlobalAveragePool_output_0_output_array, &_pool_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  ecg_Transpose_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 12, 1, 1000), AI_STRIDE_INIT(4, 1, 1, 12, 12),
  1, &ecg_Transpose_output_array, &ecg_Transpose_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  ecg_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 1000, 1, 12), AI_STRIDE_INIT(4, 1, 1, 1000, 1000),
  1, &ecg_output_array, &ecg_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_0_conversion_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &logits_QuantizeLinear_Input_0_conversion_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &logits_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 1, 1, 5, 5),
  1, &logits_QuantizeLinear_Input_output_array, &logits_QuantizeLinear_Input_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 153, 1, 1), AI_STRIDE_INIT(4, 2, 2, 306, 306),
  1, &logits_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights, AI_STATIC,
  31, 0x1,
  AI_SHAPE_INIT(4, 128, 5, 1, 1), AI_STRIDE_INIT(4, 1, 128, 640, 640),
  1, &logits_QuantizeLinear_Input_weights_array, &logits_QuantizeLinear_Input_weights_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  logits_QuantizeLinear_Input_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  logits_QuantizeLinear_Input_0_conversion_layer, 43,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &logits_QuantizeLinear_Input_0_conversion_chain,
  NULL, &logits_QuantizeLinear_Input_0_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  logits_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &logits_QuantizeLinear_Input_weights, &logits_QuantizeLinear_Input_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  logits_QuantizeLinear_Input_layer, 43,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &logits_QuantizeLinear_Input_chain,
  NULL, &logits_QuantizeLinear_Input_0_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _pool_GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_features_3_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _pool_GlobalAveragePool_output_0_layer, 37,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_pool_GlobalAveragePool_output_0_chain,
  NULL, &logits_QuantizeLinear_Input_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 31), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 31), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_features_3_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_features_3_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_3_features_3_2_Relu_output_0_weights, &_features_features_3_features_3_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_3_features_3_2_Relu_output_0_scratch0, &_features_features_3_features_3_2_Relu_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_layer, 34,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_deep_sssa8_ch_nl_pool,
  &_features_features_3_features_3_2_Relu_output_0_chain,
  NULL, &_pool_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _features_features_3_features_3_2_Relu_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _features_features_3_features_3_2_Relu_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _features_features_3_features_3_2_Relu_output_0_pad_before_value_data, _features_features_3_features_3_2_Relu_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_features_2_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_features_3_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_pad_before_layer, 31,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_features_features_3_features_3_2_Relu_output_0_pad_before_chain,
  NULL, &_features_features_3_features_3_2_Relu_output_0_layer, AI_STATIC, 
  .value = &_features_features_3_features_3_2_Relu_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_features_2_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_features_2_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_2_features_2_2_Relu_output_0_weights, &_features_features_2_features_2_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_2_features_2_2_Relu_output_0_scratch0, &_features_features_2_features_2_2_Relu_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_layer, 28,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_deep_sssa8_ch_nl_pool,
  &_features_features_2_features_2_2_Relu_output_0_chain,
  NULL, &_features_features_3_features_3_2_Relu_output_0_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _features_features_2_features_2_2_Relu_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _features_features_2_features_2_2_Relu_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _features_features_2_features_2_2_Relu_output_0_pad_before_value_data, _features_features_2_features_2_2_Relu_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_features_1_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_features_2_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_pad_before_layer, 25,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_features_features_2_features_2_2_Relu_output_0_pad_before_chain,
  NULL, &_features_features_2_features_2_2_Relu_output_0_layer, AI_STATIC, 
  .value = &_features_features_2_features_2_2_Relu_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 2, 0, 2, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_features_1_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_features_1_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_1_features_1_2_Relu_output_0_weights, &_features_features_1_features_1_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_1_features_1_2_Relu_output_0_scratch0, &_features_features_1_features_1_2_Relu_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_layer, 22,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_deep_sssa8_ch_nl_pool,
  &_features_features_1_features_1_2_Relu_output_0_chain,
  NULL, &_features_features_2_features_2_2_Relu_output_0_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _features_features_1_features_1_2_Relu_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _features_features_1_features_1_2_Relu_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _features_features_1_features_1_2_Relu_output_0_pad_before_value_data, _features_features_1_features_1_2_Relu_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_features_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_features_1_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_pad_before_layer, 19,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_features_features_1_features_1_2_Relu_output_0_pad_before_chain,
  NULL, &_features_features_1_features_1_2_Relu_output_0_layer, AI_STATIC, 
  .value = &_features_features_1_features_1_2_Relu_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 3, 0, 3, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_features_0_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_features_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_0_features_0_2_Relu_output_0_weights, &_features_features_0_features_0_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_0_features_0_2_Relu_output_0_scratch0, &_features_features_0_features_0_2_Relu_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_layer, 16,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_sssa8_ch_nl_pool,
  &_features_features_0_features_0_2_Relu_output_0_chain,
  NULL, &_features_features_1_features_1_2_Relu_output_0_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _features_features_0_features_0_2_Relu_output_0_pad_before_value_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _features_features_0_features_0_2_Relu_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _features_features_0_features_0_2_Relu_output_0_pad_before_value_data, _features_features_0_features_0_2_Relu_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &ecg_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_features_0_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_pad_before_layer, 13,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_features_features_0_features_0_2_Relu_output_0_pad_before_chain,
  NULL, &_features_features_0_features_0_2_Relu_output_0_layer, AI_STATIC, 
  .value = &_features_features_0_features_0_2_Relu_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 7, 0, 7, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  ecg_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &ecg_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &ecg_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  ecg_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &ecg_Transpose_chain,
  NULL, &_features_features_0_features_0_2_Relu_output_0_pad_before_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 112276, 1, 1),
    112276, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 24000, 1, 1),
    24000, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ECG_NETWORK_IN_NUM, &ecg_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ECG_NETWORK_OUT_NUM, &logits_QuantizeLinear_Input_0_conversion_output),
  &ecg_Transpose_layer, 0x3ab1ef3e, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 112276, 1, 1),
      112276, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 24000, 1, 1),
      24000, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ECG_NETWORK_IN_NUM, &ecg_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ECG_NETWORK_OUT_NUM, &logits_QuantizeLinear_Input_0_conversion_output),
  &ecg_Transpose_layer, 0x3ab1ef3e, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool ecg_network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_ecg_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    ecg_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 0);
    ecg_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 0);
    ecg_Transpose_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 12000);
    ecg_Transpose_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 12000);
    _features_features_0_features_0_2_Relu_output_0_pad_before_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 11832);
    _features_features_0_features_0_2_Relu_output_0_pad_before_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 11832);
    _features_features_0_features_0_2_Relu_output_0_scratch0_array.data = AI_PTR(g_ecg_network_activations_map[0] + 0);
    _features_features_0_features_0_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 0);
    _features_features_0_features_0_2_Relu_output_0_scratch1_array.data = AI_PTR(g_ecg_network_activations_map[0] + 6288);
    _features_features_0_features_0_2_Relu_output_0_scratch1_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 6288);
    _features_features_0_features_0_2_Relu_output_0_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 11800);
    _features_features_0_features_0_2_Relu_output_0_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 11800);
    _features_features_1_features_1_2_Relu_output_0_pad_before_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 0);
    _features_features_1_features_1_2_Relu_output_0_pad_before_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 0);
    _features_features_1_features_1_2_Relu_output_0_scratch0_array.data = AI_PTR(g_ecg_network_activations_map[0] + 8192);
    _features_features_1_features_1_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 8192);
    _features_features_1_features_1_2_Relu_output_0_scratch1_array.data = AI_PTR(g_ecg_network_activations_map[0] + 15104);
    _features_features_1_features_1_2_Relu_output_0_scratch1_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 15104);
    _features_features_1_features_1_2_Relu_output_0_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 15232);
    _features_features_1_features_1_2_Relu_output_0_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 15232);
    _features_features_2_features_2_2_Relu_output_0_pad_before_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 6976);
    _features_features_2_features_2_2_Relu_output_0_pad_before_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 6976);
    _features_features_2_features_2_2_Relu_output_0_scratch0_array.data = AI_PTR(g_ecg_network_activations_map[0] + 15232);
    _features_features_2_features_2_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 15232);
    _features_features_2_features_2_2_Relu_output_0_scratch1_array.data = AI_PTR(g_ecg_network_activations_map[0] + 0);
    _features_features_2_features_2_2_Relu_output_0_scratch1_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 0);
    _features_features_2_features_2_2_Relu_output_0_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 6848);
    _features_features_2_features_2_2_Relu_output_0_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 6848);
    _features_features_3_features_3_2_Relu_output_0_pad_before_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 14784);
    _features_features_3_features_3_2_Relu_output_0_pad_before_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 14784);
    _features_features_3_features_3_2_Relu_output_0_scratch0_array.data = AI_PTR(g_ecg_network_activations_map[0] + 0);
    _features_features_3_features_3_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 0);
    _features_features_3_features_3_2_Relu_output_0_scratch1_array.data = AI_PTR(g_ecg_network_activations_map[0] + 8448);
    _features_features_3_features_3_2_Relu_output_0_scratch1_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 8448);
    _features_features_3_features_3_2_Relu_output_0_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 8704);
    _features_features_3_features_3_2_Relu_output_0_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 8704);
    _pool_GlobalAveragePool_output_0_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 0);
    _pool_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 0);
    logits_QuantizeLinear_Input_scratch0_array.data = AI_PTR(g_ecg_network_activations_map[0] + 128);
    logits_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 128);
    logits_QuantizeLinear_Input_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 436);
    logits_QuantizeLinear_Input_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 436);
    logits_QuantizeLinear_Input_0_conversion_output_array.data = AI_PTR(g_ecg_network_activations_map[0] + 0);
    logits_QuantizeLinear_Input_0_conversion_output_array.data_start = AI_PTR(g_ecg_network_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool ecg_network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_ecg_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _features_features_0_features_0_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_0_features_0_2_Relu_output_0_weights_array.data = AI_PTR(g_ecg_network_weights_map[0] + 0);
    _features_features_0_features_0_2_Relu_output_0_weights_array.data_start = AI_PTR(g_ecg_network_weights_map[0] + 0);
    _features_features_0_features_0_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_0_features_0_2_Relu_output_0_bias_array.data = AI_PTR(g_ecg_network_weights_map[0] + 5760);
    _features_features_0_features_0_2_Relu_output_0_bias_array.data_start = AI_PTR(g_ecg_network_weights_map[0] + 5760);
    _features_features_1_features_1_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_1_features_1_2_Relu_output_0_weights_array.data = AI_PTR(g_ecg_network_weights_map[0] + 5888);
    _features_features_1_features_1_2_Relu_output_0_weights_array.data_start = AI_PTR(g_ecg_network_weights_map[0] + 5888);
    _features_features_1_features_1_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_1_features_1_2_Relu_output_0_bias_array.data = AI_PTR(g_ecg_network_weights_map[0] + 20224);
    _features_features_1_features_1_2_Relu_output_0_bias_array.data_start = AI_PTR(g_ecg_network_weights_map[0] + 20224);
    _features_features_2_features_2_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_2_features_2_2_Relu_output_0_weights_array.data = AI_PTR(g_ecg_network_weights_map[0] + 20480);
    _features_features_2_features_2_2_Relu_output_0_weights_array.data_start = AI_PTR(g_ecg_network_weights_map[0] + 20480);
    _features_features_2_features_2_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_2_features_2_2_Relu_output_0_bias_array.data = AI_PTR(g_ecg_network_weights_map[0] + 61440);
    _features_features_2_features_2_2_Relu_output_0_bias_array.data_start = AI_PTR(g_ecg_network_weights_map[0] + 61440);
    _features_features_3_features_3_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_3_features_3_2_Relu_output_0_weights_array.data = AI_PTR(g_ecg_network_weights_map[0] + 61952);
    _features_features_3_features_3_2_Relu_output_0_weights_array.data_start = AI_PTR(g_ecg_network_weights_map[0] + 61952);
    _features_features_3_features_3_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_3_features_3_2_Relu_output_0_bias_array.data = AI_PTR(g_ecg_network_weights_map[0] + 111104);
    _features_features_3_features_3_2_Relu_output_0_bias_array.data_start = AI_PTR(g_ecg_network_weights_map[0] + 111104);
    logits_QuantizeLinear_Input_weights_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_weights_array.data = AI_PTR(g_ecg_network_weights_map[0] + 111616);
    logits_QuantizeLinear_Input_weights_array.data_start = AI_PTR(g_ecg_network_weights_map[0] + 111616);
    logits_QuantizeLinear_Input_bias_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_bias_array.data = AI_PTR(g_ecg_network_weights_map[0] + 112256);
    logits_QuantizeLinear_Input_bias_array.data_start = AI_PTR(g_ecg_network_weights_map[0] + 112256);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_ecg_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_ECG_NETWORK_MODEL_NAME,
      .model_signature   = AI_ECG_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 14698207,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x3ab1ef3e,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_ecg_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_ECG_NETWORK_MODEL_NAME,
      .model_signature   = AI_ECG_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 14698207,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x3ab1ef3e,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_ecg_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_ecg_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_ecg_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_ecg_network_create(network, AI_ECG_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_ecg_network_data_params_get(&params) != true) {
    err = ai_ecg_network_get_error(*network);
    return err;
  }
#if defined(AI_ECG_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_ECG_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_ecg_network_init(*network, &params) != true) {
    err = ai_ecg_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_ecg_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_ecg_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_ecg_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_ecg_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= ecg_network_configure_weights(net_ctx, params);
  ok &= ecg_network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_ecg_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_ecg_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_ECG_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

