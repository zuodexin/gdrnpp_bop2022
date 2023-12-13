DATASET := robi


YOLOX_CONFIG := ./configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_robi_pbr_robi_bop_test.py
TRAINED_MODEL := ./output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_robi_pbr_robi_bop_test/model_final.pth

train_yolox:
	./det/yolox/tools/train_yolox.sh  ${YOLOX_CONFIG} 0

test_yolox:
	./det/yolox/tools/test_yolox.sh  ${YOLOX_CONFIG} 0 ${TRAINED_MODEL}


GDRN_CONFIG := ./configs/gdrn/robi/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_robi.py

train_gdrnet:
	./core/gdrn_modeling/train_gdrn.sh ${GDRN_CONFIG} 0