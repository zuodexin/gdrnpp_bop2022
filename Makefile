DATASET := robi

# robi
YOLOX_CONFIG := ./configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_robi_pbr_robi_bop_test.py
TRAINED_YOLOX := ./output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_robi_pbr_robi_bop_test/model_final.pth

train_yolox:
	./det/yolox/tools/train_yolox.sh  ${YOLOX_CONFIG} 0

test_yolox:
	./det/yolox/tools/test_yolox.sh  ${YOLOX_CONFIG} 0 ${TRAINED_YOLOX}


SRC_DET := ./output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_robi_pbr_robi_bop_test/inference/robi_test/coco_instances_results_bop.json
DST_DET := 	./output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_robi_pbr_robi_bop_test/inference/robi_test/robi_test_ours_results.json

convert_det:
	python ./core/gdrn_modeling/tools/robi/convert_det_to_our_format.py --coco_bop_input ${SRC_DET} --output ${DST_DET}


GDRN_CONFIG := ./configs/gdrn/robi/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_robi.py
TRAINED_GDRN := ./output/gdrn/robi/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_robi/model_0049319.pth

# tless
GDRN_CONFIG := ./configs/gdrn/tless/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tless_ge6011_full.py
TRAINED_GDRN := ./output/gdrn/tless/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tless_ge6011_full/model_final.pth

train_gdrnet:
	export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64'
	./core/gdrn_modeling/train_gdrn.sh ${GDRN_CONFIG} 0

test_gdrnet:
	./core/gdrn_modeling/test_gdrn.sh ${GDRN_CONFIG} 0 ${TRAINED_GDRN}

test_dataset:
	python -m core.gdrn_modeling.datasets.tless_pbr tless_1_train_pbr