if [ ! -d "maskrcnn_benchmark/datasets/PRW-v16.04.20" ]; then
	ln -s /home/zyj/Dataset/PRW-v16.04.20 maskrcnn_benchmark/datasets/PRW-v16.04.20
fi


export NGPUS=2

if [ ! -d "models" ]; then
	mkdir models
fi

#mkdir models/prw_oim
# OIM
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/reid/prw_R_50_C4.yaml" SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 16 OUTPUT_DIR "models/prw_oim"



echo "----------------------------------------------------"
echo "************Training the part extractor*************"
echo "----------------------------------------------------"


if [ ! -d "models/prw_part" ]; then
	mkdir models/prw_part
fi
cp models/prw_oim/last_checkpoint models/prw_part/
cp models/prw_oim/model_final.pth models/prw_part/
echo "models/prw_part/model_final.pth" > models/prw_part/last_checkpoint

rm -rf build/
python setup.py build develop

# RSFE
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/reid/prw_R_50_C4_part.yaml" SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 8 OUTPUT_DIR "models/prw_part"

echo "----------------------------------------------------"
echo "************Training the part estimator*************"
echo "----------------------------------------------------"

if [ ! -d "models/prw_padreg" ]; then
	mkdir models/prw_padreg
fi
cp models/prw_part/last_checkpoint models/prw_padreg/
cp models/prw_part/model_final.pth models/prw_padreg/
echo "models/prw_padreg/model_final.pth" > models/prw_padreg/last_checkpoint

rm -rf build/
python setup.py build develop

# BBA
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/reid/prw_R_50_C4_partdone_padreg.yaml" SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 8 OUTPUT_DIR "models/prw_padreg"
