python -m torch.distributed.launch --nproc_per_node=2 /code/train_MU4.py --training --trainset='CIFAR10' --distortion-metric='MSE' --loss-type='L1' --channel-type='awgn' --fine-tune=False  --model-mode=2 --C=48 --multiple-snr='10' --batch-size=256 --angle=10 --NOCM-deep=2 --alpha=0.01

CUDA_VISIBLE_DEVICES=7 python jpeg_ldpc_qam_cdma.py --dataset='AFHQ' --distortion-metric='MSE' --channel-type='awgn'

CUDA_VISIBLE_DEVICES=7 python jpeg_ldpc_qam_cdma.py --dataset='AFHQ' --distortion-metric='MSE' --channel-type='rayleigh'

CUDA_VISIBLE_DEVICES=7 python jpeg_ldpc_qam_cdma.py --dataset='CIFAR10' --distortion-metric='MSE' --channel-type='awgn'

CUDA_VISIBLE_DEVICES=7 python jpeg_ldpc_qam_cdma.py --dataset='CIFAR10' --distortion-metric='MSE' --channel-type='rayleigh'

CUDA_VISIBLE_DEVICES=7 python jpeg_ldpc_qam_noma.py --dataset='CIFAR10' --distortion-metric='MSE' --channel-type='awgn'

CUDA_VISIBLE_DEVICES=7 python jpeg_ldpc_qam_noma.py --dataset='AFHQ' --distortion-metric='MSE' --channel-type='awgn'

CUDA_VISIBLE_DEVICES=7 python jpeg_ldpc_bpsk_noma_inference.py --dataset='AFHQ' --distortion-metric='MSE' --channel-type='rayleigh'

CUDA_VISIBLE_DEVICES=7 python jpeg_ldpc_qam_cdma_inference.py --dataset='AFHQ' --distortion-metric='MSE' --channel-type='rayleigh'