# Deepfake-Disruption

### image face swapping defense against adversarial attack
```
python adversarial_swap.py --crop_size 224 --name people --Arc_path ./SimSwap/arcface_model/arcface_checkpoint.tar --pic_a_path ./SimSwap/crop_224/6.jpg --pic_b_path ./SimSwap/crop_224/trump.jpg --output_path output/
```

### video face swapping defense against adversarial attack
```
cd SimSwap
```

```
python test_video_swapspecific.py --crop_size 224 --use_mask --pic_specific_path ./demo_file/specific1.png --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/Iron_man.jpg --video_path ./demo_file/multi_people_1080p.mp4 --output_path ./output/multi_test_specific.mp4 --temp_path ./temp_results --isattack 1
```
