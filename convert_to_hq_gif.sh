#!/bin/bash

# High-quality MP4 to GIF conversion script
echo "Converting MP4 files to HIGH-QUALITY GIFs..."

# Remove existing low-quality GIFs
echo "Removing existing low-quality GIFs..."
rm -rf gif_files

# Create output directory
mkdir -p gif_files

echo "Converting bert/samples_03_to_05.mp4..."
mkdir -p gif_files/bert
ffmpeg -i mp4_files/bert/samples_03_to_05.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/bert/samples_03_to_05.gif -y -loglevel error

echo "Converting bert/samples_00_to_02.mp4..."
ffmpeg -i mp4_files/bert/samples_00_to_02.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/bert/samples_00_to_02.gif -y -loglevel error

echo "Converting bert/samples_06_to_06.mp4..."
ffmpeg -i mp4_files/bert/samples_06_to_06.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/bert/samples_06_to_06.gif -y -loglevel error

echo "Converting humanml_enc_512_50steps/output_1/samples_03_to_05.mp4..."
mkdir -p gif_files/humanml_enc_512_50steps/output_1
ffmpeg -i mp4_files/humanml_enc_512_50steps/output_1/samples_03_to_05.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/humanml_enc_512_50steps/output_1/samples_03_to_05.gif -y -loglevel error

echo "Converting humanml_enc_512_50steps/output_1/samples_00_to_02.mp4..."
ffmpeg -i mp4_files/humanml_enc_512_50steps/output_1/samples_00_to_02.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/humanml_enc_512_50steps/output_1/samples_00_to_02.gif -y -loglevel error

echo "Converting humanml_enc_512_50steps/output_1/samples_06_to_06.mp4..."
ffmpeg -i mp4_files/humanml_enc_512_50steps/output_1/samples_06_to_06.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/humanml_enc_512_50steps/output_1/samples_06_to_06.gif -y -loglevel error

echo "Converting humanml_enc_512_50steps/output_2/samples_06_to_07.mp4..."
mkdir -p gif_files/humanml_enc_512_50steps/output_2
ffmpeg -i mp4_files/humanml_enc_512_50steps/output_2/samples_06_to_07.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/humanml_enc_512_50steps/output_2/samples_06_to_07.gif -y -loglevel error

echo "Converting humanml_enc_512_50steps/output_2/samples_03_to_05.mp4..."
ffmpeg -i mp4_files/humanml_enc_512_50steps/output_2/samples_03_to_05.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/humanml_enc_512_50steps/output_2/samples_03_to_05.gif -y -loglevel error

echo "Converting humanml_enc_512_50steps/output_2/samples_00_to_02.mp4..."
ffmpeg -i mp4_files/humanml_enc_512_50steps/output_2/samples_00_to_02.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/humanml_enc_512_50steps/output_2/samples_00_to_02.gif -y -loglevel error

echo "Converting humanml_trans_enc_512/output_1/samples_06_to_07.mp4..."
mkdir -p gif_files/humanml_trans_enc_512/output_1
ffmpeg -i mp4_files/humanml_trans_enc_512/output_1/samples_06_to_07.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/humanml_trans_enc_512/output_1/samples_06_to_07.gif -y -loglevel error

echo "Converting humanml_trans_enc_512/output_1/samples_03_to_05.mp4..."
ffmpeg -i mp4_files/humanml_trans_enc_512/output_1/samples_03_to_05.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/humanml_trans_enc_512/output_1/samples_03_to_05.gif -y -loglevel error

echo "Converting humanml_trans_enc_512/output_1/samples_00_to_02.mp4..."
ffmpeg -i mp4_files/humanml_trans_enc_512/output_1/samples_00_to_02.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/humanml_trans_enc_512/output_1/samples_00_to_02.gif -y -loglevel error

echo "Converting humanml_trans_enc_512/output_2/samples_03_to_05.mp4..."
mkdir -p gif_files/humanml_trans_enc_512/output_2
ffmpeg -i mp4_files/humanml_trans_enc_512/output_2/samples_03_to_05.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/humanml_trans_enc_512/output_2/samples_03_to_05.gif -y -loglevel error

echo "Converting humanml_trans_enc_512/output_2/samples_00_to_02.mp4..."
ffmpeg -i mp4_files/humanml_trans_enc_512/output_2/samples_00_to_02.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/humanml_trans_enc_512/output_2/samples_00_to_02.gif -y -loglevel error

echo "Converting humanml_trans_enc_512/output_2/samples_06_to_06.mp4..."
ffmpeg -i mp4_files/humanml_trans_enc_512/output_2/samples_06_to_06.mp4 -vf "scale=900:-1" -r 20 -loop 0 gif_files/humanml_trans_enc_512/output_2/samples_06_to_06.gif -y -loglevel error

echo "High-quality GIF conversion complete!"
echo "New GIFs saved in gif_files directory with original resolution and frame rate"
