#! /bin/bash

if [ $# -lt 1 ]; then
    NUM="4"
else
    NUM=$1
fi

if [ $NUM -eq 6 ]; then
    # This is 6 videos, into a 3x2 formation, at half resolution
    ffmpeg -i rollout_0.mp4 -i rollout_1.mp4 -i rollout_2.mp4 -i rollout_3.mp4 -i rollout_4.mp4 -i rollout_5.mp4 -filter_complex "nullsrc=size=1024x764 [base]; [0:v] setpts=PTS-STARTPTS, scale=512x256 [upperleft]; [1:v] setpts=PTS-STARTPTS, scale=512x256 [middleleft]; [2:v] setpts=PTS-STARTPTS, scale=512x256 [lowerleft]; [3:v] setpts=PTS-STARTPTS, scale=512x256 [upperright]; [4:v] setpts=PTS-STARTPTS, scale=512x256 [middleright];[5:v] setpts=PTS-STARTPTS, scale=512x256 [lowerright]; [base][upperleft] overlay=shortest=1 [tmp0]; [tmp0][middleleft] overlay=shortest=1:y=256 [tmp1]; [tmp1][lowerleft] overlay=shortest=1:y=512 [tmp2]; [tmp2][upperright] overlay=shortest=1:x=512 [tmp3]; [tmp3][middleright] overlay=shortest=1:x=512:y=256 [tmp4]; [tmp4][lowerright] overlay=shortest=1:x=512:y=512" -c:v libx264 -y output.mp4
elif [ $NUM -eq 4 ]; then
    ffmpeg -i rollout_0.mp4 -i rollout_1.mp4 -i rollout_2.mp4 -i rollout_3.mp4 -filter_complex "nullsrc=size=1024x512 [base]; [0:v] setpts=PTS-STARTPTS, scale=512x256 [upperleft]; [1:v] setpts=PTS-STARTPTS, scale=512x256 [middleleft]; [2:v] setpts=PTS-STARTPTS, scale=512x256 [upperright]; [3:v] setpts=PTS-STARTPTS, scale=512x256 [middleright]; [base][upperleft] overlay=shortest=1 [tmp0]; [tmp0][middleleft] overlay=shortest=1:y=256 [tmp1]; [tmp1][upperright] overlay=shortest=1:x=512 [tmp3]; [tmp3][middleright] overlay=shortest=1:x=512:y=256" -c:v libx264 -y output.mp4
else
    echo "DONT KNOW HOW TO STITCH THAT MANY"
fi
