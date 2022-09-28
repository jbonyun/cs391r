#!/bin/bash 

# Use "pip show robosuite" to find directory 
ROBOSUITE_DIR=~/anaconda3/envs/py_robosuite/lib/python3.8/site-packages/robosuite

# Leave this bash script inside the base "models" directory
MODEL_DIR=$(pwd)

# Copy files
cp $MODEL_DIR/robosuite_files/__init__.py $ROBOSUITE_DIR/models/grippers/__init__.py
cp $MODEL_DIR/robosuite_files/bat_one.py $ROBOSUITE_DIR/models/grippers/bat_one.py
cp $MODEL_DIR/robosuite_files/bat_one_gripper.xml $ROBOSUITE_DIR/models/assets/grippers/bat_one_gripper.xml

# User print out
echo Files were copied to the following robosuite directory:
echo .../robosuite/models/grippers/
echo .../robosuite/models/assets/grippers/